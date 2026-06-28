"""gpu_run -- run GPU work on a CUDA device, never silently on MPS/CPU.

Policy (see CLAUDE.md "Compute / GPU execution"): training, profiling, and any
diagnostic that touches torch MUST run on a CUDA GPU whenever one is reachable --
a *local* NVIDIA GPU or a *configured remote* GPU host (e.g. a Tailnet 4090).
MPS/CPU is a last resort that must be requested explicitly, never a silent default.

This launcher decides WHERE to run a command and steers the device, so an agent
that "just runs the smoke" lands on the GPU instead of defaulting to local MPS:

    1. local CUDA available            -> run locally, TT_DEVICE=cuda
    2. else a remote host is reachable -> dispatch over SSH on the remote, --device cuda
    3. else                            -> REFUSE (exit 2) unless --allow-mps is given,
                                          in which case run locally with a loud warning.

It is hardware-agnostic on purpose: nothing about any one person's setup is baked
in. The remote is read from ``gpu-host.toml`` (git-ignored) or ``TT_GPU_*`` env
vars, so a teammate with a *local* NVIDIA GPU never configures a remote at all
(case 1 fires), while someone whose GPU is a separate box fills in the config.

Usage
-----
    uv run python scripts/gpu_run.py -- python -m pop_trainer.pretraining.train \\
        --trunk impala --pooling gap --resolution 180 --data datasets/decode-v2 --epochs 8

Everything after ``--`` is the command to run. Leave its device as ``--device auto``
(the default) -- this launcher sets ``TT_DEVICE`` so ``resolve_device`` picks CUDA
where it runs. Ad-hoc diagnostics should call ``resolve_device()`` (or read
``TT_DEVICE``) rather than hardcoding ``torch.device('mps')``.

Config (gpu-host.toml at the repo root, or TT_GPU_* env which take precedence)
-----
    [remote]
    host        = "host-or-tailnet-ip"   # required to enable remote dispatch
    user        = "username"              # required
    ssh_key     = "~/.ssh/id_ed25519"     # optional
    remote_dir  = "C:/path/to/repo"       # required: repo checkout on the remote
    remote_shell = "cmd"                  # "cmd" | "powershell" | "posix"

Env overrides: TT_GPU_HOST, TT_GPU_USER, TT_GPU_SSH_KEY, TT_GPU_REMOTE_DIR,
TT_GPU_REMOTE_SHELL, TT_ALLOW_MPS.

Audit log: every invocation appends one strict-JSON line to ``logs/gpu_run.jsonl``
(git-ignored) -- outcome (local-cuda / remote / mps-fallback / refused), resolved
device, remote, exit code, duration, operator. Override the path with ``TT_GPU_LOG``,
or disable with ``TT_GPU_LOG=off``. This is the audit trail for "is the team actually
using the GPU, or quietly falling back to MPS?"
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "gpu-host.toml"
# Per-machine audit trail (git-ignored under /logs). Override with TT_GPU_LOG.
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "gpu_run.jsonl"

# command separator + inline env-set syntax per remote shell
_REMOTE_SHELLS = {
    "cmd": " && ",
    "powershell": "; ",
    "posix": " && ",
}


def _warn(msg: str) -> None:
    print(f"[gpu_run] {msg}", file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC)


def _log_path() -> Path | None:
    """Resolve the audit-log path. ``TT_GPU_LOG=off`` (or empty) disables logging."""
    override = os.environ.get("TT_GPU_LOG")
    if override is not None:
        val = override.strip()
        if val == "" or val.lower() in ("off", "none", "0", "false"):
            return None
        return Path(val).expanduser()
    return DEFAULT_LOG_PATH


def log_run(record: dict) -> None:
    """Append one strict-JSON line to the audit log. Never raises (logging must not break a run)."""
    path = _log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as exc:  # logging is best-effort; surface but don't fail the run
        _warn(f"could not write audit log ({path}): {exc}")


def _run_logged(
    proc_argv: list[str],
    *,
    outcome: str,
    device: str,
    wrapped: list[str],
    allow_mps: bool,
    remote: str | None = None,
    env: dict | None = None,
) -> int:
    """Run a subprocess, timing it, and append one audit record covering the whole invocation."""
    start = _now()
    rc = subprocess.run(proc_argv, env=env).returncode
    end = _now()
    log_run(
        {
            "ts_start": start.isoformat(),
            "ts_end": end.isoformat(),
            "duration_s": round((end - start).total_seconds(), 3),
            "outcome": outcome,  # local-cuda | remote | mps-fallback
            "device": device,  # resolved TT_DEVICE the job saw
            "remote": remote,  # user@host when dispatched remotely
            "allow_mps": allow_mps,
            "exit_code": rc,
            "command": wrapped,
            "operator": f"{getpass.getuser()}@{platform.node() or socket.gethostname()}",
            "platform": platform.system(),
        }
    )
    return rc


def load_remote_config() -> dict:
    """Merge gpu-host.toml [remote] with TT_GPU_* env overrides (env wins)."""
    cfg: dict = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as fh:
            cfg = (tomllib.load(fh) or {}).get("remote", {}) or {}
    env_map = {
        "host": "TT_GPU_HOST",
        "user": "TT_GPU_USER",
        "ssh_key": "TT_GPU_SSH_KEY",
        "remote_dir": "TT_GPU_REMOTE_DIR",
        "remote_shell": "TT_GPU_REMOTE_SHELL",
    }
    for key, env in env_map.items():
        val = os.environ.get(env)
        if val:
            cfg[key] = val
    return cfg


def local_cuda_available() -> bool:
    """True iff this machine has a usable CUDA GPU. Guarded torch import."""
    try:
        import torch
    except Exception:  # torch missing/broken -> treat as no local cuda
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def remote_is_reachable(cfg: dict) -> bool:
    """Probe the configured remote over SSH (BatchMode, short timeout)."""
    host, user = cfg.get("host"), cfg.get("user")
    if not (host and user):
        return False
    if shutil.which("ssh") is None:
        _warn("ssh not found on PATH; cannot reach remote GPU.")
        return False
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if cfg.get("ssh_key"):
        argv += ["-i", os.path.expanduser(cfg["ssh_key"])]
    argv += [f"{user}@{host}", "echo gpu_run_ok"]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0 and "gpu_run_ok" in out.stdout


def build_remote_command(cfg: dict, wrapped: list[str]) -> str:
    """Compose the one-liner the remote shell runs: cd, set TT_DEVICE=cuda, then cmd."""
    shell = (cfg.get("remote_shell") or "cmd").strip().lower()
    if shell not in _REMOTE_SHELLS:
        raise SystemExit(
            f"[gpu_run] remote_shell must be one of {sorted(_REMOTE_SHELLS)}; got {shell!r}"
        )
    sep = _REMOTE_SHELLS[shell]
    remote_dir = cfg.get("remote_dir")
    if not remote_dir:
        raise SystemExit(
            "[gpu_run] remote_dir is required for remote dispatch "
            "(gpu-host.toml or TT_GPU_REMOTE_DIR)."
        )
    cmd = " ".join(_quote_for_remote(a) for a in wrapped)
    if shell == "cmd":
        parts = [f'cd /d "{remote_dir}"', "set TT_DEVICE=cuda", cmd]
    elif shell == "powershell":
        parts = [f'cd "{remote_dir}"', '$env:TT_DEVICE="cuda"', cmd]
    else:  # posix
        parts = [f'cd "{remote_dir}"', f"TT_DEVICE=cuda {cmd}"]
    return sep.join(parts)


def _quote_for_remote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "\t" in arg) else arg


def _explicit_mps_requested(wrapped: list[str]) -> bool:
    joined = " ".join(wrapped)
    return "--device mps" in joined or "device('mps')" in joined or 'device("mps")' in joined


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command on a CUDA GPU (local or remote); never silent MPS/CPU.",
        epilog="Put the command after `--`, e.g.: gpu_run.py -- python -m pop_trainer.x",
    )
    parser.add_argument(
        "--allow-mps",
        action="store_true",
        default=os.environ.get("TT_ALLOW_MPS", "").strip().lower() in ("1", "true", "yes"),
        help="Permit local MPS/CPU fallback when no CUDA GPU is reachable (otherwise refuse).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command to run>")
    args = parser.parse_args(argv)

    wrapped = args.command
    if wrapped and wrapped[0] == "--":
        wrapped = wrapped[1:]
    if not wrapped:
        parser.error("no command given; put it after `--`.")

    if _explicit_mps_requested(wrapped):
        _warn("the command explicitly requests MPS; gpu_run will still prefer CUDA where it runs.")

    # 1) local CUDA
    if local_cuda_available():
        _warn("local CUDA GPU detected -> running locally on cuda.")
        return _run_logged(
            wrapped,
            outcome="local-cuda",
            device="cuda",
            wrapped=wrapped,
            allow_mps=args.allow_mps,
            env={**os.environ, "TT_DEVICE": "cuda"},
        )

    # 2) configured remote GPU
    cfg = load_remote_config()
    if cfg.get("host") and cfg.get("user"):
        if remote_is_reachable(cfg):
            remote_cmd = build_remote_command(cfg, wrapped)
            remote = f"{cfg['user']}@{cfg['host']}"
            _warn(f"dispatching to remote GPU {remote} (cuda).")
            ssh_argv = ["ssh"]
            if cfg.get("ssh_key"):
                ssh_argv += ["-i", os.path.expanduser(cfg["ssh_key"])]
            ssh_argv += [remote, remote_cmd]
            return _run_logged(
                ssh_argv,
                outcome="remote",
                device="cuda",
                wrapped=wrapped,
                allow_mps=args.allow_mps,
                remote=remote,
            )
        _warn(f"remote GPU {cfg['user']}@{cfg['host']} configured but UNREACHABLE.")
    else:
        _warn("no remote GPU configured (gpu-host.toml / TT_GPU_*).")

    # 3) no CUDA anywhere
    if args.allow_mps:
        _warn(
            "WARNING: no CUDA reachable -- falling back to LOCAL MPS/CPU (--allow-mps). "
            "This is slow; results are for smoke only."
        )
        return _run_logged(
            wrapped,
            outcome="mps-fallback",
            device="auto",
            wrapped=wrapped,
            allow_mps=True,
            env={**os.environ, "TT_DEVICE": "auto"},
        )

    _warn("REFUSING to run: no CUDA GPU reachable and --allow-mps not set.")
    _warn(
        "Fix: configure a remote GPU (gpu-host.toml) or run on a CUDA box; "
        "or pass --allow-mps to accept MPS/CPU."
    )
    log_run(
        {
            "ts_start": _now().isoformat(),
            "ts_end": _now().isoformat(),
            "duration_s": 0.0,
            "outcome": "refused",
            "device": None,
            "remote": None,
            "allow_mps": False,
            "exit_code": 2,
            "command": wrapped,
            "operator": f"{getpass.getuser()}@{platform.node() or socket.gethostname()}",
            "platform": platform.system(),
        }
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
