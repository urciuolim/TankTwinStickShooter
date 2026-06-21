"""Stage-2 (frame, state)-pair COLLECTION harness for the pixel inverse-renderer effort.

COLLECTION ONLY — this module launches the frame-capable standalone build, drives BOTH
players with seeded random actions, and writes sharded, compressed ``(frame, state)`` pairs
to disk. It does NOT train any encoder; that is a separate downstream effort.

It deliberately MIRRORS the proven Stage-1 verify template
(``scripts/verify_pixel_capture.py``) for the launch + handshake + teardown path, and reuses
ONLY the additive, already-authorized pixel-frame wire channel
(``protocol.Connection.receive_state_and_frame``). It does NOT route through ``TankEnv`` and
does NOT touch the frozen RL seam (``DriverController`` socket/``actions``,
``GameController.UpdateState``'s 52-float layout, or ``protocol.send``/``receive``/``encode``/
``decode``). It opens its OWN socket per worker (bind-retry then connect to
``127.0.0.1:game_port``), exactly like the template.

-------------------------------------------------------------------------------------------
POLICY (CTO: RANDOM, both players)
-------------------------------------------------------------------------------------------
Each step both players get an INDEPENDENT 5-float action ``[move_x, move_y, aim_x, aim_y,
fire]``: the four analog channels ~ uniform[-1, 1] and ``fire`` a coin flip (0.0 / 1.0). All
draws come from ONE seeded ``numpy.random.default_rng`` per worker, seeded from
``np.random.SeedSequence(--seed).spawn(--workers)[worker_id]`` so the same ``--seed`` +
``--workers`` reproduces the same action stream bit-for-bit. The action message is sent as
``conn.send({1: p1_action, 2: p2_action})`` — INTEGER keys (json coerces to "1"/"2"), exactly
the protocol's step message.

-------------------------------------------------------------------------------------------
MAP ROTATION SCHEDULE
-------------------------------------------------------------------------------------------
The 10 shipped map configs (``exp-configs/maps/*.json``) are resolved with
``maps._resolve_map_rotation`` (``--maps all`` -> the sorted 10; a comma-list -> explicit
configs in given order). Each config is resolved to its ABSOLUTE arena path via
``env._arena_path_from_config`` (switch_arena takes the arena path, NOT the config path —
exactly what ``TankEnv`` does).

EVERY worker rotates over the SAME full ordered map list, round-robin, advancing to the next
map on each NEW episode (each (re)handshake). Worker ``k`` STARTS at offset ``k % len(maps)``
so the M workers don't all hammer the same map at the same instant — this spreads load and
keeps the GLOBAL per-map distribution even (~target/10 per map). To bound per-map balance, a
worker SKIPS (does not collect on) any map whose GLOBAL quota is already met: each worker is
given a per-map quota ``ceil(sub_target / num_maps)`` and, when advancing the rotation, skips
maps it has already filled, so the final per-map counts land within ~one episode of even. The
last partial episode may slightly overshoot a single map's quota (we never discard an
in-flight pair), which is acceptable (Director: "none wildly starved").

A "done" state (the state dict contains the ``"done"`` key, set by the build at round end /
game_maxTime) ends the current episode: the final pair for that step is still recorded, then
the worker re-handshakes for the next episode (advance rotation -> switch_arena -> restart ->
start -> read first state+frame). A fresh, monotonically-increasing ``episode_id`` is assigned
per (re)handshake; ``step_idx`` resets to 0 each episode.

-------------------------------------------------------------------------------------------
HANDSHAKE ORDER (mirrors TankEnv.reset / TankEnv._switch_arena)
-------------------------------------------------------------------------------------------
To start an episode on map M (Unity must be ``!ingame`` so it can swap while loading):
  1. ``conn.send({"switch_arena": <abs arena path>})`` ; ``conn.receive()`` -> assert
     ``arena_switched`` truthy.   (BEFORE restart — takes effect on the LoadScene restart triggers.)
  2. ``conn.send({"restart": True})`` ; ``conn.receive()``  (restart ack)
  3. ``conn.send({"start": True})``   ; ``conn.receive()``  -> assert ``"starting" in ack``
  4. ``conn.receive_state_and_frame()`` -> the episode's FIRST (state, frame). NOT ``receive``:
     the first state has a frame glued on (the coalesced unframed-recv landmine).
Each subsequent step: ``conn.send({1: a1, 2: a2})`` then ``receive_state_and_frame()``.

-------------------------------------------------------------------------------------------
PARALLELISM
-------------------------------------------------------------------------------------------
``--workers M`` processes via ``multiprocessing.get_context("spawn")`` (NEVER fork). Each
worker = its own Unity instance on DISTINCT ports (worker k: ``game_port = base + 2*k``,
``my_port = base + 2*k + 1``), its own seed, its own per-worker pixels-ON config file
(``--out/.worker_configs/worker_<k>.json``, a copy of the shipped ``config_pixels.json`` with
``timeScale`` set to ``--time-scale``), and its OWN shard files (namespaced ``shard_w{k}_{NNNN}``)
so two workers never write the same path. The worker target function is MODULE-LEVEL
(``_run_worker``, picklable for spawn) and takes a single plain-data ``WorkerSpec`` dataclass.
The main process splits ``--target`` across M (remainder spread over the first few), spawns,
joins, then aggregates the per-worker progress into the top-level manifest.

-------------------------------------------------------------------------------------------
STORAGE CONTRACT
-------------------------------------------------------------------------------------------
Shards of <= ``--shard-size`` pairs (default 10000), one ``numpy.savez_compressed`` ``.npz``
each, written to a ``.tmp`` then ``os.replace``'d to the final name (ATOMIC — a crash never
leaves a half-shard resume would trust). Per shard, parallel arrays:
  * ``frames``      uint8   (n, 360, 640, 3)   raw decoded RGB frames (top-left origin).
  * ``states``      float32 (n, 52)            the paired 52-float state.
  * ``map_ids``     int32   (n,)               index into the manifest ``maps`` list.
  * ``episode_ids`` int32   (n,)               monotonic per worker.
  * ``step_idxs``   int32   (n,)               step index within the episode.
  * ``worker_seed`` int64   scalar             this worker's entropy seed.
Frames are stored RAW (``frame_encoding == "raw_uint8"``); flat-color renders deflate hard so
npz is fine (the smoke reports measured per-pair on-disk bytes).

``manifest.json`` (``--out/manifest.json``, STRICT JSON, written atomically) records:
schema_version, resolution, frame_dtype/encoding, state_len/dtype, target,
total_pairs_collected, the per-shard list (file, worker, n, map_counts), per-map totals, the
maps list (index -> {name, config_path, arena_path}), per-worker seeds, the root seed, git_sha,
time_scale, workers, and created/updated UTC ISO timestamps. The maps list stores
``config_path``/``arena_path`` as REPO-RELATIVE POSIX strings (NOT Windows-absolute) so the
dataset is portable to GCS/Linux for CNN training; NO absolute machine path is written anywhere
in the manifest. Each worker also writes ``worker_<k>.progress.json`` so resume reads completed
counts without parsing every shard.

-------------------------------------------------------------------------------------------
RESUME (``--resume``)
-------------------------------------------------------------------------------------------
On restart each worker scans ``--out`` for its OWN complete shards (``shard_w{k}_*.npz``) and
its ``worker_<k>.progress.json``, sums pairs already collected (total + per-map), and continues
toward its sub-target WITHOUT overwriting any existing shard (new shards take fresh indices =
count of existing shards for that worker). Per-map quotas already met are honored on resume so
balance is preserved. If ``--resume`` is NOT set and ``--out`` is non-empty, the run REFUSES
with a clear error (no clobber). On ``--resume``, if a sealed ``manifest.json`` exists the run
VALIDATES that ``--workers``/``--seed`` and the maps list (same names, same order) match the
prior run BEFORE spawning — a mismatch would replay identical action streams on the first K
workers (``SeedSequence.spawn`` is a stateful counter) or mislabel positional ``map_ids``, so it
REFUSES. If no manifest exists yet (a crash before sealing), it warns and falls back to the
per-worker progress / shard scan rather than hard-failing.

-------------------------------------------------------------------------------------------
THROUGHPUT NOTE
-------------------------------------------------------------------------------------------
``timeScale`` accelerates the in-game physics clock, but the per-step GPU readback
(``cam.Render()`` + ``ReadPixels``) is a FIXED wall-clock cost largely INDEPENDENT of
``timeScale``. So timeScale 20 is NOT 4x timeScale 5 in pairs/sec — the readback dominates.
The harness measures and logs real pairs/sec per worker; trust the measured number, not a
timeScale multiplier.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# src-layout import shim so `python -m tank_twin.collect_pixels` AND a direct invocation both
# resolve the package (uv run installs it, but this keeps it robust from the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tank_twin.env import _arena_path_from_config, _build_game_cmd  # noqa: E402
from tank_twin.maps import ALL_MAPS_SENTINEL, _resolve_map_rotation  # noqa: E402
from tank_twin.protocol import Connection  # noqa: E402

# --- defaults (the build the Director froze + its shipped pixels-ON config) ---------------
DEFAULT_EXE = _REPO_ROOT / "build" / "TankTwinStickShooter.exe"
BUILD_PIXELS_CONFIG = (
    _REPO_ROOT / "build" / "TankTwinStickShooter_Data" / "StreamingAssets" / "config_pixels.json"
)
FRAME_H = 360
FRAME_W = 640
FRAME_C = 3
STATE_LEN = 52
SCHEMA_VERSION = 1

# Connection / handshake tuning (mirrors the verify template + env defaults).
_CONNECT_ATTEMPTS = 60
_SOCK_TIMEOUT = 30.0  # GPU readback per step is a fixed cost; be generous on the recv timeout.


# ===========================================================================================
# Worker spec (plain data, picklable for spawn)
# ===========================================================================================
@dataclass
class WorkerSpec:
    """Everything one worker process needs. MUST be plain data (picklable for spawn)."""

    worker_id: int
    sub_target: int
    out_dir: str
    exe: str
    config_path: str
    game_port: int
    my_port: int
    seed_entropy: int  # this worker's SeedSequence entropy (its own seed)
    arena_paths: list[str]  # absolute arena paths, rotation order
    map_names: list[str]  # parallel to arena_paths AND to the global maps list (same order)
    map_ids: list[int]  # global manifest index per rotation slot (identity here: 0..N-1)
    shard_size: int
    resume: bool


@dataclass
class WorkerResult:
    """What a worker reports back (also persisted to worker_<k>.progress.json)."""

    worker_id: int
    seed_entropy: int
    pairs_collected: int  # total ON DISK for this worker after the run (incl. resumed)
    pairs_this_run: int  # newly collected in THIS invocation
    per_map_counts: list[int]  # parallel to the global maps list
    shards: list[dict] = field(default_factory=list)  # {file, worker, n, map_counts}
    pairs_per_sec: float = 0.0
    elapsed_sec: float = 0.0


# ===========================================================================================
# Connection / handshake / teardown (mirrors scripts/verify_pixel_capture.py exactly)
# ===========================================================================================
def _connect_to_build(exe_path, config_path, *, game_port, my_port, log_path):
    """Launch the build (arg-list) and open the TCP socket. Mirrors the verify template.

    Opens its OWN socket (bind-retry loop, then connect to 127.0.0.1:game_port). Returns
    ``(Connection, Popen, log_handle)``. Does NOT route through TankEnv.
    """
    log = open(log_path, "w")  # noqa: SIM115 (held for the child's lifetime; closed by caller)
    cmd = _build_game_cmd(exe_path, game_port, config_path)
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    bound = False
    last_err = None
    for _ in range(_CONNECT_ATTEMPTS):
        try:
            sock.bind(("", my_port))
            bound = True
            break
        except OSError as exc:
            last_err = exc
            time.sleep(1)
    if not bound:
        for _ in range(_CONNECT_ATTEMPTS):
            try:
                sock.bind(("", random.randint(33000, 60000)))
                bound = True
                break
            except OSError as exc:
                last_err = exc
                time.sleep(1)
    if not bound:
        raise last_err if last_err else OSError("could not bind socket")

    connected = False
    for _ in range(_CONNECT_ATTEMPTS):
        try:
            sock.connect(("127.0.0.1", game_port))
            connected = True
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    if not connected:
        raise ConnectionError(f"could not connect to the build at 127.0.0.1:{game_port}")

    sock.settimeout(_SOCK_TIMEOUT)
    return Connection(sock), proc, log


def _switch_arena(conn, arena_path):
    """Send ``{"switch_arena": <abs arena path>}`` and verify the ack. Mirrors TankEnv."""
    conn.send({"switch_arena": str(Path(arena_path).resolve())})
    received = conn.receive()
    if not received.get("arena_switched"):
        raise RuntimeError(f"unexpected switch_arena ack from build: {received!r}")


def _handshake_episode(conn, arena_path):
    """restart -> switch_arena -> restart -> start -> read first (state, frame).

    Returns ``(state, frame)``. Mirrors the build's state machine (verified against
    ``DriverController.cs`` + a live probe):

    The CRITICAL leading ``restart`` drains the build to ``!ingame``. ``switch_arena`` is ONLY
    handled in ``ReceiveAndSendData`` (the ``!ingame`` branch); if the build is ``ingame`` (a
    fresh boot that auto-started, OR a mid-episode rollover forced before ``done``) it is
    blocked in ``SendAndReceiveData`` reading an ACTION, so a switch_arena there is swallowed as
    an action and the build replies with the next ``state`` (the smoke crash:
    "unexpected switch_arena ack ... {'state': ...}"). A leading ``restart`` is accepted in
    BOTH branches (``ingame`` -> SendAndReceiveData restart path sets ``ingame=false`` and acks
    ``restarting``; ``!ingame`` -> ReceiveAndSendData acks ``restarting`` and stays ``!ingame``),
    so after it the build is GUARANTEED ``!ingame`` and switch_arena is handled correctly. The
    build sent NO trailing frame on that restart (the restart ack is a plain control write), so
    the acks are read with the unframed ``receive``. The FIRST episode state DOES have a frame
    glued on, so it is read with ``receive_state_and_frame``.

    ``TankEnv.reset`` doesn't need this leading restart because it only rotates at an episode
    boundary where the prior episode already ended (build already ``!ingame``); this harness
    additionally forces mid-episode rollovers (quota balancing), so the explicit drain is
    required here.
    """
    # Drain to !ingame so switch_arena is handled (the ingame-rollover / boot landmine).
    conn.send({"restart": True})
    conn.receive()  # restarting ack (no frame on a control write)
    _switch_arena(conn, arena_path)
    conn.send({"restart": True})
    conn.receive()  # restarting ack
    conn.send({"start": True})
    ack = conn.receive()  # starting ack
    if "starting" not in ack:
        raise RuntimeError(f"unexpected start ack from build: {ack!r}")
    return conn.receive_state_and_frame()


def _teardown(conn, proc, log):
    """Best-effort end handshake, close socket, terminate + wait + kill. Mirrors the template."""
    try:
        conn.send({"restart": True})
        conn.receive()
        conn.send({"end": True})
        conn.receive()
    except (ConnectionError, OSError, KeyError, ValueError):
        pass
    with contextlib.suppress(OSError):
        conn.transport.close()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    with contextlib.suppress(OSError):
        log.close()


# ===========================================================================================
# Atomic IO helpers (stdlib only; STRICT json)
# ===========================================================================================
def _atomic_write_json(path: Path, obj) -> None:
    """Write ``obj`` as STRICT JSON to ``path`` atomically (tmp + os.replace)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_savez(path: Path, **arrays) -> None:
    """``np.savez_compressed`` to a ``.tmp`` then atomic-replace into ``path``."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _repo_relative_posix(path) -> str:
    """Return ``path`` as a REPO-RELATIVE forward-slash string (portable for GCS/Linux).

    The shipped exp-configs maps ARE under ``_REPO_ROOT``, so the normal result is a clean
    relative POSIX path like ``exp-configs/maps/Arenas/center_block.json``. If a path is somehow
    NOT under the repo root (shouldn't happen for shipped maps), fall back to just the basename
    rather than crashing — the manifest must never carry a Windows absolute ``C:\\...`` path.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return Path(p.name).as_posix()


def _git_sha() -> str:
    """``git rev-parse HEAD`` via arg-list; tolerate any failure (returns "unknown")."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ===========================================================================================
# Shard buffer (flushes <= shard_size pairs per .npz)
# ===========================================================================================
class _ShardBuffer:
    """Accumulates pairs and flushes a compressed shard when it reaches ``shard_size``.

    Shard filenames are namespaced by worker id (``shard_w{worker}_{NNNN}.npz``). The shard
    index continues from any pre-existing shards (resume). Records each flushed shard's
    metadata for the worker progress file / manifest.
    """

    def __init__(self, out_dir: Path, worker_id: int, shard_size: int, start_index: int):
        self.out_dir = Path(out_dir)
        self.worker_id = worker_id
        self.shard_size = shard_size
        self.next_index = start_index
        self.frames: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.map_ids: list[int] = []
        self.episode_ids: list[int] = []
        self.step_idxs: list[int] = []
        self.flushed: list[dict] = []  # {file, worker, n, map_counts}

    def __len__(self) -> int:
        return len(self.frames)

    def add(self, frame, state, map_id, episode_id, step_idx, worker_seed):
        self.frames.append(np.asarray(frame, dtype=np.uint8))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.map_ids.append(int(map_id))
        self.episode_ids.append(int(episode_id))
        self.step_idxs.append(int(step_idx))
        self._worker_seed = int(worker_seed)
        if len(self.frames) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        n = len(self.frames)
        fname = f"shard_w{self.worker_id}_{self.next_index:04d}.npz"
        fpath = self.out_dir / fname
        frames = np.stack(self.frames).astype(np.uint8)
        states = np.stack(self.states).astype(np.float32)
        map_ids = np.asarray(self.map_ids, dtype=np.int32)
        episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        step_idxs = np.asarray(self.step_idxs, dtype=np.int32)
        _atomic_savez(
            fpath,
            frames=frames,
            states=states,
            map_ids=map_ids,
            episode_ids=episode_ids,
            step_idxs=step_idxs,
            worker_seed=np.int64(self._worker_seed),
        )
        # per-map counts for this shard (over the global maps space — sized by caller via max id)
        uniq, counts = np.unique(map_ids, return_counts=True)
        map_counts = {int(u): int(c) for u, c in zip(uniq, counts, strict=True)}
        self.flushed.append(
            {"file": fname, "worker": self.worker_id, "n": n, "map_counts": map_counts}
        )
        self.next_index += 1
        self.frames.clear()
        self.states.clear()
        self.map_ids.clear()
        self.episode_ids.clear()
        self.step_idxs.clear()


# ===========================================================================================
# Resume scan (per worker)
# ===========================================================================================
def _scan_existing(out_dir: Path, worker_id: int, num_maps: int):
    """Scan ``out_dir`` for this worker's complete shards. Returns ``(n_shards, total, per_map)``.

    ``per_map`` is a length-``num_maps`` list of counts. Reads shard ``map_ids`` to rebuild the
    per-map histogram (cheap: only the small int32 array is loaded, not the frames). The
    next shard index is ``n_shards`` (fresh indices never collide with existing ones).
    """
    shards = sorted(out_dir.glob(f"shard_w{worker_id}_*.npz"))
    total = 0
    per_map = [0] * num_maps
    for sp in shards:
        try:
            with np.load(sp) as data:
                ids = data["map_ids"]
        except (OSError, ValueError, KeyError):
            # A corrupt / partial shard: skip it (atomic writes mean this should not happen,
            # but resume must be robust). It is NOT counted toward progress.
            continue
        total += int(ids.shape[0])
        u, c = np.unique(ids, return_counts=True)
        for mu, mc in zip(u, c, strict=True):
            if 0 <= int(mu) < num_maps:
                per_map[int(mu)] += int(mc)
    return len(shards), total, per_map


# ===========================================================================================
# The worker (MODULE-LEVEL; picklable for spawn)
# ===========================================================================================
def _run_worker(spec_dict: dict) -> dict:
    """Worker entry point. Collects up to ``sub_target`` pairs and returns a result dict.

    Plain-dict in / plain-dict out so spawn pickling is trivial. Rotates maps round-robin from
    a worker-specific offset, honoring a per-map quota so the global distribution stays even.
    """
    spec = WorkerSpec(**spec_dict)
    out_dir = Path(spec.out_dir)
    num_maps = len(spec.arena_paths)
    wid = spec.worker_id

    rng = np.random.default_rng(np.random.SeedSequence(spec.seed_entropy))

    # Resume accounting: existing shards + per-map already on disk.
    if spec.resume:
        start_index, already_total, per_map = _scan_existing(out_dir, wid, num_maps)
    else:
        start_index, already_total, per_map = 0, 0, [0] * num_maps

    # Per-map quota for THIS worker (ceil split so the union covers sub_target). A map is
    # "full" once its on-disk + this-run count reaches the quota; the rotation skips full maps.
    per_map_quota = math.ceil(spec.sub_target / num_maps) if num_maps else 0

    buf = _ShardBuffer(out_dir, wid, spec.shard_size, start_index)

    # Rotation cursor starts at the worker's offset so workers stagger across maps.
    cursor = wid % num_maps
    episode_id = -1  # incremented to 0 on the first handshake

    log_path = out_dir / f"gamelog_w{wid}.txt"
    collected_this_run = 0
    target_this_run = max(0, spec.sub_target - already_total)
    started = time.time()

    def _remaining_target() -> int:
        return target_this_run - collected_this_run

    def _next_map():
        """Advance the cursor to the next map whose quota is NOT yet met; return its index.

        Returns ``None`` if EVERY map is full (worker is done). Skips full maps so balance
        holds. The cursor wraps; we scan at most ``num_maps`` slots to avoid an infinite loop.
        """
        nonlocal cursor
        for _ in range(num_maps):
            cursor = (cursor + 1) % num_maps
            if per_map[cursor] < per_map_quota:
                return cursor
        return None

    if target_this_run <= 0:
        # Already at/over sub_target on resume — nothing to do.
        return _finalize_worker(spec, buf, per_map, already_total, 0, started)

    conn = proc = log = None
    try:
        # Pick the first map to collect on (honoring quotas / resume).
        first_map = cursor if per_map[cursor] < per_map_quota else _next_map()
        if first_map is None:
            return _finalize_worker(spec, buf, per_map, already_total, 0, started)
        cursor = first_map

        conn, proc, log = _connect_to_build(
            spec.exe,
            spec.config_path,
            game_port=spec.game_port,
            my_port=spec.my_port,
            log_path=log_path,
        )

        episode_id += 1
        state_dict, frame = _handshake_episode(conn, spec.arena_paths[cursor])
        step_idx = 0
        last_report = started

        while _remaining_target() > 0:
            # Record the current (state, frame) pair for the current map.
            state = state_dict["state"]
            buf.add(frame, state, cursor, episode_id, step_idx, spec.seed_entropy)
            per_map[cursor] += 1
            collected_this_run += 1

            # progress log roughly once a second
            now = time.time()
            if now - last_report >= 5.0:
                rate = collected_this_run / max(1e-9, now - started)
                print(
                    f"[w{wid}] {collected_this_run}/{target_this_run} this-run "
                    f"({already_total + collected_this_run} total) "
                    f"map={spec.map_names[cursor]} ep={episode_id} {rate:.1f} pairs/s",
                    flush=True,
                )
                last_report = now

            if _remaining_target() <= 0:
                break

            done = "done" in state_dict
            # If this map just hit its quota, force an episode rollover to a non-full map.
            map_full = per_map[cursor] >= per_map_quota

            if done or map_full:
                nxt = _next_map()
                if nxt is None:
                    break  # all maps full
                cursor = nxt
                episode_id += 1
                state_dict, frame = _handshake_episode(conn, spec.arena_paths[cursor])
                step_idx = 0
                continue

            # Drive the next step with independent random actions for BOTH players.
            a1 = _random_action(rng)
            a2 = _random_action(rng)
            conn.send({1: a1, 2: a2})
            state_dict, frame = conn.receive_state_and_frame()
            step_idx += 1

        buf.flush()
    finally:
        if conn is not None:
            _teardown(conn, proc, log)

    return _finalize_worker(spec, buf, per_map, already_total, collected_this_run, started)


def _random_action(rng) -> list[float]:
    """5-float [move_x, move_y, aim_x, aim_y, fire]: analog ~U[-1,1], fire coin flip 0/1."""
    analog = rng.uniform(-1.0, 1.0, 4)
    fire = 1.0 if rng.integers(0, 2) else 0.0
    return [float(analog[0]), float(analog[1]), float(analog[2]), float(analog[3]), fire]


def _finalize_worker(spec, buf, per_map, already_total, collected_this_run, started):
    """Flush, write the per-worker progress file, return the result dict."""
    buf.flush()
    out_dir = Path(spec.out_dir)
    total = already_total + collected_this_run
    elapsed = time.time() - started
    rate = collected_this_run / max(1e-9, elapsed)

    result = WorkerResult(
        worker_id=spec.worker_id,
        seed_entropy=spec.seed_entropy,
        pairs_collected=total,
        pairs_this_run=collected_this_run,
        per_map_counts=list(per_map),
        shards=buf.flushed,
        pairs_per_sec=rate,
        elapsed_sec=elapsed,
    )
    # Aggregate shard list across resume runs: re-scan ALL of this worker's shards on disk so
    # the progress file always reflects EVERYTHING (this run's flushed + any prior run's).
    all_shards = _all_worker_shards(out_dir, spec.worker_id)
    progress = {
        "worker_id": spec.worker_id,
        "seed_entropy": spec.seed_entropy,
        "pairs_collected": total,
        "per_map_counts": list(per_map),
        "shards": all_shards,
        "pairs_per_sec_last_run": rate,
        "elapsed_sec_last_run": elapsed,
        "updated": _utc_now_iso(),
    }
    _atomic_write_json(out_dir / f"worker_{spec.worker_id}.progress.json", progress)
    result.shards = all_shards
    return asdict(result)


def _all_worker_shards(out_dir: Path, worker_id: int) -> list[dict]:
    """List ALL of a worker's shards on disk with {file, worker, n, map_counts}."""
    shards = []
    for sp in sorted(out_dir.glob(f"shard_w{worker_id}_*.npz")):
        try:
            with np.load(sp) as data:
                ids = data["map_ids"]
        except (OSError, ValueError, KeyError):
            continue
        u, c = np.unique(ids, return_counts=True)
        map_counts = {int(mu): int(mc) for mu, mc in zip(u, c, strict=True)}
        shards.append(
            {"file": sp.name, "worker": worker_id, "n": int(ids.shape[0]), "map_counts": map_counts}
        )
    return shards


# ===========================================================================================
# Per-worker config (pixels-ON, timeScale overridden)
# ===========================================================================================
def _write_worker_config(
    base_config: dict, out_dir: Path, worker_id: int, time_scale: float, boot_arena: str
) -> Path:
    """Write a per-worker pixels-ON config with ``timeScale`` set. STRICT JSON. Returns path.

    LANDMINE (caught in the smoke): the build's ``DriverController.Awake`` resolves the
    config's ``arena_path`` RELATIVE TO THE CONFIG FILE'S DIRECTORY (same rule as
    ``env._arena_path_from_config``) and reads it AT BOOT, before the socket is up. The shipped
    ``config_pixels.json`` carries a RELATIVE ``arena_path`` (``Arenas/custom1.json``) that is
    only valid next to the shipped StreamingAssets dir — copying the config to
    ``.worker_configs/`` would make the build look for ``.worker_configs/Arenas/custom1.json``
    and crash with DirectoryNotFoundException before it ever accepts our socket. So we OVERRIDE
    ``arena_path`` with an ABSOLUTE boot arena (the first rotation map's arena, guaranteed to
    exist and dimensionally valid). The boot arena is cosmetic: we ``switch_arena`` to the real
    target map on the very first handshake regardless.
    """
    cfg_dir = out_dir / ".worker_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_config)
    cfg["timeScale"] = time_scale
    cfg["arena_path"] = str(Path(boot_arena).resolve())  # absolute -> boot-safe in any dir
    cfg_path = cfg_dir / f"worker_{worker_id}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


# ===========================================================================================
# Manifest aggregation
# ===========================================================================================
def _write_manifest(
    out_dir: Path,
    *,
    maps_meta: list[dict],
    results: list[dict],
    target: int,
    time_scale: float,
    workers: int,
    seed: int,
    frame_encoding: str,
    created: str,
):
    """Aggregate worker results into the top-level manifest.json (STRICT JSON, atomic)."""
    num_maps = len(maps_meta)
    per_map_total = [0] * num_maps
    total_pairs = 0
    shard_list = []
    seeds = {}
    for r in results:
        total_pairs += r["pairs_collected"]
        for i, c in enumerate(r["per_map_counts"]):
            per_map_total[i] += c
        shard_list.extend(r["shards"])
        seeds[str(r["worker_id"])] = r["seed_entropy"]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "resolution": {"width": FRAME_W, "height": FRAME_H},
        "frame_dtype": "uint8",
        "frame_encoding": frame_encoding,
        "state_len": STATE_LEN,
        "state_dtype": "float32",
        "target": target,
        "total_pairs_collected": total_pairs,
        "workers": workers,
        "seed": seed,
        "time_scale": time_scale,
        "git_sha": _git_sha(),
        "maps": maps_meta,  # [{index, name, config_path, arena_path}] — repo-relative POSIX paths
        "per_map_total": {maps_meta[i]["name"]: per_map_total[i] for i in range(num_maps)},
        "per_worker_seeds": seeds,
        "shards": shard_list,
        "created": created,
        "updated": _utc_now_iso(),
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


# ===========================================================================================
# CLI / orchestration
# ===========================================================================================
def _resolve_maps(maps_arg: str):
    """Resolve the --maps value to (arena_paths, map_names, config_paths) via the shared resolver.

    ``all`` -> the sorted 10 exp-configs maps. A comma-list -> those config paths in order.
    Each config -> its absolute arena path (env._arena_path_from_config) + the config stem name.
    ``arena_paths`` and ``config_paths`` are ABSOLUTE (the worker resolves switch_arena locally on
    THIS box); the MANIFEST converts them to repo-relative POSIX (see ``_repo_relative_posix``).
    """
    if maps_arg.strip().lower() == "all":
        configs = _resolve_map_rotation([ALL_MAPS_SENTINEL])
    else:
        entries = [s.strip() for s in maps_arg.split(",") if s.strip()]
        configs = _resolve_map_rotation(entries)
    if not configs:
        raise SystemExit(f"--maps resolved to no map configs: {maps_arg!r}")
    arena_paths = []
    names = []
    config_paths = []
    for cfg in configs:
        cfg = Path(cfg)
        if not cfg.is_absolute():
            # Resolve a bare name like "empty" or "empty.json" against the default maps dir.
            from tank_twin.maps import DEFAULT_MAPS_DIR

            candidate = cfg if cfg.suffix else cfg.with_suffix(".json")
            cfg = (DEFAULT_MAPS_DIR / candidate.name).resolve()
        else:
            cfg = cfg.resolve()
        arena = _arena_path_from_config(cfg).resolve()
        if not arena.exists():
            raise SystemExit(f"arena not found for map config {cfg}: {arena}")
        arena_paths.append(str(arena))
        names.append(cfg.stem)
        config_paths.append(str(cfg))
    return arena_paths, names, config_paths


def _validate_resume_against_manifest(
    out_dir: Path, *, workers: int, seed: int, map_names: list[str]
) -> None:
    """On --resume, cross-check --workers/--seed/maps against a sealed ``manifest.json``.

    WHY (P0): per-worker entropy comes from ``SeedSequence(seed).spawn(workers)``, whose first K
    children are IDENTICAL to ``spawn(K)`` (spawn is a stateful counter). So resuming with a
    DIFFERENT ``--workers`` or ``--seed`` would replay the SAME action streams on the first K
    workers — near-duplicate trajectories that silently skew per-map balance. ``map_ids`` are also
    POSITIONAL into the maps list, so a different map order/set would mislabel every pair.

    Behavior:
      * If a ``manifest.json`` exists, REFUSE (SystemExit) on any mismatch of workers, seed, or the
        maps list (same names, same order).
      * If NO manifest exists yet (a prior run crashed before sealing it, but shards may exist),
        do NOT hard-fail — fall back to the per-worker progress files / shard scan as usual, but
        PRINT A WARNING that workers/seed could not be cross-checked.
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        print(
            "WARNING: --resume but no manifest.json in --out; cannot cross-check --workers/--seed "
            "against the prior run (manifest absent — prior run likely crashed before sealing). "
            "Falling back to the per-worker progress files / shard scan.",
            flush=True,
        )
        return

    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--resume: could not read existing manifest.json: {exc}") from exc

    prior_workers = prior.get("workers")
    prior_seed = prior.get("seed")
    prior_map_names = [m.get("name") for m in prior.get("maps", [])]

    problems = []
    if prior_workers != workers:
        problems.append(f"--workers {prior_workers} (got {workers})")
    # ``seed`` was added in this fix; an older manifest may lack it. Only enforce when present.
    if prior_seed is not None and prior_seed != seed:
        problems.append(f"--seed {prior_seed} (got {seed})")
    if prior_map_names and prior_map_names != map_names:
        problems.append(f"--maps producing names {prior_map_names} in this order (got {map_names})")

    if problems:
        need_workers = prior_workers if prior_workers is not None else workers
        need_seed = prior_seed if prior_seed is not None else seed
        raise SystemExit(
            "--resume must match the prior run recorded in manifest.json. "
            f"Required to match: --workers {need_workers} --seed {need_seed}. "
            "Mismatch(es): " + "; ".join(problems) + ".\n"
            "Resuming with a different --workers or --seed would replay identical action streams "
            "on the first K workers (SeedSequence.spawn is a stateful counter), corrupting the "
            "dataset; a different maps order/set would mislabel positional map_ids."
        )


def _clean_stale_tmp(out_dir: Path) -> None:
    """Delete leftover ``*.tmp`` from a crashed prior run (a .tmp is by definition incomplete).

    Belt-and-suspenders: the resume scan only globs final ``*.npz``/``*.json``, so a stray
    ``shard_w*.npz.tmp`` / ``*.json.tmp`` is never trusted — but removing it keeps --out tidy and
    avoids confusion. Best-effort; never fails the run.
    """
    for tmp in list(out_dir.glob("*.tmp")):
        with contextlib.suppress(OSError):
            tmp.unlink()


def _split_target(target: int, workers: int) -> list[int]:
    """Split ``target`` across ``workers`` so the parts sum to exactly ``target``."""
    base = target // workers
    rem = target % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-2 (frame, state)-pair collection harness (RANDOM policy, both players)."
    )
    parser.add_argument("--target", type=int, required=True, help="total (frame, state) pairs")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--workers", type=int, default=4, help="parallel Unity workers")
    parser.add_argument("--maps", default="all", help="'all' or comma-list of map configs/names")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Unity timeScale")
    parser.add_argument("--seed", type=int, default=0, help="root seed (reproducible actions)")
    parser.add_argument("--shard-size", type=int, default=10000, help="max pairs per shard npz")
    parser.add_argument("--resume", action="store_true", help="continue a prior run (no clobber)")
    parser.add_argument("--exe", default=str(DEFAULT_EXE), help="path to the frame-capable build")
    parser.add_argument("--game-port-base", type=int, default=50000, help="worker k: base+2k / +1")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    exe_path = Path(args.exe).resolve()
    if not exe_path.exists():
        raise SystemExit(f"build exe not found: {exe_path}")
    if not BUILD_PIXELS_CONFIG.exists():
        raise SystemExit(f"pixels-ON config not found: {BUILD_PIXELS_CONFIG}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.target < 1:
        raise SystemExit("--target must be >= 1")

    # No-clobber guard: refuse a non-empty --out unless --resume.
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"--out is non-empty and --resume not set: {out_dir}\n"
            "Pass --resume to continue, or choose an empty/new --out (refusing to clobber)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_paths, map_names, config_paths = _resolve_maps(args.maps)
    num_maps = len(arena_paths)

    # RESUME VALIDATION (P0): cross-check --workers/--seed/maps against a sealed manifest BEFORE
    # spawning any worker. On resume, also sweep any stale *.tmp from a crashed prior run.
    if args.resume:
        _validate_resume_against_manifest(
            out_dir, workers=args.workers, seed=args.seed, map_names=map_names
        )
        _clean_stale_tmp(out_dir)

    # MANIFEST PORTABILITY (P0): store REPO-RELATIVE POSIX paths only — the dataset is uploaded to
    # GCS and consumed on Linux, so a Windows-absolute ``C:\...`` arena_path would be dead there.
    # The WORKER still gets ABSOLUTE arena_paths on the wire (switch_arena resolves locally here).
    maps_meta = [
        {
            "index": i,
            "name": map_names[i],
            "config_path": _repo_relative_posix(config_paths[i]),
            "arena_path": _repo_relative_posix(arena_paths[i]),
        }
        for i in range(num_maps)
    ]

    base_config = json.loads(BUILD_PIXELS_CONFIG.read_text(encoding="utf-8"))
    if not base_config.get("obs_pixels"):
        raise SystemExit("shipped config_pixels.json does not have obs_pixels=true")

    # Per-worker seeds: SeedSequence(seed).spawn(workers) -> distinct, reproducible entropy.
    child_seqs = np.random.SeedSequence(args.seed).spawn(args.workers)
    # A stable integer "seed" per worker for logging/manifest (entropy is the real driver).
    worker_entropies = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seqs]

    sub_targets = _split_target(args.target, args.workers)

    specs = []
    for k in range(args.workers):
        cfg_path = _write_worker_config(
            base_config, out_dir, k, args.time_scale, boot_arena=arena_paths[0]
        )
        spec = WorkerSpec(
            worker_id=k,
            sub_target=sub_targets[k],
            out_dir=str(out_dir),
            exe=str(exe_path),
            config_path=str(cfg_path),
            game_port=args.game_port_base + 2 * k,
            my_port=args.game_port_base + 2 * k + 1,
            seed_entropy=worker_entropies[k],
            arena_paths=arena_paths,
            map_names=map_names,
            map_ids=list(range(num_maps)),
            shard_size=args.shard_size,
            resume=args.resume,
        )
        specs.append(asdict(spec))

    created = _utc_now_iso()
    print(
        f"Stage-2 collect: target={args.target} workers={args.workers} maps={num_maps} "
        f"time_scale={args.time_scale} out={out_dir}",
        flush=True,
    )
    for k, st in enumerate(sub_targets):
        print(
            f"  worker {k}: sub_target={st} ports=({args.game_port_base + 2 * k},"
            f"{args.game_port_base + 2 * k + 1}) seed_entropy={worker_entropies[k]}",
            flush=True,
        )

    started = time.time()
    if args.workers == 1:
        # Single worker: run in-process (simpler; still spawn-safe everywhere else).
        results = [_run_worker(specs[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = pool.map(_run_worker, specs)
    elapsed = time.time() - started

    results = sorted(results, key=lambda r: r["worker_id"])
    manifest = _write_manifest(
        out_dir,
        maps_meta=maps_meta,
        results=results,
        target=args.target,
        time_scale=args.time_scale,
        workers=args.workers,
        seed=args.seed,
        frame_encoding="raw_uint8",
        created=created,
    )

    total = manifest["total_pairs_collected"]
    new_pairs = sum(r["pairs_this_run"] for r in results)
    agg_rate = new_pairs / max(1e-9, elapsed)
    print("\n=== collection complete ===", flush=True)
    print(f"total pairs on disk: {total} (target {args.target}); new this run: {new_pairs}")
    print(
        f"wall time: {elapsed:.1f}s; aggregate {agg_rate:.1f} pairs/s over {args.workers} workers"
    )
    for r in results:
        print(
            f"  worker {r['worker_id']}: total={r['pairs_collected']} "
            f"this_run={r['pairs_this_run']} {r['pairs_per_sec']:.1f} pairs/s/worker"
        )
    print("per-map totals:")
    for i in range(num_maps):
        print(f"  {map_names[i]:>16}: {manifest['per_map_total'][map_names[i]]}")
    print(f"manifest: {out_dir / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
