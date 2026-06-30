"""Per-process structured-JSONL logging setup (the cross-stack observability seam).

This is the OBSERVABILITY layer: it gives every Python process in a training run its OWN log
file and a single structured-JSON line format so the train / env / protocol layers can leave a
wall-clock-mergeable evidence trail. It is stdlib ``logging`` ONLY (no new deps) and lives in
``core`` (the dependency-free root) so every layer above can route to it.

PER-PROCESS FILE ROUTING. A training run is many processes: the training-system MAIN process,
each ``SubprocVecEnv`` WORKER (a spawned subprocess), and the eval env (which runs in the main
process behind the eval callback). Each must write to its OWN file so concurrent writers never
contend on one handle and the evidence stays partitionable by who wrote it:

* :func:`system_log_path` -> ``training-system.log`` (the main process).
* :func:`env_log_path` -> ``env-<role>-<port>.log`` for one env connection, where
  ``role`` is ``"train"`` or ``"eval"`` and ``port`` is the env's TCP port. The C# side writes
  ``unity-<role>-<port>.log`` itself; the two pair by the shared ``(role, port)``.

Both path builders are PURE (no I/O) so the exact filenames are unit-testable.

NAMED LOGGERS + ISOLATION. Each role+port gets a DISTINCT named logger
(``pop_trainer.env.<role>.<port>``); the system logger is ``pop_trainer.train``. Every configured
logger sets ``propagate=False`` so its records NEVER bubble up to the root / system handler — this
is load-bearing: an eval env's records must land in ``env-eval-<port>.log`` and must NOT leak into
``training-system.log``. :func:`setup_logger` is IDEMPOTENT (it tags the handler it owns and skips
re-attaching when the same logger is configured twice in one process), so a worker that constructs
two envs, or a re-entered factory, never double-writes.

JSONL LINE SCHEMA. Every record is one strict-JSON object per line carrying at least::

    {"ts_wall": <float>, "ts_mono": <float>, "level": <str>, "layer": <str>,
     "role": <str>, "port": <int|null>, "event": <str>, ...detail}

* ``ts_wall`` = :func:`time.time` (wall clock; the cross-file merge key — C# writes
  ``DateTime.UtcNow`` on its lines).
* ``ts_mono`` = :func:`time.monotonic` (for durations within a process).
* ``level`` = the record's level name.
* ``layer`` = which layer emitted it: ``"protocol"`` | ``"env"`` | ``"train"``.
* ``role`` = ``"train"`` | ``"eval"`` | ``"system"``.
* ``port`` = the env TCP port (int) or ``null`` for the system file.
* ``event`` = a short stable event name.
* ``...detail`` = any extra fields the call site passes via ``extra={"detail": {...}}`` (e.g.
  ``elapsed_ms`` / ``bytes`` / ``map`` / ``episode`` / ``timesteps`` / ``fps``).

The ``role`` / ``port`` / ``layer`` are bound into the :class:`JsonlFormatter` at setup, so call
sites only pass ``event`` + an optional ``detail`` dict (``logger.info(event, extra={"detail":
{...}})``) and never repeat the fixed context.

Cross-platform: stdlib + ``pathlib`` only, no fork/forkserver assumptions. The file handler is
opened by :func:`setup_logger` IN THE CALLING PROCESS, so a spawned worker (which re-runs the env
factory, see ``rl.train``) opens its OWN handle to its OWN file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

__all__ = [
    "LAYER_PROTOCOL",
    "LAYER_ENV",
    "LAYER_TRAIN",
    "ROLE_TRAIN",
    "ROLE_EVAL",
    "ROLE_SYSTEM",
    "SYSTEM_LOGGER_NAME",
    "LOG_LEVEL_ENV_VAR",
    "JsonlFormatter",
    "env_log_path",
    "system_log_path",
    "unity_log_path",
    "env_logger_name",
    "setup_logger",
    "setup_env_logger",
    "setup_system_logger",
    "level_from_env",
]

# Layer tags (which layer emitted a record).
LAYER_PROTOCOL = "protocol"
LAYER_ENV = "env"
LAYER_TRAIN = "train"

# Role tags (which side of the run a logger belongs to).
ROLE_TRAIN = "train"
ROLE_EVAL = "eval"
ROLE_SYSTEM = "system"

# The single named logger for the training-system main process.
SYSTEM_LOGGER_NAME = "pop_trainer.train"

# Marker stashed on a handler this module owns, so idempotent setup can recognize + skip it.
_HANDLER_TAG = "_pop_trainer_jsonl"

# Env var that cranks the whole run to DEBUG (a single switch; see rl.train).
LOG_LEVEL_ENV_VAR = "POP_LOG_LEVEL"


# --- pure path + name builders (no I/O; unit-tested for exact filenames) ----------------------


def env_log_path(log_dir: str | Path, role: str, port: int) -> Path:
    """The per-env Python log path: ``<log_dir>/env-<role>-<port>.log``. PURE (no I/O).

    ``role`` is ``"train"`` or ``"eval"``; ``port`` is the env's TCP port. The C# side writes a
    paired ``unity-<role>-<port>.log`` for the same connection.
    """
    return Path(log_dir) / f"env-{role}-{int(port)}.log"


def system_log_path(log_dir: str | Path) -> Path:
    """The training-system main-process log path: ``<log_dir>/training-system.log``. PURE."""
    return Path(log_dir) / "training-system.log"


def unity_log_path(log_dir: str | Path, role: str, port: int) -> Path:
    """The per-Unity-instance C# log path: ``<log_dir>/unity-<role>-<port>.log``. PURE (no I/O).

    Passed to ``core.launch.build_launch_cmd(..., unity_log_path=...)`` as Unity's ``-logFile`` so
    each build writes its OWN C# log instead of clobbering the shared default ``Player.log``. Pairs
    with the Python-side :func:`env_log_path` for the same ``(role, port)``.
    """
    return Path(log_dir) / f"unity-{role}-{int(port)}.log"


def env_logger_name(role: str, port: int) -> str:
    """The DISTINCT named logger for one env connection: ``pop_trainer.env.<role>.<port>``.

    Distinct per role+port so an eval logger never shares a logger object (and thus never shares a
    handler / file) with a train logger in the same process.
    """
    return f"pop_trainer.env.{role}.{int(port)}"


# --- the strict-JSON formatter ---------------------------------------------------------------


class JsonlFormatter(logging.Formatter):
    """A :class:`logging.Formatter` that emits one strict-JSON object per record (JSONL).

    ``role`` / ``port`` / ``layer`` are FIXED context bound at construction, so call sites only
    supply ``event`` (the log message) + an optional ``detail`` dict via
    ``extra={"detail": {...}}``. The ``detail`` keys are merged into the record at the top level.
    A ``detail["layer"]`` override is honored (the SHARED env file carries both ``env`` and
    ``protocol`` records, distinguished by their per-record ``layer``); the timestamp / level /
    role / port / event schema keys are PROTECTED — a stray detail key by those names never clobbers
    them.

    Emits strict JSON (``json.dumps``): no trailing commas, no leading-dot floats — Python's
    ``json`` is fine and Python parses these lines back. ``default=str`` keeps a non-JSON detail
    value (e.g. a ``Path`` or a ``WallLayout``) from raising in the logging path.
    """

    # Schema keys a caller's detail must never overwrite (``layer`` is intentionally NOT here — a
    # per-record layer override is the documented way the shared env file tags protocol vs env).
    _PROTECTED_KEYS = frozenset({"ts_wall", "ts_mono", "level", "role", "port", "event"})

    def __init__(self, *, layer: str, role: str, port: int | None) -> None:
        super().__init__()
        self._layer = layer
        self._role = role
        self._port = port

    def format(self, record: logging.LogRecord) -> str:
        detail = getattr(record, "detail", None)
        detail = detail if isinstance(detail, dict) else {}
        payload: dict = {
            "ts_wall": record.created if record.created is not None else time.time(),
            "ts_mono": time.monotonic(),
            "level": record.levelname,
            "layer": detail.get("layer", self._layer),
            "role": self._role,
            "port": self._port,
            "event": record.getMessage(),
        }
        for key, value in detail.items():
            if key not in self._PROTECTED_KEYS and key != "layer":
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# --- setup (opens the per-process file handle; idempotent) -----------------------------------


def setup_logger(
    *,
    name: str,
    log_dir: str | Path,
    path: str | Path,
    layer: str,
    role: str,
    port: int | None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Attach a per-process JSONL file handler to the named logger and return it (IDEMPOTENT).

    Creates ``log_dir`` if needed, opens ``path`` for append in THIS process, and attaches a
    :class:`logging.FileHandler` with a :class:`JsonlFormatter` bound to ``(layer, role, port)``.
    The logger's level is set to ``level`` and ``propagate`` is set to ``False`` so its records
    never bubble up to the root / system handler (the eval-isolation guarantee).

    Idempotent: the handler this module attaches is tagged, so a second call for the SAME logger in
    the SAME process re-uses the existing handler (it only updates the level) instead of attaching a
    duplicate — a worker that builds two envs, or a re-entered factory, never double-writes.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    existing = next((h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)), None)
    if existing is not None:
        existing.setLevel(level)
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(path), mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(JsonlFormatter(layer=layer, role=role, port=port))
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)
    return logger


def setup_env_logger(
    log_dir: str | Path, role: str, port: int, *, level: int = logging.INFO
) -> logging.Logger:
    """Set up the per-env logger for ``(role, port)`` routed to ``env-<role>-<port>.log``.

    The shared per-connection logger: the env layer logs its handshake / step milestones on it AND
    the :class:`~pop_trainer.core.protocol.Connection` for the SAME connection is handed this logger
    so protocol + env records for one socket land in one file. ``layer`` defaults to ``env`` on the
    logger; the protocol layer overrides the per-record layer via the formatter it carries — both
    share the file, distinguished by the ``layer`` field. Distinct named logger per role+port with
    ``propagate=False`` keeps eval records out of ``training-system.log``.
    """
    return setup_logger(
        name=env_logger_name(role, port),
        log_dir=log_dir,
        path=env_log_path(log_dir, role, port),
        layer=LAYER_ENV,
        role=role,
        port=port,
        level=level,
    )


def setup_system_logger(log_dir: str | Path, *, level: int = logging.INFO) -> logging.Logger:
    """Set up the training-system main-process logger routed to ``training-system.log``.

    ``role == "system"``, ``port == None``, ``layer == "train"``. ``propagate=False`` keeps it from
    catching env/eval records (which use their own non-propagating loggers).
    """
    return setup_logger(
        name=SYSTEM_LOGGER_NAME,
        log_dir=log_dir,
        path=system_log_path(log_dir),
        layer=LAYER_TRAIN,
        role=ROLE_SYSTEM,
        port=None,
        level=level,
    )


def level_from_env(*, debug: bool = False, env_value: str | None = None) -> int:
    """Resolve the log level from the single DEBUG switch (PURE; no ``os.environ`` read).

    The ONE switch cranks everything to DEBUG. Returns ``logging.DEBUG`` when ``debug`` is set OR
    when ``env_value`` (the caller-read ``POP_LOG_LEVEL``) names a recognized level at DEBUG, else
    ``logging.INFO``. A recognized non-DEBUG ``env_value`` (e.g. ``"WARNING"``) maps to that level;
    an unrecognized value falls back to ``INFO``. The caller reads the env var and passes it here so
    this stays pure / testable.
    """
    if debug:
        return logging.DEBUG
    if env_value:
        named = logging.getLevelNamesMapping().get(env_value.strip().upper())
        if isinstance(named, int):
            return named
    return logging.INFO
