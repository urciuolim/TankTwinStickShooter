"""Contract tests for ``pop_trainer.core.logging_setup`` (the cross-stack observability seam).

Logging is process-global, so each test uses a UNIQUE port (and therefore a unique named logger)
and tears its handlers down, so handlers never leak across the run. We parse emitted lines with
``json.loads`` to prove they are strict JSON and carry the required schema, prove the eval/system
file-routing isolation the trainer relies on, and prove a logger attached to ``TankEnv`` is purely
observational (identical step/reset behavior logger-on vs logger-off).
"""

import json
import logging
from pathlib import Path

import numpy as np

# Reuse the env package's fake-transport harness rather than inventing a new socket fake.
# pytest puts ``tests/pop_trainer`` on sys.path (no ``__init__`` above the leaf test packages),
# so the sibling module is importable as ``env.test_tank_env``.
from env.test_tank_env import (
    ScriptedTransport,
    _flat_state,
    _reset_blobs,
    _state_and_frame_bytes,
)

from pop_trainer.core import protocol as P
from pop_trainer.core import state as S
from pop_trainer.core.logging_setup import (
    JsonlFormatter,
    env_log_path,
    env_logger_name,
    level_from_env,
    setup_env_logger,
    setup_logger,
    setup_system_logger,
    system_log_path,
    unity_log_path,
)
from pop_trainer.env.tank_env import TankEnv

REQUIRED_KEYS = ("ts_wall", "ts_mono", "level", "layer", "role", "port", "event")


def _drop_logger(name):
    """Detach every handler from a named logger so a test never leaks into the next one."""
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _read_lines(path):
    """Parse a JSONL file into a list of dicts (one per non-blank line)."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- (a) pure path + name builders ------------------------------------------------------


def test_env_log_path_is_exact():
    d = Path("/some/log/dir")
    assert env_log_path(d, "train", 50000) == d / "env-train-50000.log"
    assert env_log_path(d, "eval", 50000) == d / "env-eval-50000.log"


def test_system_log_path_is_exact():
    d = Path("/some/log/dir")
    assert system_log_path(d) == d / "training-system.log"


def test_unity_log_path_is_exact():
    d = Path("/some/log/dir")
    assert unity_log_path(d, "eval", 50007) == d / "unity-eval-50007.log"


def test_env_logger_name_is_exact():
    assert env_logger_name("eval", 50007) == "pop_trainer.env.eval.50007"


def test_path_builders_accept_str_log_dir():
    # ``log_dir`` may be a str; the result is still a Path with the exact filename.
    assert env_log_path("logs", "train", 50001) == Path("logs") / "env-train-50001.log"


# --- (b) JSONL records: strict JSON + required schema -----------------------------------


def test_jsonl_record_carries_required_fields_and_merged_detail(tmp_path):
    port = 51001
    logger = setup_env_logger(tmp_path, "train", port)
    try:
        logger.info("some_event", extra={"detail": {"bytes": 42}})
    finally:
        _drop_logger(env_logger_name("train", port))

    lines = _read_lines(env_log_path(tmp_path, "train", port))
    assert len(lines) == 1
    rec = lines[0]

    for key in REQUIRED_KEYS:
        assert key in rec, f"missing required key {key!r}"
    assert isinstance(rec["ts_wall"], float)
    assert isinstance(rec["ts_mono"], float)
    assert isinstance(rec["port"], int)
    assert isinstance(rec["level"], str)
    assert isinstance(rec["layer"], str)
    assert isinstance(rec["role"], str)
    assert isinstance(rec["event"], str)

    assert rec["event"] == "some_event"
    assert rec["role"] == "train"
    assert rec["port"] == port
    assert rec["layer"] == "env"
    # The detail dict is merged at the top level.
    assert rec["bytes"] == 42


def test_jsonl_detail_layer_override_lands_in_layer_field(tmp_path):
    # The shared env file carries both env and protocol records, distinguished by a per-record
    # ``layer`` override.
    port = 51002
    logger = setup_env_logger(tmp_path, "train", port)
    try:
        logger.info("recv", extra={"detail": {"layer": "protocol"}})
    finally:
        _drop_logger(env_logger_name("train", port))

    rec = _read_lines(env_log_path(tmp_path, "train", port))[0]
    assert rec["layer"] == "protocol"


def test_jsonl_protected_key_in_detail_does_not_clobber_event(tmp_path):
    # A stray detail key by a protected schema name never overwrites the real value.
    port = 51003
    logger = setup_env_logger(tmp_path, "train", port)
    try:
        logger.info("real_event", extra={"detail": {"event": "hijack"}})
    finally:
        _drop_logger(env_logger_name("train", port))

    rec = _read_lines(env_log_path(tmp_path, "train", port))[0]
    assert rec["event"] == "real_event"


def test_jsonl_formatter_serializes_non_json_detail_value():
    # ``default=str`` keeps a non-JSON detail value (e.g. a Path) from raising in the logging path.
    fmt = JsonlFormatter(layer="env", role="train", port=51004)
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.detail = {"path": Path("/a/b")}
    rec = json.loads(fmt.format(record))
    assert rec["path"] == str(Path("/a/b"))


# --- (c) eval isolation -----------------------------------------------------------------


def test_eval_records_route_only_to_eval_file_not_system(tmp_path):
    port = 51010
    sys_logger = setup_system_logger(tmp_path)
    eval_logger = setup_env_logger(tmp_path, "eval", port)
    try:
        sys_logger.info("system_event")
        eval_logger.info("eval_event")
    finally:
        _drop_logger("pop_trainer.train")
        _drop_logger(env_logger_name("eval", port))

    eval_recs = _read_lines(env_log_path(tmp_path, "eval", port))
    sys_recs = _read_lines(system_log_path(tmp_path))

    # The eval event landed ONLY in env-eval-<port>.log.
    eval_events = [r["event"] for r in eval_recs]
    assert "eval_event" in eval_events
    assert all(r["role"] == "eval" for r in eval_recs)

    # ...and did NOT leak into training-system.log.
    sys_events = [r["event"] for r in sys_recs]
    assert "eval_event" not in sys_events
    assert "system_event" in sys_events

    # The isolation guarantee: the eval logger never propagates to the root/system handler.
    assert eval_logger.propagate is False


# --- (d) logging on vs off does not change TankEnv behavior -----------------------------


def _make_env(blobs, **kwargs):
    transport = ScriptedTransport(blobs)
    conn = P.Connection(transport)
    return TankEnv(connection=conn, **kwargs), transport


def test_logger_on_vs_off_is_behavior_identical(tmp_path):
    port = 51020
    raw0 = _flat_state(0.0)
    raw1 = [float(i) for i in range(S.STATE_LEN)]
    action = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)

    def run(logger):
        blobs = _reset_blobs(raw0, fill=(7, 8, 9)) + [_state_and_frame_bytes(raw1, fill=(1, 2, 3))]
        env, _ = _make_env(blobs, logger=logger)
        env.reset(seed=0)
        return env.step(action.copy())

    off_obs, off_reward, off_term, off_trunc, off_info = run(None)

    logger = setup_env_logger(tmp_path, "train", port)
    try:
        on_obs, on_reward, on_term, on_trunc, on_info = run(logger)
    finally:
        _drop_logger(env_logger_name("train", port))

    # The 5-tuple is identical with logging on vs off.
    assert np.array_equal(on_obs, off_obs)
    assert on_reward == off_reward
    assert on_term is off_term
    assert on_trunc is off_trunc
    assert on_info["state"] == off_info["state"]
    assert on_info["p1_action"] == off_info["p1_action"]
    assert on_info["p2_action"] == off_info["p2_action"]

    # The logger-on run wrote a non-empty env log; nothing else changed.
    on_path = env_log_path(tmp_path, "train", port)
    assert on_path.exists()
    assert on_path.read_text(encoding="utf-8").strip() != ""


def test_logger_off_writes_no_file(tmp_path):
    port = 51021
    raw0 = _flat_state(0.0)
    blobs = _reset_blobs(raw0)
    env, _ = _make_env(blobs, logger=None)
    env.reset(seed=0)
    # The logger-off path never touches the filesystem.
    assert not env_log_path(tmp_path, "train", port).exists()


# --- (e) level_from_env pure switch -----------------------------------------------------


def test_level_from_env_debug_flag():
    assert level_from_env(debug=True) == logging.DEBUG


def test_level_from_env_recognized_names():
    assert level_from_env(env_value="DEBUG") == logging.DEBUG
    assert level_from_env(env_value="WARNING") == logging.WARNING


def test_level_from_env_unrecognized_falls_back_to_info():
    assert level_from_env(env_value="bogus") == logging.INFO


def test_level_from_env_default_is_info():
    assert level_from_env() == logging.INFO


# --- bonus: setup idempotency -----------------------------------------------------------


def test_setup_logger_is_idempotent(tmp_path):
    name = "pop_trainer.test.idempotent.51030"
    try:
        setup_logger(
            name=name,
            log_dir=tmp_path,
            path=tmp_path / "idem.log",
            layer="env",
            role="train",
            port=51030,
        )
        setup_logger(
            name=name,
            log_dir=tmp_path,
            path=tmp_path / "idem.log",
            layer="env",
            role="train",
            port=51030,
        )
        # Two calls for the same name attach only ONE tagged handler.
        handlers = logging.getLogger(name).handlers
        tagged = [h for h in handlers if getattr(h, "_pop_trainer_jsonl", False)]
        assert len(tagged) == 1
    finally:
        _drop_logger(name)
