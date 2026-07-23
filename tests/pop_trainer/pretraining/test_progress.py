"""Unit tests for the PURE progress helpers (no tqdm rendering, no real clock).

Covers the captured-mode throttle decision (``should_emit``) and the progress line formatter
(``format_progress_line`` / ``format_duration``). The tqdm bar itself is presentation and is not
exercised here.
"""

from __future__ import annotations

from pop_trainer.pretraining.progress import (
    format_duration,
    format_progress_line,
    should_emit,
)


def test_should_emit_first_call_always():
    # No prior emit -> emit (the first line of the epoch always lands).
    assert should_emit(now=0.0, last_emit_time=None, throttle_seconds=5.0, step=1, total_steps=10)


def test_should_emit_after_throttle_elapsed():
    assert should_emit(now=10.0, last_emit_time=4.0, throttle_seconds=5.0, step=3, total_steps=10)


def test_should_not_emit_mid_throttle():
    # Only 2s since the last line, throttle is 5s, not the final step -> suppress.
    assert not should_emit(
        now=6.0, last_emit_time=4.0, throttle_seconds=5.0, step=3, total_steps=10
    )


def test_should_emit_on_final_step_despite_throttle():
    # Final step lands even though we are mid-throttle, so the last line is never lost.
    assert should_emit(now=4.5, last_emit_time=4.0, throttle_seconds=5.0, step=10, total_steps=10)


def test_format_duration():
    assert format_duration(0) == "0:00"
    assert format_duration(65) == "1:05"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(-5) == "0:00"


def test_format_progress_line_content():
    line = format_progress_line(
        epoch=1,
        total_epochs=3,
        step=5,
        total_steps=20,
        loss=0.1234,
        probe=0.5678,
        elapsed_s=65.0,
        eta_s=195.0,
    )
    # No embedded newline (the printer adds it).
    assert "\n" not in line
    # Exact required substrings + known-value formatting.
    assert "epoch 1/3" in line
    assert "step 5/20" in line
    assert "loss=0.1234" in line
    assert "probe=0.5678" in line
    assert "elapsed=1:05" in line
    assert "ETA=3:15" in line
