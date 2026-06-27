"""Training progress reporting for the decoder harness — PRESENTATION ONLY.

A dual-mode reporter so a run is visible whether it is watched live or captured to a file:

* INTERACTIVE (``sys.stderr.isatty()``): a live per-epoch ``tqdm`` bar over batches with the
  running combined / probe loss in the postfix.
* NON-TTY (redirected / captured over SSH): THROTTLED, NEWLINE-terminated progress LINES (no
  ``\r`` overwrite) so a captured log stays tail-able.

The throttle decision and the line format are PURE functions (time is injected, no I/O) and are the
testable core; :class:`ProgressReporter` owns the single TTY-vs-non-TTY branch (the one tqdm config
site) and exposes an ``on_batch`` callback the training loop calls. The callback is a pure
observability side-channel — it receives only ints / floats and returns nothing, so it cannot
perturb training (loss math, optimizer steps, RNG, or the results / checkpoint output).

tqdm + stdlib only; nothing from ``env`` / ``rl``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from tqdm import tqdm

__all__ = [
    "should_emit",
    "format_progress_line",
    "format_duration",
    "ProgressReporter",
]


def should_emit(
    *,
    now: float,
    last_emit_time: float | None,
    throttle_seconds: float,
    step: int,
    total_steps: int,
) -> bool:
    """PURE: should a captured-mode progress line be emitted at ``step`` right now?

    Time is INJECTED (``now`` / ``last_emit_time``) — this never reads the clock. Emits when:

    * it is the FIRST line of the epoch (``last_emit_time is None``), or
    * at least ``throttle_seconds`` have elapsed since the last line, or
    * it is the FINAL step of the epoch (``step >= total_steps``), so the last line always lands.

    Otherwise returns ``False`` (mid-throttle). No I/O, no side effects.
    """
    if last_emit_time is None:
        return True
    if total_steps > 0 and step >= total_steps:
        return True
    return (now - last_emit_time) >= throttle_seconds


def format_duration(seconds: float) -> str:
    """PURE: format ``seconds`` as ``H:MM:SS`` (``MM:SS`` under an hour). Negative -> ``0:00``."""
    secs = int(max(0.0, seconds))
    hours, rem = divmod(secs, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def format_progress_line(
    *,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    probe: float,
    elapsed_s: float,
    eta_s: float,
) -> str:
    """PURE: the captured-mode progress line (no trailing newline — the printer adds it).

    Form: ``epoch E/Es step S/Ss loss=…f probe=…f elapsed=H:MM:SS ETA=H:MM:SS``. Losses use a
    fixed 4-decimal format; times are ``format_duration``. ``epoch`` / ``step`` are 1-based for
    display.
    """
    return (
        f"epoch {epoch}/{total_epochs} "
        f"step {step}/{total_steps} "
        f"loss={loss:.4f} probe={probe:.4f} "
        f"elapsed={format_duration(elapsed_s)} ETA={format_duration(eta_s)}"
    )


class ProgressReporter:
    """Per-epoch progress reporter owning the single TTY-vs-non-TTY decision (one tqdm site).

    Construct once per run with the run shape, then per epoch call :meth:`epoch_start` (with that
    epoch's batch count), pass :attr:`on_batch` to ``train_one_epoch``, then :meth:`epoch_close`.

    * TTY -> a ``tqdm`` bar over batches on ``sys.stderr`` (default in-place ``\r`` overwrite),
      the running losses in the postfix.
    * non-TTY -> throttled, newline-terminated ``print(line, file=sys.stderr, flush=True)`` lines
      via :func:`should_emit` + :func:`format_progress_line` (tail-able; NO ``\r``).

    ``enabled=False`` makes every method a no-op (the ``--no-progress`` path). The clock and
    isatty probe are INJECTED so the class is testable without a real TTY / wall clock.
    """

    def __init__(
        self,
        *,
        total_epochs: int,
        enabled: bool = True,
        throttle_seconds: float = 5.0,
        is_tty: bool | None = None,
        time_fn: Callable[[], float] | None = None,
    ):
        import time

        self.total_epochs = total_epochs
        self.enabled = enabled
        self.throttle_seconds = throttle_seconds
        self.is_tty = sys.stderr.isatty() if is_tty is None else is_tty
        self._time = time_fn if time_fn is not None else time.monotonic

        self._epoch = 0
        self._total_steps = 0
        self._epoch_start_t = 0.0
        self._last_emit_time: float | None = None
        self._bar: tqdm | None = None

    def epoch_start(self, epoch: int, total_steps: int) -> None:
        """Begin reporting epoch ``epoch`` (1-based) over ``total_steps`` batches."""
        if not self.enabled:
            return
        self._epoch = epoch
        self._total_steps = total_steps
        self._epoch_start_t = self._time()
        self._last_emit_time = None
        if self.is_tty:
            self._bar = tqdm(
                total=total_steps,
                file=sys.stderr,
                unit="batch",
                desc=f"epoch {epoch}/{self.total_epochs}",
                leave=False,
            )

    def on_batch(self, step: int, loss: float, probe: float) -> None:
        """``train_one_epoch`` callback: ``step`` (1-based) just finished with running losses.

        TTY: advance the bar + update the postfix. Non-TTY: emit a throttled, newline-terminated
        line. A no-op when disabled. Touches no tensor / optimizer / RNG — pure observability.
        """
        if not self.enabled:
            return
        if self.is_tty:
            if self._bar is not None:
                self._bar.update(1)
                self._bar.set_postfix(loss=f"{loss:.4f}", probe=f"{probe:.4f}", refresh=False)
            return

        now = self._time()
        if not should_emit(
            now=now,
            last_emit_time=self._last_emit_time,
            throttle_seconds=self.throttle_seconds,
            step=step,
            total_steps=self._total_steps,
        ):
            return
        elapsed = now - self._epoch_start_t
        eta = (elapsed / step) * (self._total_steps - step) if step > 0 else 0.0
        line = format_progress_line(
            epoch=self._epoch,
            total_epochs=self.total_epochs,
            step=step,
            total_steps=self._total_steps,
            loss=loss,
            probe=probe,
            elapsed_s=elapsed,
            eta_s=eta,
        )
        print(line, file=sys.stderr, flush=True)
        self._last_emit_time = now

    def epoch_close(self) -> None:
        """End the current epoch's reporting (close the TTY bar if any). No-op when disabled."""
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.close()
            self._bar = None
