"""Budget-based shaped reward + episode-boundary logic (pure, stdlib-only).

This is env/domain logic, NOT core: it turns the wire signals one step reports
(``winner`` / ``done`` / ``max_steps`` / lost connection) plus a per-episode time budget
into the gymnasium ``(reward, terminated, truncated)`` triple that :meth:`TankEnv.step`
returns. It is a PURE function — no socket, no gym, no numpy — so the reward is unit-testable
in isolation from the environment wiring.

The reward has two components:

* a per-step TIME PENALTY that accrues EVERY non-lost-connection step. The full-episode
  ``time_total`` budget (e.g. ``-1.0``) is spread over ``max_steps`` (e.g. 300) by
  :func:`time_penalty_per_step` -> ``-1/300`` per step.
* a WIN / LOSS TERMINAL ADDED (not overwriting) on the decided step: ``+ win_reward`` when
  P1 (the agent) won, ``+ loss_reward`` when the opponent won. ADDING means a late loss is
  worse than ``loss_reward`` alone (the accrued time penalty stacks on top).
* a DRAW (explicit ``winner == -1``) or a winner-less ``done`` terminal contributes only the
  accrued time penalty — no terminal component.

Episode boundaries (gymnasium semantics):

* ``terminated`` — the game DECIDED the round this step (a winner, or a bare ``done``). A
  real game result.
* ``truncated`` — a time-limit cutoff (``max_steps_reached``) on a step the game did NOT
  decide, OR a LOST CONNECTION. Neither is a decided game.
* lost connection -> reward ``0.0``, ``truncated`` (no shaping accrues); the game is counted
  as ending with no winner.

``survivor`` mode flips the terminal COMPONENT: the flip keys off whether a winner KEY was
reported, NOT the winner VALUE. Any reported winner (``0``, ``1``, or ``-1`` draw) scores
``loss_reward``; only a winner-key-ABSENT terminal (a bare ``done`` the game resolved without
a winner) scores ``win_reward`` (you survived). The accrued time penalty is KEPT under the
flip. Under this asymmetry a ``winner != -1`` decided game becomes a loss for BOTH players.

SANITY EQUIVALENCE: with ``time_penalty == 0`` and ``win/loss == ±1`` this reduces to the
plain ±1 terminal (continuing / draw -> ``0.0``; P1 win -> ``+1``; opponent win -> ``-1``).
"""

from __future__ import annotations

# Player index of the agent under training (matches core.state.PLAYER_1). Kept local so this
# pure module imports nothing — it is the boundary translation, not the state schema.
PLAYER_1 = 0


def time_penalty_per_step(time_total: float, max_steps: int) -> float:
    """Convert a full-episode time budget to the per-step time penalty.

    ``time_total`` (e.g. ``-1.0``) spread over ``max_steps`` (e.g. 300) -> ``-1/300`` per
    step. Accrues every non-lost-connection step (continuing, draw, and the decided step).
    Raises ``ValueError`` on a non-positive ``max_steps`` (a zero budget per step is
    undefined).
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    return time_total / max_steps


def shaped_step_reward(
    winner: int | None = None,
    *,
    done: bool = False,
    survivor: bool = False,
    time_penalty: float = 0.0,
    win_reward: float = 1.0,
    loss_reward: float = -1.0,
    max_steps_reached: bool = False,
    lost_connection: bool = False,
) -> tuple[float, bool, bool]:
    """Compute ``(reward, terminated, truncated)`` for one env step.

    Args:
        winner: ``0`` (P1/agent), ``1`` (opponent), ``-1`` (explicit draw), or ``None`` when
            the game reported no winner this step.
        done: the raw ``"done" in received`` flag (round over, possibly with no winner key).
        survivor: survivor-mode terminal flip (see module docstring; keys off winner-key
            PRESENCE, not the winner value).
        time_penalty: the per-step time penalty (already converted from the budget by
            :func:`time_penalty_per_step`). Accrues on EVERY non-lost-connection step.
        win_reward / loss_reward: the terminal ADDED on a decided P1 / opponent win.
        max_steps_reached: the env hit its step-count cap this step (truncation).
        lost_connection: the transport dropped -> reward ``0.0``, ``truncated`` (no shaping).
    """
    # Transport failure: a flat reward-0 truncation. NOT a decided game, so no shaping
    # accrues (the game ends with no winner; the env then reconnects).
    if lost_connection:
        return 0.0, False, True

    game_done = done or winner is not None

    # The time penalty accrues on every step (continuing, draw, decided, or truncation).
    reward = time_penalty
    terminated = False

    if game_done:
        terminated = True
        if winner is not None and winner != -1:
            # ADD (not overwrite) the terminal so a late loss is worse than loss_reward alone.
            reward += win_reward if winner == PLAYER_1 else loss_reward

    # Survivor flips the terminal COMPONENT (key-presence rule): any reported winner ->
    # loss_reward, a winner-key-absent terminal -> win_reward. Applied AFTER the winner
    # branch, replacing the terminal component while KEEPING the accrued time penalty.
    if survivor and game_done:
        winner_reported = winner is not None
        reward = time_penalty + (loss_reward if winner_reported else win_reward)

    # Time-limit cutoff is a truncation, never a termination. If the game ALSO decided this
    # step, the real terminal result takes precedence (stay terminated).
    truncated = bool(max_steps_reached) and not terminated

    return reward, terminated, truncated
