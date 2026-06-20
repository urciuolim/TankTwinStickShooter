"""Terminal-reward + episode-boundary logic (pure, stdlib-only).

Extracted from the 2021 ``TankEnv.step`` reward branch (``PythonScripts/tank_env.py``)
and UPGRADED to gymnasium semantics: instead of the legacy single ``done`` flag this
returns the ``(reward, terminated, truncated)`` triple gymnasium's 5-tuple ``step``
expects. The reward NUMBERS are preserved byte-for-byte from 2021; only the
done -> (terminated, truncated) split is new.

Mapping (P1 is player index 0, the agent under training; the opponent is index 1):

* ``winner == 0``  -> reward ``+1``, terminated (agent won)
* ``winner == 1``  -> reward ``-1``, terminated (opponent won)
* ``winner == -1`` (an explicit draw signalled this step) -> ``time_reward``, terminated
* no winner/done this step -> ``time_reward``, neither terminated nor truncated
* ``max_steps`` reached -> ``truncated`` (NOT terminated) — a time-limit cutoff
* lost connection -> ``truncated`` (NOT terminated) — a transport failure, not a game result

``survivor`` mode flips the terminal reward exactly as the legacy did. CRUCIAL
detail preserved from 2021: the legacy keyed the flip off whether a ``"winner"``
KEY was reported (``"winner" in info``), NOT off the winner VALUE. The legacy set
``info["winner"]`` whenever a winner key arrived — INCLUDING an explicit
``winner == -1`` draw. So under survivor, ANY step that reported a winner key
(``0``, ``1``, or ``-1``) scores ``-1``, and only a winner-key-absent terminal
(a bare ``done`` / timeout the game resolved without a winner) scores ``+1``
(you survived). A ``winner != -1`` decided game also becomes ``-1`` for BOTH
players because the survivor flip runs AFTER the winner branch — that asymmetry
is preserved deliberately.

No numpy/sb3/gym/torch — plain Python so the test net needs no heavy deps.
"""

PLAYER_1 = 0


def step_reward(
    winner=None,
    *,
    done=False,
    survivor=False,
    time_reward=0.0,
    max_steps_reached=False,
    lost_connection=False,
):
    """Compute ``(reward, terminated, truncated)`` for one env step.

    Parameters mirror the signals the env reads off the wire / its own counters:

    * ``winner``: ``0`` (P1/agent), ``1`` (opponent), ``-1`` (explicit draw), or
      ``None`` when the game did not report a winner this step.
    * ``done``: the raw legacy ``"done" in received`` flag (the game declared the
      round over even without a winner key — e.g. a timeout the game resolved).
    * ``survivor``: survivor-mode reward flip (see module docstring).
    * ``time_reward``: per-step shaping reward when the round continues / draws.
    * ``max_steps_reached``: the env hit its own step-count cap this step.
    * ``lost_connection``: the transport dropped (legacy returned a no-winner end).

    Reward numbers are identical to the 2021 ``TankEnv.step``.
    """
    # Transport failure: legacy returned (state, 0, done=True, {"lost_connection":True}).
    # Under gymnasium this is a truncation (not a real terminal game result), reward 0.
    if lost_connection:
        return 0.0, False, True

    # The game declared a winner this step, or otherwise declared the round done.
    game_done = done or winner is not None

    reward = time_reward
    terminated = False

    if game_done:
        terminated = True
        if winner is not None and winner != -1:
            reward = 1 if winner == PLAYER_1 else -1

    # Survivor mode flips the terminal reward (matches legacy: applied to the same
    # `done` set, AFTER the winner branch). The legacy keyed off `"winner" in info`,
    # i.e. whether a winner KEY was reported at all — a `winner == -1` draw counts as
    # "winner present" and scores -1. Only a winner-key-absent terminal scores +1.
    if survivor and game_done:
        winner_reported = winner is not None
        reward = -1 if winner_reported else 1

    # Time-limit cutoff: truncation, never termination. The legacy did not end the
    # episode here (the wrapper did); gymnasium models it explicitly. If the game
    # also ended this very step, a real terminal result takes precedence (stay
    # terminated); otherwise it's a pure truncation with the step's shaping reward.
    truncated = bool(max_steps_reached) and not terminated

    return reward, terminated, truncated


# --- budget-based reward (CTO; the new env path) --------------------------------------
#
# The legacy step_reward above OVERWRITES the reward to ±1 on a decided game; the budget
# reward instead ACCRUES per-step penalties EVERY step and ADDS the ±1 terminal on the
# decided step (so a late loss is worse than ±1 alone). The budgets live in
# tank_twin.config.RewardConfig as FULL-EPISODE totals; the two helpers below convert
# them to per-step values (the env knows max_episode_length and has the action in hand),
# and shaped_step_reward composes the per-step penalties with the ±1 terminal.
#
# BACKWARD-COMPAT (hard requirement): with time_penalty == 0 and action_cost == 0 and
# win/loss == ±1, shaped_step_reward returns the SAME (reward, terminated, truncated) as
# the legacy step_reward — the existing truth-table tests pin step_reward and stay green
# (it is untouched), and new tests pin the budgets==0 equivalence.

# The 5-dim action box is [-1, 1]^5, so the maximum L1 magnitude is 5.0 (a saturated
# action). RewardConfig.action_norm defaults to this so L1(action)/action_norm in [0, 1].
L1_MAX = 5.0


def time_penalty_per_step(time_total, max_episode_length):
    """Convert a full-episode time budget to the per-step time penalty.

    ``time_total`` (e.g. ``-1.0``) spread over ``max_episode_length`` steps (e.g. 300)
    -> ``-1/300`` per step. Accrues every step (continuing, draw, and the decided step).
    """
    return time_total / max_episode_length


def action_cost_per_step(action, action_total, max_episode_length, action_norm=L1_MAX):
    """Per-step action cost = ``(action_total / max_episode_length) * (L1(action)/norm)``.

    Linear in the action's L1 magnitude: ``0`` at zero action; at a CONSTANT MAX action
    (``L1 == action_norm``) it equals ``action_total / max_episode_length`` per step, so a
    full episode at max action totals ``action_total`` (e.g. ``-0.1``). ``action`` is any
    iterable of the 5 action dims; only its magnitude matters (sign-insensitive — L1).
    """
    l1 = sum(abs(float(a)) for a in action)
    return (action_total / max_episode_length) * (l1 / action_norm)


def shaped_step_reward(
    winner=None,
    *,
    done=False,
    survivor=False,
    time_penalty=0.0,
    action_cost=0.0,
    win_reward=1.0,
    loss_reward=-1.0,
    max_steps_reached=False,
    lost_connection=False,
):
    """Budget-based ``(reward, terminated, truncated)`` — penalties accrue, ±1 is ADDED.

    Differs from :func:`step_reward` in ONE way: instead of overwriting the reward to ±1
    on a decided game, it ACCRUES ``time_penalty + action_cost`` every step and ADDS the
    terminal (``win_reward`` if P1 won, ``loss_reward`` if the opponent won) on the
    decided step. A draw / winner-less ``done`` terminal contributes only the accrued
    per-step penalties (no ±1). The ``terminated`` / ``truncated`` split and the
    ``survivor`` / ``lost_connection`` / ``max_steps`` semantics match ``step_reward``.

    BACKWARD-COMPAT: with ``time_penalty == 0``, ``action_cost == 0`` and
    ``win/loss == ±1`` this returns the legacy ``step_reward`` numbers byte-for-byte
    (continuing/draw -> 0.0; P1 win -> +1; opp win -> -1).

    Args:
        winner: ``0`` (P1/agent), ``1`` (opponent), ``-1`` (explicit draw), or ``None``.
        done: raw legacy ``"done" in received`` flag (round over, possibly no winner).
        survivor: survivor-mode terminal flip (same key-presence rule as ``step_reward``).
        time_penalty: per-step time penalty (already converted from the budget). Accrues
            on EVERY non-lost-connection step.
        action_cost: per-step action cost (already converted; magnitude-scaled). Accrues
            on EVERY non-lost-connection step, INCLUDING through to a loss.
        win_reward / loss_reward: the terminal added on a decided P1 / opponent win.
        max_steps_reached: env hit its step-count cap this step (truncation).
        lost_connection: transport dropped -> reward ``0.0``, truncated (no shaping).
    """
    # Transport failure: a flat reward-0 truncation, exactly like step_reward — NOT a
    # game result, so no shaping accrues either.
    if lost_connection:
        return 0.0, False, True

    game_done = done or winner is not None

    # Per-step penalties accrue on every step (continuing, draw, decided, or truncation).
    reward = time_penalty + action_cost
    terminated = False

    if game_done:
        terminated = True
        if winner is not None and winner != -1:
            # ADD (not overwrite) the terminal so a late loss is worse than -1 alone.
            reward += win_reward if winner == PLAYER_1 else loss_reward

    # Survivor flips the terminal COMPONENT (same key-presence rule as the legacy): any
    # reported winner -> loss_reward, a winner-key-absent terminal -> win_reward. Applied
    # AFTER the winner branch (matching step_reward), replacing the terminal component
    # while the accrued per-step penalties are KEPT.
    if survivor and game_done:
        winner_reported = winner is not None
        reward = (time_penalty + action_cost) + (loss_reward if winner_reported else win_reward)

    truncated = bool(max_steps_reached) and not terminated

    return reward, terminated, truncated
