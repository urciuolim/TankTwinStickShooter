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
