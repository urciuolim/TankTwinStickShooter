"""Pure trajectory-diff for the timeScale-determinism gate (M1 Workstream C, part 2).

This module answers ONE question, numerically: are two recorded Unity trajectories
the same simulation? It is the verdict half of the determinism harness — the
:mod:`tank_twin.determinism_harness` drives the real build and feeds the two
``list[list[float]]`` trajectories (one 52-float state per step) it captured here.

Kept deliberately PURE so it is unit-testable with no Unity, no socket, no torch:
only ``numpy`` + stdlib. The diff computes, between two trajectories:

* exact / bitwise equality of every captured state,
* the max absolute per-element difference (over the overlapping prefix),
* the first step index at which they diverge (``None`` if identical),
* a per-field-group breakdown of the max diff (P1 tank, P1 bullets, P2 tank,
  P2 bullets), so a divergence can be attributed to movement vs. bullets vs. a
  particular player,
* the RL invariants the trainer actually cares about: same trajectory length,
  same winner, same total reward — derived through the SAME field map / reward
  logic the env uses (:data:`tank_twin.observation` field layout and
  :func:`tank_twin.rewards.step_reward`).

The 52-float state field map (from :mod:`tank_twin.observation`): P1 occupies
indices 0..25 ``[pos_x, pos_y, vec_x, vec_y, aim_x, aim_y, <5 bullets x 4>]`` and
P2 occupies 26..51 with the same shape. Absent bullets are reported by Unity as
the sentinel ``-100`` in every slot.

IMPORTANT (per the board decision): this module MEASURES, it does not JUDGE. There
is no pass/fail epsilon here — the CTO sets the pass bar after seeing the numbers.
``max_abs_diff == 0.0`` plus ``bitwise_identical is True`` is the only unambiguous
statement this module makes; everything else is a measurement for a human to weigh.

numpy + stdlib only; no sb3 / gym / torch / socket.
"""

from dataclasses import dataclass, field

import numpy as np

from tank_twin.rewards import step_reward

# --- 52-float state field map (mirrors tank_twin.observation) -----------------
STATE_LEN = 52
P1_TANK = slice(0, 6)  # pos_x, pos_y, vec_x, vec_y, aim_x, aim_y
P1_BULLETS = slice(6, 26)  # 5 bullets x (pos_x, pos_y, vec_x, vec_y)
P2_TANK = slice(26, 32)
P2_BULLETS = slice(32, 52)

# Named field groups for the per-group divergence breakdown.
FIELD_GROUPS = {
    "p1_tank": P1_TANK,
    "p1_bullets": P1_BULLETS,
    "p2_tank": P2_TANK,
    "p2_bullets": P2_BULLETS,
}

# Player index of P1 / the agent under training — matches rewards.PLAYER_1.
PLAYER_1 = 0


@dataclass(frozen=True)
class InvariantResult:
    """The RL-level invariants: do two trajectories agree on what the trainer sees?

    These are the signals that actually matter to self-play training even if the
    raw floats wobble: the episode is the same LENGTH, resolves to the same WINNER,
    and yields the same TOTAL terminal+shaping REWARD (computed via the real
    :func:`tank_twin.rewards.step_reward`, P1's perspective).
    """

    same_length: bool
    length_a: int
    length_b: int
    same_winner: bool
    winner_a: int | None
    winner_b: int | None
    same_total_reward: bool
    total_reward_a: float
    total_reward_b: float

    @property
    def all_match(self) -> bool:
        """True iff every RL invariant agrees between the two trajectories."""
        return self.same_length and self.same_winner and self.same_total_reward


@dataclass(frozen=True)
class DiffResult:
    """Structured result of diffing two trajectories. Pure measurement, no verdict.

    ``first_divergence_step`` is the index of the first state that differs at all
    (``None`` when bitwise-identical over the compared prefix). ``max_abs_diff`` is
    the largest absolute per-element difference over the overlapping prefix (0.0
    when identical). ``per_group_max_diff`` attributes that to a field group.
    ``compared_steps`` is the length of the overlap actually diffed (a
    length-mismatch is recorded in :attr:`invariants`, not silently ignored).
    """

    bitwise_identical: bool
    max_abs_diff: float
    first_divergence_step: int | None
    compared_steps: int
    per_group_max_diff: dict[str, float] = field(default_factory=dict)
    invariants: InvariantResult | None = None

    def to_dict(self) -> dict:
        """Plain-dict view for strict-JSON serialization (no numpy scalars)."""
        inv = self.invariants
        return {
            "bitwise_identical": bool(self.bitwise_identical),
            "max_abs_diff": float(self.max_abs_diff),
            "first_divergence_step": (
                None if self.first_divergence_step is None else int(self.first_divergence_step)
            ),
            "compared_steps": int(self.compared_steps),
            "per_group_max_diff": {k: float(v) for k, v in self.per_group_max_diff.items()},
            "invariants": (
                None
                if inv is None
                else {
                    "same_length": bool(inv.same_length),
                    "length_a": int(inv.length_a),
                    "length_b": int(inv.length_b),
                    "same_winner": bool(inv.same_winner),
                    "winner_a": inv.winner_a,
                    "winner_b": inv.winner_b,
                    "same_total_reward": bool(inv.same_total_reward),
                    "total_reward_a": float(inv.total_reward_a),
                    "total_reward_b": float(inv.total_reward_b),
                    "all_match": bool(inv.all_match),
                }
            ),
        }


def _as_array(trajectory) -> np.ndarray:
    """Coerce a ``list[list[float]]`` trajectory to a ``(steps, 52)`` float64 array.

    float64 so the diff is computed at full precision (the wire delivers JSON
    numbers, parsed as Python floats). Raises ``ValueError`` on a ragged / wrong-width
    trajectory rather than letting numpy build an object array silently.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    # An empty trajectory ([]) coerces to shape (0,); normalize it to (0, 52) so the
    # rest of the diff treats it as zero steps of valid-width state.
    if arr.size == 0:
        return arr.reshape(0, STATE_LEN)
    if arr.ndim != 2 or arr.shape[1] != STATE_LEN:
        raise ValueError(f"trajectory must be (steps, {STATE_LEN}); got shape {arr.shape!r}")
    return arr


def _winner_from_trajectory(winners) -> int | None:
    """Reduce a per-step list of winner signals to a single episode winner.

    ``winners`` is the per-step ``winner`` the wire reported (``None`` for steps
    with no winner key). The episode winner is the last non-``None`` entry — Unity
    reports the winner on the terminal step. ``None`` means no winner was ever
    reported (a max-time draw the game resolved without a winner key).
    """
    if winners is None:
        return None
    decided = [w for w in winners if w is not None]
    return decided[-1] if decided else None


def _total_reward(winners) -> float:
    """Total P1-perspective reward for a trajectory via the real reward logic.

    Runs :func:`tank_twin.rewards.step_reward` per step with ``time_reward=0`` (the
    determinism configs carry no shaping), so the total is just the terminal reward:
    ``+1`` P1 win, ``-1`` P1 loss, ``0`` draw / undecided. Using the SAME function
    the env uses keeps this honest — the invariant is "the trainer would have seen
    the same reward," not a re-derivation.
    """
    if winners is None:
        return 0.0
    total = 0.0
    for w in winners:
        reward, _terminated, _truncated = step_reward(winner=w, time_reward=0.0)
        total += reward
    return float(total)


def compute_invariants(
    trajectory_a,
    trajectory_b,
    winners_a=None,
    winners_b=None,
) -> InvariantResult:
    """Compare the RL-level invariants of two trajectories (length / winner / reward).

    ``winners_*`` are optional per-step winner lists (``None`` per step where no
    winner key arrived). When omitted, the winner is ``None`` and the total reward
    is ``0.0`` — i.e. "no decided result captured," which still lets the harness
    assert length agreement.
    """
    arr_a = _as_array(trajectory_a)
    arr_b = _as_array(trajectory_b)
    len_a, len_b = arr_a.shape[0], arr_b.shape[0]

    winner_a = _winner_from_trajectory(winners_a)
    winner_b = _winner_from_trajectory(winners_b)
    total_a = _total_reward(winners_a)
    total_b = _total_reward(winners_b)

    return InvariantResult(
        same_length=(len_a == len_b),
        length_a=len_a,
        length_b=len_b,
        same_winner=(winner_a == winner_b),
        winner_a=winner_a,
        winner_b=winner_b,
        same_total_reward=(total_a == total_b),
        total_reward_a=total_a,
        total_reward_b=total_b,
    )


def diff_trajectories(
    trajectory_a,
    trajectory_b,
    winners_a=None,
    winners_b=None,
) -> DiffResult:
    """Diff two trajectories element-wise and roll up the RL invariants.

    Both trajectories are ``list[list[float]]`` (one 52-float state per step). The
    element diff runs over the OVERLAPPING prefix (``min(len_a, len_b)`` steps); any
    length mismatch is surfaced via :attr:`DiffResult.invariants` (``same_length``),
    not hidden. ``first_divergence_step`` is the first prefix index with ANY nonzero
    element diff. ``per_group_max_diff`` attributes the max diff to a field group
    (p1_tank / p1_bullets / p2_tank / p2_bullets).

    Returns a :class:`DiffResult`. This is pure measurement — no pass/fail epsilon.
    """
    arr_a = _as_array(trajectory_a)
    arr_b = _as_array(trajectory_b)

    invariants = compute_invariants(trajectory_a, trajectory_b, winners_a, winners_b)

    n = min(arr_a.shape[0], arr_b.shape[0])
    if n == 0:
        # Nothing to compare element-wise. Bitwise-identical iff both are empty.
        return DiffResult(
            bitwise_identical=(arr_a.shape[0] == arr_b.shape[0] == 0),
            max_abs_diff=0.0,
            first_divergence_step=None,
            compared_steps=0,
            per_group_max_diff={name: 0.0 for name in FIELD_GROUPS},
            invariants=invariants,
        )

    pa = arr_a[:n]
    pb = arr_b[:n]

    # Bitwise equality is the strongest claim: identical IEEE-754 bits, prefix-wide
    # AND no length mismatch hiding past the overlap.
    bitwise_identical = bool(np.array_equal(pa, pb)) and (arr_a.shape[0] == arr_b.shape[0])

    abs_diff = np.abs(pa - pb)  # (n, 52)
    max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0

    # First step with any nonzero element difference (NaN-safe: NaN != NaN counts).
    per_step_differs = np.any((pa != pb) | (np.isnan(pa) != np.isnan(pb)), axis=1)
    diverged = np.nonzero(per_step_differs)[0]
    first_divergence_step = int(diverged[0]) if diverged.size else None

    per_group_max_diff = {
        name: (float(abs_diff[:, sl].max()) if abs_diff[:, sl].size else 0.0)
        for name, sl in FIELD_GROUPS.items()
    }

    return DiffResult(
        bitwise_identical=bitwise_identical,
        max_abs_diff=max_abs_diff,
        first_divergence_step=first_divergence_step,
        compared_steps=n,
        per_group_max_diff=per_group_max_diff,
        invariants=invariants,
    )
