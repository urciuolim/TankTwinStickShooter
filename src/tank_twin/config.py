"""Typed reward configuration (pure, stdlib-only).

The CTO's budget-based reward, parameterized by EPISODE BUDGETS so the knobs read as
"how much total reward/penalty does a FULL episode contribute" rather than opaque
per-step constants. :class:`RewardConfig` holds those budgets; the per-step conversion
(budget / ``max_episode_length``) lives in :mod:`tank_twin.rewards` so this module stays
a tiny pure dataclass with a strict-JSON loader and NOTHING heavy imported.

Budgets (full-episode totals), with ``max_episode_length`` = the env ``max_steps`` (300):

* ``win_reward`` / ``loss_reward``: the ±1 terminal added on the decided step.
* ``time_total``: total time penalty over a full episode (default ``-1.0`` -> ``-1/300``
  per step). Accrues EVERY step.
* ``action_total``: total action-cost budget at CONSTANT MAX action over a full episode
  (default ``-0.1``). The per-step action cost scales LINEARLY with the action's L1
  magnitude: ``(action_total / max_episode_length) * (L1(action) / action_norm)`` — zero
  at zero action, ``action_total`` over a full episode at constant max action.
* ``action_norm``: the L1 normalizer (``L1_MAX``; default ``5.0`` = the 5-dim action box's
  max L1, so a saturated action has ``L1/action_norm == 1``).

stdlib-only (``json`` + ``dataclasses``) so this stays in the pure import path: no
numpy / sb3 / torch here. STRICT json (``json.load``/``json.loads`` reject trailing
commas / leading-dot floats — Python's ``json`` does this natively; we add no tolerance).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = ["RewardConfig"]


@dataclass(frozen=True)
class RewardConfig:
    """Budget-based reward knobs (full-episode totals; see module docstring).

    Defaults are the CTO's new reward:
    ``win=+1, loss=-1, time_total=-1.0, action_total=-0.1, action_norm=5.0``.
    Setting ``time_total`` and ``action_total`` to ``0.0`` (with ``win/loss = ±1``)
    reproduces the legacy ±1-only terminal reward byte-for-byte.
    """

    win_reward: float = 1.0
    loss_reward: float = -1.0
    time_total: float = -1.0
    action_total: float = -0.1
    action_norm: float = 5.0

    def to_dict(self) -> dict:
        """Plain-dict view (e.g. for the run manifest). Float-cast for JSON-stability."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> RewardConfig:
        """Build from a dict, ignoring nothing and REJECTING unknown keys.

        Unknown keys raise ``TypeError`` (a misspelled knob must fail loudly, not be
        silently dropped). Missing keys fall back to the dataclass defaults.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"unknown RewardConfig keys: {sorted(unknown)}")
        return cls(**{k: data[k] for k in data})

    @classmethod
    def from_json(cls, text: str) -> RewardConfig:
        """Parse a STRICT-JSON object string into a RewardConfig.

        ``json.loads`` rejects trailing commas / leading-dot floats (no tolerance added).
        The top-level value must be a JSON object.
        """
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError(f"RewardConfig JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> RewardConfig:
        """Load a RewardConfig from a STRICT-JSON file at ``path``."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
