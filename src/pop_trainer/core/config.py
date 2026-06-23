"""Typed config dataclasses for a training run (run / env / reward).

Frozen dataclasses with STRICT-JSON loaders. stdlib-only (``dataclasses`` + ``json`` +
``pathlib``) so this stays in the pure import path — nothing heavy (no numpy / torch / sb3).

Three configs:

* :class:`RewardConfig` — budget-based reward knobs (full-episode totals).
* :class:`EnvConfig` — environment knobs (pixels/square, episode cap, socket address).
* :class:`RunConfig` — top-level run identity (name + seed) nesting an ``EnvConfig`` and a
  ``RewardConfig``.

Every config shares the same loader API: ``to_dict`` / ``from_dict`` (REJECTS unknown keys
with ``TypeError``; missing keys fall back to defaults) / ``from_json`` (STRICT
``json.loads`` — a trailing comma or leading-dot float RAISES) / ``load(path)``. Nested
configs round-trip through the dict form.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

__all__ = ["RewardConfig", "EnvConfig", "RunConfig"]


def _reject_unknown(cls, data: dict) -> None:
    """Raise ``TypeError`` if ``data`` has keys that are not fields of ``cls``."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise TypeError(f"unknown {cls.__name__} keys: {sorted(unknown)}")


@dataclass(frozen=True)
class RewardConfig:
    """Budget-based reward knobs (full-episode totals).

    * ``win_reward`` / ``loss_reward``: the +/-1 terminal added on the decided step.
    * ``time_total``: total time penalty over a full episode (accrues every step).
    """

    win_reward: float = 1.0
    loss_reward: float = -1.0
    time_total: float = -1.0

    def to_dict(self) -> dict:
        """Plain-dict view; float-cast for JSON stability."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> RewardConfig:
        """Build from a dict; REJECT unknown keys (``TypeError``); missing -> defaults."""
        _reject_unknown(cls, data)
        return cls(**dict(data))

    @classmethod
    def from_json(cls, text: str) -> RewardConfig:
        """Parse a STRICT-JSON object string into a RewardConfig."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError(f"RewardConfig JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> RewardConfig:
        """Load a RewardConfig from a STRICT-JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EnvConfig:
    """Environment knobs.

    * ``env_p``: pixels per game square for any world->grid coordinate mapping.
    * ``max_steps``: episode step cap (also the reward budgets' ``max_episode_length``).
    * ``game_ip`` / ``game_port``: the Unity simulator's TCP address.
    * ``sock_timeout``: per-recv socket timeout in seconds.
    """

    env_p: int = 3
    max_steps: int = 300
    game_ip: str = "127.0.0.1"
    game_port: int = 50000
    sock_timeout: float = 10.0

    def to_dict(self) -> dict:
        """Plain-dict view (JSON-ready scalars)."""
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> EnvConfig:
        """Build from a dict; REJECT unknown keys (``TypeError``); missing -> defaults."""
        _reject_unknown(cls, data)
        return cls(**dict(data))

    @classmethod
    def from_json(cls, text: str) -> EnvConfig:
        """Parse a STRICT-JSON object string into an EnvConfig."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError(f"EnvConfig JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> EnvConfig:
        """Load an EnvConfig from a STRICT-JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunConfig:
    """Top-level run identity: a name + seed nesting an ``EnvConfig`` and ``RewardConfig``.

    ``env`` and ``reward`` round-trip through their own dict forms inside ``to_dict`` /
    ``from_dict``, so the whole config is one STRICT-JSON object on disk.
    """

    run_name: str
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self) -> dict:
        """Plain-dict view with the nested configs expanded to their own dicts."""
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "env": self.env.to_dict(),
            "reward": self.reward.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        """Build from a dict; REJECT unknown keys; nested ``env`` / ``reward`` are configs.

        ``env`` / ``reward`` may be omitted (defaults) or provided as nested dicts, which are
        themselves validated by ``EnvConfig.from_dict`` / ``RewardConfig.from_dict`` (so an
        unknown nested key also raises ``TypeError``).
        """
        _reject_unknown(cls, data)
        kwargs: dict = {k: v for k, v in data.items() if k not in ("env", "reward")}
        if "env" in data:
            kwargs["env"] = EnvConfig.from_dict(data["env"])
        if "reward" in data:
            kwargs["reward"] = RewardConfig.from_dict(data["reward"])
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> RunConfig:
        """Parse a STRICT-JSON object string into a RunConfig."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError(f"RunConfig JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load a RunConfig from a STRICT-JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
