"""``pop_trainer.utils`` — a LEAF CLI toolbox over the rest of the stack.

Each module here is a self-contained developer/operator tool run as ``python -m
pop_trainer.utils.<tool>`` (e.g. :mod:`pop_trainer.utils.model_info` inspects a trained SB3 PPO
checkpoint). The tools INSPECT / report on artifacts the production components produce; they are
conveniences, never part of any runtime code path.

Boundary: ``utils`` is a one-way sink. It MAY import ``core`` / ``models`` / ``rl`` (and the other
components) plus torch / stable-baselines3 / gymnasium / stdlib, but NOTHING in ``pop_trainer``
outside ``utils`` may import ``pop_trainer.utils`` — the stack never depends on the tools, so the
tools can reach across components without creating a cycle. This ``__init__`` is intentionally
import-light: importing the package pulls in no heavy deps; each tool's imports live in its own
module.
"""
