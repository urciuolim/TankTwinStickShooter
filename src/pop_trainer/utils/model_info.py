"""``model_info`` — inspect a trained SB3 PPO checkpoint's architecture + parameter counts.

Run as ``python -m pop_trainer.utils.model_info <ckpt.zip>``. Loads the checkpoint on CPU (the
schedules are SKIPPED on load via ``custom_objects`` so no env / learning-rate context is needed),
then reports:

* the observation space and the action space,
* the full ``model.policy`` repr,
* the ACTIVE encoder trunk class name
  (``model.policy.features_extractor.encoder.trunk.__class__.__name__``),
* per-section parameter counts (total / trainable / frozen).

PARAM-COUNT CORRECTNESS (the load-bearing detail): SB3's ``ActorCriticPolicy`` SHARES the one
features-extractor instance across its ``features_extractor`` / ``pi_features_extractor`` /
``vf_features_extractor`` references, so a naive sum over each named child's ``parameters()``
triple-counts the encoder. Every count here is DEDUPED by ``id(p)`` first, so the policy total is
exactly ``sum(p.numel() for unique p in model.policy.parameters())`` and the extractor section is
counted once. The numeric work lives in :func:`collect_model_info`, which RETURNS a
:class:`ModelInfo` (so it is testable without scraping stdout); :func:`format_model_info` renders it
and :func:`main` prints it.

Boundary: ``utils`` is a LEAF — it MAY import stable-baselines3 / torch / stdlib here. Nothing in
``pop_trainer`` outside ``utils`` imports this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from stable_baselines3 import PPO

if TYPE_CHECKING:  # type-only — keep the module import-light, no torch needed to import this file
    from torch.nn import Parameter

__all__ = [
    "ModelInfo",
    "SectionCounts",
    "collect_model_info",
    "format_model_info",
    "load_ppo_cpu",
    "main",
]

# Passed to PPO.load so the saved schedules are not reconstructed on load: we only inspect the
# network, never train, so a constant 0.0 learning_rate / clip_range / lr_schedule is enough to
# satisfy the load path without needing the original schedule closures.
_INSPECT_CUSTOM_OBJECTS: dict = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}


@dataclass(frozen=True)
class SectionCounts:
    """Parameter counts for one named section of the policy, DEDUPED by ``id(p)``.

    ``total`` is the unique-parameter element count, ``trainable`` those with ``requires_grad``,
    and ``frozen`` the rest (``total == trainable + frozen``). The dedupe matters because SB3
    shares the features extractor across the policy's pi/vf references — counting it once.
    """

    name: str
    total: int
    trainable: int
    frozen: int


@dataclass(frozen=True)
class ModelInfo:
    """The structured inspection result for one checkpoint (what :func:`main` prints).

    ``observation_space`` / ``action_space`` are the gymnasium spaces' ``repr``; ``policy_repr`` is
    the full ``model.policy`` repr; ``trunk_class`` is the active encoder trunk class name (or
    ``None`` if the policy has no ``features_extractor.encoder.trunk`` path); ``policy`` is the
    overall id-deduped count and ``extractor`` the (also id-deduped) features-extractor section, so
    a reader / a test can confirm the shared extractor is counted once.
    """

    checkpoint: str
    observation_space: str
    action_space: str
    policy_repr: str
    trunk_class: str | None
    policy: SectionCounts
    extractor: SectionCounts


def load_ppo_cpu(path: str | Path) -> PPO:
    """Load an SB3 PPO checkpoint on CPU with the inspect-only ``custom_objects`` (no schedules).

    ``PPO.load`` rebuilds the policy network from the zip; ``device="cpu"`` keeps it off the GPU
    and :data:`_INSPECT_CUSTOM_OBJECTS` skips reconstructing the saved learning-rate / clip-range
    schedules (we only read the network). No env is attached — ``.predict`` / inspection work
    without one.
    """
    return PPO.load(str(path), device="cpu", custom_objects=_INSPECT_CUSTOM_OBJECTS)


def _dedup_by_id(params: Iterable[Parameter]) -> Iterator[Parameter]:
    """Yield each parameter at most once, identity-deduped by ``id(p)`` (preserving first order).

    SB3 shares ONE features-extractor instance across the policy's features / pi / vf references,
    so the same ``Parameter`` objects appear under multiple names; summing per-name
    double/triple-counts them. Deduping by ``id`` collapses the aliases so each is counted once.
    """
    seen: set[int] = set()
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        yield p


def _count_section(name: str, params: Iterable[Parameter]) -> SectionCounts:
    """Total / trainable / frozen element counts over ``params``, id-deduped.

    See :func:`_dedup_by_id` — the shared features extractor is counted exactly once.
    """
    total = trainable = 0
    for p in _dedup_by_id(params):
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return SectionCounts(name=name, total=total, trainable=trainable, frozen=total - trainable)


def _resolve_trunk_class(policy: object) -> str | None:
    """Return ``policy.features_extractor.encoder.trunk.__class__.__name__`` or ``None`` if absent.

    The Phase-1 ``CnnPolicy`` uses our :class:`~pop_trainer.rl.extractor.EncoderExtractor`, whose
    ``.encoder.trunk`` is the active vision trunk (cnn / resnet / gn-cnn). A policy built
    with a different extractor has no such path, so this returns ``None`` rather than raising.
    """
    extractor = getattr(policy, "features_extractor", None)
    encoder = getattr(extractor, "encoder", None)
    trunk = getattr(encoder, "trunk", None)
    return None if trunk is None else type(trunk).__name__


def collect_model_info(model: PPO, *, checkpoint: str | Path = "") -> ModelInfo:
    """Compute the inspection :class:`ModelInfo` for a loaded PPO (PURE; no I/O, no print).

    The overall ``policy`` section is id-deduped over ``model.policy.parameters()`` so its ``total``
    equals the sum of ``p.numel()`` over the UNIQUE policy parameters (the shared extractor counted
    once); the ``extractor`` section is the id-deduped count over the features extractor alone. Both
    sections split trainable vs frozen by ``requires_grad``. Returning a dataclass keeps the numbers
    testable without scraping stdout.
    """
    policy = model.policy
    policy_section = _count_section("policy", policy.parameters())

    extractor = getattr(policy, "features_extractor", None)
    extractor_params: Iterable[Parameter] = extractor.parameters() if extractor is not None else ()
    extractor_section = _count_section("features_extractor", extractor_params)

    return ModelInfo(
        checkpoint=str(checkpoint),
        observation_space=repr(model.observation_space),
        action_space=repr(model.action_space),
        policy_repr=repr(policy),
        trunk_class=_resolve_trunk_class(policy),
        policy=policy_section,
        extractor=extractor_section,
    )


def _format_section(section: SectionCounts) -> str:
    """One-line ``name: total=.. trainable=.. frozen=..`` summary for a section."""
    return (
        f"  {section.name}: total={section.total:,} "
        f"trainable={section.trainable:,} frozen={section.frozen:,}"
    )


def format_model_info(info: ModelInfo) -> str:
    """Render a :class:`ModelInfo` to the multi-line report :func:`main` prints."""
    lines = [
        f"checkpoint:        {info.checkpoint}",
        f"observation space: {info.observation_space}",
        f"action space:      {info.action_space}",
        f"encoder trunk:     {info.trunk_class if info.trunk_class is not None else '(none)'}",
        "",
        "policy:",
        repr_indent(info.policy_repr),
        "",
        "parameters (id-deduped; the shared features extractor is counted once):",
        _format_section(info.policy),
        _format_section(info.extractor),
    ]
    return "\n".join(lines)


def repr_indent(text: str, prefix: str = "  ") -> str:
    """Indent every line of a (possibly multi-line) repr by ``prefix`` for the report body."""
    return "\n".join(prefix + line for line in text.splitlines())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.utils.model_info",
        description=(
            "Inspect a trained SB3 PPO checkpoint: print its observation/action spaces, the full "
            "policy, the active encoder trunk, and id-deduped (total/trainable/frozen) parameter "
            "counts. Loads on CPU with schedules skipped — no env or training context needed."
        ),
    )
    parser.add_argument("ckpt", type=Path, help="path to the SB3 PPO checkpoint (*.zip)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the checkpoint named on the CLI, compute the report, print it, return 0."""
    args = _parse_args(argv)
    model = load_ppo_cpu(args.ckpt)
    info = collect_model_info(model, checkpoint=args.ckpt)
    print(format_model_info(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
