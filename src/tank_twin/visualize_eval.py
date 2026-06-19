"""Standalone eval action visualizer: ONE model vs. the random opponent (M1 Workstream B).

Plays greedy (``deterministic=True``) episodes of a trained PPO model through a real
``TankEnv`` (against the random opponent — agent-vs-agent is DEFERRED to the league wave)
and renders, per step, what the agent SEES and DOES:

* (a) the agent's **36x60x3 synthetic observation** (the exact grid the CnnPolicy is fed),
  upscaled to a viewable RGB image. Channel map (from :mod:`tank_twin.observation`):
  R = self / P1, G = walls, B = opponent / P2.
* (b) five labeled horizontal gauges for the model's action vector (range -1..1), one per
  :data:`ACTION_LABELS` entry. The ``shoot`` gauge flags fire > 0.5 (the Unity
  ``triggerThreshold`` in ``PlayerController.cs``). Optionally a second, dimmer gauge row
  for the opponent's random action draw (label "opponent (random)").

Output: matplotlib with the headless **Agg** backend, saved as an animated **GIF** via
``PillowWriter`` (pillow ships as a matplotlib dependency — NO ffmpeg needed) to
``runs/<run-name>/eval_viz.gif`` by default. ``--live`` opens an interactive window
instead; ``--boardroom`` also copies the GIF into ``boardroom/`` for the port-8777 server.

CLI::

    python -m tank_twin.visualize_eval --model-path ... --game-path ... \
        [--episodes N] [--run-name ...] [--live] [--boardroom] [--out ...]

Design: the action -> gauge mapping is the PURE :func:`action_to_gauges` (a stdlib-only
dataclass list), unit-testable with NO matplotlib / torch / Unity. matplotlib is imported
lazily INSIDE the render functions, and torch / sb3 / the env only inside :func:`main`, so
``import tank_twin.visualize_eval`` (and the pure test) stays torch/matplotlib-free.

This module reuses :func:`tank_twin.evaluate.evaluate_winrate`'s greedy-loop SHAPE and the
``PretrainedNatureCNN`` extractor (importable from :mod:`tank_twin.features`, so
``PPO.load`` resolves the custom extractor) rather than re-implementing them. ``env.close()``
runs in a ``finally`` (the Workstream-B teardown fix) so a real Unity run never orphans the
``TankTwinStickShooter.exe`` / holds the socket port.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only: keep the pure path torch/sb3/matplotlib-free at runtime
    import numpy as np

__all__ = [
    "ACTION_LABELS",
    "SHOOT_INDEX",
    "SHOOT_THRESHOLD",
    "Gauge",
    "action_to_gauges",
    "main",
]

# Action layout, traced from PlayerController.cs (the AI branch reads myInput[0..4]):
#   [0] desired velocity x   [1] desired velocity y
#   [2] desired aim x        [3] desired aim y
#   [4] fire (Shoot() when myInput[4] > triggerThreshold == 0.5)
ACTION_LABELS: tuple[str, ...] = ("move_x", "move_y", "aim_x", "aim_y", "shoot")
SHOOT_INDEX = 4
SHOOT_THRESHOLD = 0.5  # PlayerController.triggerThreshold: fire when value > 0.5

# Gauge value range (the env's Box action space is [-1, 1] on every dim).
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# Colors for the gauge bars (matplotlib-style names; the renderer maps them to colors).
# Held here (not in the renderer) so the PURE mapping decides color and the test can assert
# it without importing matplotlib.
_COLOR_NEUTRAL = "steelblue"  # a directional axis (move / aim)
_COLOR_SHOOT_ON = "crimson"  # shoot gauge, fire > threshold (FIRING)
_COLOR_SHOOT_OFF = "gray"  # shoot gauge, below threshold (holding fire)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"
BOARDROOM_DIR = _REPO_ROOT / "boardroom"
BOARDROOM_GIF = BOARDROOM_DIR / "eval_viz.gif"


@dataclass(frozen=True)
class Gauge:
    """One labeled horizontal gauge for a single action dimension (PURE; no matplotlib).

    Attributes:
        label: the action-dimension label (an :data:`ACTION_LABELS` entry).
        value: the raw action value, CLAMPED into ``[-1, 1]`` (a model can predict slightly
            outside the Box during exploration; we clamp for display only — the wire value
            is untouched).
        fraction: ``value`` remapped to ``[0, 1]`` (``-1 -> 0``, ``0 -> 0.5``, ``1 -> 1``),
            i.e. the bar's fill position along the -1..1 track. Convenient for drawing.
        color: the bar color name (directional axes share one color; the shoot gauge is
            red when firing, gray otherwise).
        is_shoot: True for the ``shoot`` dimension (drawn against the threshold).
        firing: True only for the shoot gauge when ``value`` > :data:`SHOOT_THRESHOLD`.
    """

    label: str
    value: float
    fraction: float
    color: str
    is_shoot: bool
    firing: bool


def _clamp(value: float, low: float, high: float) -> float:
    return high if value > high else (low if value < low else value)


def action_to_gauges(action: Sequence[float]) -> list[Gauge]:
    """Map a 5-D action vector to five labeled :class:`Gauge`s. PURE (stdlib only).

    For each dimension this clamps the value into ``[-1, 1]`` (display-only), computes its
    ``[0, 1]`` fill ``fraction``, and assigns a color: directional axes (move / aim) share a
    neutral color; the ``shoot`` dimension is red when firing (value > 0.5) and gray
    otherwise. No matplotlib / numpy / torch — testable headless.

    Args:
        action: the model's action vector; must have exactly 5 entries (the env's Box(5)).

    Returns:
        A list of five :class:`Gauge`, in :data:`ACTION_LABELS` order.

    Raises:
        ValueError: if ``action`` does not have exactly 5 entries.
    """
    values = [float(v) for v in action]
    if len(values) != len(ACTION_LABELS):
        raise ValueError(
            f"action must have {len(ACTION_LABELS)} entries (one per {ACTION_LABELS}), "
            f"got {len(values)}"
        )

    gauges: list[Gauge] = []
    span = ACTION_MAX - ACTION_MIN
    for index, (label, raw) in enumerate(zip(ACTION_LABELS, values, strict=True)):
        clamped = _clamp(raw, ACTION_MIN, ACTION_MAX)
        fraction = (clamped - ACTION_MIN) / span
        is_shoot = index == SHOOT_INDEX
        if is_shoot:
            firing = clamped > SHOOT_THRESHOLD
            color = _COLOR_SHOOT_ON if firing else _COLOR_SHOOT_OFF
        else:
            firing = False
            color = _COLOR_NEUTRAL
        gauges.append(
            Gauge(
                label=label,
                value=clamped,
                fraction=fraction,
                color=color,
                is_shoot=is_shoot,
                firing=firing,
            )
        )
    return gauges


# --- rendering (matplotlib; imported lazily so the pure path stays matplotlib-free) ----


def _draw_gauges(ax, gauges: list[Gauge], *, title: str, dim: bool = False) -> None:
    """Draw one row group of horizontal gauges onto a matplotlib Axes.

    Each gauge is a -1..1 track with a marker at its value, a centered zero line, and a
    label + numeric value. ``dim`` lowers alpha (used for the opponent's random row). No
    return value; mutates ``ax``.
    """
    n = len(gauges)
    ax.set_xlim(ACTION_MIN, ACTION_MAX)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([g.label for g in gauges])
    ax.invert_yaxis()  # first label (move_x) on top
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_title(title, fontsize=10)
    base_alpha = 0.4 if dim else 1.0
    for y, g in enumerate(gauges):
        # Bar from zero to value so direction (sign) reads at a glance.
        ax.barh(y, g.value, height=0.5, color=g.color, alpha=base_alpha, zorder=2)
        # Shoot threshold marker on the shoot row.
        if g.is_shoot:
            ax.plot(
                [SHOOT_THRESHOLD, SHOOT_THRESHOLD],
                [y - 0.3, y + 0.3],
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=base_alpha,
            )
        suffix = "  FIRE" if g.firing else ""
        ax.text(
            ACTION_MAX,
            y,
            f" {g.value:+.2f}{suffix}",
            va="center",
            ha="left",
            fontsize=8,
            alpha=base_alpha,
        )


def _render_frame(
    fig,
    obs: np.ndarray,
    agent_gauges: list[Gauge],
    opp_gauges: list[Gauge] | None,
    *,
    episode: int,
    step: int,
) -> None:
    """Render ONE step's panel (obs image + gauge rows) into ``fig`` (cleared first)."""
    import numpy as np

    fig.clf()
    has_opp = opp_gauges is not None
    nrows = 3 if has_opp else 2
    gs = fig.add_gridspec(nrows, 1, height_ratios=[3, 2, 2] if has_opp else [3, 2])

    # (a) the agent's 36x60x3 observation, upscaled (nearest) for a crisp pixel grid.
    ax_obs = fig.add_subplot(gs[0, 0])
    image = np.asarray(obs)
    ax_obs.imshow(image, interpolation="nearest", aspect="auto")
    ax_obs.set_title(
        f"agent observation 36x60x3  (R=self  G=walls  B=opponent)  ep {episode} step {step}",
        fontsize=10,
    )
    ax_obs.set_xticks([])
    ax_obs.set_yticks([])

    # (b) the model's action gauges.
    ax_act = fig.add_subplot(gs[1, 0])
    _draw_gauges(ax_act, agent_gauges, title="agent action (greedy)")

    if has_opp:
        ax_opp = fig.add_subplot(gs[2, 0])
        _draw_gauges(ax_opp, opp_gauges, title="opponent (random)", dim=True)

    fig.tight_layout()


def visualize_episodes(
    model,
    env,
    *,
    out_path: str | Path,
    n_episodes: int = 1,
    seed: int | None = 0,
    show_opponent: bool = True,
    live: bool = False,
    fps: int = 8,
) -> Path:
    """Play ``n_episodes`` greedy episodes and render obs + action gauges to a GIF.

    Reuses :func:`tank_twin.evaluate.evaluate_winrate`'s loop shape (greedy
    ``predict(deterministic=True)`` until ``terminated or truncated``), but instead of
    counting wins it captures a frame per step. The opponent's random action is the SAME
    draw the env makes on the wire (``env.np_random.uniform`` in ``TankEnv.step``); to mirror
    it for display we read it back from the env's recorded transport when available.

    Args:
        model: a built/loaded SB3 model with ``.predict``.
        env: a built ``TankEnv`` (caller owns its lifecycle; ``main`` closes it).
        out_path: GIF output path (created; parents made).
        n_episodes: number of greedy episodes to record.
        seed: seed for the first reset (opponent reproducibility), as in evaluate_winrate.
        show_opponent: also draw the opponent's random action as a dimmed second row.
        live: if True, also show the frames in an interactive window (TkAgg) as they render.
        fps: GIF frames per second.

    Returns:
        The GIF path (``out_path``).
    """
    import matplotlib

    matplotlib.use("TkAgg" if live else "Agg")  # interactive window vs. headless GIF
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 8))
    writer = PillowWriter(fps=fps)
    frame_count = 0

    if live:
        plt.ion()
        plt.show()

    try:
        with writer.saving(fig, str(out), dpi=110):
            for episode in range(n_episodes):
                reset_seed = seed if (episode == 0 and seed is not None) else None
                obs, _info = env.reset(seed=reset_seed)
                terminated = truncated = False
                step = 0
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    agent_gauges = action_to_gauges(_as_list(action))

                    obs, _reward, terminated, truncated, _info = env.step(action)
                    step += 1

                    opp_gauges = None
                    if show_opponent:
                        opp_action = _last_opponent_action(env)
                        if opp_action is not None:
                            opp_gauges = action_to_gauges(opp_action)

                    _render_frame(
                        fig,
                        obs,
                        agent_gauges,
                        opp_gauges,
                        episode=episode,
                        step=step,
                    )
                    writer.grab_frame()
                    frame_count += 1
                    if live:
                        plt.pause(1.0 / max(fps, 1))
    finally:
        plt.close(fig)

    if frame_count == 0:
        raise RuntimeError(
            "visualize_episodes captured 0 frames — the env terminated/truncated before any "
            "step. Check the model/env wiring."
        )
    return out


def _as_list(action) -> list[float]:
    """Coerce a model action (numpy array / list / scalar-ish) to a plain float list."""
    try:
        return [float(v) for v in action]
    except TypeError:
        return [float(action)]


def _last_opponent_action(env) -> list[float] | None:
    """Read the opponent's last random action off the env's transport, if observable.

    ``TankEnv.step`` sends ``{1: agent, 2: opponent}`` over the wire; the opponent draw is
    the ``"2"`` entry of the most recent message. We read it from the recorded transport
    (real ``socket.socket`` does not record, so this returns ``None`` and the opponent row is
    simply omitted — the agent gauges still render). This does NOT touch the wire or the
    52-float layout; it only inspects what the env already sent.
    """
    conn = getattr(env, "conn", None)
    transport = getattr(conn, "transport", None)
    received = getattr(transport, "received", None)
    if not received:
        return None
    last = received[-1]
    opp = last.get("2") if isinstance(last, dict) else None
    if opp is None:
        return None
    try:
        return [float(v) for v in opp]
    except (TypeError, ValueError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tank_twin.visualize_eval",
        description=(
            "Render a trained PPO model's greedy eval (obs + action gauges) vs. the random "
            "opponent to an animated GIF."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved SB3 model (.zip) to visualize.",
    )
    parser.add_argument(
        "--game-path",
        type=Path,
        required=True,
        help="Path to the Unity build executable to launch and play against.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of greedy episodes to record (default: 1).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="m1-local",
        help="Run name -> default output runs/<run-name>/eval_viz.gif.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GIF path (default: runs/<run-name>/eval_viz.gif).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the first reset (opponent reproducibility).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="GIF frames per second (default: 8).",
    )
    parser.add_argument(
        "--no-opponent",
        dest="show_opponent",
        action="store_false",
        help="Do not draw the opponent's random-action row (default: draw it, dimmed).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open an interactive window instead of (in addition to) saving the GIF.",
    )
    parser.add_argument(
        "--boardroom",
        action="store_true",
        help="Also copy the GIF to boardroom/eval_viz.gif (served at /eval_viz.gif).",
    )
    parser.add_argument(
        "--game-port",
        type=int,
        default=50000,
        help="TCP port for the Unity build (bump it to dodge a TIME_WAIT-held port).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for inference (auto -> cuda if available, else cpu).",
    )
    return parser


def _resolve_device(device: str) -> str:
    """Map ``auto`` -> ``cuda`` when available, else ``cpu``; pass others through."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main(argv: Sequence[str] | None = None) -> Path:
    """CLI entry point: load a model, launch Unity, render the GIF, return its path.

    Imports torch / sb3 / the env lazily so ``import tank_twin.visualize_eval`` (and the
    pure gauge-mapping test) stays cheap and torch/matplotlib-free. Reuses the
    ``PretrainedNatureCNN`` extractor (so ``PPO.load`` resolves the custom features class)
    and the Workstream-B ``env.close()`` (in a ``finally``) so no Unity orphan is left.
    """
    import shutil

    from stable_baselines3 import PPO

    # Importing features registers PretrainedNatureCNN so PPO.load can resolve the saved
    # custom feature-extractor class (mirrors evaluate.py loading models/m1-smoke.zip).
    import tank_twin.features  # noqa: F401
    from tank_twin.env import TankEnv

    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    out_path = (
        args.out if args.out is not None else DEFAULT_RUNS_DIR / args.run_name / "eval_viz.gif"
    )

    model = PPO.load(str(args.model_path), device=device)
    env = TankEnv(game_path=str(args.game_path), game_port=args.game_port)
    try:
        gif_path = visualize_episodes(
            model,
            env,
            out_path=out_path,
            n_episodes=args.episodes,
            seed=args.seed,
            show_opponent=args.show_opponent,
            live=args.live,
            fps=args.fps,
        )
    finally:
        env.close()

    print(f"Wrote eval visualization to {gif_path}")

    if args.boardroom:
        BOARDROOM_GIF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gif_path, BOARDROOM_GIF)
        print(f"Copied to {BOARDROOM_GIF} (boardroom server serves /eval_viz.gif)")

    return gif_path


if __name__ == "__main__":
    main()
