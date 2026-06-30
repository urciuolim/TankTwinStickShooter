"""Runnable play app: drive any pairing of human / rule-based / trained-RL players through the
live Tank game.

``python -m pop_trainer.play`` launches the real Unity build WINDOWED, opens a TCP socket to it,
wraps that socket in :class:`pop_trainer.core.protocol.Connection`, constructs a
:class:`pop_trainer.env.tank_env.TankEnv` over it, and runs ONE episode with the two policies named
on ``--player1`` / ``--player2`` so a human can watch (and play). BOTH policies are driven by this
module via a small per-player ADAPTER keyed by ``obs_kind``: a ``state`` / ``keyboard`` adapter acts
on the 52-float state (player1 the unflipped ``info["state"]``, player2 its FLIPPED first-person
view via :func:`core.state.split_state_for_opponent`); a ``pixels`` (RL) adapter acts on the
rendered pixel FRAME (player1 the frame as-is, player2 the FLIPPED frame via
:meth:`TankEnv.player2_frame`).
Both actions are passed to ``env.step(a1, a2)`` — the env is a bare pure transport that owns neither
player.

Three player FORMS on each slot:

* ``human`` — keyboard play (:class:`pop_trainer.agents.HumanAgent`); a SINGLE shared
  :class:`~pop_trainer.agents.KeyboardListener` + state feeds BOTH humans (two players, one
  keyboard). ``pynput`` is the optional ``human`` extra, imported lazily by the listener.
* a rule-based selector from :data:`pop_trainer.agents.AGENT_SELECTORS`
  (``aggressive-coverage`` / ``wall-hugger`` / ``opponent-shadower`` / ``random`` / ``noop``).
* ``rl:<checkpoint.zip>`` — a trained Stable-Baselines3 PPO model loaded from that path (a bare
  ``*.zip`` path is also accepted). Any matchup works: rl-vs-human, rl-vs-rulebased, rl-vs-rl, etc.

BOUNDARY (INTENTIONAL, CTO-APPROVED RELAXATION): ``play`` is a COMPOSITION-ROOT APP, like
``rl/train.py``. At module top it imports ``env`` + ``agents`` + ``core`` only. It is a LEAF entry
point — imported by NOTHING. To load a trained PPO checkpoint it reaches ``stable_baselines3``
DIRECTLY (it does NOT import ``pop_trainer.rl`` at all); that direct sb3 dependency is the
deliberate relaxation of demo's old "torch-free" rule, and the repo-steward must NOT treat it as a
violation. Still FORBIDDEN: ``rl``, ``models``, ``data``, ``pretraining``, anything from
``tank_twin``. ``stable_baselines3`` / ``torch`` are imported LAZILY — ONLY inside
:func:`make_rl_player` — so human / rule-based play stays torch-free and fast; the module-top
imports are ``env`` + ``agents`` + ``core`` only (a unit test enforces it).

The episode-driving logic is the pure :func:`run_play_episode` (an already-built env + two player
adapters in, a :class:`PlayResult` out); the subprocess launch and socket connect live in
:func:`main`, which the tests do NOT exercise.

PIXELS MUST BE ON. ``TankEnv.reset``/``step`` always read a length-prefixed pixel frame after each
state JSON, but the shipped ``unity/Assets/StreamingAssets/config.json`` has no ``obs_pixels`` key,
so the build defaults ``obsPixels=false`` and sends NO frame — the env would block waiting for
bytes that never arrive. Play therefore launches the build with
``unity/Assets/StreamingAssets/demo_config.json``, which sets ``"obs_pixels": true`` plus
``obs_pixels_width``/``obs_pixels_height``, and DERIVES the env's ``frame_shape`` (H, W, 3) from
those SAME keys via :func:`core.obs.frame_shape_from_config` — so the env always MATCHES whatever
resolution the launched config declares (one source of truth, no hand-synced constant). That
config lives inside StreamingAssets next to the ``Arenas/`` directory so its relative ``arena_path``
(``Arenas/custom1.json``) resolves against the config directory — exactly how
``DriverController.ResolveArenaPath`` resolves it. It also drops ``timeScale`` from the shipped 20
to a human-watchable 2.

HUMAN PLAY. Passing ``--player1 human`` and/or ``--player2 human`` makes that player
keyboard-driven: a SINGLE shared :class:`~pop_trainer.agents.KeyboardListener` + state feeds BOTH
agents, started before the episode and stopped on teardown. When a player is human the default
config switches to ``human_config.json`` (real-time ``timeScale: 1``) unless ``--config`` is given.
RL players do NOT change the config default (they run at the bots cadence unless a human is also
present or ``--config`` is passed).
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pop_trainer import agents
from pop_trainer.agents import AGENT_SELECTORS, make_agent  # the canonical selector registry
from pop_trainer.core import agent as core_agent
from pop_trainer.core import state as S
from pop_trainer.core.config import EnvConfig
from pop_trainer.core.launch import build_launch_cmd, connect, default_build_path
from pop_trainer.core.obs import frame_shape_from_config
from pop_trainer.core.protocol import Connection, WallLayout
from pop_trainer.env.tank_env import TankEnv

# --- launch / wire defaults -------------------------------------------------------------

# The repo root is three parents up from this file (src/pop_trainer/play.py -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
# OS-aware default to the launchable build binary (.exe / .app inner / bare); --exe overrides it.
DEFAULT_EXE = default_build_path(_REPO_ROOT / "unity")
# The play config enables obs_pixels and lives in StreamingAssets so its relative arena_path
# resolves against the config directory (DriverController.ResolveArenaPath).
DEFAULT_CONFIG = _REPO_ROOT / "unity" / "Assets" / "StreamingAssets" / "demo_config.json"
# Real-time human play needs a slower, more forgiving cadence than the watch-the-bots default:
# timeScale 1, a responsive ai_actionFreq, and more health for a duel. Used as the default config
# when EITHER player is "human" (unless --config is explicitly overridden).
HUMAN_CONFIG = _REPO_ROOT / "unity" / "Assets" / "StreamingAssets" / "human_config.json"
DEFAULT_PORT = 50000

# The pixel frame_shape (H, W, 3) is DERIVED from the launched config's obs_pixels_* at runtime
# (frame_shape_from_config), so setting obs_pixels_width/height in a config auto-matches the env
# with no manual sync. There is no hardcoded resolution constant here anymore.

DEFAULT_MAX_STEPS = 600
DEFAULT_SEED = 0


# --- player forms -----------------------------------------------------------------------
# The string -> agent registry (``AGENT_SELECTORS`` + ``make_agent``) is owned by
# ``pop_trainer.agents`` (the single source of truth) and imported above. The default pairing is
# player1 = aggressive-coverage (sweeps the arena, aim-sweeps + fires) vs player2 =
# opponent-shadower (the same coverage movement but the aim layer tracks player1 and fires):
# two visibly different policies.
DEFAULT_PLAYER1 = "aggressive-coverage"
DEFAULT_PLAYER2 = "opponent-shadower"

# The "human" form is NOT a registry seed-factory (it needs a SHARED keyboard listener + state that
# a per-agent seed factory cannot express), so it is a SPECIAL PATH in main, NOT an AGENT_SELECTORS
# entry.
HUMAN_SELECTOR = "human"
# An "rl:" player spec is "rl:<path/to/checkpoint.zip>"; a bare "*.zip" path is also accepted.
RL_PREFIX = "rl:"


@dataclass(frozen=True)
class PlayerSpec:
    """A parsed ``--player1`` / ``--player2`` string, classified into one of the three forms.

    Exactly one form per spec:

    * ``human`` is ``True`` for the ``human`` form (keyboard play; ``selector`` / ``rl_path`` None).
    * ``selector`` is an :data:`AGENT_SELECTORS` key for the rule-based form.
    * ``rl_path`` is the checkpoint path for the ``rl:<path>`` (or bare ``*.zip``) form.

    ``raw`` keeps the original string for the printed trace.
    """

    raw: str
    human: bool = False
    selector: str | None = None
    rl_path: Path | None = None

    @property
    def is_human(self) -> bool:
        return self.human

    @property
    def is_rl(self) -> bool:
        return self.rl_path is not None


def parse_player_spec(value: str) -> PlayerSpec:
    """Classify a ``--player1`` / ``--player2`` string into a :class:`PlayerSpec`.

    Accepts exactly three forms and rejects everything else with an actionable message naming the
    valid forms:

    * ``human`` -> the keyboard-play form.
    * any :data:`AGENT_SELECTORS` key -> the rule-based form.
    * ``rl:<path>`` (the ``rl:`` prefix stripped to the checkpoint path) OR a bare ``*.zip`` path
      -> the trained-RL form. The path's existence is NOT checked here (the RL factory validates it
      when the player is actually built); a structurally-invalid string is rejected here.

    The closed ``choices=`` of the old demo could not express ``rl:<path>`` — this validator
    replaces it (used as argparse ``type=``).
    """
    if value == HUMAN_SELECTOR:
        return PlayerSpec(raw=value, human=True)
    if value in AGENT_SELECTORS:
        return PlayerSpec(raw=value, selector=value)
    if value.startswith(RL_PREFIX):
        path = value[len(RL_PREFIX) :]
        if not path:
            raise argparse.ArgumentTypeError(
                f"invalid player {value!r}: 'rl:' needs a checkpoint path, "
                f"e.g. rl:runs/m1/model.zip"
            )
        return PlayerSpec(raw=value, rl_path=Path(path))
    if value.endswith(".zip"):
        return PlayerSpec(raw=value, rl_path=Path(value))

    valid = ", ".join(sorted(AGENT_SELECTORS))
    raise argparse.ArgumentTypeError(
        f"invalid player {value!r}: expected 'human', one of [{valid}], "
        f"or 'rl:<checkpoint.zip>' (a bare '*.zip' path is also accepted)"
    )


# --- per-player adapters ----------------------------------------------------------------
# A player adapter maps the per-step (frame, state) pair + the player slot (1 or 2) to a 5-float
# action, giving each form the RIGHT perspective. State-reading agents always act as PLAYER_1 on
# their own view (the self-play convention), so player2 is handed the FLIPPED state; an RL pixels
# adapter is handed the FLIPPED frame for player2. The loop computes BOTH players' actions from the
# SAME current frame/state before stepping (a simultaneous move).


class StateAdapter:
    """Drives a state-reading agent (rule-based OR human) on the 52-float state.

    player1 acts on the unflipped state; player2 acts on ``split_state_for_opponent`` of it (its
    first-person view). A ``HumanAgent`` ignores the passed view (it polls its shared
    ``KeyboardState``) but is handed one anyway — the same path, no special-case.
    """

    obs_kind = "state"

    def __init__(self, agent: core_agent.Agent, *, slot: int):
        self.agent = agent
        self.slot = slot

    def act(self, frame, state):  # noqa: ARG002 (state adapters ignore the frame)
        view = state if self.slot == S.PLAYER_1 else S.split_state_for_opponent(np.asarray(state))
        return self.agent.act(view)

    def reset(self) -> None:
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset()

    def set_map(self, layout) -> None:
        set_map = getattr(self.agent, "set_map", None)
        if callable(set_map):
            set_map(layout)


class PixelsAdapter:
    """Drives a trained RL model on the rendered pixel FRAME.

    player1 acts on the frame as-is; player2 acts on ``env.player2_frame()`` (the R/B-swapped
    first-person frame), handed in by the loop. The action is
    ``predict(frame, deterministic=True)[0]`` — SB3's CnnPolicy does its own HWC->NCHW + normalize,
    so the RAW ``(H, W, 3)`` uint8 frame is fed unchanged (no transpose / no normalize). ``predict``
    is injectable so a test can drive the adapter with a fake (no real SB3 / no real ``.zip``).
    """

    obs_kind = "pixels"

    def __init__(self, predict, *, slot: int):
        self._predict = predict
        self.slot = slot

    def act(self, frame, state):  # noqa: ARG002 (RL adapters ignore the state)
        action, _ = self._predict(frame, deterministic=True)
        return action


def make_rl_player(checkpoint: str | Path, *, slot: int, model=None) -> PixelsAdapter:
    """Build a :class:`PixelsAdapter` over a trained SB3 PPO checkpoint (the ONLY trained-model
    seam).

    This is the SOLE place ``stable_baselines3`` / ``torch`` are imported (via
    ``from stable_baselines3 import PPO``; ``pop_trainer.rl`` is NOT imported), and the import is
    LAZY (inside the function) — invoked only when an ``rl:`` player is requested, so human /
    rule-based play stays torch-free. ``PPO.load(checkpoint)`` is predict-only (no env needed for
    ``.predict``). The ``checkpoint`` path must exist (an actionable error otherwise).

    ``model`` is the test seam: pass an already-built model (any object with a ``.predict``) to
    inject a fake ``predict`` WITHOUT touching the filesystem or SB3; when ``None`` the checkpoint
    is validated and loaded here.
    """
    if model is None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"RL checkpoint not found at {path} — pass an existing PPO '*.zip' "
                f"(e.g. --player1 rl:runs/m1/model.zip)"
            )
        from stable_baselines3 import PPO

        model = PPO.load(str(path))
    return PixelsAdapter(model.predict, slot=slot)


def build_players(
    spec1: PlayerSpec,
    spec2: PlayerSpec,
    state: agents.KeyboardState,
    *,
    seed: int | None = None,
):
    """Build the (player1, player2) ADAPTER pair from the two parsed specs.

    Handles all three forms and shares the ONE ``state`` across any human players:

    * ``human`` -> a :class:`~pop_trainer.agents.HumanAgent` over the player's keymap + the shared
      ``state``, wrapped in a :class:`StateAdapter`.
    * a rule-based selector -> :func:`make_agent`, wrapped in a :class:`StateAdapter`.
    * ``rl:<path>`` -> :func:`make_rl_player` (the lazy ``stable_baselines3`` import; the checkpoint
      loads here).

    Pure with respect to the keyboard: it takes an already-built ``state`` and never constructs a
    listener / touches ``pynput``. RL loading is NOT injectable here — the RL-adapter unit tests
    build :func:`make_rl_player` directly with a fake ``model``.
    """
    map1, map2 = agents.player1_mapping(), agents.player2_mapping()
    p1 = _build_one(spec1, slot=S.PLAYER_1, mapping=map1, state=state, seed=seed)
    p2 = _build_one(spec2, slot=S.PLAYER_2, mapping=map2, state=state, seed=seed)
    return p1, p2


def _build_one(spec: PlayerSpec, *, slot: int, mapping, state, seed):
    """Build ONE player adapter for ``spec`` in the given slot (shared by both slots)."""
    if spec.is_human:
        return StateAdapter(agents.HumanAgent(mapping, state), slot=slot)
    if spec.is_rl:
        return make_rl_player(spec.rl_path, slot=slot)
    return StateAdapter(make_agent(spec.selector, seed=seed), slot=slot)


# --- pure episode loop (unit-tested against a fake transport) ----------------------------


@dataclass
class PlayResult:
    """The outcome of one play episode (everything the trace / a test needs).

    * ``steps`` — number of ``env.step`` calls made.
    * ``terminated`` / ``truncated`` — the gymnasium boundary that ended the episode.
    * ``winner`` — the reported winner int (``0`` == player1, ``1`` == player2, ``-1`` ==
      draw) on a decided terminal, else ``None``.
    * ``outcome`` — the env's free-form result tag (``"win"`` / ``"loss"`` / ``"draw"``) from
      player1's perspective on a decided terminal, else ``None``.
    * ``total_reward`` — the summed player1 step reward.
    * ``map`` — the tracked :class:`pop_trainer.core.protocol.WallLayout` (``info["map"]``), or
      ``None`` when no arena/walls were configured.
    """

    steps: int
    terminated: bool
    truncated: bool
    winner: int | None
    outcome: str | None
    total_reward: float
    map: WallLayout | None


def run_play_episode(
    env: TankEnv,
    player1,
    player2,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> PlayResult:
    """Drive ONE episode of an already-built bare ``env`` with the player1 / player2 adapters.

    ``env`` is a bare pure-transport :class:`TankEnv` that owns neither player; this loop drives
    BOTH adapters. It resets each adapter if it exposes ``reset``, calls ``env.reset()``, then each
    step holds BOTH the current pixel FRAME and the current 52-float STATE: it computes player1's
    action from the current ``(frame, state)`` and player2's from the SAME current ``(frame,
    state)`` (a simultaneous move) — a ``state`` adapter reads the state with the right view, a
    ``pixels`` adapter reads the frame (``env.player2_frame()`` for player2, the FLIP of the env's
    CURRENT frame). Both actions feed ``env.step(a1, a2)`` until ``terminated or truncated`` or
    ``max_steps`` is reached. The transport lives inside ``env``, so this is unit-testable against a
    fake-backed ``TankEnv`` with no subprocess and no live socket.
    """
    _maybe_reset(player1)
    _maybe_reset(player2)

    frame, info = env.reset()
    state_vec = info["state"]
    tracked_map = info.get("map")

    # Hand the static layout to a map-aware adapter (the OPTIONAL ``set_map`` hook). A map-agnostic
    # adapter does not forward it (StateAdapter probes the wrapped agent; PixelsAdapter has none).
    if tracked_map is not None:
        for player in (player1, player2):
            set_map = getattr(player, "set_map", None)
            if callable(set_map):
                set_map(tracked_map)

    steps = 0
    total_reward = 0.0
    terminated = truncated = False
    winner: int | None = None
    outcome: str | None = None

    while steps < max_steps:
        # Both actions come from the SAME current frame/state (simultaneous move). player2's pixels
        # view is env.player2_frame() — the FLIP of the env's CURRENT frame (this step's frame).
        a1 = player1.act(frame, state_vec)
        a2 = player2.act(env.player2_frame(), state_vec)
        frame, reward, terminated, truncated, info = env.step(a1, a2)
        steps += 1
        total_reward += float(reward)
        if "map" in info and info["map"] is not None:
            tracked_map = info["map"]
        if "state" in info:
            state_vec = info["state"]
        if terminated or truncated:
            winner = info.get("winner")
            outcome = info.get("outcome")
            break

    return PlayResult(
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        winner=winner,
        outcome=outcome,
        total_reward=total_reward,
        map=tracked_map,
    )


def _maybe_reset(player: object) -> None:
    """Call ``player.reset()`` if the (optional) method exists — ``StateAdapter`` forwards it to a
    seeded / stateful agent for a deterministic replay; ``PixelsAdapter`` has none."""
    reset = getattr(player, "reset", None)
    if callable(reset):
        reset()


# --- live launch + socket connect (NOT unit-tested; the shared seam lives in core.launch) -


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pop_trainer.play",
        description=(
            "Play the Tank Twin Stick Shooter through TankEnv with any pairing of player forms: "
            "'human' (keyboard), a rule-based selector, or 'rl:<checkpoint.zip>' (a trained PPO "
            "model). Example: --player1 rl:runs/m1/model.zip --player2 human."
        ),
    )
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="path to the Unity build")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port for the build")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "config JSON for the build (MUST enable obs_pixels; the env frame_shape is derived "
            "from its obs_pixels_width/height); defaults to human_config.json when a player is "
            "'human', else demo_config.json"
        ),
    )
    _player_help = (
        "player form: 'human' (keyboard), a rule-based selector "
        "[" + ", ".join(sorted(AGENT_SELECTORS)) + "], or 'rl:<checkpoint.zip>' (a trained PPO "
        "model; a bare '*.zip' path also works)"
    )
    parser.add_argument(
        "--player1",
        type=parse_player_spec,
        default=parse_player_spec(DEFAULT_PLAYER1),
        help="player1 " + _player_help,
    )
    parser.add_argument(
        "--player2",
        type=parse_player_spec,
        default=parse_player_spec(DEFAULT_PLAYER2),
        help="player2 " + _player_help,
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def _print_trace(result: PlayResult, *, player1: str, player2: str) -> None:
    """Print a short human-readable summary of the episode."""
    print(f"player1={player1}  vs  player2={player2}")
    print(f"steps:        {result.steps}")
    print(f"terminated:   {result.terminated}")
    print(f"truncated:    {result.truncated}")
    print(f"winner:       {result.winner}")
    print(f"outcome (p1): {result.outcome}")
    print(f"total reward: {result.total_reward:.4f}")
    if result.map is None:
        print("map:          (no walls tracked)")
    else:
        m = result.map
        print(f"map:          {m.map_id} ({len(m.occupied)} wall cells)")


def main(argv: list[str] | None = None) -> int:
    """Launch the build, connect, run one episode, print the trace, tear down cleanly."""
    args = _parse_args(argv)
    spec1: PlayerSpec = args.player1
    spec2: PlayerSpec = args.player2

    human_play = spec1.is_human or spec2.is_human
    # A human player needs real-time cadence; pick human_config.json by default, unless the user
    # explicitly passed --config. RL players do not change the default cadence.
    if args.config is not None:
        config = args.config
    else:
        config = HUMAN_CONFIG if human_play else DEFAULT_CONFIG

    if not args.exe.exists():
        print(f"error: build not found at {args.exe}", file=sys.stderr)
        return 2
    if not config.exists():
        print(f"error: config not found at {config}", file=sys.stderr)
        return 2

    # DERIVE the env's pixel frame_shape from the config we actually launch with (its obs_pixels_*),
    # so a 64x64 config auto-yields a (64, 64, 3) env with no hardcoded constant to drift.
    frame_shape = frame_shape_from_config(config)

    # For human play, ONE listener owns ONE shared KeyboardState that both HumanAgents read; the
    # listener is started before the episode and stopped in the teardown. pynput is imported lazily
    # by KeyboardListener, so a missing 'human' extra fails here with an actionable message.
    listener = agents.KeyboardListener() if human_play else None
    state = listener.state if listener is not None else agents.KeyboardState()
    # Build the adapters before launching the build: an RL player loads its checkpoint here (the
    # lazy stable_baselines3 import), so a missing '*.zip' errors BEFORE any subprocess is spawned.
    try:
        player1, player2 = build_players(spec1, spec2, state, seed=args.seed)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cmd = build_launch_cmd(args.exe, args.port, config)
    print("launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)  # noqa: S603 (arg-list, trusted local build path)

    env: TankEnv | None = None
    try:
        if listener is not None:
            listener.start()
        connection = Connection(connect(args.port))
        env = TankEnv(
            connection=connection,
            frame_shape=frame_shape,
            env_config=EnvConfig(max_steps=args.max_steps),
            seed=args.seed,
        )
        result = run_play_episode(env, player1, player2, max_steps=args.max_steps)
        _print_trace(result, player1=spec1.raw, player2=spec2.raw)
    finally:
        if listener is not None:
            with contextlib.suppress(Exception):
                listener.stop()
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        _terminate(proc)
    return 0


def _terminate(proc: subprocess.Popen) -> None:
    """Reap the build subprocess, escalating to ``kill`` if it does not exit promptly."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
