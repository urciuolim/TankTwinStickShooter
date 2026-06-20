"""Gymnasium single-agent environment wrapping the Unity simulator over the socket.

M1 Wave 2 (task 2.1): the gym -> gymnasium port of the 2021 ``PythonScripts/tank_env.py``
``TankEnv``, single-agent + image-based + random-opponent slice. This module COMPOSES
the already-migrated pure modules rather than duplicating their logic:

* :func:`tank_twin.arenas.load_level` builds the wall-baked obs grid + dims (geometry),
* :func:`tank_twin.observation.draw_state` renders the raw-52 state into the RGB grid,
* :func:`tank_twin.rewards.step_reward` turns the wire signals into
  ``(reward, terminated, truncated)`` (gymnasium semantics),
* :mod:`tank_twin.protocol` (``Connection`` / ``encode`` / ``decode``) does the strict-JSON
  wire I/O over an INJECTABLE transport.

What changed from 2021 (deliberately, and ONLY the env API — never the wire):

* gym -> gymnasium: ``reset(*, seed, options) -> (obs, info)`` and
  ``step(action) -> (obs, reward, terminated, truncated, info)`` (the legacy single
  ``done`` is split by ``rewards.step_reward``).
* The random opponent draws from ``self.np_random`` (seeded via ``super().reset(seed=...)``)
  instead of a bare ``np.random`` — so ``reset(seed=0)`` is reproducible end to end.
* The transport is injected: tests drive a FAKE in-process socket (no Unity, no
  subprocess). The real path still launches the Unity build via a ``subprocess`` arg-list
  (``[game_path, str(port)]``) and wraps the socket in ``protocol.Connection``.

RL seam: the wire contract is BYTE-IDENTICAL to 2021 — the handshake
(``{"restart":True}`` -> ack, ``{"start":True}`` -> ``"starting"`` ack, first ``state``),
the step message (``{1: action, 2: opp_action}`` with INTEGER keys), and the inbound
``state`` / ``winner`` / ``done`` keys all match. The legacy ``PythonScripts/tank_env.py``
is untouched. No C# / 52-float-layout changes.

This module imports gymnasium + numpy + the pure tank_twin modules; it stays sb3/torch-FREE
(it is just the env — the trainer / feature extractor import torch elsewhere).
"""

import contextlib
import json
import socket
import subprocess
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces

from tank_twin.arenas import load_level
from tank_twin.observation import draw_state
from tank_twin.protocol import Connection
from tank_twin.rewards import step_reward
from tank_twin.state import flip_state

# Resolved relative to the repo root so the env works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The build's StreamingAssets config.json — the SINGLE SOURCE of the arena the
# standalone build actually simulates when launched with NO ``--config`` (it falls
# back to this file). It declares ``"arena_path": "Arenas/custom1.json"`` (resolved
# relative to the config dir). The env's DEFAULT reads THIS SAME config so the obs
# wall grid matches the arena the build plays out of the box (single-source).
DEFAULT_CONFIG_PATH = _REPO_ROOT / "Assets" / "StreamingAssets" / "config.json"

# A bare arena JSON, kept for back-compat / tests that want an explicit arena with no
# config indirection. NOT the default any more: it MISMATCHED the build (the build
# simulates custom1, default.json has a different interior wall layout), which made the
# obs G (wall) channel disagree with the live game. The default is now config-derived.
DEFAULT_LEVEL_PATH = _REPO_ROOT / "Assets" / "Arenas" / "default.json"

# Player index of the agent under training (P1). Matches rewards.PLAYER_1 / draw_state.
PLAYER_1 = 0


def _build_game_cmd(game_path, game_port, config_path=None):
    """Build the Unity-launch subprocess ARG-LIST (never a shell string).

    Returns ``[game_path, str(game_port)]`` and, when ``config_path`` is set, appends
    ``["--config", <absolute config path>]`` so the build's
    ``DriverController.ResolveConfigPath`` loads the SAME config the env reads its arena
    from. The path is made absolute via ``Path.resolve()`` so the build resolves the
    config's relative ``arena_path`` against the right directory regardless of the
    build's working directory. When ``config_path`` is ``None`` the list is exactly the
    legacy ``[game_path, port]`` and the build falls back to its StreamingAssets config.

    Forwarding ``--config`` is config PLUMBING, not an RL-seam change: it does NOT touch
    the wire bytes, the socket/``actions`` path, or ``GameController.UpdateState()``'s
    52-float state layout. ``DriverController`` already parses ``--config``; this only
    chooses which config file the build reads.
    """
    cmd = [str(game_path), str(game_port)]
    if config_path is not None:
        cmd += ["--config", str(Path(config_path).resolve())]
    return cmd


def _arena_path_from_config(config_path):
    """Resolve a config's ``arena_path`` to an absolute arena JSON path (STRICT JSON).

    Mirrors ``DriverController.ResolveArenaPath``: parse the config with strict
    ``json.load`` (no trailing commas / leading-dot floats — Python's ``json`` rejects
    what Unity's Newtonsoft tolerates), read ``arena_path``, and resolve it RELATIVE TO
    THE CONFIG FILE'S DIRECTORY (an absolute ``arena_path`` is used as-is). This is the
    single source: the same config feeds both the build's simulated arena (via
    ``--config``) and the env's obs wall grid (via this resolved arena path).

    Reading a config to pick an arena for the obs is config plumbing, not an RL-seam
    change — the wire/state layout are untouched.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    arena_path = config["arena_path"]
    arena = Path(arena_path)
    if arena.is_absolute():
        return arena
    return config_path.parent / arena_path


class TankEnv(gymnasium.Env):
    """Single-agent, image-based, random-opponent gymnasium env over the Unity socket.

    Composes the migrated pure modules; see the module docstring. Construct with a real
    ``game_path`` (launches Unity + opens a socket) OR with ``game_path=None`` and an
    injected ``transport`` (a socket-like fake) so tests run with no Unity/subprocess.

    Args:
        game_path: path to the Unity build. ``None`` skips the subprocess launch (test /
            injected-transport mode).
        transport: optional socket-like object (``sendall(bytes)`` / ``recv(int)``). When
            given it is wrapped in ``protocol.Connection`` and NO socket is constructed —
            this is the test seam. When ``None`` and ``game_path`` is set, a real TCP
            socket is opened to the launched Unity build.
        config_path: a game config JSON (the same shape as the build's StreamingAssets
            ``config.json``). When set it is the SINGLE SOURCE of the arena: (a) it is
            FORWARDED to the build launch as ``--config <abspath>`` so the build simulates
            this config's arena, and (b) the env reads the obs wall grid from the SAME
            config's ``arena_path`` (resolved relative to the config dir). When ``None``
            and no ``level_path`` is given, the env defaults to reading the build's
            StreamingAssets config (``DEFAULT_CONFIG_PATH`` -> custom1) but does NOT
            forward ``--config`` (the build falls back to that same StreamingAssets
            config on its own) — so the default obs matches the default build arena,
            single-sourced. Forwarding ``--config`` is config plumbing, NOT an RL-seam
            change (the wire bytes / 52-float state are untouched).
        level_path: an explicit arena JSON (back-compat / tests). Bypasses the config
            indirection: the env reads this arena directly and does NOT forward
            ``--config``. PRECEDENCE: ``config_path`` (if given) wins; else ``level_path``
            (if given); else the default config (custom1, single-sourced with the build).
        env_p: pixels per game-grid square. **3** so the obs is (36, 60, 3) for the CNN.
        rand_opp: random opponent (the only opponent mode this single-agent port ships).
        max_steps: step-count cap; reaching it truncates (never terminates).
        time_reward: per-step shaping reward while the round continues.
        survivor: survivor-mode reward flip (passed through to ``rewards.step_reward``).
        game_ip / game_port / my_port / sock_timeout / num_connection_attempts /
        game_log_path: real-socket connection params (ignored when a transport is injected).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        game_path=None,
        *,
        transport=None,
        config_path=None,
        level_path=None,
        env_p=3,
        rand_opp=True,
        image_based=True,
        max_steps=300,
        time_reward=0.0,
        survivor=False,
        game_ip="127.0.0.1",
        game_port=50000,
        my_port=None,
        sock_timeout=10.0,
        num_connection_attempts=60,
        game_log_path="gamelog.txt",
    ):
        super().__init__()

        if not image_based:
            raise NotImplementedError(
                "TankEnv (M1 Wave 2 port) is image-based only; raw-state MLP did not work in 2021."
            )
        if not rand_opp:
            raise NotImplementedError(
                "TankEnv (M1 Wave 2 port) ships the random-opponent path only; "
                "the PFSP/ELO league opponent path lands in a later wave."
            )

        self.rand_opp = rand_opp
        self.image_based = image_based
        self.p = env_p
        self.max_steps = max_steps
        self.time_reward = time_reward
        self.survivor = survivor

        # --- arena single-source: ONE config feeds both the build and the obs ----------
        # PRECEDENCE: config_path (forwarded to the build AND read for the obs arena) >
        # level_path (explicit arena, no --config forwarding) > default config (the
        # build's StreamingAssets config -> custom1, NOT forwarded because the build
        # falls back to that same file on its own). This is what makes the default obs
        # match the arena the build simulates out of the box.
        if config_path is not None:
            # config_path set: forward it to the build AND derive the obs arena from it.
            self.config_path = Path(config_path).resolve()
            level = _arena_path_from_config(self.config_path)
        elif level_path is not None:
            # explicit arena, no config indirection -> do NOT forward --config.
            self.config_path = None
            level = level_path
        else:
            # default: read the build's StreamingAssets config for the obs arena, but do
            # NOT forward --config (the build falls back to this same file). Single source.
            self.config_path = None
            level = _arena_path_from_config(DEFAULT_CONFIG_PATH)

        # --- observation geometry (composed from arenas.load_level) ----------
        arena = load_level(level, p=self.p)
        self.dims = arena.dims
        # The wall-baked grid: draw_state reads its shape + the persisted G (wall) channel.
        self.wall_grid = arena.state
        self.state = arena.state
        self.observation_space = spaces.Box(low=0, high=255, shape=arena.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # --- transport / connection params -----------------------------------
        self.game_path = game_path
        self.game_ip = game_ip
        self.game_port = game_port
        self.my_port = my_port if my_port else game_port + 1
        self.sock_timeout = sock_timeout
        self.num_connection_attempts = num_connection_attempts
        self.game_log_path = game_log_path

        self.game_p = None
        self.game_log = None
        self.raw_state = None
        self.step_counter = 0
        self.last_winner = -1

        # Injected transport (test seam) vs. real socket launch.
        if transport is not None:
            self.conn = Connection(transport)
        elif game_path is not None:
            self.conn = self._connect_to_unity()
        else:
            # No transport and no game_path: nothing to talk to. Allowed for
            # introspection (spaces are set), but reset/step will fail loudly.
            self.conn = None

    # --- real-socket launch (preserved from legacy connect_to_unity) ----------

    def _connect_to_unity(self):
        """Launch the Unity build (subprocess arg-list) and open the TCP socket.

        Preserves the legacy ``[game_path, str(game_port)]`` launch + bind/connect
        retry loop, wrapped so the rest of the env talks to a ``protocol.Connection``.
        When ``self.config_path`` is set the arg-list also carries
        ``--config <abspath>`` (built by :func:`_build_game_cmd`) so the build simulates
        the SAME config the env read its obs arena from. Forwarding ``--config`` is config
        plumbing, not an RL-seam change (the wire bytes / 52-float state are untouched).
        Only reached on the real path (``game_path`` set, no injected transport).
        """
        import random
        import time

        # Long-lived handle: held open for the Unity subprocess's lifetime (it is the
        # child's stdout/stderr sink) and closed in close(); a context manager doesn't fit.
        self.game_log = open(self.game_log_path, "w")  # noqa: SIM115
        game_cmd_list = _build_game_cmd(self.game_path, self.game_port, self.config_path)
        self.game_p = subprocess.Popen(game_cmd_list, stdout=self.game_log, stderr=self.game_log)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        os_error = None
        bound = False
        for _ in range(self.num_connection_attempts):
            try:
                sock.bind(("", self.my_port))
                bound = True
                break
            except OSError as exc:
                os_error = exc
                time.sleep(1)
        if not bound:
            for _ in range(self.num_connection_attempts):
                try:
                    sock.bind(("", random.randint(33000, 60000)))
                    bound = True
                    break
                except OSError as exc:
                    os_error = exc
                    time.sleep(1)
            if not bound:
                raise os_error if os_error else OSError("could not bind socket")

        connected = False
        for _ in range(self.num_connection_attempts):
            try:
                sock.connect((self.game_ip, self.game_port))
                connected = True
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        if not connected:
            raise ConnectionError(f"could not connect to Unity at {self.game_ip}:{self.game_port}")

        sock.settimeout(self.sock_timeout)
        return Connection(sock)

    # --- gymnasium API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Run the legacy restart/start/first-state handshake; return ``(obs, info)``.

        Seeds ``self.np_random`` via ``super().reset(seed=seed)`` (so the random
        opponent + any env randomness are reproducible), then performs the
        BYTE-IDENTICAL 2021 handshake: ``{"restart":True}`` -> ack, ``{"start":True}`` ->
        ``"starting"`` ack, first ``{"state":[...52...]}``. The first state is rendered to
        the RGB grid via ``observation.draw_state``.
        """
        super().reset(seed=seed)

        self.conn.send({"restart": True})
        self.conn.receive()  # restart ack
        self.conn.send({"start": True})
        received = self.conn.receive()  # start ack
        if "starting" not in received:
            raise RuntimeError(f"unexpected start ack from game: {received!r}")

        received = self.conn.receive()  # first state
        self.raw_state = np.array(received["state"])
        self.state = draw_state(self.raw_state, self.wall_grid, self.dims, self.p)
        self.step_counter = 0
        self.last_winner = -1

        return self.state, {}

    def step(self, action):
        """Send ``{1: action, 2: opp_action}``, receive state, return the 5-tuple.

        The opponent action is drawn from ``self.np_random`` (NOT a bare ``np.random``)
        so a seeded ``reset`` makes the whole episode reproducible. The reward / episode
        boundary is computed by ``rewards.step_reward`` (gymnasium ``terminated`` /
        ``truncated``). On a dropped connection the step truncates with reward 0 (legacy
        no-winner end), via ``step_reward(lost_connection=True)``.
        """
        action = np.asarray(action, dtype=np.float32)
        opp_action = self.np_random.uniform(-1.0, 1.0, 5)

        message = {1: action.tolist(), 2: opp_action.tolist()}

        try:
            self.conn.send(message)
            received = self.conn.receive()
        except ConnectionError:
            reward, terminated, truncated = step_reward(
                lost_connection=True,
                survivor=self.survivor,
                time_reward=self.time_reward,
            )
            return self.state, reward, terminated, truncated, {"lost_connection": True}

        self.raw_state = np.array(received["state"])
        self.state = draw_state(self.raw_state, self.state, self.dims, self.p)
        self.step_counter += 1

        winner = int(received["winner"]) if "winner" in received else None
        done = bool("done" in received)
        max_steps_reached = self.step_counter >= self.max_steps

        reward, terminated, truncated = step_reward(
            winner=winner,
            done=done,
            survivor=self.survivor,
            time_reward=self.time_reward,
            max_steps_reached=max_steps_reached,
        )

        info = {}
        if winner is not None:
            info["winner"] = winner
            self.last_winner = winner

        # Enrich info on terminal steps so the evaluator can read the game result
        # UNAMBIGUOUSLY without inspecting the reward sign (M1 Wave 3). This only adds
        # free-form gymnasium ``info`` keys — it does NOT touch the wire bytes, the
        # 52-float layout, the obs, or the reward value. ``terminated`` is a real game
        # result (win/loss/draw); ``truncated`` (max_steps / lost connection) is NOT a
        # decided game and gets no outcome.
        if terminated:
            if winner == PLAYER_1:
                info["outcome"] = "win"
            elif winner is not None and winner != -1:
                info["outcome"] = "loss"
            else:
                # An explicit draw (winner == -1) or a winner-less ``done`` terminal.
                info["outcome"] = "draw"

        return self.state, reward, terminated, truncated, info

    def render(self):
        """Not implemented (``metadata['render_modes'] == []``)."""
        raise NotImplementedError()

    def close(self):
        """Release the transport and REAP the Unity subprocess (gymnasium lifecycle).

        This is the fix for the leaked ``TankTwinStickShooter.exe`` / held socket port: a
        real Unity run launches a child process + binds a socket, and nothing tore them
        down, so an interrupted/finished run orphaned the process and held the port into
        TIME_WAIT. ``close`` now ALWAYS reaps:

        1. best-effort end handshake (``{"restart":True}`` -> ack, ``{"end":True}`` -> ack)
           so Unity can exit cleanly; transport errors are swallowed (a dropped connection
           or a fake test transport without these replies must not block teardown),
        2. close the socket / ``Connection`` transport (frees the bound port),
        3. terminate the Unity subprocess GRACEFULLY (``terminate()``, wait briefly, then
           ``kill()`` if it is still alive) and ``wait()`` to reap the zombie,
        4. close the captured game-log handle.

        IDEMPOTENT and safe with no subprocess (injected ``transport``) or no connection
        (introspection-only construction): every attribute is nulled after release, so a
        second ``close()`` is a no-op. This is purely ``gymnasium.Env.close`` lifecycle —
        it does NOT touch the wire protocol bytes or the 52-float state layout, so it is
        NOT an RL-seam change.
        """
        if self.conn is not None:
            try:
                self.conn.send({"restart": True})
                self.conn.receive()
                self.conn.send({"end": True})
                self.conn.receive()
            except (ConnectionError, OSError, KeyError, ValueError):
                pass
            transport = getattr(self.conn, "transport", None)
            close = getattr(transport, "close", None)
            if callable(close):
                with contextlib.suppress(OSError):
                    close()
            self.conn = None

        if self.game_p is not None:
            # Graceful -> forceful reap. terminate() (SIGTERM / TerminateProcess), give the
            # child a short window to exit, then kill() if it is still alive. Either way
            # wait() so we never leave a zombie / orphaned TankTwinStickShooter.exe.
            if self.game_p.poll() is None:
                self.game_p.terminate()
                try:
                    self.game_p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.game_p.kill()
                    self.game_p.wait()
            self.game_p = None

        if self.game_log is not None:
            self.game_log.close()
            self.game_log = None

    # --- helper retained for parity / readability -----------------------------

    def opp_view(self):
        """The opponent's first-person (R<->B flipped) view of the current grid.

        Unused by the random-opponent path (the opponent action is a uniform draw), but
        kept so the league opponent path (a later wave) can compose ``state.flip_state``
        here without re-deriving it.
        """
        return flip_state(self.state)
