"""Map rotation via the additive switch_arena message + obs reload (Feature 1).

Drives ``TankEnv`` with a FAKE Unity transport — NO Unity, NO subprocess, NO GPU. Extends
the ``ScriptedUnity`` fake from ``tests/test_env.py`` so it answers the additive handshake
message ``{"switch_arena": <path>}`` with the strict-JSON ack ``{"arena_switched": true}``,
mirroring the pinned Unity wire contract (commit: "unity: additive switch_arena ...").

Pins:

* (a) the wire carries ``switch_arena`` with the expected ABSOLUTE arena path on a rotating
  reset (and the no-rotation env sends NONE — the existing handshake stays byte-identical);
* (b) obs == game across a switch: after ``reset()`` rotates to map B, the env's
  ``wall_grid`` G (wall) channel equals ``load_level(mapB_arena).state``'s G channel;
* (c) round-robin order across N resets cycles map0 -> map1 -> ... -> map0.

Uses three REAL exp-configs maps with DIFFERENT interior walls but identical (36, 60, 3)
dims (empty / center_block / four_pillars) so the G-channel array comparison is meaningful.
"""

from pathlib import Path

import numpy as np

from tank_twin.arenas import G, load_level
from tank_twin.env import TankEnv, _arena_path_from_config

# Reuse the canned-protocol fake transport + canonical states from the env contract test.
from test_env import FIRST_STATE, ScriptedUnity, _frame

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAPS_DIR = _REPO_ROOT / "exp-configs" / "maps"

# Three real map CONFIGS with DIFFERENT interior walls but identical (36,60,3) dims.
MAP_CONFIGS = [
    _MAPS_DIR / "empty.json",
    _MAPS_DIR / "center_block.json",
    _MAPS_DIR / "four_pillars.json",
]
# Their resolved ABSOLUTE arena paths (what the env both loads obs from AND sends on the wire).
MAP_ARENAS = [_arena_path_from_config(c) for c in MAP_CONFIGS]


class RotatingScriptedUnity(ScriptedUnity):
    """``ScriptedUnity`` that ALSO answers the additive ``switch_arena`` handshake message.

    Mirrors the Unity side: a ``{"switch_arena": "<abs path>"}`` message (sent while the
    game is ``!ingame``, before restart/start) is acked with strict JSON
    ``{"arena_switched": true}``. Everything else (restart/start/end/step) is inherited
    unchanged, so the per-step ``{1:action,2:opp}`` + 52-float read are untouched.
    """

    def _respond(self, msg):
        if msg.get("switch_arena") is not None:
            self._pending.append(_frame({"arena_switched": True}))
        else:
            super()._respond(msg)


def _g_channel(arena_path):
    """The G (wall) channel of an arena's loaded obs grid (the load-bearing invariant target)."""
    return load_level(arena_path, p=3).state[:, :, G]


def _make_rotating_env(map_rotation=MAP_CONFIGS, **kwargs):
    """A TankEnv driven by the rotating fake transport, rotating over ``map_rotation``."""
    transport = RotatingScriptedUnity(**kwargs)
    return TankEnv(game_path=None, transport=transport, map_rotation=map_rotation)


# --- (a) wire carries switch_arena with the absolute arena path ------------------------


def test_rotating_reset_sends_switch_arena_before_handshake():
    env = _make_rotating_env()
    env.reset(seed=0)
    sent = env.conn.transport.received
    # switch_arena is sent FIRST (before the restart/start handshake), carrying map0's path.
    assert sent[0] == {"switch_arena": str(MAP_ARENAS[0].resolve())}
    assert sent[1] == {"restart": True}
    assert sent[2] == {"start": True}


def test_switch_arena_path_is_absolute():
    env = _make_rotating_env()
    env.reset(seed=0)
    switch_msg = env.conn.transport.received[0]
    assert Path(switch_msg["switch_arena"]).is_absolute()


# --- (d) the NO-rotation env sends NO switch_arena (byte-identical handshake) ----------


def test_no_rotation_env_sends_no_switch_arena():
    # map_rotation None: reset() must be byte-identical to today — sent[0] == {"restart":True},
    # and NO message anywhere carries switch_arena.
    env = TankEnv(game_path=None, transport=RotatingScriptedUnity())
    env.reset(seed=0)
    sent = env.conn.transport.received
    assert sent[0] == {"restart": True}
    assert sent[1] == {"start": True}
    assert all("switch_arena" not in m for m in sent)


def test_empty_rotation_list_disables_rotation():
    # An empty list is falsy -> treated as no rotation (back-compat).
    env = TankEnv(game_path=None, transport=RotatingScriptedUnity(), map_rotation=[])
    assert env.map_rotation is None
    env.reset(seed=0)
    assert all("switch_arena" not in m for m in env.conn.transport.received)


# --- (b) obs == game across a switch: wall_grid G channel matches the new arena --------


def test_obs_wall_grid_matches_first_map_after_reset():
    env = _make_rotating_env()
    env.reset(seed=0)
    # After the first rotating reset (index 0 -> empty), the obs walls == empty's walls.
    assert np.array_equal(env.wall_grid[:, :, G], _g_channel(MAP_ARENAS[0]))


def test_obs_wall_grid_swaps_to_second_map_on_second_reset():
    env = _make_rotating_env()
    env.reset(seed=0)  # map0 (empty)
    env.reset(seed=0)  # map1 (center_block)
    # LOAD-BEARING INVARIANT: obs walls equal center_block's walls after the switch.
    assert np.array_equal(env.wall_grid[:, :, G], _g_channel(MAP_ARENAS[1]))
    # And they DIFFER from map0 (the switch actually changed the obs).
    assert not np.array_equal(_g_channel(MAP_ARENAS[1]), _g_channel(MAP_ARENAS[0]))


def test_obs_wall_grid_matches_wire_arena_each_reset():
    # Single-source: the arena reloaded into the obs is the SAME arena sent on the wire,
    # every reset. Walk all three maps and assert obs G channel == the sent arena's G channel.
    env = _make_rotating_env()
    for i in range(len(MAP_CONFIGS)):
        env.reset(seed=0)
        sent_arena = Path(env.conn.transport.received[-3]["switch_arena"])
        # received[-3] is the switch_arena for THIS reset (then restart, then start).
        assert sent_arena == MAP_ARENAS[i].resolve()
        assert np.array_equal(env.wall_grid[:, :, G], _g_channel(sent_arena))


# --- (c) round-robin order across N resets cycles map0 -> ... -> map0 ------------------


def test_round_robin_cycles_through_all_maps_and_wraps():
    env = _make_rotating_env()
    seen = []
    # 2 full cycles + 1 to prove the wrap returns to map0.
    for _ in range(len(MAP_CONFIGS) * 2 + 1):
        env.reset(seed=0)
        # The switch_arena message for this reset is the 3rd-from-last (then restart, start).
        seen.append(Path(env.conn.transport.received[-3]["switch_arena"]))
    expected = [a.resolve() for a in MAP_ARENAS]
    assert seen[0] == expected[0]
    assert seen[1] == expected[1]
    assert seen[2] == expected[2]
    assert seen[3] == expected[0]  # wrapped back to map0
    assert seen[: len(MAP_CONFIGS)] == expected
    assert seen[len(MAP_CONFIGS) : len(MAP_CONFIGS) * 2] == expected


def test_first_reset_uses_index_zero():
    env = _make_rotating_env()
    assert env._rotation_index == -1  # before any reset
    env.reset(seed=0)
    assert env._rotation_index == 0  # first reset -> index 0


# --- ack verification: a missing/false arena_switched ack raises -----------------------


def test_missing_arena_switched_ack_raises():
    class NoAck(RotatingScriptedUnity):
        def _respond(self, msg):
            if msg.get("switch_arena") is not None:
                # Wrong/absent ack key -> the env must raise (mirrors the 'starting' check).
                self._pending.append(_frame({"not_switched": True}))
            else:
                ScriptedUnity._respond(self, msg)

    env = TankEnv(game_path=None, transport=NoAck(), map_rotation=MAP_CONFIGS)
    try:
        env.reset(seed=0)
    except RuntimeError as exc:
        assert "switch_arena" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on a missing arena_switched ack")


# --- the rotation config-list resolves to arenas in the constructor --------------------


def test_constructor_resolves_config_list_to_arenas():
    env = _make_rotating_env()
    assert env.map_rotation == MAP_ARENAS


# --- rotation suppression: the eval-desync fix at the ENV layer ------------------------
#
# set_rotation_enabled(False) (and reset(options={"rotate": False})) make a rotating reset
# do the PLAIN restart/start handshake on the CURRENT arena: NO switch_arena on the wire and
# the rotation index is NOT advanced (so the next ENABLED reset rotates from where training
# left off). This is the primitive the eval callback uses to avoid the live rollout-boundary
# desync. These pin it directly on the env + fake transport (no Unity).


def test_suppressed_reset_sends_no_switch_arena():
    env = _make_rotating_env()
    env.set_rotation_enabled(False)
    env.reset(seed=0)
    sent = env.conn.transport.received
    # Plain handshake only — restart/start, and NO switch_arena anywhere.
    assert sent[0] == {"restart": True}
    assert sent[1] == {"start": True}
    assert all("switch_arena" not in m for m in sent)


def test_suppressed_reset_does_not_advance_rotation_index():
    env = _make_rotating_env()
    assert env._rotation_index == -1
    env.set_rotation_enabled(False)
    env.reset(seed=0)
    env.reset(seed=0)
    # The suppressed resets consumed NO rotation slots: the index is untouched.
    assert env._rotation_index == -1


def test_rotation_resumes_in_order_after_suppression():
    # The proven round-robin order is PRESERVED across a suppressed eval block: rotate to
    # map0, suppress (eval) twice, re-enable, and the next reset rotates to map1 (NOT map2).
    env = _make_rotating_env()
    env.reset(seed=0)  # map0 (index 0)
    assert env._rotation_index == 0
    env.set_rotation_enabled(False)
    env.reset(seed=0)  # suppressed (eval episode 1)
    env.reset(seed=0)  # suppressed (eval episode 2 / buffer repair)
    assert env._rotation_index == 0  # unchanged
    env.set_rotation_enabled(True)
    env.reset(seed=0)  # next training reset -> rotates to map1
    assert env._rotation_index == 1
    sent_arena = Path(env.conn.transport.received[-3]["switch_arena"])
    assert sent_arena == MAP_ARENAS[1].resolve()


def test_reset_options_rotate_false_suppresses_switch_arena():
    # The per-call override matches the durable flag: options={"rotate": False} also pins the
    # reset to the current arena (no switch_arena, index not advanced).
    env = _make_rotating_env()
    env.reset(options={"rotate": False})
    sent = env.conn.transport.received
    assert all("switch_arena" not in m for m in sent)
    assert env._rotation_index == -1


def test_set_rotation_enabled_returns_previous_value():
    # Returns the PRIOR value so the callback can restore it in a finally.
    env = _make_rotating_env()
    assert env.set_rotation_enabled(False) is True  # was enabled by default
    assert env.set_rotation_enabled(True) is False  # was just disabled


def test_first_state_after_switch_renders_on_new_walls():
    # End-to-end-ish (fake socket): after the switch + handshake, the rendered obs is in
    # the declared space and the persisted walls are the new map's (not the constructor's).
    env = _make_rotating_env()
    env.reset(seed=0)  # empty
    obs, _info = env.reset(seed=0)  # center_block
    assert env.observation_space.contains(obs)
    assert np.array_equal(env.wall_grid[:, :, G], _g_channel(MAP_ARENAS[1]))
    # The first state's raw floats round-tripped (sanity that the handshake completed).
    assert env.raw_state is not None
    assert len(env.raw_state) == len(FIRST_STATE)
