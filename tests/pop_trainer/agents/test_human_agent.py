"""Tests for ``pop_trainer.agents.human_agent`` — the pure pressed-key -> action mapping.

NO hardware and NO real ``pynput`` listener: a :class:`KeyboardState` is constructed with a chosen
pressed-key set and the :class:`HumanAgent` mapping is asserted per channel. The numpad-by-vk seam
is exercised by injecting BOTH the vk tokens AND the Num-Lock-OFF arrow-name fallbacks and asserting
the SAME aim — proving the Num-Lock-independent path. The optional/lazy ``pynput`` contract is
exercised by importing the module without ``pynput`` and by forcing the import to fail.
"""

import builtins
import importlib

import numpy as np
import pytest

from pop_trainer.agents.human_agent import (
    PYNPUT_MISSING_MSG,
    HumanAgent,
    KeyboardListener,
    KeyboardState,
    player1_mapping,
    player2_mapping,
)


def _p1(pressed):
    return HumanAgent(player1_mapping(), KeyboardState(pressed))


def _p2(pressed):
    return HumanAgent(player2_mapping(), KeyboardState(pressed))


# --- player 1: WASD move ------------------------------------------------------------------


def test_p1_move_x_right_and_left():
    assert _p1({"d"}).act(None)[0] == 1.0
    assert _p1({"a"}).act(None)[0] == -1.0
    assert _p1({"a", "d"}).act(None)[0] == 0.0


def test_p1_move_y_up_and_down():
    assert _p1({"w"}).act(None)[1] == 1.0
    assert _p1({"s"}).act(None)[1] == -1.0
    assert _p1({"w", "s"}).act(None)[1] == 0.0


def test_p1_keys_are_case_insensitive():
    # The listener lowercases chars; a state built with an uppercase token must still match.
    assert _p1({"D"}).act(None)[0] == 1.0


# --- player 1: TFGH aim -------------------------------------------------------------------


def test_p1_aim_tfgh():
    assert _p1({"h"}).act(None)[2] == 1.0  # aim_x +
    assert _p1({"f"}).act(None)[2] == -1.0  # aim_x -
    assert _p1({"t"}).act(None)[3] == 1.0  # aim_y +
    assert _p1({"g"}).act(None)[3] == -1.0  # aim_y -


# --- player 1: fire (left shift) ----------------------------------------------------------


def test_p1_fire_shift():
    # pynput reports left shift with name "shift" on Windows; the mapping also accepts "shift_l".
    assert _p1({"shift"}).act(None)[4] == 1.0
    assert _p1({"shift_l"}).act(None)[4] == 1.0
    assert _p1(set()).act(None)[4] == 0.0


# --- player 1: diagonals are UN-normalized, all-released is zero --------------------------


def test_p1_diagonal_is_unnormalized():
    # W+D down -> move (1, 1); diagonals are intentionally left un-normalized (the wire coerces).
    assert _p1({"w", "d"}).act(None) == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_all_released_is_zero_action():
    assert _p1(set()).act(None) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert _p2(set()).act(None) == [0.0, 0.0, 0.0, 0.0, 0.0]


# --- player 2: IJKL move ------------------------------------------------------------------


def test_p2_move_ijkl():
    assert _p2({"l"}).act(None)[0] == 1.0  # move_x +
    assert _p2({"j"}).act(None)[0] == -1.0  # move_x -
    assert _p2({"i"}).act(None)[1] == 1.0  # move_y +
    assert _p2({"k"}).act(None)[1] == -1.0  # move_y -


# --- player 2: numpad aim BY VK (Num Lock ON path) ----------------------------------------


def test_p2_aim_numpad_by_vk():
    # Inject the raw virtual-key codes (Num Lock ON): NUMPAD8=104, NUMPAD5=101, NUMPAD4=100,
    # NUMPAD6=102. This is the load-bearing numpad-by-vk seam.
    assert _p2({104}).act(None)[3] == 1.0  # aim_y +  (numpad 8)
    assert _p2({101}).act(None)[3] == -1.0  # aim_y -  (numpad 5)
    assert _p2({100}).act(None)[2] == -1.0  # aim_x -  (numpad 4)
    assert _p2({102}).act(None)[2] == 1.0  # aim_x +  (numpad 6)


# --- player 2: numpad aim via the Num-Lock-OFF fallbacks (SAME aim) ------------------------


def test_p2_aim_numpad_off_fallbacks_match_vk():
    # Num Lock OFF: numpad 8/4/6 arrive as arrow Keys, numpad 5 as VK_CLEAR (vk 12) / Key.begin.
    # Each must produce the SAME aim channel as its vk — proving Num-Lock independence.
    assert _p2({"up"}).act(None)[3] == 1.0  # numpad 8 off -> Key.up
    assert _p2({"left"}).act(None)[2] == -1.0  # numpad 4 off -> Key.left
    assert _p2({"right"}).act(None)[2] == 1.0  # numpad 6 off -> Key.right
    assert _p2({12}).act(None)[3] == -1.0  # numpad 5 off (Windows) -> VK_CLEAR
    assert _p2({"begin"}).act(None)[3] == -1.0  # numpad 5 off on a build exposing Key.begin


# --- player 2: fire (numpad enter == enter) -----------------------------------------------


def test_p2_fire_enter():
    # Numpad-Enter shares VK_RETURN with main Enter and is not separable -> matches Key.enter.
    assert _p2({"enter"}).act(None)[4] == 1.0
    assert _p2(set()).act(None)[4] == 0.0


# --- obs is ignored -----------------------------------------------------------------------


def test_act_ignores_obs():
    agent = _p1({"w", "d", "shift"})
    expected = [1.0, 1.0, 0.0, 0.0, 1.0]
    assert agent.act(None) == expected
    assert agent.act(np.arange(52, dtype=np.float32)) == expected
    assert agent.act({"state": [0.0] * 52, "anything": 1}) == expected


# --- shared state is read live ------------------------------------------------------------


def test_agent_reads_shared_state_live():
    # The agent holds a reference to the shared state; press/release after construction is seen.
    state = KeyboardState()
    agent = HumanAgent(player1_mapping(), state)
    assert agent.act(None)[0] == 0.0
    state.press("d")
    assert agent.act(None)[0] == 1.0
    state.release("d")
    assert agent.act(None)[0] == 0.0


# --- optional/lazy pynput contract --------------------------------------------------------


def test_module_imports_without_pynput(monkeypatch):
    # Forcing pynput absent and re-importing the module must still succeed: the only pynput import
    # is lazy (inside KeyboardListener), never at module top.
    monkeypatch.setitem(__import__("sys").modules, "pynput", None)
    monkeypatch.setitem(__import__("sys").modules, "pynput.keyboard", None)
    mod = importlib.import_module("pop_trainer.agents.human_agent")
    reloaded = importlib.reload(mod)
    assert reloaded.HumanAgent is not None
    # The pure mapping still works with no pynput present.
    agent = reloaded.HumanAgent(reloaded.player1_mapping(), reloaded.KeyboardState({"d"}))
    assert agent.act(None)[0] == 1.0


def test_listener_without_pynput_raises_actionable_error(monkeypatch):
    # Genuinely exercise the lazy guard: make `from pynput import keyboard` fail, then construct a
    # KeyboardListener and assert the error names pynput and the 'human' extra.
    real_import = builtins.__import__

    def _no_pynput(name, *args, **kwargs):
        if name == "pynput" or name.startswith("pynput."):
            raise ImportError("No module named 'pynput'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pynput)
    with pytest.raises(RuntimeError) as excinfo:
        KeyboardListener()  # __post_init__ probes pynput and raises
    msg = str(excinfo.value)
    assert "pynput" in msg
    assert "human" in msg
    assert msg == PYNPUT_MISSING_MSG
