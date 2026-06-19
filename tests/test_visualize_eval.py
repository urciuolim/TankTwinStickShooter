"""Pure test of the eval visualizer's action -> gauge mapping (M1 Workstream B).

Exercises :func:`tank_twin.visualize_eval.action_to_gauges` and the action-layout
constants with NO matplotlib / torch / Unity / display. The visualizer imports matplotlib
ONLY inside its render functions (and torch/sb3/env only inside ``main``), so importing the
module for this test stays cheap and headless — the test asserts that, too.
"""

import subprocess
import sys

import pytest

from tank_twin.visualize_eval import (
    ACTION_LABELS,
    SHOOT_INDEX,
    SHOOT_THRESHOLD,
    Gauge,
    action_to_gauges,
)


def test_action_labels_match_playercontroller_layout():
    # Traced from PlayerController.cs: [0,1]=desired velocity, [2,3]=desired aim, [4]=fire.
    assert ACTION_LABELS == ("move_x", "move_y", "aim_x", "aim_y", "shoot")
    assert SHOOT_INDEX == 4
    assert ACTION_LABELS[SHOOT_INDEX] == "shoot"
    assert SHOOT_THRESHOLD == 0.5


def test_importing_module_does_not_import_matplotlib_or_torch():
    # The pure path must stay matplotlib/torch-free: importing the module (and using the pure
    # gauge mapping) must NOT pull in matplotlib or torch. Checked in a FRESH subprocess
    # because sys.modules is process-global — sibling tests in this suite import torch/sb3,
    # so an in-process check would be contaminated by import order.
    code = (
        "import sys; import tank_twin.visualize_eval as v; "
        "v.action_to_gauges([0,0,0,0,1]); "
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into the pure path'; "
        "assert 'torch' not in sys.modules, 'torch leaked into the pure path'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_returns_one_gauge_per_label_in_order():
    gauges = action_to_gauges([0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(gauges) == len(ACTION_LABELS)
    assert [g.label for g in gauges] == list(ACTION_LABELS)
    assert all(isinstance(g, Gauge) for g in gauges)


def test_wrong_length_action_raises():
    with pytest.raises(ValueError, match="5 entries"):
        action_to_gauges([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="5 entries"):
        action_to_gauges([0.0] * 6)


def test_fraction_maps_minus_one_to_zero_zero_to_half_one_to_one():
    # value -1 -> fraction 0.0, 0 -> 0.5, +1 -> 1.0 (the -1..1 -> 0..1 fill position).
    g_min = action_to_gauges([-1.0, -1.0, -1.0, -1.0, -1.0])
    g_mid = action_to_gauges([0.0, 0.0, 0.0, 0.0, 0.0])
    g_max = action_to_gauges([1.0, 1.0, 1.0, 1.0, 1.0])
    assert g_min[0].fraction == pytest.approx(0.0)
    assert g_mid[0].fraction == pytest.approx(0.5)
    assert g_max[0].fraction == pytest.approx(1.0)


def test_value_is_clamped_into_minus_one_one_for_display():
    # A model may predict slightly outside the Box; gauges clamp for DISPLAY only.
    over = action_to_gauges([5.0, -5.0, 2.0, -2.0, 9.0])
    assert over[0].value == pytest.approx(1.0)  # move_x clamped to +1
    assert over[1].value == pytest.approx(-1.0)  # move_y clamped to -1
    assert over[0].fraction == pytest.approx(1.0)
    assert over[1].fraction == pytest.approx(0.0)


def test_shoot_gauge_flags_firing_above_threshold():
    # shoot value > 0.5 -> firing True; the four directional gauges are never firing.
    gauges = action_to_gauges([0.0, 0.0, 0.0, 0.0, 0.75])
    shoot = gauges[SHOOT_INDEX]
    assert shoot.is_shoot is True
    assert shoot.firing is True
    for g in gauges[:SHOOT_INDEX]:
        assert g.is_shoot is False
        assert g.firing is False


def test_shoot_gauge_not_firing_at_or_below_threshold():
    # Strictly greater-than the threshold fires (matches PlayerController: value > 0.5).
    at = action_to_gauges([0.0, 0.0, 0.0, 0.0, 0.5])
    below = action_to_gauges([0.0, 0.0, 0.0, 0.0, 0.49])
    assert at[SHOOT_INDEX].firing is False  # exactly 0.5 is NOT > 0.5
    assert below[SHOOT_INDEX].firing is False


def test_shoot_color_changes_with_firing_state_directional_color_is_constant():
    firing = action_to_gauges([0.3, -0.3, 0.0, 0.0, 1.0])
    holding = action_to_gauges([0.3, -0.3, 0.0, 0.0, 0.0])
    # The shoot bar color differs between firing and holding.
    assert firing[SHOOT_INDEX].color != holding[SHOOT_INDEX].color
    # The directional gauges keep one constant color regardless of shoot state.
    assert firing[0].color == holding[0].color
    assert firing[0].color == firing[1].color == firing[2].color == firing[3].color


def test_accepts_numpy_like_via_float_coercion():
    # action_to_gauges coerces entries via float(): a tuple of python floats is the contract,
    # and any sequence of float-able values works (no numpy import in the pure path).
    gauges = action_to_gauges((0.1, 0.2, 0.3, 0.4, 0.6))
    assert gauges[0].value == pytest.approx(0.1)
    assert gauges[SHOOT_INDEX].firing is True  # 0.6 > 0.5
