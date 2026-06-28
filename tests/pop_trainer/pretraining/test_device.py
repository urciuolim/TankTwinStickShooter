"""resolve_device: --device parsing, auto-detection, and the TT_DEVICE env override."""

from __future__ import annotations

import pytest

from pop_trainer.pretraining import device as device_mod
from pop_trainer.pretraining.device import resolve_device


@pytest.fixture(autouse=True)
def _clear_tt_device(monkeypatch):
    monkeypatch.delenv("TT_DEVICE", raising=False)


def _force_accel(monkeypatch, *, cuda: bool, mps: bool):
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod.torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("explicit", ["cuda", "mps", "cpu"])
def test_explicit_device_returned_verbatim(explicit):
    assert resolve_device(explicit) == explicit


def test_explicit_device_wins_over_env(monkeypatch):
    monkeypatch.setenv("TT_DEVICE", "cuda")
    # explicit request must not be overridden by the env (the escape hatch)
    assert resolve_device("mps") == "mps"


@pytest.mark.parametrize("value", ["cuda", "mps", "cpu"])
def test_auto_honors_tt_device(monkeypatch, value):
    _force_accel(monkeypatch, cuda=False, mps=False)  # detection would say cpu
    monkeypatch.setenv("TT_DEVICE", value)
    assert resolve_device("auto") == value


def test_auto_tt_device_case_insensitive(monkeypatch):
    _force_accel(monkeypatch, cuda=False, mps=False)
    monkeypatch.setenv("TT_DEVICE", "CUDA")
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_detection_when_env_unset(monkeypatch):
    _force_accel(monkeypatch, cuda=True, mps=False)
    assert resolve_device("auto") == "cuda"
    _force_accel(monkeypatch, cuda=False, mps=True)
    assert resolve_device("auto") == "mps"
    _force_accel(monkeypatch, cuda=False, mps=False)
    assert resolve_device("auto") == "cpu"


def test_auto_ignores_empty_or_auto_env(monkeypatch):
    _force_accel(monkeypatch, cuda=True, mps=False)
    monkeypatch.setenv("TT_DEVICE", "auto")
    assert resolve_device("auto") == "cuda"
    monkeypatch.setenv("TT_DEVICE", "")
    assert resolve_device("auto") == "cuda"


def test_invalid_tt_device_raises(monkeypatch):
    _force_accel(monkeypatch, cuda=False, mps=False)
    monkeypatch.setenv("TT_DEVICE", "gpu")
    with pytest.raises(ValueError, match="TT_DEVICE"):
        resolve_device("auto")


def test_invalid_explicit_request_raises():
    with pytest.raises(ValueError, match="--device"):
        resolve_device("gpu")
