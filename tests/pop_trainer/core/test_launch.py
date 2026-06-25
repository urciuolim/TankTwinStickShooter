"""Contract tests for ``pop_trainer.core.launch.default_build_path`` (the OS-aware resolver)."""

from pathlib import Path

import pytest

from pop_trainer.core.launch import default_build_path


def test_win32_returns_exe(monkeypatch, tmp_path):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "win32")
    assert default_build_path(tmp_path) == tmp_path / "build" / "TankTwinStickShooter.exe"


def test_linux_bare_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "linux")
    assert default_build_path(tmp_path) == tmp_path / "build" / "TankTwinStickShooter"


def test_linux_prefers_existing_x86_64_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "linux")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    x86 = build_dir / "TankTwinStickShooter.x86_64"
    x86.write_bytes(b"")
    assert default_build_path(tmp_path) == x86


def test_darwin_returns_inner_binary_regardless_of_name(monkeypatch, tmp_path):
    # The inner executable's name comes from PlayerSettings and need NOT match the bundle name;
    # the resolver must return whatever single binary lives under Contents/MacOS/.
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "darwin")
    macos = tmp_path / "build" / "TankTwinStickShooter.app" / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    inner = macos / "SomeOtherProductName"
    inner.write_bytes(b"")
    assert default_build_path(tmp_path) == inner


def test_darwin_fallback_when_nothing_built(monkeypatch, tmp_path):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "darwin")
    expected = (
        tmp_path
        / "build"
        / "TankTwinStickShooter.app"
        / "Contents"
        / "MacOS"
        / "TankTwinStickShooter"
    )
    assert default_build_path(tmp_path) == expected


def test_darwin_fallback_when_bundle_ambiguous(monkeypatch, tmp_path):
    # Zero or multiple entries under Contents/MacOS/ is ambiguous -> conventional fallback.
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "darwin")
    macos = tmp_path / "build" / "TankTwinStickShooter.app" / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    (macos / "binary_a").write_bytes(b"")
    (macos / "binary_b").write_bytes(b"")
    expected = macos / "TankTwinStickShooter"
    assert default_build_path(tmp_path) == expected


def test_accepts_str_repo_root(monkeypatch, tmp_path):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", "win32")
    assert default_build_path(str(tmp_path)) == tmp_path / "build" / "TankTwinStickShooter.exe"


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_returns_path_without_requiring_build_to_exist(monkeypatch, tmp_path, platform):
    monkeypatch.setattr("pop_trainer.core.launch.sys.platform", platform)
    result = default_build_path(tmp_path)
    assert isinstance(result, Path)
    assert not result.exists()
