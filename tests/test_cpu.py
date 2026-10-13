"""Unit tests for container-aware CPU inspection and thread resolution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wyoming_vietnamese.cpu import (
    _MAX_WINDOWS_WORKERS,
    _PHYSICAL_CORES_CACHE,
    _PROBE_TIMEOUT_SECONDS,
    _count_physical_cores,
    _count_physical_cores_darwin,
    _count_physical_cores_freebsd,
    _count_physical_cores_linux,
    _count_physical_cores_win32,
    _count_physical_cores_win32_ctypes,
    _count_physical_cores_win32_powershell,
    _cpu_count_affinity_set,
    _cpu_count_cgroup,
    _find_cgroup_v1_quota,
    _find_cgroup_v2_quota,
    _parse_cpuinfo_cores,
    _parse_cpuinfo_processors,
    _parse_proc_cgroup_paths,
    _read_cgroup_v1_quota,
    _read_cgroup_v2_quota,
    available_cpu_count,
    resolve_cpu_threads,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear physical cores cache between test cases."""
    _PHYSICAL_CORES_CACHE.clear()


def test_read_cgroup_v2_quota(tmp_path: Path) -> None:
    """Test reading cgroup v2 quota from cpu.max file."""
    assert _read_cgroup_v2_quota(tmp_path) is None

    cpu_max = tmp_path / "cpu.max"
    for content, expected in [
        ("max 100000\n", None),
        ("200000 100000\n", 2),
        ("150000 100000\n", 2),
        ("invalid\n", None),
        ("not_a_number 100000\n", None),
        ("-1 100000\n", None),
    ]:
        cpu_max.write_text(content, encoding="utf-8")
        assert _read_cgroup_v2_quota(tmp_path) == expected


def test_read_cgroup_v1_quota(tmp_path: Path) -> None:
    """Test reading cgroup v1 quota from cpu.cfs_quota_us and cpu.cfs_period_us."""
    assert _read_cgroup_v1_quota(tmp_path) is None

    quota_file = tmp_path / "cpu.cfs_quota_us"
    period_file = tmp_path / "cpu.cfs_period_us"

    quota_file.write_text("-1\n", encoding="utf-8")
    period_file.write_text("100000\n", encoding="utf-8")
    assert _read_cgroup_v1_quota(tmp_path) is None

    quota_file.write_text("300000\n", encoding="utf-8")
    period_file.write_text("100000\n", encoding="utf-8")
    assert _read_cgroup_v1_quota(tmp_path) == 3

    quota_file.write_text("bad\n", encoding="utf-8")
    assert _read_cgroup_v1_quota(tmp_path) is None


def test_parse_proc_cgroup_paths(tmp_path: Path) -> None:
    """Test parsing cgroup paths from /proc/self/cgroup file."""
    non_existent = tmp_path / "non_existent"
    assert _parse_proc_cgroup_paths(non_existent) == ([], [])

    proc_cgroup = tmp_path / "cgroup"
    content = (
        "0::/docker/abc123def456\n"
        "1:name=systemd:/user.slice\n"
        "2:cpu,cpuacct:/docker/abc123def456\n"
        "3:memory:/\n"
        "corrupted_line\n"
    )
    proc_cgroup.write_text(content, encoding="utf-8")
    v2_paths, v1_paths = _parse_proc_cgroup_paths(proc_cgroup)
    assert v2_paths == ["docker/abc123def456"]
    assert v1_paths == ["docker/abc123def456"]


def test_find_cgroup_v2_quota(tmp_path: Path) -> None:
    """Test finding cgroup v2 quota walking up the hierarchy."""
    nested = tmp_path / "docker" / "container"
    nested.mkdir(parents=True)
    (nested / "cpu.max").write_text("200000 100000\n", encoding="utf-8")

    quota = _find_cgroup_v2_quota(tmp_path, ["docker/container"])
    assert quota == 2

    (nested / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (tmp_path / "docker" / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    quota = _find_cgroup_v2_quota(tmp_path, ["docker/container"])
    assert quota == 4

    (nested / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    assert _find_cgroup_v2_quota(tmp_path, ["docker/container"]) == 2
    (nested / "cpu.max").write_text("600000 100000\n", encoding="utf-8")
    assert _find_cgroup_v2_quota(tmp_path, ["docker/container"]) == 4

    assert _find_cgroup_v2_quota(tmp_path, ["non_existent"]) is None


def test_find_cgroup_v1_quota(tmp_path: Path) -> None:
    """Test finding cgroup v1 quota walking up the hierarchy."""
    cpu_dir = tmp_path / "cpu"
    nested = cpu_dir / "docker" / "container"
    nested.mkdir(parents=True)
    (nested / "cpu.cfs_quota_us").write_text("200000\n", encoding="utf-8")
    (nested / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")

    quota = _find_cgroup_v1_quota(tmp_path, ["docker/container"])
    assert quota == 2

    (cpu_dir / "cpu.cfs_quota_us").write_text("400000\n", encoding="utf-8")
    (cpu_dir / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    assert _find_cgroup_v1_quota(tmp_path, ["docker/container"]) == 2
    (nested / "cpu.cfs_quota_us").write_text("600000\n", encoding="utf-8")
    assert _find_cgroup_v1_quota(tmp_path, ["docker/container"]) == 4

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert _find_cgroup_v1_quota(empty_root, ["docker"]) is None


def test_cpu_count_cgroup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test overall cgroup CPU quota detection logic."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "darwin")
    assert _cpu_count_cgroup() is None

    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "linux")
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.Path",
        lambda *p: tmp_path / "nonexistent" if not p or p == ("/sys/fs/cgroup",) else Path(*p),
    )
    assert _cpu_count_cgroup() is None

    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "cpu.max").write_text("800000 100000\n", encoding="utf-8")
    child = cgroup_root / "docker"
    child.mkdir()
    (child / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    proc_cgroup = tmp_path / "proc_cgroup"
    proc_cgroup.write_text("0::/docker\n", encoding="utf-8")
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.Path",
        lambda *p: (
            cgroup_root
            if p == ("/sys/fs/cgroup",)
            else proc_cgroup
            if p == ("/proc/self/cgroup",)
            else Path(*p)
        ),
    )
    assert _cpu_count_cgroup() == 2


def test_cpu_count_affinity_set_from_sched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test querying CPU affinity mask from os.sched_getaffinity."""
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.os.sched_getaffinity",
        lambda _pid: {0, 1},
        raising=False,
    )
    assert _cpu_count_affinity_set() == {0, 1}


def test_cpu_count_affinity_set_from_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test querying CPU affinity mask from psutil fallback."""

    def _raise_error(_pid: int) -> set[int]:
        raise OSError("affinity failed")

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.os.sched_getaffinity",
        _raise_error,
        raising=False,
    )
    mock_psutil = MagicMock()
    mock_proc = MagicMock()
    mock_proc.cpu_affinity.return_value = [2, 3]
    mock_psutil.Process.return_value = mock_proc
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.importlib.import_module",
        lambda name: mock_psutil if name == "psutil" else MagicMock(),
    )
    assert _cpu_count_affinity_set() == {2, 3}


def test_cpu_count_affinity_set_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CPU affinity mask returns None when unavailable."""

    def _raise_error(_pid: int) -> set[int]:
        raise OSError("affinity failed")

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.os.sched_getaffinity",
        _raise_error,
        raising=False,
    )

    def _raise_import(_name: str) -> None:
        raise ImportError("no psutil")

    monkeypatch.setattr("wyoming_vietnamese.cpu.importlib.import_module", _raise_import)
    assert _cpu_count_affinity_set() is None


def test_parse_cpuinfo_cores_and_processors() -> None:
    """Test parsing Linux cpuinfo cores and processor fallbacks."""
    content = (
        "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n\n"
        "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 0\n\n"
        "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 1\n\n"
        "processor\t: 3\nphysical id\t: 0\ncore id\t\t: 1\n\n"
    )
    cores = _parse_cpuinfo_cores(content, cpu_set=None)
    assert len(cores) == 2

    cores_filtered = _parse_cpuinfo_cores(content, cpu_set={0, 1})
    assert len(cores_filtered) == 1

    arm_content = "processor\t: 0\n\nprocessor\t: 1\n\nprocessor\t: 2\n\n"
    assert len(_parse_cpuinfo_cores(arm_content)) == 0
    processors = _parse_cpuinfo_processors(arm_content, cpu_set={1, 2})
    assert processors == {1, 2}


def test_count_physical_cores_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test physical core counting on Linux."""
    cpuinfo_file = tmp_path / "cpuinfo"
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.Path",
        lambda p: cpuinfo_file if p == "/proc/cpuinfo" else Path(p),
    )

    with pytest.raises(ValueError, match="not found"):
        _count_physical_cores_linux()

    content = (
        "processor : 0\ncore id : 0\nphysical id : 0\n\n"
        "processor : 1\ncore id : 1\nphysical id : 0\n\n"
    )
    cpuinfo_file.write_text(content, encoding="utf-8")
    assert _count_physical_cores_linux() == 2

    arm_content = "processor : 0\n\nprocessor : 1\n\n"
    cpuinfo_file.write_text(arm_content, encoding="utf-8")
    assert _count_physical_cores_linux() == 2

    cpuinfo_file.write_text("bogus info\n", encoding="utf-8")
    with pytest.raises(ValueError, match="could not find any physical core"):
        _count_physical_cores_linux()


def test_count_physical_cores_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test physical core counting on macOS / Darwin."""
    calls: list[dict[str, object]] = []

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(["sysctl"], 0, stdout="8\n", stderr="")

    monkeypatch.setattr("wyoming_vietnamese.cpu.subprocess.run", fake_run)
    assert _count_physical_cores_darwin() == 8
    assert calls[0].get("timeout") == _PROBE_TIMEOUT_SECONDS

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["sysctl"], 1, stdout="", stderr="error"
        ),
    )
    with pytest.raises(ValueError, match=r"sysctl hw\.physicalcpu"):
        _count_physical_cores_darwin()

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _count_physical_cores_darwin()


def test_count_physical_cores_freebsd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test physical core counting on FreeBSD."""
    calls: list[dict[str, object]] = []

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(["sysctl"], 0, stdout="6\n", stderr="")

    monkeypatch.setattr("wyoming_vietnamese.cpu.subprocess.run", fake_run)
    assert _count_physical_cores_freebsd() == 6
    assert calls[0].get("timeout") == _PROBE_TIMEOUT_SECONDS

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["sysctl"], 1, stdout="", stderr="error"
        ),
    )
    with pytest.raises(ValueError, match=r"sysctl kern\.smp\.cores"):
        _count_physical_cores_freebsd()

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _count_physical_cores_freebsd()


def test_count_physical_cores_win32_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test physical core counting on Windows using PowerShell."""
    calls: list[dict[str, object]] = []

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(["powershell.exe"], 0, stdout="4\n", stderr="")

    monkeypatch.setattr("wyoming_vietnamese.cpu.subprocess.run", fake_run)
    assert _count_physical_cores_win32_powershell() == 4
    assert calls[0].get("timeout") == _PROBE_TIMEOUT_SECONDS

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["powershell.exe"], 1, stdout="", stderr="error"
        ),
    )
    with pytest.raises(ValueError, match="PowerShell failed"):
        _count_physical_cores_win32_powershell()

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(
                cmd=["powershell.exe"], timeout=_PROBE_TIMEOUT_SECONDS
            )
        ),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _count_physical_cores_win32_powershell()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test for WinDLL absence")
def test_count_physical_cores_win32_ctypes_non_windows() -> None:
    """Test Win32 ctypes raises on non-Windows platforms."""
    with pytest.raises(RuntimeError, match="WinDLL not available"):
        _count_physical_cores_win32_ctypes()


def test_count_physical_cores_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Windows core count fallback cascade."""
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_win32_ctypes",
        MagicMock(side_effect=RuntimeError("no ctypes")),
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_win32_powershell",
        lambda: 8,
    )
    assert _count_physical_cores_win32() == 8

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_win32_powershell",
        MagicMock(side_effect=ValueError("no powershell")),
    )
    calls: list[dict[str, object]] = []

    def fake_run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            ["wmic"], 0, stdout="Node,NumberOfCores\nMYPC,4\n", stderr=""
        )

    monkeypatch.setattr("wyoming_vietnamese.cpu.subprocess.run", fake_run)
    assert _count_physical_cores_win32() == 4
    assert calls[0].get("timeout") == _PROBE_TIMEOUT_SECONDS

    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(["wmic"], 1, stdout="", stderr="error"),
    )
    with pytest.raises(ValueError, match="could not detect physical cores on Windows"):
        _count_physical_cores_win32()

    # Test fallback when PowerShell times out, falling through to wmic
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_win32_powershell",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(
                cmd=["powershell.exe"], timeout=_PROBE_TIMEOUT_SECONDS
            )
        ),
    )
    calls.clear()
    monkeypatch.setattr("wyoming_vietnamese.cpu.subprocess.run", fake_run)
    assert _count_physical_cores_win32() == 4

    # Test fallback when both PowerShell and wmic time out
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu.subprocess.run",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["wmic"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    with pytest.raises(ValueError, match="could not detect physical cores on Windows"):
        _count_physical_cores_win32()


def test_count_physical_cores_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _count_physical_cores platform dispatch and cache behavior."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "darwin")
    monkeypatch.setattr("wyoming_vietnamese.cpu._count_physical_cores_darwin", lambda: 8)

    assert _count_physical_cores() == 8
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_darwin",
        MagicMock(side_effect=RuntimeError("should use cache")),
    )
    assert _count_physical_cores() == 8

    _PHYSICAL_CORES_CACHE.clear()
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "unknown_os")
    assert _count_physical_cores() is None


def test_count_physical_cores_timeout_suppression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that subprocess timeouts fall through to None in _count_physical_cores."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "win32")
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_win32",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["wmic"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    assert _count_physical_cores() is None

    _PHYSICAL_CORES_CACHE.clear()
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "darwin")
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_darwin",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    assert _count_physical_cores() is None

    _PHYSICAL_CORES_CACHE.clear()
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "freebsd14")
    monkeypatch.setattr(
        "wyoming_vietnamese.cpu._count_physical_cores_freebsd",
        MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=_PROBE_TIMEOUT_SECONDS)
        ),
    )
    assert _count_physical_cores() is None


def test_available_cpu_count_container_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test available_cpu_count accounts for cgroup container limits."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.cpu_count", lambda: 64)
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.process_cpu_count", lambda: 64)
    monkeypatch.setattr("wyoming_vietnamese.cpu._cpu_count_cgroup", lambda: 2)

    assert available_cpu_count() == 2


def test_available_cpu_count_windows_capping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Windows caps maximum CPUs to _MAX_WINDOWS_WORKERS."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.sys.platform", "win32")
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.cpu_count", lambda: 128)
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.process_cpu_count", lambda: 128)
    monkeypatch.setattr("wyoming_vietnamese.cpu._cpu_count_cgroup", lambda: None)

    assert available_cpu_count() == _MAX_WINDOWS_WORKERS


def test_available_cpu_count_physical_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test available_cpu_count with only_physical_cores=True."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.cpu_count", lambda: 16)
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.process_cpu_count", lambda: 16)
    monkeypatch.setattr("wyoming_vietnamese.cpu._cpu_count_cgroup", lambda: None)
    monkeypatch.setattr("wyoming_vietnamese.cpu._count_physical_cores", lambda _affinity: 8)

    assert available_cpu_count(only_physical_cores=False) == 16
    assert available_cpu_count(only_physical_cores=True) == 8


def test_resolve_cpu_threads_validation_and_capping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_cpu_threads auto mode, explicit capping, and input validation."""
    monkeypatch.setattr("wyoming_vietnamese.cpu.available_cpu_count", lambda **_kwargs: 4)

    with pytest.raises(ValueError, match="must not be negative"):
        resolve_cpu_threads(-1)

    assert resolve_cpu_threads(0) == 4
    assert resolve_cpu_threads(2) == 2
    assert resolve_cpu_threads(8) == 4

    called_with: dict[str, bool] = {}

    def _mock_available(*, only_physical_cores: bool = False) -> int:
        called_with["only_physical_cores"] = only_physical_cores
        return 2

    monkeypatch.setattr("wyoming_vietnamese.cpu.available_cpu_count", _mock_available)
    assert resolve_cpu_threads(0, only_physical_cores=True) == 2
    assert called_with.get("only_physical_cores") is True


def test_resolve_cpu_threads_uses_cpu_count_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU detection remains positive when affinity data is unavailable."""
    monkeypatch.setattr("wyoming_vietnamese.cpu._cpu_count_affinity_set", lambda: None)
    monkeypatch.setattr("wyoming_vietnamese.cpu._cpu_count_cgroup", lambda: None)
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.process_cpu_count", lambda: None)
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.cpu_count", lambda: 2)
    assert resolve_cpu_threads(0) == 2
    monkeypatch.setattr("wyoming_vietnamese.cpu.os.cpu_count", lambda: None)
    assert resolve_cpu_threads(0) == 1
