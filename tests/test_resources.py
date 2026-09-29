"""Tests for container-aware resource detection and TTS scaling policy."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from wyoming_vietnamese import resources
from wyoming_vietnamese.resources import DeviceResources, DeviceTier


@pytest.mark.parametrize(
    ("memory_bytes", "expected"),
    [
        (1024**3, DeviceTier.WEAK),
        (2 * 1024**3, DeviceTier.WEAK),
        (4 * 1024**3, DeviceTier.AVERAGE),
        (6 * 1024**3, DeviceTier.AVERAGE),
        (8 * 1024**3, DeviceTier.STRONG),
        (16 * 1024**3, DeviceTier.STRONG),
        (None, DeviceTier.WEAK),
    ],
)
def test_classify_memory_tier(memory_bytes: int | None, expected: DeviceTier) -> None:
    """Classify memory tiers independently of CPU capacity."""
    assert resources._classify(memory_bytes) is expected


def test_read_memory_limit_ignores_invalid_and_unlimited_values(tmp_path: Path) -> None:
    """Read finite positive cgroup memory limits and ignore invalid sentinels."""
    limit_path = tmp_path / "memory.limit"
    assert resources._read_memory_limit(limit_path) is None
    for content in ("max", "0", "-1", str(1 << 60), "invalid"):
        limit_path.write_text(content, encoding="ascii")
        assert resources._read_memory_limit(limit_path) is None
    limit_path.write_text(str(512 * 1024**2), encoding="ascii")
    assert resources._read_memory_limit(limit_path) == 512 * 1024**2
    limit_path.write_text(" 1048576\n", encoding="ascii")
    assert resources._read_memory_limit(limit_path) == 1048576


def test_available_memory_uses_smallest_host_or_cgroup_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Account for other processes and cgroup usage when measuring available RAM."""
    monkeypatch.setattr(resources, "_host_available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources, "_linux_available_memory_bytes", lambda: [2 * 1024**3])
    assert resources.available_memory_bytes() == 2 * 1024**3

    monkeypatch.setattr(resources.sys, "platform", "darwin")
    assert resources.available_memory_bytes() == 8 * 1024**3
    monkeypatch.setattr(resources, "_host_available_memory_bytes", lambda: None)
    assert resources.available_memory_bytes() is None


def test_parse_proc_memory_paths(tmp_path: Path) -> None:
    """Parse memory controller paths from both cgroup versions."""
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text(
        "0::/docker/container\n"
        "1:cpu,cpuacct:/cpu/container\n"
        "2:memory:/memory/container\n"
        "malformed\n",
        encoding="utf-8",
    )
    assert resources._parse_proc_memory_paths(proc_cgroup) == (
        ["docker/container"],
        ["memory/container"],
    )


def test_hierarchy_memory_available_uses_leaf_and_parent(tmp_path: Path) -> None:
    """Find remaining memory under the leaf and its parent cgroup limits."""
    root = tmp_path / "cgroup"
    child = root / "parent" / "child"
    child.mkdir(parents=True)
    (root / "memory.max").write_text("max", encoding="ascii")
    (root / "parent" / "memory.max").write_text(str(4 * 1024**3), encoding="ascii")
    (root / "parent" / "memory.current").write_text(str(1 * 1024**3), encoding="ascii")
    (child / "memory.max").write_text(str(2 * 1024**3), encoding="ascii")
    (child / "memory.current").write_text(str(512 * 1024**2), encoding="ascii")
    assert resources._hierarchy_memory_available(
        root,
        ["parent/child"],
        "memory.max",
        "memory.current",
    ) == [
        2 * 1024**3 - 512 * 1024**2,
        4 * 1024**3 - 1 * 1024**3,
    ]


def test_linux_memory_available_checks_cgroup_v2_and_v1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inspect process leaf paths for both cgroup layouts."""
    root = tmp_path / "cgroup"
    (root / "tenant").mkdir(parents=True)
    (root / "tenant" / "memory.max").write_text(str(3 * 1024**3), encoding="ascii")
    (root / "tenant" / "memory.current").write_text(str(1 * 1024**3), encoding="ascii")
    v1_root = root / "memory"
    (v1_root / "tenant").mkdir(parents=True)
    (v1_root / "tenant" / "memory.limit_in_bytes").write_text(str(2 * 1024**3), encoding="ascii")
    (v1_root / "tenant" / "memory.usage_in_bytes").write_text(str(512 * 1024**2), encoding="ascii")
    proc_cgroup = tmp_path / "proc-cgroup"
    proc_cgroup.write_text("0::/tenant\n1:memory:/tenant\n", encoding="utf-8")
    original_path = Path

    def mapped_path(*parts: str) -> Path:
        if parts == ("/sys/fs/cgroup",):
            return root
        if parts == ("/proc/self/cgroup",):
            return proc_cgroup
        return original_path(*parts)

    monkeypatch.setattr(resources, "Path", mapped_path)
    assert resources._linux_available_memory_bytes() == [
        2 * 1024**3,
        2 * 1024**3 - 512 * 1024**2,
    ]
    assert resources._linux_memory_capacity_bytes() == [3 * 1024**3, 2 * 1024**3]


def test_host_available_memory_uses_psutil_on_any_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read currently available host RAM through psutil on any platform."""
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=40_960),
    )
    monkeypatch.setattr(resources.sys, "platform", "unknown")
    assert resources._host_available_memory_bytes() == 40_960


def test_host_available_memory_handles_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return an unknown value when psutil cannot inspect available host RAM."""

    def fail_probe() -> SimpleNamespace:
        raise OSError("memory probe failed")

    monkeypatch.setattr(resources.psutil, "virtual_memory", fail_probe)
    assert resources._host_available_memory_bytes() is None


def test_device_resource_policy_scales_voice_and_cache_limits() -> None:
    """Scale voice warmup by device tier and cache limits by memory tier."""
    weak = DeviceResources(2, 1024**3, 1024**3, DeviceTier.WEAK)
    average = DeviceResources(4, 3 * 1024**3, 4 * 1024**3, DeviceTier.AVERAGE)
    strong = DeviceResources(8, 8 * 1024**3, 16 * 1024**3, DeviceTier.STRONG)

    assert weak.max_tts_voices == 1
    assert weak.ram_cache_limits() == (64, 16 * 1024**2, 2 * 1024**2)
    assert weak.disk_cache_limits() == (256, 128 * 1024**2, 8 * 1024**2, 2_592_000.0)
    assert average.max_tts_voices == 2
    assert average.ram_cache_limits() == (256, 64 * 1024**2, 8 * 1024**2)
    assert average.disk_cache_limits() == (1_024, 256 * 1024**2, 8 * 1024**2, 2_592_000.0)
    assert strong.max_tts_voices is None
    assert strong.ram_cache_limits() == (2_048, 512 * 1024**2, 8 * 1024**2)
    assert strong.disk_cache_limits() == (2_048, 512 * 1024**2, 8 * 1024**2, 2_592_000.0)


@pytest.mark.parametrize(
    ("engine", "voice_count", "device_memory_gib", "ram_cache_mib", "expected_mib"),
    [
        ("nghitts", 1, 1, 16, 572),
        ("nghitts", 5, 4, 64, 1_276),
        ("nghitts", 5, 16, 512, 2_236),
        ("zerotts", 1, 4, 64, 1_276),
        ("zerotts", 8, 16, 512, 2_236),
    ],
)
def test_minimum_startup_memory_includes_engine_estimate_and_reserve(
    engine: str,
    voice_count: int,
    device_memory_gib: int,
    ram_cache_mib: int,
    expected_mib: int,
) -> None:
    """Estimate each engine's measured memory footprint plus other-service headroom."""
    assert (
        resources.minimum_startup_memory_bytes(
            engine,
            voice_count,
            device_memory_gib * 1024**3,
            ram_cache_mib * 1024**2,
        )
        == expected_mib * 1024**2
    )


def test_startup_memory_warning_includes_available_memory_and_reserve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn without blocking when an engine estimate plus reserve will not fit."""
    required_bytes = resources.minimum_startup_memory_bytes(
        "zerotts",
        1,
        4 * 1024**3,
        64 * 1024**2,
    )
    resources.warn_if_low_startup_memory(
        "zerotts",
        1,
        required_bytes - 1,
        4 * 1024**3,
        64 * 1024**2,
    )
    assert "Low available RAM for zerotts" in caplog.text
    assert "ram_cache_max_mb=64" in caplog.text
    assert "continuing startup" in caplog.text


def test_startup_memory_warning_skips_when_available_ram_is_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report that the check was skipped when no platform RAM reading is available."""
    resources.warn_if_low_startup_memory("nghitts", 1, None, 2 * 1024**3, 16 * 1024**2)
    assert "skipping the startup memory check" in caplog.text


def test_four_core_device_with_sixteen_gib_ram_uses_strong_policy() -> None:
    """CPU count does not lower resident voice or cache capacity when RAM is ample."""
    device = DeviceResources(4, 16 * 1024**3, 16 * 1024**3, DeviceTier.STRONG)

    assert device.max_tts_voices is None
    assert device.ram_cache_limits() == (2_048, 512 * 1024**2, 8 * 1024**2)
    assert device.disk_cache_limits() == (2_048, 512 * 1024**2, 8 * 1024**2, 2_592_000.0)


def test_detect_device_resources_logs_profile(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Combine CPU and memory probes and report their resource classification."""
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 8)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(resources, "memory_capacity_bytes", lambda: 16 * 1024**3)
    with caplog.at_level("INFO", logger=resources.__name__):
        detected = resources.detect_device_resources()
    assert detected == DeviceResources(8, 8 * 1024**3, 16 * 1024**3, DeviceTier.STRONG)
    assert "tier=strong" in caplog.text
    assert "memory_tier=strong" in caplog.text
