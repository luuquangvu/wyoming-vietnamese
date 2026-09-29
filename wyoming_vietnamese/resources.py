"""Container-aware hardware classification and runtime resource policy."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import psutil

from .const import TtsEngine
from .cpu import available_cpu_count

_LOGGER = logging.getLogger(__name__)
_GIB = 1024**3
_MIB = 1024**2
_ENGINE_MEMORY_ESTIMATES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        TtsEngine.NGHITTS: (300 * _MIB, 100 * _MIB),
        TtsEngine.ZEROTTS: (700 * _MIB, 0),
    }
)


class DeviceTier(StrEnum):
    """Broad device classes used to scale optional TTS resources."""

    WEAK = "weak"
    AVERAGE = "average"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    """Memory thresholds and TTS resource limits for one tier."""

    memory_below_bytes: int | None
    minimum_system_reserve_bytes: int
    max_tts_voices: int | None
    max_ram_cache_entries: int
    max_ram_cache_bytes: int
    max_ram_cache_item_bytes: int
    max_disk_cache_entries: int
    max_disk_cache_bytes: int
    max_disk_cache_item_bytes: int
    cache_idle_seconds: float


DEVICE_POLICIES: Mapping[DeviceTier, DevicePolicy] = MappingProxyType(
    {
        DeviceTier.WEAK: DevicePolicy(
            memory_below_bytes=2 * _GIB,
            minimum_system_reserve_bytes=256 * _MIB,
            max_tts_voices=1,
            max_ram_cache_entries=64,
            max_ram_cache_bytes=16 * 1024 * 1024,
            max_ram_cache_item_bytes=2 * 1024 * 1024,
            max_disk_cache_entries=256,
            max_disk_cache_bytes=128 * 1024 * 1024,
            max_disk_cache_item_bytes=8 * 1024 * 1024,
            cache_idle_seconds=2_592_000.0,
        ),
        DeviceTier.AVERAGE: DevicePolicy(
            memory_below_bytes=6 * _GIB,
            minimum_system_reserve_bytes=512 * _MIB,
            max_tts_voices=2,
            max_ram_cache_entries=256,
            max_ram_cache_bytes=64 * 1024 * 1024,
            max_ram_cache_item_bytes=8 * 1024 * 1024,
            max_disk_cache_entries=1_024,
            max_disk_cache_bytes=256 * 1024 * 1024,
            max_disk_cache_item_bytes=8 * 1024 * 1024,
            cache_idle_seconds=2_592_000.0,
        ),
        DeviceTier.STRONG: DevicePolicy(
            memory_below_bytes=None,
            minimum_system_reserve_bytes=_GIB,
            max_tts_voices=None,
            max_ram_cache_entries=2_048,
            max_ram_cache_bytes=512 * 1024 * 1024,
            max_ram_cache_item_bytes=8 * 1024 * 1024,
            max_disk_cache_entries=2_048,
            max_disk_cache_bytes=512 * 1024 * 1024,
            max_disk_cache_item_bytes=8 * 1024 * 1024,
            cache_idle_seconds=2_592_000.0,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DeviceResources:
    """Detected CPU and memory capacity available to this process."""

    cpu_count: int
    memory_bytes: int | None
    memory_capacity_bytes: int | None
    tier: DeviceTier

    @property
    def max_tts_voices(self) -> int | None:
        """Return the TTS voices to keep resident, or None for all configured voices."""
        return DEVICE_POLICIES[self.tier].max_tts_voices

    def ram_cache_limits(self) -> tuple[int, int, int]:
        """Return in-memory audio-cache entry, total-byte, and item-byte limits."""
        policy = DEVICE_POLICIES[self.tier]
        return (
            policy.max_ram_cache_entries,
            policy.max_ram_cache_bytes,
            min(policy.max_ram_cache_bytes, policy.max_ram_cache_item_bytes),
        )

    def disk_cache_limits(self) -> tuple[int, int, int, float]:
        """Return persistent audio-cache entry, byte, item-byte, and idle limits."""
        policy = DEVICE_POLICIES[self.tier]
        return (
            policy.max_disk_cache_entries,
            policy.max_disk_cache_bytes,
            policy.max_disk_cache_item_bytes,
            policy.cache_idle_seconds,
        )


def _host_available_memory_bytes() -> int | None:
    """Return currently available host RAM through psutil's cross-platform API."""
    try:
        memory_bytes = int(psutil.virtual_memory().available)
    except OSError, ValueError, NotImplementedError, psutil.Error:
        return None
    return memory_bytes if memory_bytes >= 0 else None


def _host_memory_capacity_bytes() -> int | None:
    """Return total host RAM through psutil's cross-platform system memory API."""
    try:
        memory_bytes = int(psutil.virtual_memory().total)
    except OSError, ValueError, NotImplementedError, psutil.Error:
        return None
    return memory_bytes if memory_bytes > 0 else None


def _read_memory_limit(path: Path) -> int | None:
    """Read a positive cgroup memory limit, ignoring unlimited sentinels."""
    try:
        raw_limit = path.read_text(encoding="ascii").strip()
        limit = int(raw_limit)
    except OSError, ValueError:
        return None
    return None if limit <= 0 or limit >= 1 << 60 else limit


def _read_memory_usage(path: Path) -> int | None:
    """Read current cgroup memory use, or return None when it is unavailable."""
    try:
        usage = int(path.read_text(encoding="ascii").strip())
    except OSError, ValueError:
        return None
    return usage if usage >= 0 else None


def _parse_proc_memory_paths(proc_cgroup: Path) -> tuple[list[str], list[str]]:
    """Extract cgroup v2 and v1 memory-controller paths from procfs."""
    v2_paths: list[str] = []
    v1_paths: list[str] = []
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2_paths, v1_paths
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        _, controllers, raw_path = parts
        relative_path = raw_path.strip().lstrip("/")
        if not relative_path:
            if not controllers:
                v2_paths.append("")
            elif "memory" in controllers.split(","):
                v1_paths.append("")
            continue
        if not controllers:
            v2_paths.append(relative_path)
        elif "memory" in controllers.split(","):
            v1_paths.append(relative_path)
    return v2_paths, v1_paths


def _hierarchy_memory_available(
    root: Path,
    relative_paths: list[str],
    limit_filename: str,
    usage_filename: str,
) -> list[int]:
    """Calculate remaining memory under each cgroup limit in the process hierarchy."""
    available: list[int] = []
    for relative_path in relative_paths:
        current = root / relative_path
        while current.is_relative_to(root):
            limit = _read_memory_limit(current / limit_filename)
            usage = _read_memory_usage(current / usage_filename)
            if limit is not None and usage is not None:
                available.append(max(0, limit - usage))
            if current == root:
                break
            current = current.parent
    return available


def _hierarchy_memory_limits(root: Path, relative_paths: list[str], filename: str) -> list[int]:
    """Read finite memory limits from each process cgroup and its parents."""
    limits: list[int] = []
    for relative_path in relative_paths:
        current = root / relative_path
        while current.is_relative_to(root):
            if (limit := _read_memory_limit(current / filename)) is not None:
                limits.append(limit)
            if current == root:
                break
            current = current.parent
    return limits


def _linux_available_memory_bytes() -> list[int]:
    """Return remaining memory for active Linux cgroup v1 and v2 limits."""
    cgroup_root = Path("/sys/fs/cgroup")
    if not cgroup_root.is_dir():
        return []
    v2_paths, v1_paths = _parse_proc_memory_paths(Path("/proc/self/cgroup"))
    available = _hierarchy_memory_available(
        cgroup_root,
        v2_paths,
        "memory.max",
        "memory.current",
    )
    for mount in (cgroup_root / "memory", cgroup_root / "memory,memory"):
        if mount.is_dir():
            available.extend(
                _hierarchy_memory_available(
                    mount,
                    v1_paths,
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                )
            )
    return available


def _linux_memory_capacity_bytes() -> list[int]:
    """Return finite cgroup v1 and v2 memory caps for the current process."""
    cgroup_root = Path("/sys/fs/cgroup")
    if not cgroup_root.is_dir():
        return []
    v2_paths, v1_paths = _parse_proc_memory_paths(Path("/proc/self/cgroup"))
    limits = _hierarchy_memory_limits(cgroup_root, v2_paths, "memory.max")
    for mount in (cgroup_root / "memory", cgroup_root / "memory,memory"):
        if mount.is_dir():
            limits.extend(_hierarchy_memory_limits(mount, v1_paths, "memory.limit_in_bytes"))
    return limits


def available_memory_bytes() -> int | None:
    """Return the minimum currently available host or container memory."""
    host_available = _host_available_memory_bytes()
    candidates = [host_available] if host_available is not None else []
    if sys.platform == "linux":
        candidates.extend(_linux_available_memory_bytes())
    return min(candidates, default=None)


def memory_capacity_bytes() -> int | None:
    """Return the minimum detectable host or container memory capacity."""
    host_capacity = _host_memory_capacity_bytes()
    candidates = [host_capacity] if host_capacity is not None else []
    if sys.platform == "linux":
        candidates.extend(_linux_memory_capacity_bytes())
    return min(candidates, default=None)


def minimum_startup_memory_bytes(
    tts_engine: str,
    voice_count: int,
    device_memory_bytes: int | None,
    ram_cache_bytes: int,
) -> int:
    """Return engine, full RAM-cache, and system-reserve memory estimates."""
    if voice_count < 1:
        raise ValueError("voice_count must be at least one")
    if ram_cache_bytes < 0:
        raise ValueError("ram_cache_bytes must not be negative")
    try:
        base_bytes, per_additional_voice_bytes = _ENGINE_MEMORY_ESTIMATES[tts_engine]
    except KeyError as err:
        raise ValueError(f"Unsupported TTS engine for memory estimate: {tts_engine}") from err
    additional_voice_count = voice_count - 1 if tts_engine == TtsEngine.NGHITTS else 0
    reserve_tier = _classify(device_memory_bytes)
    reserve_bytes = DEVICE_POLICIES[reserve_tier].minimum_system_reserve_bytes
    return (
        base_bytes
        + additional_voice_count * per_additional_voice_bytes
        + ram_cache_bytes
        + reserve_bytes
    )


def warn_if_low_startup_memory(
    tts_engine: str,
    voice_count: int,
    available_bytes: int | None,
    device_memory_bytes: int | None,
    ram_cache_bytes: int,
) -> None:
    """Warn when available RAM is below engine, cache, and shared-service estimates."""
    if available_bytes is None:
        _LOGGER.warning("Could not determine available RAM; skipping the startup memory check")
        return

    required_bytes = minimum_startup_memory_bytes(
        tts_engine,
        voice_count,
        device_memory_bytes,
        ram_cache_bytes,
    )
    reserve_tier = _classify(device_memory_bytes)
    reserve_bytes = DEVICE_POLICIES[reserve_tier].minimum_system_reserve_bytes
    if available_bytes < required_bytes:
        _LOGGER.warning(
            "Low available RAM for %s: available_mb=%.0f estimated_minimum_mb=%.0f "
            "(engine_mb=%.0f ram_cache_max_mb=%.0f system_reserve_mb=%.0f); "
            "continuing startup",
            tts_engine,
            available_bytes / _MIB,
            required_bytes / _MIB,
            (required_bytes - ram_cache_bytes - reserve_bytes) / _MIB,
            ram_cache_bytes / _MIB,
            reserve_bytes / _MIB,
        )


def _classify(memory_bytes: int | None) -> DeviceTier:
    """Classify memory capacity, treating unavailable measurements conservatively."""
    if memory_bytes is None:
        return DeviceTier.WEAK
    for tier in (DeviceTier.WEAK, DeviceTier.AVERAGE):
        memory_below_bytes = DEVICE_POLICIES[tier].memory_below_bytes
        if memory_below_bytes is not None and memory_bytes <= memory_below_bytes:
            return tier
    return DeviceTier.STRONG


def detect_device_resources() -> DeviceResources:
    """Inspect process-visible CPU and RAM and log the resulting resource tier."""
    cpu_count = available_cpu_count()
    memory_bytes = available_memory_bytes()
    capacity_bytes = memory_capacity_bytes()
    tier = _classify(capacity_bytes)
    _LOGGER.info(
        "Detected device resources: memory_tier=%s cpu_count=%d available_memory_mb=%s "
        "memory_capacity_mb=%s",
        tier,
        cpu_count,
        f"{memory_bytes / (1024 * 1024):.0f}" if memory_bytes is not None else "unknown",
        f"{capacity_bytes / (1024 * 1024):.0f}" if capacity_bytes is not None else "unknown",
    )
    return DeviceResources(cpu_count, memory_bytes, capacity_bytes, tier)
