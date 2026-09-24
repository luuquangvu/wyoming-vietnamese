"""CPU feature inspection and runtime architecture variant detection."""

from __future__ import annotations

import contextlib
import importlib
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from .const import ZeroTtsVariant

_MAX_WINDOWS_WORKERS: Final[int] = 61 if sys.version_info >= (3, 10) else 60
_PHYSICAL_CORES_CACHE: Final[dict[frozenset[int] | None, int]] = {}
_PROBE_TIMEOUT_SECONDS: Final[float] = 5.0

_SSE4_CPU_FLAGS: Final[frozenset[str]] = frozenset({"sse4_1", "sse4_2", "popcnt"})
_AVX_CPU_FLAGS: Final[frozenset[str]] = frozenset({"avx"} | _SSE4_CPU_FLAGS)
_AVX2_CPU_FLAGS: Final[frozenset[str]] = frozenset({"avx2", "fma", "f16c", "bmi2"} | _AVX_CPU_FLAGS)
_AVX512_CPU_FLAGS: Final[frozenset[str]] = frozenset(
    {"avx512f", "avx512vl", "avx512bw", "avx512dq"} | _AVX2_CPU_FLAGS
)
_ARM_DOTPROD_CPU_FLAGS: Final[frozenset[str]] = frozenset({"asimddp"})

_DARWIN_FLAG_MAP: Final[dict[str, str]] = {
    "avx1.0": "avx",
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
}


def get_darwin_cpu_flags() -> set[str]:
    """Parse and return CPU flags from sysctl on macOS.

    Returns:
        Set of lowercased CPU feature flags detected via sysctl, or an empty set.
    """
    if platform.system().lower() != "darwin":
        return set()

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        flags = {"asimd"}
        with contextlib.suppress(Exception):
            out = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm.FEAT_DotProd"],
                capture_output=True,
                text=True,
                check=False,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
            if out.returncode == 0 and out.stdout.strip() == "1":
                flags.update({"asimddp", "dotprod"})
        return flags

    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features", "machdep.cpu.features"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if out.returncode == 0:
            raw_flags = {f.lower() for f in out.stdout.split()}
            return raw_flags | {_DARWIN_FLAG_MAP[f] for f in raw_flags if f in _DARWIN_FLAG_MAP}
    return set()


def get_host_cpu_flags() -> set[str]:
    """Parse and return CPU flags present across all cores from /proc/cpuinfo or sysctl.

    Returns:
        Intersection set of lowercased CPU feature flags common to all online CPU cores.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return get_darwin_cpu_flags()

    core_flags: list[set[str]] = []
    with contextlib.suppress(OSError):
        for block in cpuinfo.read_text(encoding="utf-8", errors="replace").split("\n\n"):
            block_flags: set[str] = set()
            for line in block.splitlines():
                if line.startswith(("flags", "Features")):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        block_flags.update(parts[1].strip().split())
            if block_flags:
                core_flags.append(block_flags)

    return set.intersection(*core_flags) if core_flags else set()


def can_host_execute_variant(variant: str, flags: set[str] | None = None) -> bool:
    """Return whether the current host CPU can execute the specified variant.

    Args:
        variant: Architecture variant identifier ('compat', 'sse4', 'avx',
            'avx2', 'avx512', 'arm_dotprod', or 'native').
        flags: Optional pre-fetched host CPU flags; if None, queries host flags automatically.

    Returns:
        True if the host processor satisfies all instruction set requirements.
    """
    if variant in (ZeroTtsVariant.COMPAT, ZeroTtsVariant.NATIVE):
        return True

    machine = platform.machine().lower()
    cpu_flags = get_host_cpu_flags() if flags is None else flags
    if not cpu_flags:
        return False

    if variant == ZeroTtsVariant.ARM_DOTPROD:
        if machine not in ("aarch64", "arm64"):
            return False
        return bool(_ARM_DOTPROD_CPU_FLAGS.intersection(cpu_flags) or "dotprod" in cpu_flags)

    if machine not in ("x86_64", "amd64", "x64"):
        return False

    if variant == ZeroTtsVariant.AVX512:
        return _AVX512_CPU_FLAGS.issubset(cpu_flags)
    if variant == ZeroTtsVariant.AVX2:
        return _AVX2_CPU_FLAGS.issubset(cpu_flags)
    if variant == ZeroTtsVariant.AVX:
        return _AVX_CPU_FLAGS.issubset(cpu_flags)
    if variant == ZeroTtsVariant.SSE4:
        return _SSE4_CPU_FLAGS.issubset(cpu_flags)
    return False


def detect_cpu_variant() -> str:
    """Detect the optimal ZeroTTS GGML CPU variant for the current host processor.

    Returns:
        One of 'avx512', 'avx2', 'avx', 'sse4', 'arm_dotprod', or 'compat'.
    """
    machine = platform.machine().lower()
    flags = get_host_cpu_flags()

    if machine in ("aarch64", "arm64"):
        if can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD, flags=flags):
            return ZeroTtsVariant.ARM_DOTPROD
        return ZeroTtsVariant.COMPAT

    if can_host_execute_variant(ZeroTtsVariant.AVX512, flags=flags):
        return ZeroTtsVariant.AVX512

    if can_host_execute_variant(ZeroTtsVariant.AVX2, flags=flags):
        return ZeroTtsVariant.AVX2

    if can_host_execute_variant(ZeroTtsVariant.AVX, flags=flags):
        return ZeroTtsVariant.AVX

    if can_host_execute_variant(ZeroTtsVariant.SSE4, flags=flags):
        return ZeroTtsVariant.SSE4

    return ZeroTtsVariant.COMPAT


def get_supported_variants_for_host() -> list[str]:
    """Return all buildable CPU variants supported for the current machine architecture.

    Returns:
        List of supported variant names for the host architecture.
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        return [
            ZeroTtsVariant.COMPAT,
            ZeroTtsVariant.SSE4,
            ZeroTtsVariant.AVX,
            ZeroTtsVariant.AVX2,
            ZeroTtsVariant.AVX512,
        ]
    if machine in ("aarch64", "arm64"):
        return [
            ZeroTtsVariant.COMPAT,
            ZeroTtsVariant.ARM_DOTPROD,
        ]
    return [ZeroTtsVariant.COMPAT]


def _read_cgroup_v2_quota(cgroup_dir: Path) -> int | None:
    """Read and calculate CPU quota from a cgroup v2 directory.

    Args:
        cgroup_dir: Directory containing 'cpu.max'.

    Returns:
        Integer CPU limit if active and positive, or None.
    """
    cpu_max_file = cgroup_dir / "cpu.max"
    if not cpu_max_file.is_file():
        return None
    try:
        content = cpu_max_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    parts = content.split()
    if len(parts) != 2:
        return None
    quota_str, period_str = parts
    if quota_str == "max":
        return None
    try:
        quota = int(quota_str)
        period = int(period_str)
    except ValueError:
        return None
    return math.ceil(quota / period) if quota > 0 and period > 0 else None


def _read_cgroup_v1_quota(cgroup_dir: Path) -> int | None:
    """Read and calculate CPU quota from a cgroup v1 directory.

    Args:
        cgroup_dir: Directory containing 'cpu.cfs_quota_us' and 'cpu.cfs_period_us'.

    Returns:
        Integer CPU limit if active and positive, or None.
    """
    quota_file = cgroup_dir / "cpu.cfs_quota_us"
    period_file = cgroup_dir / "cpu.cfs_period_us"
    if not (quota_file.is_file() and period_file.is_file()):
        return None
    try:
        quota_str = quota_file.read_text(encoding="utf-8").strip()
        period_str = period_file.read_text(encoding="utf-8").strip()
        quota = int(quota_str)
        period = int(period_str)
    except OSError, ValueError:
        return None
    if quota > 0 and period > 0:
        return math.ceil(quota / period)
    return None


def _parse_proc_cgroup_paths(proc_cgroup: Path) -> tuple[list[str], list[str]]:
    """Parse cgroup paths for v2 and v1 cpu controllers from /proc/self/cgroup.

    Args:
        proc_cgroup: Path to '/proc/self/cgroup'.

    Returns:
        Tuple of (v2_relative_paths, v1_cpu_relative_paths).
    """
    cgroup_paths: list[str] = []
    cgroup_v1_cpu_paths: list[str] = []
    if not proc_cgroup.is_file():
        return cgroup_paths, cgroup_v1_cpu_paths

    with contextlib.suppress(OSError):
        for line in proc_cgroup.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            _cgroup_id, controllers, path = parts
            path_clean = path.strip().lstrip("/")
            if not path_clean:
                continue
            if not controllers:
                cgroup_paths.append(path_clean)
            elif any(c in controllers.split(",") for c in ("cpu", "cpuacct")):
                cgroup_v1_cpu_paths.append(path_clean)

    return cgroup_paths, cgroup_v1_cpu_paths


def _collect_hierarchy_quotas(
    reader: Callable[[Path], int | None],
    root: Path,
    relative_paths: list[str],
) -> list[int]:
    """Collect all active quotas along relative paths from leaf up to root.

    Args:
        reader: Function reading quota from a cgroup directory.
        root: Base cgroup mount path.
        relative_paths: Relative cgroup paths to inspect from leaf to parent.

    Returns:
        List of detected integer CPU quotas.
    """
    quotas: list[int] = []
    for rel_path in relative_paths:
        curr = root / rel_path
        while curr != root and curr.is_relative_to(root):
            quota = reader(curr)
            if quota is not None:
                quotas.append(quota)
            curr = curr.parent
    return quotas


def _find_cgroup_v2_quota(cgroup_root: Path, cgroup_paths: list[str]) -> int | None:
    """Find active cgroup v2 quotas along process hierarchy paths and return minimum.

    Args:
        cgroup_root: Path to cgroup root mount.
        cgroup_paths: Relative cgroup paths to inspect from leaf to parent.

    Returns:
        Minimum integer CPU limit if active and positive, or None.
    """
    quotas = _collect_hierarchy_quotas(_read_cgroup_v2_quota, cgroup_root, cgroup_paths)
    return min(quotas, default=None)


def _find_cgroup_v1_quota(cgroup_root: Path, cgroup_v1_paths: list[str]) -> int | None:
    """Find active cgroup v1 quotas along process hierarchy paths and return minimum.

    Args:
        cgroup_root: Path to cgroup root mount.
        cgroup_v1_paths: Relative cgroup paths for cpu controller.

    Returns:
        Minimum integer CPU limit if active and positive, or None.
    """
    quotas: list[int] = []
    for candidate_dir in (cgroup_root / "cpu", cgroup_root / "cpu,cpuacct"):
        if not candidate_dir.is_dir():
            continue
        if (root_quota := _read_cgroup_v1_quota(candidate_dir)) is not None:
            quotas.append(root_quota)
        quotas.extend(
            _collect_hierarchy_quotas(_read_cgroup_v1_quota, candidate_dir, cgroup_v1_paths)
        )
    return min(quotas, default=None)


def _cpu_count_cgroup() -> int | None:
    """Return CPU count limit imposed by cgroup bandwidth quotas (CFS), if any.

    Checks:
    - Cgroup v2 (`cpu.max`) at root and current process cgroup
    - Cgroup v1 (`cpu.cfs_quota_us` and `cpu.cfs_period_us`) at root and current process cgroup

    Returns:
        Minimum integer CPU limit across active cgroups, or None.
    """
    if sys.platform != "linux":
        return None

    cgroup_root = Path("/sys/fs/cgroup")
    if not cgroup_root.is_dir():
        return None

    quotas: list[int] = []
    if (root_v2 := _read_cgroup_v2_quota(cgroup_root)) is not None:
        quotas.append(root_v2)

    cgroup_paths, cgroup_v1_paths = _parse_proc_cgroup_paths(Path("/proc/self/cgroup"))
    if (quota_v2 := _find_cgroup_v2_quota(cgroup_root, cgroup_paths)) is not None:
        quotas.append(quota_v2)

    if (quota_v1 := _find_cgroup_v1_quota(cgroup_root, cgroup_v1_paths)) is not None:
        quotas.append(quota_v1)

    return min(quotas, default=None)


def _cpu_count_affinity_set() -> set[int] | None:
    """Return the set of logical CPU IDs allowed by current process affinity.

    Returns:
        Set of allowed logical CPU indices, or None if affinity cannot be determined.
    """
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError, NotImplementedError):
            return set(os.sched_getaffinity(0))

    with contextlib.suppress(Exception):
        psutil = importlib.import_module("psutil")
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            affinity = proc.cpu_affinity()
            if affinity is not None:
                return set(affinity)

    return None


def _parse_cpuinfo_cores(
    content: str,
    cpu_set: set[int] | None = None,
) -> set[tuple[str | None, str]]:
    """Parse distinct (physical id, core id) pairs from /proc/cpuinfo content.

    Args:
        content: Text content of /proc/cpuinfo.
        cpu_set: Optional set of allowed logical CPU IDs to filter.

    Returns:
        Set of (physical id, core id) tuples.
    """
    cores: set[tuple[str | None, str]] = set()
    processor: int | None = None
    core_id: str | None = None
    physical_id: str | None = None

    for line in [*content.splitlines(), ""]:
        if not line.strip():
            if core_id is not None and (
                cpu_set is None or (processor is not None and processor in cpu_set)
            ):
                cores.add((physical_id, core_id))
            processor = core_id = physical_id = None
            continue

        key, sep, value = line.partition(":")
        if not sep:
            continue
        key_str = key.strip()
        val_str = value.strip()
        if key_str == "processor":
            with contextlib.suppress(ValueError):
                processor = int(val_str)
        elif key_str == "core id":
            core_id = val_str
        elif key_str == "physical id":
            physical_id = val_str

    return cores


def _parse_cpuinfo_processors(
    content: str,
    cpu_set: set[int] | None = None,
) -> set[int]:
    """Parse distinct logical processor IDs from /proc/cpuinfo content.

    Args:
        content: Text content of /proc/cpuinfo.
        cpu_set: Optional set of allowed logical CPU IDs to filter.

    Returns:
        Set of logical processor IDs.
    """
    processors: set[int] = set()
    for line in content.splitlines():
        if line.startswith("processor"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    p_id = int(parts[1].strip())
                    if cpu_set is None or p_id in cpu_set:
                        processors.add(p_id)
    return processors


def _count_physical_cores_linux(cpu_set: set[int] | None = None) -> int:
    """Return the number of distinct physical cores on Linux by parsing /proc/cpuinfo.

    Args:
        cpu_set: Optional set of allowed logical CPU IDs to filter and collapse SMT siblings.

    Returns:
        Number of physical cores found.

    Raises:
        ValueError: If no physical cores could be identified.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        raise ValueError("/proc/cpuinfo not found")

    content = cpuinfo.read_text(encoding="utf-8", errors="replace")
    if cores := _parse_cpuinfo_cores(content, cpu_set):
        return len(cores)

    if processors := _parse_cpuinfo_processors(content, cpu_set):
        return len(processors)

    raise ValueError("could not find any physical core in /proc/cpuinfo")


def _count_physical_cores_darwin() -> int:
    """Return the number of physical cores on Darwin/macOS via sysctl.

    Returns:
        Number of physical cores.

    Raises:
        ValueError: If sysctl fails or returns non-positive output.
    """
    out = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if out.returncode == 0 and out.stdout.strip():
        with contextlib.suppress(ValueError):
            val = int(out.stdout.strip())
            if val > 0:
                return val
    raise ValueError("sysctl hw.physicalcpu returned non-positive or failed")


def _count_physical_cores_freebsd() -> int:
    """Return the number of physical cores on FreeBSD via sysctl.

    Returns:
        Number of physical cores.

    Raises:
        ValueError: If sysctl fails or returns non-positive output.
    """
    out = subprocess.run(
        ["sysctl", "-n", "kern.smp.cores"],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if out.returncode == 0 and out.stdout.strip():
        with contextlib.suppress(ValueError):
            val = int(out.stdout.strip())
            if val > 0:
                return val
    raise ValueError("sysctl kern.smp.cores returned non-positive or failed")


def _count_physical_cores_win32_powershell() -> int:
    """Return physical core count on Windows using PowerShell.

    Returns:
        Number of physical processor cores.

    Raises:
        ValueError: If PowerShell command fails or returns no valid core counts.
    """
    out = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance -ClassName Win32_Processor).NumberOfCores",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if out.returncode == 0 and out.stdout.strip():
        val = sum(int(x) for x in out.stdout.splitlines() if x.strip().isdigit())
        if val > 0:
            return val
    raise ValueError("PowerShell failed to return core count")


def _count_physical_cores_win32_ctypes() -> int:
    """Return physical core count on Windows using GetLogicalProcessorInformationEx.

    Returns:
        Number of physical processor cores.

    Raises:
        RuntimeError: If WinDLL is unavailable or processor info cannot be retrieved.
    """
    import ctypes
    from ctypes import wintypes

    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise RuntimeError("WinDLL not available on this platform")

    relation_processor_core = 0
    kernel32 = win_dll("kernel32", use_last_error=True)
    get_logical_proc_info = kernel32.GetLogicalProcessorInformationEx
    get_logical_proc_info.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_logical_proc_info.restype = wintypes.BOOL

    class SystemLogicalProcessorInformationEx(ctypes.Structure):
        """Structure header for GetLogicalProcessorInformationEx."""

        _fields_ = [
            ("Relationship", wintypes.DWORD),
            ("Size", wintypes.DWORD),
        ]

    returned_length = wintypes.DWORD()
    returned_length_ref = ctypes.byref(returned_length)
    if get_logical_proc_info(relation_processor_core, None, returned_length_ref):
        raise RuntimeError("unexpected successful buffer sizing call")

    buf = ctypes.create_string_buffer(returned_length.value)
    if not get_logical_proc_info(relation_processor_core, buf, returned_length_ref):
        get_last_error = getattr(ctypes, "get_last_error", lambda: 0)
        win_error = getattr(ctypes, "WinError", OSError)
        raise win_error(get_last_error())

    offset = 0
    physical_core_count = 0
    header_size = ctypes.sizeof(SystemLogicalProcessorInformationEx)

    while offset < returned_length.value:
        remaining = returned_length.value - offset
        if remaining < header_size:
            raise RuntimeError("truncated processor information record")
        proc_info = SystemLogicalProcessorInformationEx.from_buffer_copy(buf.raw, offset)
        record_size = proc_info.Size
        if record_size < header_size or record_size > remaining:
            raise RuntimeError("invalid processor information record size")
        if proc_info.Relationship == relation_processor_core:
            physical_core_count += 1
        offset += record_size

    if physical_core_count < 1:
        raise RuntimeError("no physical cores detected")
    return physical_core_count


def _count_physical_cores_win32() -> int:
    """Return physical core count on Windows via ctypes, PowerShell, or wmic.

    Returns:
        Number of physical processor cores.

    Raises:
        ValueError: If physical core count could not be detected on Windows.
    """
    with contextlib.suppress(Exception):
        return _count_physical_cores_win32_ctypes()
    with contextlib.suppress(Exception):
        return _count_physical_cores_win32_powershell()
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["wmic", "CPU", "Get", "NumberOfCores", "/Format:csv"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if lines := [
            line.split(",")[1]
            for line in out.stdout.splitlines()
            if line and line != "Node,NumberOfCores"
        ]:
            val = sum(int(x) for x in lines if x.isdigit())
            if val > 0:
                return val
    raise ValueError("could not detect physical cores on Windows")


def _count_physical_cores(cpu_set: set[int] | None = None) -> int | None:
    """Count physical CPU cores on the system, caching results per affinity set.

    Args:
        cpu_set: Optional set of allowed logical CPU IDs to filter on Linux.

    Returns:
        Number of physical cores found, or None if detection failed.
    """
    cache_key = None if cpu_set is None else frozenset(cpu_set)
    if cache_key in _PHYSICAL_CORES_CACHE:
        return _PHYSICAL_CORES_CACHE[cache_key]

    count: int | None = None
    with contextlib.suppress(Exception):
        current_platform = sys.platform
        if current_platform == "linux":
            count = _count_physical_cores_linux(cpu_set)
        elif current_platform == "darwin":
            count = _count_physical_cores_darwin()
        elif current_platform.startswith("freebsd"):
            count = _count_physical_cores_freebsd()
        elif current_platform == "win32":
            count = _count_physical_cores_win32()

    if count is not None and count >= 1:
        _PHYSICAL_CORES_CACHE[cache_key] = count
        return count
    return None


def available_cpu_count(*, only_physical_cores: bool = False) -> int:
    """Return the real number of CPUs available to the current process.

    Accounts for:
    - Total host logical CPUs (os.cpu_count()), capped on Windows to 61
    - CPU affinity constraints of the process (os.process_cpu_count / os.sched_getaffinity)
    - Cgroup CPU quota/bandwidth limits (CFS quotas in Docker, Kubernetes, systemd)
    - Physical CPU core count if `only_physical_cores` is True

    Args:
        only_physical_cores: When True, constrain to detected physical cores.

    Returns:
        Number of available CPUs (at least 1).
    """
    os_cpu_count = os.cpu_count() or 1
    if sys.platform == "win32":
        os_cpu_count = min(os_cpu_count, _MAX_WINDOWS_WORKERS)

    if hasattr(os, "process_cpu_count"):
        proc_cpus = os.process_cpu_count()
        cpu_count_affinity = proc_cpus if proc_cpus is not None else os_cpu_count
    else:
        affinity_set = _cpu_count_affinity_set()
        cpu_count_affinity = len(affinity_set) if affinity_set is not None else os_cpu_count

    cpu_count_cgroup = _cpu_count_cgroup()

    candidates: list[int] = [os_cpu_count, cpu_count_affinity]
    if cpu_count_cgroup is not None:
        candidates.append(cpu_count_cgroup)

    if only_physical_cores:
        affinity_set = _cpu_count_affinity_set() if sys.platform == "linux" else None
        physical_count = _count_physical_cores(affinity_set)
        if physical_count is not None and physical_count >= 1:
            candidates.append(physical_count)

    return max(1, min(candidates))


def resolve_cpu_threads(
    configured_threads: int,
    *,
    only_physical_cores: bool = False,
) -> int:
    """Resolve auto mode and cap inference threads to process-available CPUs.

    Args:
        configured_threads: Explicit thread count (0 for auto detection).
        only_physical_cores: When True, cap auto and explicit threads to physical cores.

    Returns:
        Resolved thread count (at least 1).

    Raises:
        ValueError: If configured_threads is negative.
    """
    if configured_threads < 0:
        raise ValueError("configured threads must not be negative")
    available_threads = available_cpu_count(only_physical_cores=only_physical_cores)
    return min(configured_threads, available_threads) if configured_threads else available_threads
