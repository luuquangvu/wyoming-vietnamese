"""CPU feature inspection and runtime architecture variant detection."""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Final

from .const import ZeroTtsVariant

_LOGGER = logging.getLogger(__name__)

_AVX2_CPU_FLAGS: Final[frozenset[str]] = frozenset({"avx2", "fma", "f16c", "bmi2"})
_AVX512_CPU_FLAGS: Final[frozenset[str]] = frozenset(
    {"avx512f", "avx512vl", "avx512bw", "avx512dq"} | _AVX2_CPU_FLAGS
)


def get_darwin_cpu_flags() -> set[str]:
    """Parse and return CPU flags from sysctl on macOS.

    Returns:
        Set of lowercased CPU feature flags detected via sysctl, or an empty set.
    """
    if platform.system().lower() != "darwin":
        return set()
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features", "machdep.cpu.features"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return {f.lower() for f in out.stdout.split()}
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
        variant: Architecture variant identifier ('compat', 'avx2', 'avx512', or 'native').
        flags: Optional pre-fetched host CPU flags; if None, queries host flags automatically.

    Returns:
        True if the host processor satisfies all instruction set requirements.
    """
    if variant in (ZeroTtsVariant.COMPAT, ZeroTtsVariant.NATIVE):
        return True

    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64", "x64"):
        return False

    cpu_flags = get_host_cpu_flags() if flags is None else flags
    if not cpu_flags:
        return False

    if variant == ZeroTtsVariant.AVX512:
        return _AVX512_CPU_FLAGS.issubset(cpu_flags)
    if variant == ZeroTtsVariant.AVX2:
        return _AVX2_CPU_FLAGS.issubset(cpu_flags)
    return False


def detect_cpu_variant() -> str:
    """Detect the optimal ZeroTTS GGML CPU variant for the current host processor.

    Returns:
        One of 'avx512', 'avx2', 'compat', or 'native'. The
        'ZEROTTS_CPU_VARIANT' override accepts the same values.
    """
    if env_variant := os.environ.get("ZEROTTS_CPU_VARIANT", "").strip().lower():
        if env_variant in {
            ZeroTtsVariant.AVX512,
            ZeroTtsVariant.AVX2,
            ZeroTtsVariant.COMPAT,
            ZeroTtsVariant.NATIVE,
        }:
            return env_variant
        _LOGGER.warning(
            "Ignoring unknown ZEROTTS_CPU_VARIANT=%r; falling back to auto-detection",
            env_variant,
        )

    flags = get_host_cpu_flags()
    if can_host_execute_variant(ZeroTtsVariant.AVX512, flags=flags):
        return ZeroTtsVariant.AVX512

    if can_host_execute_variant(ZeroTtsVariant.AVX2, flags=flags):
        return ZeroTtsVariant.AVX2

    return ZeroTtsVariant.COMPAT


def get_supported_variants_for_host() -> list[str]:
    """Return all buildable CPU variants supported for the current machine architecture.

    Returns:
        List of supported variant names ('compat', 'avx2', 'avx512').
    """
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        return [ZeroTtsVariant.COMPAT, ZeroTtsVariant.AVX2, ZeroTtsVariant.AVX512]
    return [ZeroTtsVariant.COMPAT]
