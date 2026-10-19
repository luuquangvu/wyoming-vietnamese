"""ZeroTTS GGML runtime engine implementation for high-quality Vietnamese speech synthesis."""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import json
import logging
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort
    from zerotts.codec import MossCodecDecoder, MossStreamingDecoder

from .const import (
    ZEROTTS_REPO_URL,
    ZEROTTS_SOURCE_COMMIT,
    ZeroTtsAudio,
    ZeroTtsCodebook,
    ZeroTtsEnv,
    ZeroTtsFile,
    ZeroTtsVariant,
)
from .cpu import (
    detect_cpu_variant,
    resolve_cpu_threads,
)
from .tts_model import (
    ZEROTTS_VOICES_BY_ID,
    ZeroTtsVoiceSpec,
    is_zerotts_edge_variant,
    resolve_quality_codebooks,
    resolve_zerotts_codec_int8,
)

_LOGGER = logging.getLogger(__name__)

_X86_VARIANT_FALLBACK_HIERARCHY: Final[tuple[str, ...]] = (
    ZeroTtsVariant.AVX512,
    ZeroTtsVariant.AVX2,
    ZeroTtsVariant.AVX,
    ZeroTtsVariant.SSE4,
    ZeroTtsVariant.COMPAT,
)
_ARM_VARIANT_FALLBACK_HIERARCHY: Final[tuple[str, ...]] = (
    ZeroTtsVariant.ARM_DOTPROD,
    ZeroTtsVariant.COMPAT,
)
_ZEROTTS_RUNTIME_DEPENDENCIES: Final[tuple[str, ...]] = (
    "libggml-base.so.0",
    "libggml-cpu.so.0",
    "libggml.so.0",
)
_ZEROTTS_BUILD_TIMEOUT_SECONDS: Final[float] = 1800.0
_PROCESS_TERMINATION_TIMEOUT_SECONDS: Final[float] = 5.0
_SIGKILL_GRACE_PERIOD_SECONDS: Final[float] = 15.0
_WORKER_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 2.0


class _ZeroTtsSampling(ctypes.Structure):
    """Sampling parameter configuration for ZeroTTS GGML frame generation."""

    _fields_ = [
        ("text_temperature", ctypes.c_float),
        ("text_topk", ctypes.c_int32),
        ("audio_temperature", ctypes.c_float),
        ("audio_topk", ctypes.c_int32),
        ("audio_topp", ctypes.c_float),
        ("audio_repetition_penalty", ctypes.c_float),
    ]


class _ZeroTtsHParams(ctypes.Structure):
    """Hyperparameters of the loaded ZeroTTS GGUF model."""

    _fields_ = [
        ("d_model", ctypes.c_int32),
        ("n_heads", ctypes.c_int32),
        ("n_layers", ctypes.c_int32),
        ("d_ff", ctypes.c_int32),
        ("local_n_layers", ctypes.c_int32),
        ("local_n_heads", ctypes.c_int32),
        ("local_d_ff", ctypes.c_int32),
        ("n_depth", ctypes.c_int32),
        ("num_codebooks", ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("n_voice_queries", ctypes.c_int32),
        ("cross_qk_norm", ctypes.c_int32),
        ("rope_theta", ctypes.c_float),
        ("layer_norm_eps", ctypes.c_float),
        ("qk_norm_eps", ctypes.c_float),
    ]


def _get_variant_fallback_order(optimal_variant: str) -> list[str]:
    """Return ordered list of fallback variants starting from the detected optimal variant."""
    if optimal_variant == ZeroTtsVariant.NATIVE:
        return [ZeroTtsVariant.NATIVE, ZeroTtsVariant.COMPAT]
    if optimal_variant in _ARM_VARIANT_FALLBACK_HIERARCHY:
        idx = _ARM_VARIANT_FALLBACK_HIERARCHY.index(optimal_variant)
        return list(_ARM_VARIANT_FALLBACK_HIERARCHY[idx:])
    if optimal_variant in _X86_VARIANT_FALLBACK_HIERARCHY:
        idx = _X86_VARIANT_FALLBACK_HIERARCHY.index(optimal_variant)
        return list(_X86_VARIANT_FALLBACK_HIERARCHY[idx:])
    return [optimal_variant, ZeroTtsVariant.COMPAT]


def get_zerotts_candidate_lib_paths(
    custom_lib_path: Path | None = None,
    base_dir: Path | None = None,
) -> list[tuple[str, Path]]:
    """Enumerate candidate ZeroTTS GGML library paths in preference order.

    Args:
        custom_lib_path: First candidate path; a missing file falls back to configured
            and default locations.
        base_dir: Optional directory containing variant subdirectories. When provided,
            only this directory is searched after the custom path.

    Returns:
        List of tuples containing (variant_description, path_candidate).
    """
    candidates: list[tuple[str, Path]] = []
    if custom_lib_path is not None:
        candidates.append(("custom", custom_lib_path))

    if env_path := os.environ.get("ZEROTTS_LIB_PATH", "").strip():
        candidates.append(("env", Path(env_path).expanduser()))

    optimal_variant = detect_cpu_variant()
    variant_order = _get_variant_fallback_order(optimal_variant)

    repo_root = Path(__file__).resolve().parent.parent
    base_dirs = (
        [base_dir]
        if base_dir is not None
        else [
            Path("/app/lib"),
            repo_root / "lib",
            Path(__file__).resolve().parent / "lib",
        ]
    )

    candidates.extend(
        (var, base_dir / var / ZeroTtsFile.LIBZEROTTS_SO)
        for var, base_dir in itertools.product(variant_order, base_dirs)
    )
    for base_dir in base_dirs:
        symlink_candidate = base_dir / ZeroTtsFile.LIBZEROTTS_SO
        if symlink_candidate.is_symlink():
            with contextlib.suppress(OSError):
                target_str = os.readlink(symlink_candidate)
                for var in variant_order:
                    if var in Path(target_str).parts:
                        candidates.append((f"symlink-{var}", symlink_candidate))
                        break
    return candidates


def _restore_or_remove_rollback(item: Path, orig_path: Path) -> None:
    """Restore rollback item to orig_path if missing, otherwise remove it.

    Args:
        item: Rollback file or directory to restore or delete.
        orig_path: Destination path where the file or directory originally lived.
    """
    with contextlib.suppress(OSError):
        if not orig_path.exists() and not orig_path.is_symlink():
            item.replace(orig_path)
        elif item.is_dir() and not item.is_symlink():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink()


def _extract_rollback_original_name(name: str, tag: str, pid: int | None) -> str | None:
    """Extract original filename from rollback pattern (e.g. .<orig><tag><pid>).

    Args:
        name: Name of the filesystem entry to inspect.
        tag: Rollback marker tag (e.g. '.bak.' or '.stale.').
        pid: Optional process ID filter.

    Returns:
        The extracted original name if pattern matches, None otherwise.
    """
    if pid is not None:
        suffix = f"{tag}{pid}"
        return name[1 : -len(suffix)] if name.endswith(suffix) else None
    if tag in name:
        orig, _, trailing = name[1:].rpartition(tag)
        if orig and trailing.isdigit():
            return orig
    return None


def _is_tmp_publish_file(name: str, pid: int | None) -> bool:
    """Return whether filename matches temporary publish file pattern.

    Args:
        name: Name of the filesystem entry to inspect.
        pid: Optional process ID filter.

    Returns:
        True if the name matches a temporary publish file, False otherwise.
    """
    if pid is not None:
        return name.endswith(f".tmp.{pid}")
    return name[1:].rpartition(".tmp.")[2].isdigit() if ".tmp." in name else False


def _handle_workspace_entry(item: Path, directory: Path, pid: int | None) -> None:
    """Clean transient build artifacts or restore rollback copies for a single entry.

    Args:
        item: Filesystem item path to evaluate.
        directory: Directory containing the item.
        pid: Optional process ID filter.
    """
    name = item.name
    if pid is not None:
        is_build_dir = name.startswith((f"_build_{pid}_", f"_build_all_{pid}_"))
    else:
        is_build_dir = name.startswith(("_build_", "_build_all_"))

    if is_build_dir:
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                item.unlink()
        return

    if not name.startswith("."):
        return

    for tag in (".bak.", ".stale."):
        orig_name = _extract_rollback_original_name(name, tag, pid)
        if orig_name is not None:
            _restore_or_remove_rollback(item, directory / orig_name)
            return

    if _is_tmp_publish_file(name, pid):
        with contextlib.suppress(OSError):
            item.unlink()


def _clean_zerotts_build_workspace(directory: Path, pid: int | None = None) -> None:
    """Remove transient build directories and restore rollback copies in workspace.

    Args:
        directory: Directory containing potential transient build artifacts to clean.
        pid: Optional process ID of the build process to filter process-specific artifacts.
    """
    if not directory.exists() or not directory.is_dir():
        return
    for item in directory.iterdir():
        _handle_workspace_entry(item, directory, pid)


def _is_process_group_alive(pgid: int) -> bool:
    """Return True if any processes remain alive in the process group.

    Args:
        pgid: Process group ID to check.

    Returns:
        True if any process in the process group is running, False otherwise.
    """
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _signal_process_group(pid: int, sig: signal.Signals) -> None:
    """Send termination signal to process group or taskkill tree on Windows.

    Args:
        pid: Process ID or process group ID to signal.
        sig: Signal to send (SIGTERM or SIGKILL).
    """
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(pid, sig)
    elif platform.system() == "Windows":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                timeout=_PROCESS_TERMINATION_TIMEOUT_SECONDS,
            )


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    """Terminate child process and all descendant processes in its process group.

    Args:
        proc: The running subprocess whose process tree should be terminated.
    """
    pid = proc.pid if isinstance(getattr(proc, "pid", None), int) else None
    if pid is not None:
        _signal_process_group(pid, signal.SIGTERM)

    with contextlib.suppress(ProcessLookupError, OSError, AttributeError):
        proc.terminate()

    with contextlib.suppress(subprocess.TimeoutExpired, AttributeError):
        proc.wait(timeout=_SIGKILL_GRACE_PERIOD_SECONDS)

    # Check whether any members of the build process group remain before workspace cleanup,
    # and escalate to SIGKILL when descendants are still running, even if the direct child exited.
    if pid is not None and _is_process_group_alive(pid):
        _signal_process_group(pid, signal.SIGKILL)

    with contextlib.suppress(ProcessLookupError, OSError, AttributeError):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=_PROCESS_TERMINATION_TIMEOUT_SECONDS)


def _abort_process_and_clean_workspace(
    proc: subprocess.Popen[str],
    dest_dir: Path,
    target_dir: Path,
) -> None:
    """Terminate child process tree and clean transient build artifacts from workspace.

    Args:
        proc: The running subprocess to terminate.
        dest_dir: Target variant directory to clean.
        target_dir: Base destination directory to clean.
    """
    pid = getattr(proc, "pid", None)
    pid_val = pid if isinstance(pid, int) else None

    # Ensure termination errors do not skip workspace cleanup or prevent caller from raising timeout
    with contextlib.suppress(Exception):
        _terminate_process_tree(proc)

    _clean_zerotts_build_workspace(dest_dir, pid=pid_val)
    if dest_dir != target_dir:
        _clean_zerotts_build_workspace(target_dir, pid=pid_val)


def _run_zerotts_build_script(
    build_script: Path,
    target_dir: Path,
    *,
    variant: str,
    native: bool,
    publish_subdir: bool = False,
) -> Path:
    """Execute tools/build_zerotts.py via subprocess to compile libzerotts.so.

    Args:
        build_script: Path to tools/build_zerotts.py.
        target_dir: Destination base directory for compiled library.
        variant: CPU architecture optimization variant.
        native: Whether to optimize for the host CPU.
        publish_subdir: Whether to publish into a variant-named subdirectory.

    Returns:
        Path to the compiled libzerotts.so file.

    Raises:
        RuntimeError: If the build times out or fails.
        FileNotFoundError: If the compiled library file is missing after build.
    """
    actual_variant = ZeroTtsVariant.NATIVE if native else variant
    if actual_variant not in ZeroTtsVariant:
        raise ValueError(f"Invalid ZeroTTS variant: {actual_variant}")
    dest_dir = target_dir / actual_variant if publish_subdir else target_dir
    cmd = [
        sys.executable,
        str(build_script),
        str(dest_dir),
        "--commit",
        ZEROTTS_SOURCE_COMMIT,
        "--repo-url",
        ZEROTTS_REPO_URL,
    ]
    if native or variant == ZeroTtsVariant.NATIVE:
        cmd.append("--native")
    else:
        cmd.extend(["--variant", variant])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=_ZEROTTS_BUILD_TIMEOUT_SECONDS)
    except BaseException as err:
        _abort_process_and_clean_workspace(proc, dest_dir, target_dir)
        if isinstance(err, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"ZeroTTS GGML build timed out after {_ZEROTTS_BUILD_TIMEOUT_SECONDS:g} seconds"
            ) from err
        raise

    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to build ZeroTTS GGML shared library: {stderr or stdout}"
        ) from None
    candidate = dest_dir / ZeroTtsFile.LIBZEROTTS_SO
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{ZeroTtsFile.LIBZEROTTS_SO} not found after build at: {candidate}"
        ) from None
    return candidate


def _unload_cdll(lib: ctypes.CDLL) -> None:
    """Close the underlying dlopen handle of a CDLL instance to prevent library leakage.

    Args:
        lib: Loaded ctypes shared library instance to close.
    """
    handle = getattr(lib, "_handle", None)
    if isinstance(handle, int):
        with contextlib.suppress(Exception):
            import _ctypes

            _ctypes.dlclose(handle)
        with contextlib.suppress(Exception):
            lib.__dict__["_handle"] = None


def _load_zerotts_candidate(candidate: Path) -> ctypes.CDLL:
    """Load a ZeroTTS library with GGML dependencies from the same variant directory.

    The container exposes ``/app/lib`` through ``LD_LIBRARY_PATH`` while the
    optimized libraries are published below ``/app/lib/<variant>``.  Loading
    only the top-level library lets the dynamic loader satisfy its SONAMEs from
    a stale root-level GGML copy before it considers the library's ``$ORIGIN``
    runpath.  Preloading the matching dependencies by absolute path keeps the
    selected ZeroTTS variant and its GGML kernels together.

    Dependencies are loaded locally without ``RTLD_GLOBAL`` and are cleanly unloaded
    if candidate loading fails, preventing cross-variant library and symbol reuse
    before fallback candidates are attempted.

    Args:
        candidate: Path to candidate libzerotts.so file.

    Returns:
        Loaded ctypes.CDLL shared library instance.

    Raises:
        OSError: If loading the candidate or any required dependency fails.
    """
    resolved_candidate = candidate.resolve()
    dependency_paths = [resolved_candidate.parent / name for name in _ZEROTTS_RUNTIME_DEPENDENCIES]
    existing_deps = [path for path in dependency_paths if path.is_file()]
    missing_deps = [
        name
        for path, name in zip(dependency_paths, _ZEROTTS_RUNTIME_DEPENDENCIES, strict=False)
        if not path.is_file()
    ]
    if (existing_deps or any(resolved_candidate.parent.glob("libggml*.so*"))) and missing_deps:
        missing_names = ", ".join(missing_deps)
        raise OSError(
            f"Incomplete ZeroTTS runtime dependencies for {resolved_candidate}: "
            f"missing {missing_names}"
        )

    if existing_deps:
        loaded_dependencies: list[ctypes.CDLL] = []
        try:
            loaded_dependencies.extend(
                ctypes.CDLL(str(dependency.resolve())) for dependency in dependency_paths
            )
            return ctypes.CDLL(str(resolved_candidate))
        except Exception:
            for dep in reversed(loaded_dependencies):
                _unload_cdll(dep)
            raise
    return ctypes.CDLL(str(resolved_candidate))


def _build_zerotts_ggml_lib(
    target_dir: Path,
    *,
    variant: str = ZeroTtsVariant.COMPAT,
    native: bool = False,
    publish_subdir: bool = False,
) -> Path:
    """Build the ZeroTTS GGML shared library from upstream source.

    Args:
        target_dir: Destination directory where the built shared library should be placed.
        variant: CPU architecture optimization variant ('compat', 'avx2', 'avx512', 'native').
        native: Whether to optimize for host processor architecture instead of portable baseline.
        publish_subdir: Whether to publish into a variant-named subdirectory under target_dir.

    Returns:
        Path to the compiled libzerotts.so file.

    Raises:
        RuntimeError: If building the library fails.
        FileNotFoundError: If the compiled library cannot be found after build.
    """
    actual_variant = ZeroTtsVariant.NATIVE if native else variant
    try:
        from tools.build_zerotts import build_zerotts_ggml_lib

        return build_zerotts_ggml_lib(
            target_dir,
            repo_url=ZEROTTS_REPO_URL,
            commit=ZEROTTS_SOURCE_COMMIT,
            variant=actual_variant,
            native=native,
            publish_subdir=publish_subdir,
        )
    except ImportError as import_err:
        repo_root = Path(__file__).resolve().parent.parent
        build_script = repo_root / "tools" / "build_zerotts.py"
        if build_script.is_file():
            return _run_zerotts_build_script(
                build_script,
                target_dir,
                variant=actual_variant,
                native=native,
                publish_subdir=publish_subdir,
            )
        raise RuntimeError("ZeroTTS build script tools/build_zerotts.py not found") from import_err


def resolve_zerotts_ggml_lib(
    custom_lib_path: Path | None = None,
    *,
    build_dir: Path | None = None,
) -> ctypes.CDLL:
    """Load the ZeroTTS GGML library, building it locally if no candidate exists.

    Args:
        custom_lib_path: First candidate path; a missing file falls back to configured
            and default locations.
        build_dir: Optional directory to search and build the GGML library in. The
            production default remains ``<repository>/lib``.

    Returns:
        Loaded ctypes.CDLL instance with configured argument and return types.

    Raises:
        FileNotFoundError: If the shared library cannot be located or built.
        RuntimeError: If loading the shared library fails.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = get_zerotts_candidate_lib_paths(custom_lib_path, build_dir)

    loaded_lib: ctypes.CDLL | None = None
    last_error: OSError | None = None
    last_attempted_path: Path | None = None

    for var_name, cand in candidates:
        if not cand.is_file():
            continue
        last_attempted_path = cand
        try:
            loaded_lib = _load_zerotts_candidate(cand)
            _LOGGER.info(
                "Loaded ZeroTTS GGML runtime shared library (%s variant): %s",
                var_name,
                cand,
            )
            break
        except OSError as err:
            last_error = err
            _LOGGER.warning(
                "Failed to load ZeroTTS candidate %s (%s variant): %s",
                cand,
                var_name,
                err,
            )

    if loaded_lib is None:
        if last_error is not None:
            _LOGGER.warning(
                "Failed to load candidate ZeroTTS library from %s: %s; falling back to local build",
                last_attempted_path,
                last_error,
            )

        target_dir = build_dir or (repo_root / "lib")
        optimal_variant = detect_cpu_variant()
        resolved_lib_path = _build_zerotts_ggml_lib(
            target_dir,
            variant=optimal_variant,
            publish_subdir=True,
        )
        try:
            loaded_lib = _load_zerotts_candidate(resolved_lib_path)
        except OSError as err:
            raise RuntimeError(
                f"Failed to load {ZeroTtsFile.LIBZEROTTS_SO} from {resolved_lib_path}: {err}"
            ) from err

    return _configure_zerotts_lib_signatures(loaded_lib)


def _configure_zerotts_lib_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Configure ctypes argument and return types for ZeroTTS C API functions."""
    lib.zerotts_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.zerotts_init_from_file.restype = ctypes.c_void_p
    lib.zerotts_free.argtypes = [ctypes.c_void_p]
    lib.zerotts_free.restype = None
    lib.zerotts_hparams_of.argtypes = [ctypes.c_void_p]
    lib.zerotts_hparams_of.restype = ctypes.POINTER(_ZeroTtsHParams)
    lib.zerotts_sampling_defaults.argtypes = [ctypes.POINTER(_ZeroTtsSampling)]
    lib.zerotts_sampling_defaults.restype = None
    lib.zerotts_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.zerotts_begin.restype = ctypes.c_int
    lib.zerotts_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.POINTER(_ZeroTtsSampling),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.zerotts_frame.restype = ctypes.c_int
    lib.zerotts_advance.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.zerotts_advance.restype = ctypes.c_int
    lib.zerotts_timings.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.zerotts_timings.restype = None
    if hasattr(lib, "zerotts_set_max_codebooks"):
        lib.zerotts_set_max_codebooks.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.zerotts_set_max_codebooks.restype = None
    return lib


def _default_non_pipelined_codec_threads(total: int) -> int:
    """Return default ONNX codec thread count for non-pipelined execution.

    Args:
        total: Available CPU thread count.

    Returns:
        Recommended codec thread count.
    """
    if total <= 2:
        return 1
    return 2 if total <= 4 else min(total // 2, 4)


def _default_pipelined_codec_threads(total: int) -> int:
    """Return default ONNX codec thread count for pipelined execution.

    Args:
        total: Available CPU thread count.

    Returns:
        Recommended codec thread count.
    """
    if total <= 4:
        return 1
    if total <= 8:
        return 2
    return 3 if total <= 12 else 4


def _read_env_thread_override(env_var_name: str) -> int | None:
    """Read and validate an integer thread count from an environment variable.

    Args:
        env_var_name: Name of the environment variable.

    Returns:
        Positive integer thread count if valid and set, None otherwise.
    """
    if env_var_name not in os.environ:
        return None
    with contextlib.suppress(ValueError):
        val = int(os.environ[env_var_name].strip())
        if val > 0:
            return val
    return None


def _resolve_dual_override_threads(
    effective_codec: int,
    effective_ggml: int,
    total: int,
    *,
    opportunistic: bool,
) -> tuple[int, int]:
    """Resolve thread allocation when both codec and GGML overrides are specified.

    Args:
        effective_codec: Resolved codec thread count override.
        effective_ggml: Resolved GGML thread count override.
        total: Available CPU threads budget (at least 2).
        opportunistic: Whether opportunistic oversubscription is allowed for <= 4 cores.

    Returns:
        Tuple of (ggml_threads, codec_threads).
    """
    c_threads = min(max(1, effective_codec), total - 1)
    g_max = total if (opportunistic and 2 < total <= 4 and c_threads == 1) else total - c_threads
    g_threads = min(max(1, effective_ggml), g_max)
    return g_threads, c_threads


def _resolve_explicit_pipelined_threads(
    effective_codec: int | None,
    effective_ggml: int | None,
    total: int,
    *,
    opportunistic: bool = True,
) -> tuple[int, int] | None:
    """Resolve thread allocation when explicit overrides are specified.

    Args:
        effective_codec: Resolved codec thread count override or None.
        effective_ggml: Resolved GGML thread count override or None.
        total: Available CPU threads budget (at least 2).
        opportunistic: Whether opportunistic oversubscription is allowed for <= 4 cores.

    Returns:
        Tuple of (ggml_threads, codec_threads) if overrides exist, None otherwise.
    """
    if effective_codec is not None and effective_ggml is not None:
        return _resolve_dual_override_threads(
            effective_codec, effective_ggml, total, opportunistic=opportunistic
        )

    if effective_codec is not None:
        c_requested = max(1, effective_codec)
        c_threads = min(c_requested, total - 1)
        g_threads = (
            total
            if (opportunistic and 2 < total <= 4 and c_threads == 1)
            else max(1, total - c_threads)
        )
        return g_threads, c_threads

    if effective_ggml is not None:
        g_requested = max(1, effective_ggml)
        g_max = total if (opportunistic and 2 < total <= 4) else total - 1
        g_threads = min(g_requested, g_max)
        c_threads = (
            1
            if (opportunistic and 2 < total <= 4 and g_threads == total)
            else max(1, total - g_threads)
        )
        return g_threads, c_threads

    return None


def resolve_zerotts_thread_budget(
    total_threads: int,
    *,
    pipelined: bool = True,
    codec_threads: int | None = None,
    ggml_threads: int | None = None,
    opportunistic: bool = True,
) -> tuple[int, int]:
    """Dynamically allocate CPU threads between GGML frame generation and ONNX codec.

    When pipelined streaming overlaps GGML autoregression with ONNX codec decoding,
    allocating full threads to both causes severe thread oversubscription on many-core systems.
    However, on edge quad-core CPUs (e.g. Intel Celeron N5105 / J4125 / Raspberry Pi 4),
    the neural codec decoding runs for only ~15 ms every 16 frames (~1.28 s). Under opportunistic
    budgeting (opportunistic=True, default), GGML is allocated all available cores (total_threads)
    while the codec is assigned 1 thread, allowing the CPU to execute GGML at full capacity for >98%
    of the generation time while the OS CFS scheduler timeslices during brief codec bursts.

    Strict partitioning (ggml + codec <= total_threads) is enforced with opportunistic=False,
    allowing callers on resource-constrained or strict-quota systems to retain strict budgeting.

    Note on single-core systems: When pipelined is True and total_threads <= 1,
    strict budget compliance is impossible because both runtimes require at least 1 thread.
    In this case, (1, 1) is returned as an explicit minimum-runnable fallback; engines
    should disable pipelining on single-thread systems to avoid concurrent thread contention.

    Args:
        total_threads: Total CPU threads budget available for the engine (at least 1).
        pipelined: Whether frame generation and codec decoding run concurrently.
        codec_threads: Optional explicit override for MOSS ONNX decoder threads.
        ggml_threads: Optional explicit override for GGML autoregressive transformer threads.
        opportunistic: Whether to allow opportunistic 1-thread oversubscription on <= 4 cores.

    Returns:
        Tuple of (ggml_threads, codec_threads), both >= 1.
    """
    total = max(1, total_threads)
    effective_codec = (
        codec_threads
        if codec_threads is not None
        else _read_env_thread_override(ZeroTtsEnv.CODEC_THREADS)
    )
    effective_ggml = (
        ggml_threads
        if ggml_threads is not None
        else _read_env_thread_override(ZeroTtsEnv.GGML_THREADS)
    )

    if not pipelined:
        c_threads = (
            max(1, effective_codec)
            if effective_codec is not None
            else _default_non_pipelined_codec_threads(total)
        )
        g_threads = max(1, effective_ggml) if effective_ggml is not None else total
        return g_threads, c_threads

    if total <= 1:
        return 1, 1

    explicit = _resolve_explicit_pipelined_threads(
        effective_codec, effective_ggml, total, opportunistic=opportunistic
    )
    if explicit is not None:
        return explicit

    if total <= 2:
        return 1, 1

    if total <= 4:
        return (total, 1) if opportunistic else (total - 1, 1)

    c_threads = _default_pipelined_codec_threads(total)
    return total - c_threads, c_threads


def _init_null_voice_embedding(model_dir: Path, hp: _ZeroTtsHParams) -> np.ndarray:
    """Initialize null voice latent embedding from model assets or default zeros.

    Args:
        model_dir: Directory containing ZeroTTS model files.
        hp: GGML engine hyperparameters.

    Returns:
        Float32 numpy array representing null voice embedding.
    """
    null_voice_path = model_dir / ZeroTtsFile.NULL_VOICE_EMB
    if null_voice_path.is_file():
        return np.ascontiguousarray(np.load(null_voice_path).reshape(-1), dtype=np.float32)
    flat_dim = int(hp.n_voice_queries) * int(hp.d_model)
    return np.zeros(flat_dim, dtype=np.float32)


def _resolve_engine_options(
    num_threads: int,
    pipelined: bool | None,
    codec_threads: int | None,
    opportunistic: bool = True,
    max_codebooks: int | None = None,
) -> tuple[int, bool, int, int, int]:
    """Resolve runtime threading, pipelining, and codebook depth flags.

    Args:
        num_threads: Requested CPU thread count.
        pipelined: Pipelining override.
        codec_threads: Codec thread count override.
        opportunistic: Whether to allow opportunistic thread budgeting on <= 4 cores.
        max_codebooks: Optional upper bound on generated acoustic codebooks.

    Returns:
        Tuple of (total_threads, pipelined, ggml_threads, codec_threads, max_codebooks).
    """
    total_threads = resolve_cpu_threads(num_threads)
    effective_pipelined = True if pipelined is None else pipelined
    is_pipelined = effective_pipelined and total_threads > 1

    ggml_thr, codec_thr = resolve_zerotts_thread_budget(
        total_threads,
        pipelined=is_pipelined,
        codec_threads=codec_threads,
        opportunistic=opportunistic,
    )

    if max_codebooks is not None:
        effective_cbks = max(ZeroTtsCodebook.MIN, min(ZeroTtsCodebook.MAX, max_codebooks))
    else:
        env_cbk = os.environ.get(ZeroTtsEnv.MAX_CODEBOOKS, "").strip()
        if env_cbk.isdigit():
            effective_cbks = max(ZeroTtsCodebook.MIN, min(ZeroTtsCodebook.MAX, int(env_cbk)))
        else:
            quality = os.environ.get(ZeroTtsEnv.QUALITY, "").strip().lower()
            effective_cbks = resolve_quality_codebooks(quality, is_edge=is_zerotts_edge_variant())

    return total_threads, is_pipelined, ggml_thr, codec_thr, effective_cbks


def _load_text_processors(
    model_dir: Path,
) -> tuple[Callable[..., np.ndarray], Callable[[str], str]]:
    """Load tokenizer and text normalization function for ZeroTTS.

    Args:
        model_dir: Directory containing ZeroTTS configuration and models.

    Returns:
        Tuple of (tokenizer, text_normalizer).

    Raises:
        FileNotFoundError: If config.json is missing.
    """
    config_path = model_dir / ZeroTtsFile.CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing ZeroTTS config.json at: {config_path}")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))

    from zerotts.text_norm import normalize_vi_text
    from zerotts.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(config_data, model_dir)
    return tokenizer, normalize_vi_text


class ZeroTtsGgmlEngine:
    """ZeroTTS speech synthesis engine using native GGML runtime and MOSS neural codec."""

    def __init__(
        self,
        model_dir: Path,
        gguf_path: Path,
        voices: Sequence[ZeroTtsVoiceSpec],
        num_threads: int = 0,
        custom_lib_path: Path | None = None,
        seed: int = 1234,
        codec_threads: int | None = None,
        pipelined: bool | None = None,
        codec_int8: bool | None = None,
        opportunistic: bool = True,
        max_codebooks: int | None = None,
    ) -> None:
        """Initialize the ZeroTTS GGML engine from local weights and configurations.

        Args:
            model_dir: Base directory containing ZeroTTS model assets (tokenizer, voices, codec).
            gguf_path: Path to the quantized GGUF model weights file.
            voices: Sequence of configured voice specifications to expose.
            num_threads: CPU thread count for inference.
            custom_lib_path: Optional explicit path to libzerotts.so.
            seed: Default random generator seed for deterministic frame sampling.
            codec_threads: Optional CPU thread count for MOSS ONNX decoder.
            pipelined: Whether to overlap GGML frame synthesis and ONNX codec decoding.
            codec_int8: Optional override for MOSS neural codec INT8 precision.
            opportunistic: Whether to allow opportunistic thread budgeting on <= 4 cores.
            max_codebooks: Optional upper bound on generated acoustic codebooks.

        Raises:
            ValueError: If no voice is configured.
            FileNotFoundError: If required model files are missing.
            RuntimeError: If GGML context initialization fails.
        """
        if not voices:
            raise ValueError("At least one ZeroTTS voice must be configured")
        if not gguf_path.is_file():
            raise FileNotFoundError(f"Missing ZeroTTS GGUF model weights: {gguf_path}")

        self.sample_rate: int = ZeroTtsAudio.SAMPLE_RATE
        self.chunks_are_sentences: bool = False
        (
            self.total_threads,
            self.pipelined,
            self.threads,
            self.codec_threads,
            self.max_codebooks,
        ) = _resolve_engine_options(
            num_threads=num_threads,
            pipelined=pipelined,
            codec_threads=codec_threads,
            opportunistic=opportunistic,
            max_codebooks=max_codebooks,
        )
        self.codec_int8: bool = resolve_zerotts_codec_int8(
            codec_int8,
            model_name=gguf_path.name,
        )

        self.model_dir: Path = model_dir
        self.gguf_path: Path = gguf_path
        self._ctx: ctypes.c_void_p | None = None
        self._busy_worker: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None

        self._lib = resolve_zerotts_ggml_lib(custom_lib_path)
        self._ctx = self._lib.zerotts_init_from_file(
            str(self.gguf_path.resolve()).encode("utf-8"),
            self.threads,
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to initialize ZeroTTS GGML context from: {self.gguf_path}")

        try:
            hp_ptr = self._lib.zerotts_hparams_of(self._ctx)
            self._hp = hp_ptr.contents
            self.num_codebooks: int = int(self._hp.num_codebooks)

            if hasattr(self._lib, "zerotts_set_max_codebooks"):
                self._lib.zerotts_set_max_codebooks(self._ctx, self.max_codebooks)
            else:
                self.max_codebooks = self.num_codebooks

            self._sp = _ZeroTtsSampling()
            self._lib.zerotts_sampling_defaults(ctypes.byref(self._sp))

            self._tokenizer, self._normalize_fn = _load_text_processors(model_dir)
            self._codec = self._init_neural_codec(model_dir)

            _LOGGER.info(
                "ZeroTTS thread allocation: ggml_threads=%d codec_threads=%d "
                "pipelined=%s (budget=%d)",
                self.threads,
                self.codec_threads,
                self.pipelined,
                self.total_threads,
            )

            self._null_voice_emb = _init_null_voice_embedding(model_dir, self._hp)
            self._init_voices(voices)

            self._seed: int = seed
            self._rng = np.random.default_rng(self._seed)
            self._au_buf: np.ndarray = np.empty(self.num_codebooks, dtype=np.float32)
            self._au_buf_ptr = self._au_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        except Exception:
            self.close()
            raise

    def _init_neural_codec(self, model_dir: Path) -> MossCodecDecoder:
        """Initialize the MOSS neural codec decoder.

        Args:
            model_dir: Model assets directory.

        Returns:
            Configured MossCodecDecoder instance.
        """
        from zerotts.codec import MossCodecDecoder

        codec_dir = model_dir / "onnx" / "codec"
        decoder = MossCodecDecoder(
            codec_dir,
            intra_op_num_threads=self.codec_threads,
        )
        if self.codec_int8:
            int8_step_path = codec_dir / ZeroTtsFile.DECODE_STEP_INT8_ONNX
            if int8_step_path.is_file():
                if not hasattr(decoder, "_decode_step_sess"):
                    _LOGGER.warning(
                        "ZeroTTS INT8 codec binding unavailable; decoder missing _decode_step_sess",
                    )
                else:
                    try:
                        int8_sess = self._create_int8_decode_session(int8_step_path)
                    except Exception as err:
                        _LOGGER.warning(
                            "Failed to create ZeroTTS INT8 decode session (%s); "
                            "retaining FP32 decoding",
                            err,
                        )
                    else:
                        decoder._decode_step_sess = int8_sess
                        _LOGGER.info(
                            "ZeroTTS bound optimized INT8 neural codec: %s",
                            int8_step_path.name,
                        )
            else:
                _LOGGER.debug(
                    "ZeroTTS INT8 codec enabled but %s not found; using FP32 baseline",
                    int8_step_path,
                )
        return decoder

    def _create_int8_decode_session(self, int8_step_path: Path) -> ort.InferenceSession:
        """Create an ONNX Runtime inference session configured for INT8 decode step.

        Args:
            int8_step_path: Path to the quantized INT8 decode step ONNX model.

        Returns:
            Configured onnxruntime InferenceSession instance.
        """
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.codec_threads
        sess_options.inter_op_num_threads = 1
        return ort.InferenceSession(
            str(int8_step_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _init_voices(self, voices: Sequence[ZeroTtsVoiceSpec]) -> None:
        """Initialize voice mapping catalogs and load voice embeddings.

        Args:
            voices: Sequence of configured voice specifications.
        """
        self._voice_embeddings: dict[str, np.ndarray] = {}
        self._voices_by_id: dict[str, ZeroTtsVoiceSpec] = {voice.id: voice for voice in voices}
        self._voices_by_name: dict[str, ZeroTtsVoiceSpec] = {voice.name: voice for voice in voices}
        self._default_voice = voices[0]

        self._preset_voices: Mapping[str, Mapping[str, object]] = {
            voice.name: {"description": voice.description} for voice in voices
        }

        for voice in voices:
            self._load_voice_embedding(voice.id)

    def _load_voice_embedding(self, voice_id: str) -> None:
        """Load and cache the precomputed voice latent embedding.

        Args:
            voice_id: Normalized voice identifier.

        Raises:
            FileNotFoundError: If the voice pack file does not exist.
            ValueError: If the voice embedding element count does not match model hyperparameters.
        """
        voice_npz_path = self.model_dir / "voices" / voice_id / "voice.npz"
        if not voice_npz_path.is_file():
            raise FileNotFoundError(
                f"Voice pack file not found for {voice_id} at: {voice_npz_path}"
            )
        with np.load(voice_npz_path) as npz:
            raw_emb = npz["voice_emb"]
            emb = np.ascontiguousarray(raw_emb.reshape(-1), dtype=np.float32)

        expected_size = int(self._hp.n_voice_queries) * int(self._hp.d_model)
        if emb.size != expected_size:
            raise ValueError(
                f"Invalid voice embedding size for {voice_id}: "
                f"expected {expected_size}, got {emb.size}"
            )

        self._voice_embeddings[voice_id] = emb
        if voice_id in self._voices_by_id:
            display_name = self._voices_by_id[voice_id].name
            self._voice_embeddings[display_name] = emb

    def _resolve_voice_embedding(self, voice: str | None) -> np.ndarray:
        """Resolve precomputed voice latent array from voice name or ID.

        Args:
            voice: Voice name, ID, or None for the default voice.

        Returns:
            Flat contiguous float32 numpy array representing the voice conditioning latents.

        Raises:
            ValueError: If an unsupported voice name or ID is provided.
        """
        if voice is None:
            voice = self._default_voice.name

        if voice in self._voice_embeddings:
            return self._voice_embeddings[voice]

        if voice in self._voices_by_id:
            self._load_voice_embedding(voice)
            return self._voice_embeddings[voice]

        catalog_voice = ZEROTTS_VOICES_BY_ID.get(voice)
        if catalog_voice is not None and catalog_voice.name in self._voice_embeddings:
            return self._voice_embeddings[catalog_voice.name]

        raise ValueError(f"Unsupported ZeroTTS voice: {voice!r}")

    @staticmethod
    def _decode_audio_chunk(
        stream: MossStreamingDecoder, buf: list[np.ndarray] | np.ndarray
    ) -> np.ndarray | None:
        """Decode a sequence of acoustic code frames into an audio waveform array.

        Args:
            stream: Active MOSS streaming codec decoder instance.
            buf: List of code frames or contiguous array of shape (1, K, T) to decode.

        Returns:
            Decoded 1D float32 audio array if audio was generated, or None.
        """
        chunk_codes = buf if isinstance(buf, np.ndarray) else np.stack(buf, axis=-1)
        chunk_audio = stream.decode_chunk(chunk_codes)
        audio_out = np.asarray(chunk_audio, dtype=np.float32).ravel()
        return audio_out if audio_out.size else None

    def _execute_frame_step(
        self,
        t: int,
        forbid_eoa: bool,
        codes_buf: ctypes.Array[ctypes.c_int32],
        is_eoa: ctypes.c_int,
    ) -> None:
        """Execute a single native GGML frame generation step.

        Args:
            t: Generation step index.
            forbid_eoa: Flag indicating whether end-of-audio is disallowed.
            codes_buf: Ctypes array buffer to store codebook tokens.
            is_eoa: Ctypes integer flag indicating whether end-of-audio was emitted.

        Raises:
            RuntimeError: If zerotts_frame returns non-zero.
        """
        cu = float(self._rng.random())
        self._rng.random(out=self._au_buf, dtype=np.float32)

        rc = self._lib.zerotts_frame(
            self._ctx,
            forbid_eoa,
            ctypes.byref(self._sp),
            cu,
            self._au_buf_ptr,
            codes_buf,
            ctypes.byref(is_eoa),
        )
        if rc != 0:
            raise RuntimeError(f"zerotts_frame failed at step {t}")

    @staticmethod
    def _advance_tail_countdown(
        tail_left: int, is_eoa: bool, extra_frames: int = 1
    ) -> tuple[int, bool]:
        """Advance the end-of-audio frame countdown.

        Args:
            tail_left: Remaining tail frame count (-1 if EOA has not occurred).
            is_eoa: Whether the current frame triggered end-of-audio.
            extra_frames: Additional frames allowed after EOA triggers.

        Returns:
            Tuple of (updated tail_left, should_stop flag).
        """
        if tail_left < 0:
            return (extra_frames, False) if is_eoa else (-1, False)
        updated = tail_left - 1
        return updated, updated <= 0

    @staticmethod
    def _zero_residual_codebooks(
        codes_buf: ctypes.Array[ctypes.c_int32], max_codebooks: int, total_codebooks: int
    ) -> None:
        """Zero out residual codebooks beyond the maximum codebook count."""
        for idx in range(max_codebooks, total_codebooks):
            codes_buf[idx] = 0

    def _generate_code_frames(
        self, stop_event: threading.Event | None = None
    ) -> Iterator[np.ndarray]:
        """Generate acoustic code frames sequentially from the initialized GGML context.

        Args:
            stop_event: Optional event signaled when synthesis is cancelled early.

        Yields:
            Numpy array of shape (k,) containing codebook indices for each frame.

        Raises:
            RuntimeError: If zerotts_frame or zerotts_advance returns non-zero.
        """
        k = self.num_codebooks
        min_frames = 4
        max_frames = 1500
        eoa_extra = 1

        codes_buf = (ctypes.c_int32 * k)()
        is_eoa = ctypes.c_int(0)
        tail_left = -1

        for t in range(max_frames):
            if stop_event is not None and stop_event.is_set():
                break
            forbid_eoa = (t < min_frames) or (tail_left >= 0)
            self._execute_frame_step(t, forbid_eoa, codes_buf, is_eoa)
            self._zero_residual_codebooks(codes_buf, self.max_codebooks, k)

            yield np.frombuffer(codes_buf, dtype=np.int32).copy()

            if stop_event is not None and stop_event.is_set():
                break

            tail_left, should_stop = self._advance_tail_countdown(
                tail_left, is_eoa.value != 0, eoa_extra
            )
            if should_stop:
                break

            rc = self._lib.zerotts_advance(self._ctx, codes_buf, t)
            if rc != 0:
                raise RuntimeError(f"zerotts_advance failed at step {t}")

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield synthesized audio waveforms incrementally using progressive MOSS decoding.

        Args:
            text: Input Vietnamese utterance.
            voice: Voice identifier or display name, or None for the first configured voice.
            seed: Optional random generator seed for deterministic frame sampling.

        Yields:
            Float32 numpy audio arrays representing synthesized speech chunks.

        Raises:
            ValueError: If the requested voice is unsupported.
            RuntimeError: If the GGML context is closed or native generation fails.
        """
        yield from self.infer_stream_normalized(
            self.normalize(text, voice=voice),
            voice=voice,
            seed=seed,
        )

    def _prepare_synthesis_session(
        self,
        text: str,
        voice: str | None,
        seed: int | None,
    ) -> bool:
        """Validate state and initialize native GGML context for utterance synthesis.

        Args:
            text: Normalized text to synthesize.
            voice: Voice identifier or display name.
            seed: Optional random generator seed.

        Returns:
            True if text tokens were generated and native session started, False if empty.

        Raises:
            RuntimeError: If GGML context is closed, previous worker is busy, or begin fails.
        """
        if self._ctx is None:
            raise RuntimeError("ZeroTTS GGML context is closed")

        if self._busy_worker is not None and self._busy_worker.is_alive():
            self._busy_worker.join(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            if self._busy_worker.is_alive():
                raise RuntimeError("Previous ZeroTTS synthesis worker is still executing")
            self._busy_worker = None

        self._rng = np.random.default_rng(self._seed if seed is None else seed)

        text_tokens = self._tokenizer(text)
        text_ids = np.ascontiguousarray(text_tokens, dtype=np.int32)
        if len(text_ids) == 0:
            return False

        voice_emb = self._resolve_voice_embedding(voice)
        voice_emb_ptr = voice_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        text_ids_ptr = text_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        ret = self._lib.zerotts_begin(
            self._ctx,
            text_ids_ptr,
            len(text_ids),
            voice_emb_ptr,
        )
        if ret != 0:
            raise RuntimeError(f"zerotts_begin failed with status code {ret}")
        return True

    def _infer_stream_sequential(
        self,
        stream: MossStreamingDecoder,
    ) -> Iterator[np.ndarray]:
        """Synthesize and decode audio chunks synchronously in the caller thread.

        Args:
            stream: Streaming neural codec decoder.

        Yields:
            Decoded audio chunks as float32 numpy arrays.
        """
        first_chunk_target = 2
        cap_chunk_target = 16
        chunk_buf = np.empty((1, self.num_codebooks, cap_chunk_target), dtype=np.int32)
        target_chunk = first_chunk_target
        count = 0
        for frame in self._generate_code_frames():
            chunk_buf[0, :, count] = frame
            count += 1
            if count >= target_chunk:
                audio_out = self._decode_audio_chunk(stream, chunk_buf[:, :, :count])
                if audio_out is not None:
                    yield audio_out
                count = 0
                target_chunk = min(cap_chunk_target, target_chunk * 2)

        if count > 0:
            audio_out = self._decode_audio_chunk(stream, chunk_buf[:, :, :count])
            if audio_out is not None:
                yield audio_out

    @staticmethod
    def _enqueue_chunk(
        chunk_queue: queue.Queue[np.ndarray | Exception | None],
        item: np.ndarray | Exception | None,
        stop_event: threading.Event,
    ) -> bool:
        """Enqueue an item while monitoring cancellation.

        Args:
            chunk_queue: Inter-thread queue.
            item: Code slice, exception, or None sentinel.
            stop_event: Event signaling cancellation.

        Returns:
            True if item was successfully placed in queue, False if stopped.
        """
        while not stop_event.is_set():
            try:
                chunk_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run_producer_loop(
        self,
        chunk_queue: queue.Queue[np.ndarray | Exception | None],
        stop_event: threading.Event,
        producer_error: list[Exception],
    ) -> None:
        """Generate acoustic code frames and push chunk slices to pipeline queue.

        Args:
            chunk_queue: Inter-thread queue for generated code chunks.
            stop_event: Event signaling cancellation.
            producer_error: Destination list to capture unhandled worker exception.
        """
        first_chunk_target = 2
        cap_chunk_target = 16
        try:
            chunk_buf = np.empty((1, self.num_codebooks, cap_chunk_target), dtype=np.int32)
            target_chunk = first_chunk_target
            count = 0
            for frame in self._generate_code_frames(stop_event=stop_event):
                if stop_event.is_set():
                    return
                chunk_buf[0, :, count] = frame
                count += 1
                if count >= target_chunk:
                    chunk_slice = chunk_buf[:, :, :count].copy(order="C")
                    if not self._enqueue_chunk(chunk_queue, chunk_slice, stop_event):
                        return
                    count = 0
                    target_chunk = min(cap_chunk_target, target_chunk * 2)

            if count > 0:
                chunk_slice = chunk_buf[:, :, :count].copy(order="C")
                if not self._enqueue_chunk(chunk_queue, chunk_slice, stop_event):
                    return
            if not stop_event.is_set():
                self._enqueue_chunk(chunk_queue, None, stop_event)
        except Exception as exc:
            producer_error.append(exc)
            self._enqueue_chunk(chunk_queue, exc, stop_event)

    def _drain_and_join_worker(
        self,
        worker: threading.Thread,
        chunk_queue: queue.Queue[np.ndarray | Exception | None],
        stop_event: threading.Event,
    ) -> None:
        """Signal stop event, drain remaining queue items, and join the producer worker.

        Args:
            worker: The active producer thread.
            chunk_queue: Inter-thread queue.
            stop_event: Event signaling cancellation.
        """
        stop_event.set()
        deadline = time.monotonic() + _WORKER_SHUTDOWN_TIMEOUT_SECONDS
        while worker.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(queue.Empty):
                chunk_queue.get_nowait()
            remaining = deadline - time.monotonic()
            worker.join(timeout=min(0.05, max(0.001, remaining)))
        if worker.is_alive():
            self._busy_worker = worker
            _LOGGER.warning(
                "ZeroTTS synthesis worker did not terminate within %g seconds",
                _WORKER_SHUTDOWN_TIMEOUT_SECONDS,
            )
        else:
            self._busy_worker = None

    @staticmethod
    def _check_worker_empty_queue(
        worker: threading.Thread,
        producer_error: list[Exception],
    ) -> bool:
        """Check if producer worker has terminated on empty queue condition.

        Args:
            worker: Active producer thread.
            producer_error: List capturing producer exceptions.

        Returns:
            True if worker terminated normally and consumer should break.

        Raises:
            Exception: If worker terminated with an error.
        """
        if not worker.is_alive():
            if producer_error:
                raise producer_error[0] from None
            return True
        return False

    def _consume_pipeline_queue(
        self,
        stream: MossStreamingDecoder,
        chunk_queue: queue.Queue[np.ndarray | Exception | None],
        worker: threading.Thread,
        producer_error: list[Exception],
    ) -> Iterator[np.ndarray]:
        """Consume code slices from queue and yield decoded audio waveform chunks.

        Args:
            stream: Streaming neural codec decoder.
            chunk_queue: Inter-thread queue.
            worker: Active producer thread.
            producer_error: List capturing producer exceptions.

        Yields:
            Decoded audio chunks as float32 numpy arrays.

        Raises:
            Exception: If producer encountered an error during frame generation.
        """
        while True:
            try:
                item = chunk_queue.get(timeout=0.1)
            except queue.Empty:
                if self._check_worker_empty_queue(worker, producer_error):
                    break
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            audio_out = self._decode_audio_chunk(stream, item)
            if audio_out is not None:
                yield audio_out
        if producer_error:
            raise producer_error[0]

    def _infer_stream_pipelined(
        self,
        stream: MossStreamingDecoder,
    ) -> Iterator[np.ndarray]:
        """Coordinate producer thread and decode audio asynchronously in caller thread.

        Args:
            stream: Streaming neural codec decoder.

        Yields:
            Decoded audio chunks as float32 numpy arrays.
        """
        chunk_queue: queue.Queue[np.ndarray | Exception | None] = queue.Queue(maxsize=4)
        stop_event = threading.Event()
        producer_error: list[Exception] = []

        worker = threading.Thread(
            target=self._run_producer_loop,
            args=(chunk_queue, stop_event, producer_error),
            daemon=True,
        )
        self._busy_worker = worker
        worker.start()

        try:
            yield from self._consume_pipeline_queue(stream, chunk_queue, worker, producer_error)
        finally:
            self._drain_and_join_worker(worker, chunk_queue, stop_event)

    def infer_stream_normalized(
        self,
        text: str,
        *,
        voice: str | None = None,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio for text that has already passed through the configured normalizer.

        Args:
            text: Normalized text to synthesize.
            voice: Voice identifier or display name.
            seed: Optional random generator seed.

        Yields:
            Decoded audio chunks as float32 numpy arrays.
        """
        if not self._prepare_synthesis_session(text, voice, seed):
            return

        stream = self._codec.streaming_decoder()
        try:
            if not self.pipelined:
                yield from self._infer_stream_sequential(stream)
            else:
                yield from self._infer_stream_pipelined(stream)
        finally:
            stream.close()

    def normalize(self, text: str, *, voice: str | None = None) -> str:
        """Return text after the normalizer used by inference."""
        del voice
        return text if self._normalize_fn is None else self._normalize_fn(text)

    def _defer_context_cleanup(
        self, busy_worker: threading.Thread, ctx_to_free: ctypes.c_void_p
    ) -> None:
        """Schedule asynchronous native context deallocation after worker thread termination.

        Args:
            busy_worker: Still-active worker thread.
            ctx_to_free: Native context pointer to release.
        """
        lib = self._lib

        def _deferred_cleanup() -> None:
            """Wait for worker exit before deallocating native context."""
            busy_worker.join()
            lib.zerotts_free(ctx_to_free)

        self._cleanup_thread = threading.Thread(
            target=_deferred_cleanup,
            name="ZeroTtsDeferredContextFree",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _handle_unresponsive_worker(self, busy_worker: threading.Thread) -> None:
        """Handle a worker thread that failed to stop within shutdown timeout.

        Args:
            busy_worker: The active worker thread that did not terminate.
        """
        _LOGGER.warning(
            "ZeroTTS synthesis worker still executing after %g seconds; "
            "scheduling deferred native context deallocation",
            _WORKER_SHUTDOWN_TIMEOUT_SECONDS,
        )
        ctx_to_free = self._ctx
        self._ctx = None
        self._busy_worker = None
        if ctx_to_free is not None:
            self._defer_context_cleanup(busy_worker, ctx_to_free)

    def close(self) -> None:
        """Release native model resources and tensor buffers."""
        busy_worker = getattr(self, "_busy_worker", None)
        if busy_worker is not None and busy_worker.is_alive():
            busy_worker.join(timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            if busy_worker.is_alive():
                self._handle_unresponsive_worker(busy_worker)
                return
            self._busy_worker = None
        if getattr(self, "_ctx", None) is not None:
            self._lib.zerotts_free(self._ctx)
            self._ctx = None
