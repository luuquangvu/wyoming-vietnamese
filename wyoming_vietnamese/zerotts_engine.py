"""ZeroTTS GGML runtime engine implementation for high-quality Vietnamese speech synthesis."""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from zerotts.codec import MossStreamingDecoder

from .const import (
    ZEROTTS_REPO_URL,
    ZEROTTS_SOURCE_COMMIT,
    ZeroTtsAudio,
    ZeroTtsFile,
    ZeroTtsVariant,
)
from .cpu import detect_cpu_variant
from .tts_model import ZEROTTS_VOICES_BY_ID, ZeroTtsVoiceSpec

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
    return lib


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
    ) -> None:
        """Initialize the ZeroTTS GGML engine from local weights and configurations.

        Args:
            model_dir: Base directory containing ZeroTTS model assets (tokenizer, voices, codec).
            gguf_path: Path to the quantized GGUF model weights file.
            voices: Sequence of configured voice specifications to expose.
            num_threads: CPU thread count for inference.
            custom_lib_path: Optional explicit path to libzerotts.so.
            seed: Default random generator seed for deterministic frame sampling.

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
        self.threads: int = num_threads
        self.model_dir: Path = model_dir
        self.gguf_path: Path = gguf_path
        self._ctx: ctypes.c_void_p | None = None

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

            self._sp = _ZeroTtsSampling()
            self._lib.zerotts_sampling_defaults(ctypes.byref(self._sp))

            config_path = model_dir / ZeroTtsFile.CONFIG
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing ZeroTTS config.json at: {config_path}")
            config_data = json.loads(config_path.read_text(encoding="utf-8"))

            from zerotts.codec import MossCodecDecoder
            from zerotts.text_norm import normalize_vi_text
            from zerotts.tokenizer import load_tokenizer

            self._tokenizer = load_tokenizer(config_data, model_dir)
            self._normalize_fn = normalize_vi_text

            codec_dir = model_dir / "onnx" / "codec"
            self._codec = MossCodecDecoder(
                codec_dir,
                intra_op_num_threads=self.threads,
            )

            null_voice_path = model_dir / ZeroTtsFile.NULL_VOICE_EMB
            if null_voice_path.is_file():
                self._null_voice_emb = np.ascontiguousarray(
                    np.load(null_voice_path).reshape(-1), dtype=np.float32
                )
            else:
                flat_dim = int(self._hp.n_voice_queries) * int(self._hp.d_model)
                self._null_voice_emb = np.zeros(flat_dim, dtype=np.float32)

            self._voice_embeddings: dict[str, np.ndarray] = {}
            self._voices_by_id: dict[str, ZeroTtsVoiceSpec] = {voice.id: voice for voice in voices}
            self._voices_by_name: dict[str, ZeroTtsVoiceSpec] = {
                voice.name: voice for voice in voices
            }
            self._default_voice = voices[0]

            self._preset_voices: Mapping[str, Mapping[str, object]] = {
                voice.name: {"description": voice.description} for voice in voices
            }

            for voice in voices:
                self._load_voice_embedding(voice.id)

            self._seed: int = seed
            self._rng = np.random.default_rng(self._seed)
        except Exception:
            self.close()
            raise

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
        stream: MossStreamingDecoder, buf: list[np.ndarray]
    ) -> np.ndarray | None:
        """Decode a sequence of acoustic code frames into an audio waveform array.

        Args:
            stream: Active MOSS streaming codec decoder instance.
            buf: List of code frames to decode.

        Returns:
            Decoded 1D float32 audio array if audio was generated, or None.
        """
        chunk_audio = stream.decode_chunk(np.stack(buf, axis=-1))
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
        au = self._rng.random(self.num_codebooks, dtype=np.float32)
        au_ptr = au.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rc = self._lib.zerotts_frame(
            self._ctx,
            forbid_eoa,
            ctypes.byref(self._sp),
            cu,
            au_ptr,
            codes_buf,
            ctypes.byref(is_eoa),
        )
        if rc != 0:
            raise RuntimeError(f"zerotts_frame failed at step {t}")

    def _generate_code_frames(self) -> Iterator[np.ndarray]:
        """Generate acoustic code frames sequentially from the initialized GGML context.

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
            forbid_eoa = (t < min_frames) or (tail_left >= 0)
            self._execute_frame_step(t, forbid_eoa, codes_buf, is_eoa)

            eoa_started = False
            if tail_left < 0 and is_eoa.value != 0:
                tail_left = eoa_extra
                eoa_started = True

            yield np.array(codes_buf, dtype=np.int64)

            if tail_left >= 0 and not eoa_started:
                tail_left -= 1
                if tail_left <= 0:
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

    def infer_stream_normalized(
        self,
        text: str,
        *,
        voice: str | None = None,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio for text that has already passed through the configured normalizer."""
        if self._ctx is None:
            raise RuntimeError("ZeroTTS GGML context is closed")

        self._rng = np.random.default_rng(self._seed if seed is None else seed)

        text_tokens = self._tokenizer(text)
        text_ids = np.ascontiguousarray(text_tokens, dtype=np.int32)
        if len(text_ids) == 0:
            return

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

        first_chunk_target = 2
        cap_chunk_target = 16

        stream = self._codec.streaming_decoder()
        buf: list[np.ndarray] = []
        target_chunk = first_chunk_target

        try:
            for frame in self._generate_code_frames():
                buf.append(frame)
                if len(buf) >= target_chunk:
                    audio_out = self._decode_audio_chunk(stream, buf)
                    if audio_out is not None:
                        yield audio_out
                    buf.clear()
                    target_chunk = min(cap_chunk_target, target_chunk * 2)

            if buf:
                audio_out = self._decode_audio_chunk(stream, buf)
                if audio_out is not None:
                    yield audio_out
        finally:
            stream.close()

    def normalize(self, text: str, *, voice: str | None = None) -> str:
        """Return text after the normalizer used by inference."""
        del voice
        return text if self._normalize_fn is None else self._normalize_fn(text)

    def close(self) -> None:
        """Release native model resources and tensor buffers."""
        if getattr(self, "_ctx", None) is not None:
            self._lib.zerotts_free(self._ctx)
            self._ctx = None
