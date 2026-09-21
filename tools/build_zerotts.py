"""Canonical build script for ZeroTTS GGML runtime shared library."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

try:
    from wyoming_vietnamese.const import (
        ZEROTTS_REPO_URL,
        ZEROTTS_SOURCE_COMMIT,
        ZeroTtsFile,
        ZeroTtsVariant,
    )
    from wyoming_vietnamese.cpu import (
        can_host_execute_variant,
        get_supported_variants_for_host,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from wyoming_vietnamese.const import (
        ZEROTTS_REPO_URL,
        ZEROTTS_SOURCE_COMMIT,
        ZeroTtsFile,
        ZeroTtsVariant,
    )
    from wyoming_vietnamese.cpu import (
        can_host_execute_variant,
        get_supported_variants_for_host,
    )

_can_host_execute_variant = can_host_execute_variant

_LOGGER = logging.getLogger(__name__)

_SAFE_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SAFE_REPO_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?$"
)
_GGML_LIB_GLOB: Final = "libggml*.so*"


def _validate_git_argument(value: str, pattern: re.Pattern[str], param_name: str) -> str:
    """Validate an argument passed to git commands to prevent argument injection."""
    clean = value.strip()
    if clean.startswith("-") or not pattern.match(clean):
        raise ValueError(f"Invalid or unsafe {param_name}: {value!r}")
    return clean


def _checkout_repo(clone_dest: Path, repo_url: str, commit: str) -> None:
    """Clone upstream ZeroTTS repository and checkout pinned commit.

    Args:
        clone_dest: Local directory destination for git clone.
        repo_url: Upstream repository URL.
        commit: Target Git commit hash to checkout.
    """
    safe_repo_url = _validate_git_argument(repo_url, _SAFE_REPO_URL_PATTERN, "repo_url")
    safe_commit = _validate_git_argument(commit, _SAFE_COMMIT_PATTERN, "commit")

    subprocess.run(
        [
            "git",
            "clone",
            "--recurse-submodules",
            "--",
            safe_repo_url,
            str(clone_dest),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone_dest),
            "checkout",
            "--detach",
            safe_commit,
            "--",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone_dest),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
        capture_output=True,
    )


_COMMON_DISABLED_CMAKE_FLAGS: Final = [
    "-DGGML_NATIVE=OFF",
    "-DGGML_NATIVE_DEFAULT=OFF",
    "-DGGML_AVX512_BF16=OFF",
    "-DGGML_AVX512_VBMI=OFF",
    "-DGGML_AVX512_VNNI=OFF",
    "-DGGML_AMX_TILE=OFF",
    "-DGGML_AMX_INT8=OFF",
    "-DGGML_AMX_BF16=OFF",
]
_ENABLE_SSE42: Final = "-DGGML_SSE42=ON"
_DISABLE_AVX2_FAMILY: Final = [
    "-DGGML_AVX2=OFF",
    "-DGGML_FMA=OFF",
    "-DGGML_F16C=OFF",
    "-DGGML_BMI2=OFF",
]
_DISABLE_AVX512_FAMILY: Final = [
    "-DGGML_AVX_VNNI=OFF",
    "-DGGML_AVX512=OFF",
]
_ENABLE_AVX2_FAMILY: Final = [
    "-DGGML_AVX=ON",
    "-DGGML_AVX2=ON",
    "-DGGML_FMA=ON",
    "-DGGML_F16C=ON",
    "-DGGML_BMI2=ON",
]


def _get_variant_build_flags(variant: str) -> tuple[list[str], list[str]]:
    """Return CMake definition flags and extra compiler flags for a build variant.

    Args:
        variant: Variant identifier ('compat', 'sse4', 'avx', 'avx2', 'avx512',
            'arm_dotprod', or 'native').

    Returns:
        Tuple containing (cmake_definitions, gxx_extra_flags).
    """
    if variant == ZeroTtsVariant.NATIVE:
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64"):
            return (["-DGGML_NATIVE=ON"], ["-mcpu=native"])
        return (["-DGGML_NATIVE=ON"], ["-march=native"])

    if variant == ZeroTtsVariant.ARM_DOTPROD:
        cmake_flags = [
            *_COMMON_DISABLED_CMAKE_FLAGS,
            "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod",
        ]
        gxx_flags = ["-march=armv8.2-a+dotprod"]
        return (cmake_flags, gxx_flags)

    if variant == ZeroTtsVariant.AVX512:
        cmake_flags = [
            *_COMMON_DISABLED_CMAKE_FLAGS,
            _ENABLE_SSE42,
            *_ENABLE_AVX2_FAMILY,
            "-DGGML_AVX512=ON",
        ]
        gxx_flags = [
            "-mavx512f",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-mbmi2",
        ]
        return (cmake_flags, gxx_flags)

    if variant == ZeroTtsVariant.AVX2:
        cmake_flags = [
            *_COMMON_DISABLED_CMAKE_FLAGS,
            _ENABLE_SSE42,
            *_ENABLE_AVX2_FAMILY,
            *_DISABLE_AVX512_FAMILY,
        ]
        gxx_flags = ["-mavx2", "-mfma", "-mf16c", "-mbmi2"]
        return (cmake_flags, gxx_flags)

    if variant == ZeroTtsVariant.AVX:
        cmake_flags = [
            *_COMMON_DISABLED_CMAKE_FLAGS,
            _ENABLE_SSE42,
            "-DGGML_AVX=ON",
            *_DISABLE_AVX2_FAMILY,
            *_DISABLE_AVX512_FAMILY,
        ]
        gxx_flags = ["-mavx"]
        return (cmake_flags, gxx_flags)

    if variant == ZeroTtsVariant.SSE4:
        cmake_flags = [
            *_COMMON_DISABLED_CMAKE_FLAGS,
            _ENABLE_SSE42,
            "-DGGML_AVX=OFF",
            *_DISABLE_AVX2_FAMILY,
            *_DISABLE_AVX512_FAMILY,
        ]
        gxx_flags = ["-msse4.2", "-mpopcnt"]
        return (cmake_flags, gxx_flags)

    # COMPAT baseline
    cmake_flags = [
        *_COMMON_DISABLED_CMAKE_FLAGS,
        "-DGGML_SSE42=OFF",
        "-DGGML_AVX=OFF",
        *_DISABLE_AVX2_FAMILY,
        *_DISABLE_AVX512_FAMILY,
    ]
    return (cmake_flags, [])


def _is_toolchain_variant_supported(variant: str) -> bool:
    """Check if the compiler toolchain supports the compiler flags for a variant.

    Args:
        variant: Optimization variant identifier ('compat', 'sse4', 'avx', 'avx2',
            'avx512', 'arm_dotprod', or 'native').

    Returns:
        True if the host toolchain can compile with the variant flags, False otherwise.
    """
    _, extra_flags = _get_variant_build_flags(variant)
    if not extra_flags:
        return True
    try:
        res = subprocess.run(
            ["g++", *extra_flags, "-E", "-x", "c++", os.devnull, "-o", os.devnull],
            capture_output=True,
            check=False,
        )
        return res.returncode == 0
    except OSError, subprocess.SubprocessError:
        return False


def _compile_libraries(
    clone_dest: Path,
    staging_dir: Path,
    *,
    variant: str = ZeroTtsVariant.COMPAT,
    native: bool = False,
) -> None:
    """Build GGML vendor dependencies and compile libzerotts.so into staging directory.

    Args:
        clone_dest: Directory containing checked-out ZeroTTS sources.
        staging_dir: Destination directory where output libraries will be placed.
        variant: CPU architecture optimization variant.
        native: Whether to optimize for host processor architecture (native=True)
            or target a portable instruction set baseline (native=False).
    """
    actual_variant = ZeroTtsVariant.NATIVE if native else variant
    cmake_build_dir = clone_dest / "cpp" / f"build_{actual_variant}"
    cmake_cmd = [
        "cmake",
        "-B",
        str(cmake_build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    cmake_defs, extra_gxx = _get_variant_build_flags(actual_variant)
    cmake_cmd.extend(cmake_defs)
    cmake_cmd.extend(
        [
            "-DCMAKE_BUILD_RPATH=$ORIGIN",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            str(clone_dest / "cpp"),
        ]
    )
    subprocess.run(
        cmake_cmd,
        check=True,
        capture_output=True,
    )
    jobs = str(min(os.cpu_count() or 4, 8))
    subprocess.run(
        ["cmake", "--build", str(cmake_build_dir), f"-j{jobs}"],
        check=True,
        capture_output=True,
    )

    gxx_cmd = [
        "g++",
        "-O3",
        "-fPIC",
        "-shared",
        *extra_gxx,
        f"-I{clone_dest / 'cpp' / 'include'}",
        f"-I{clone_dest / 'cpp' / 'vendor-ggml' / 'include'}",
        f"-I{clone_dest / 'cpp' / 'vendor-ggml' / 'src'}",
        f"-I{clone_dest / 'cpp' / 'vendor-ggml' / 'src' / 'ggml-cpu'}",
        str(clone_dest / "cpp" / "src" / "zerotts.cpp"),
        f"-L{cmake_build_dir / 'vendor-ggml' / 'src'}",
        "-lggml",
        "-lggml-base",
        "-lggml-cpu",
        "-fopenmp",
        "-Wl,-rpath,$ORIGIN",
        "-o",
        str(staging_dir / ZeroTtsFile.LIBZEROTTS_SO),
    ]
    subprocess.run(gxx_cmd, check=True, capture_output=True)

    _copy_ggml_libraries(cmake_build_dir / "vendor-ggml" / "src", staging_dir)


def _copy_single_library(ggml_so: Path, vendor_src_dir: Path, target_dir: Path) -> Path | None:
    """Copy or link a single GGML shared library into target_dir.

    Args:
        ggml_so: Candidate library file or symlink in vendor build directory.
        vendor_src_dir: Directory containing compiled GGML vendor libraries.
        target_dir: Destination directory where library is placed.

    Returns:
        The destination Path if copied or linked, or None if skipped.
    """
    try:
        resolved_so = ggml_so.resolve(strict=True)
    except OSError as err:
        _LOGGER.warning("Skipping unresolvable GGML candidate %s: %s", ggml_so, err)
        return None
    if not resolved_so.is_file():
        _LOGGER.warning("Skipping non-file GGML candidate %s", ggml_so)
        return None

    dest_file = target_dir / ggml_so.name
    if dest_file.is_symlink() or dest_file.exists():
        dest_file.unlink()

    if ggml_so.is_symlink():
        link_target = os.readlink(ggml_so)
        target_path = Path(link_target)
        if (
            not target_path.is_absolute()
            and target_path.parent == Path(".")
            and (vendor_src_dir / link_target).exists()
        ):
            dest_file.symlink_to(link_target)
            return dest_file

    dest_file.write_bytes(resolved_so.read_bytes())
    return dest_file


def _copy_ggml_libraries(vendor_src_dir: Path, target_dir: Path) -> list[Path]:
    """Copy GGML shared libraries to target directory without dangling builder symlinks.

    Args:
        vendor_src_dir: Directory containing compiled GGML vendor libraries.
        target_dir: Destination directory where libraries and valid relative links are placed.

    Returns:
        List of copied library destination paths.
    """
    copied: list[Path] = []
    for ggml_so in vendor_src_dir.glob(_GGML_LIB_GLOB):
        copied_path = _copy_single_library(ggml_so, vendor_src_dir, target_dir)
        if copied_path is not None:
            copied.append(copied_path)
    return copied


def _verify_library_loadable(lib_path: Path, variant: str = ZeroTtsVariant.COMPAT) -> None:
    """Verify that the built shared library and its dependencies can be loaded.

    Args:
        lib_path: Path to the compiled shared library.
        variant: Optimization variant of the compiled library.

    Raises:
        RuntimeError: If loading the shared library fails due to missing dependencies.
        FileNotFoundError: If the compiled library file does not exist.
    """
    if not lib_path.is_file():
        raise FileNotFoundError(f"Candidate library file not found: {lib_path}")

    if _can_host_execute_variant(variant):
        try:
            ctypes.CDLL(str(lib_path.resolve()))
        except OSError as err:
            raise RuntimeError(
                f"Built library {lib_path.name} failed runtime loadability check: {err}"
            ) from err
    else:
        _LOGGER.info(
            "Skipping runtime load check for %s (variant %s exceeds host CPU)",
            lib_path.name,
            variant,
        )


def _publish_single_file(src: Path, target_dir: Path) -> None:
    """Publish a single library file or symlink atomically into target_dir.

    Args:
        src: Staged source file or symlink to publish.
        target_dir: Destination directory.
    """
    dst = target_dir / src.name
    temp_dst = target_dir / f".{src.name}.tmp.{os.getpid()}"
    try:
        try:
            src.replace(dst)
            return
        except OSError as err:
            if err.errno != 18:
                raise

        if temp_dst.is_symlink() or temp_dst.exists():
            temp_dst.unlink()

        if src.is_symlink():
            link_target = os.readlink(src)
            temp_dst.symlink_to(link_target)
        else:
            shutil.copy2(src, temp_dst)

        temp_dst.replace(dst)
    finally:
        if temp_dst.is_symlink() or temp_dst.exists():
            temp_dst.unlink()


def _rollback_published(
    newly_published: list[Path],
    backups: list[tuple[Path, Path]],
) -> None:
    """Roll back published and backed-up library files after a publication error.

    Args:
        newly_published: Destination paths that did not exist before publishing.
        backups: Pairs of original destination and backup paths.
    """
    for new_file in newly_published:
        if new_file.is_symlink() or new_file.exists():
            new_file.unlink()
    for dst, backup_dst in reversed(backups):
        if backup_dst.is_symlink() or backup_dst.exists():
            backup_dst.replace(dst)


def _cleanup_backups(backups: list[tuple[Path, Path]]) -> None:
    """Remove temporary backup files created during library publication.

    Args:
        backups: Pairs of original destination and backup paths.
    """
    for _, backup_dst in backups:
        if backup_dst.is_symlink() or backup_dst.exists():
            backup_dst.unlink()


def _library_sort_key(p: Path) -> tuple[int, int, str]:
    """Sort order for staged libraries: vendor files, vendor symlinks, then libzerotts.so.

    Args:
        p: Path to candidate staged library file or symlink.

    Returns:
        Tuple representing priority sort order and file name.
    """
    is_zerotts = 1 if p.name == ZeroTtsFile.LIBZEROTTS_SO else 0
    is_link = 1 if p.is_symlink() else 0
    return (is_zerotts, is_link, p.name)


def _publish_libraries(staging_dir: Path, target_dir: Path) -> None:
    """Publish verified staged libraries to target_dir in dependency order.

    All existing library files are backed up during publishing. If publishing
    any file fails, already-published files are rolled back to preserve the
    previous library set atomically.

    Args:
        staging_dir: Staging directory containing verified libraries.
        target_dir: Destination directory where libraries are published.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    staged_entries = [p for p in staging_dir.iterdir() if p.is_file() or p.is_symlink()]
    sorted_entries = sorted(staged_entries, key=_library_sort_key)
    backups: list[tuple[Path, Path]] = []
    newly_published: list[Path] = []

    try:
        for entry in sorted_entries:
            dst = target_dir / entry.name
            if dst.is_symlink() or dst.exists():
                backup_dst = target_dir / f".{entry.name}.bak.{os.getpid()}"
                if backup_dst.is_symlink() or backup_dst.exists():
                    backup_dst.unlink()
                dst.replace(backup_dst)
                backups.append((dst, backup_dst))
            else:
                newly_published.append(dst)
            _publish_single_file(entry, target_dir)
    except Exception:
        _rollback_published(newly_published, backups)
        raise
    else:
        _cleanup_backups(backups)


def _format_process_error(err: subprocess.CalledProcessError) -> str:
    """Format a subprocess.CalledProcessError with captured stdout and stderr.

    Args:
        err: The CalledProcessError raised by subprocess.run.

    Returns:
        A human-readable string containing command, exit code, stdout, and stderr.
    """
    lines = [str(err)]
    if err.stdout:
        stdout_text = (
            err.stdout.decode("utf-8", errors="replace").strip()
            if isinstance(err.stdout, bytes)
            else str(err.stdout).strip()
        )
        if stdout_text:
            lines.append(f"Standard output:\n{stdout_text}")
    if err.stderr:
        stderr_text = (
            err.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(err.stderr, bytes)
            else str(err.stderr).strip()
        )
        if stderr_text:
            lines.append(f"Standard error:\n{stderr_text}")
    return "\n".join(lines)


def build_zerotts_ggml_lib(
    target_dir: Path,
    *,
    repo_url: str = ZEROTTS_REPO_URL,
    commit: str = ZEROTTS_SOURCE_COMMIT,
    variant: str = ZeroTtsVariant.COMPAT,
    native: bool = False,
    publish_subdir: bool = False,
) -> Path:
    """Build the ZeroTTS GGML shared library from upstream source pinned to a commit.

    Args:
        target_dir: Destination directory where the built shared library should be placed.
        repo_url: Git clone URL for the ZeroTTS repository.
        commit: Pinned Git commit SHA to checkout.
        variant: CPU architecture optimization variant ('compat', 'sse4', 'avx',
            'avx2', 'avx512', 'arm_dotprod', 'native').
        native: Whether to optimize for host processor architecture instead of portable baseline.
        publish_subdir: Whether to publish into a variant-named subdirectory under target_dir.

    Returns:
        Path to the compiled libzerotts.so file.

    Raises:
        RuntimeError: If cloning, checkout, or compilation fails.
        FileNotFoundError: If the compiled library cannot be found after build.
    """
    actual_variant = ZeroTtsVariant.NATIVE if native else variant
    if not _is_toolchain_variant_supported(actual_variant):
        raise RuntimeError(
            f"Compiler toolchain does not support required flags for variant {actual_variant}"
        )
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    destination_dir = target_dir / actual_variant if publish_subdir else target_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    build_dir = Path(tempfile.mkdtemp(prefix="_build_", dir=target_dir))
    clone_dest = build_dir / "ZeroTTS"
    staging_dir = build_dir / "stage"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        _LOGGER.info(
            "Building ZeroTTS GGML shared library (%s) into %s (commit %s)...",
            actual_variant,
            destination_dir,
            commit,
        )
        print(
            f"--> Building ZeroTTS GGML shared library ({actual_variant}) "
            f"into {destination_dir} (commit {commit})..."
        )
        _checkout_repo(clone_dest, repo_url, commit)
        _compile_libraries(clone_dest, staging_dir, variant=actual_variant, native=native)

        staged_candidate = staging_dir / ZeroTtsFile.LIBZEROTTS_SO
        if not staged_candidate.is_file():
            raise FileNotFoundError(
                f"{ZeroTtsFile.LIBZEROTTS_SO} not found after build at: {staged_candidate}"
            )
        _verify_library_loadable(staged_candidate, variant=actual_variant)
        _publish_libraries(staging_dir, destination_dir)
    except subprocess.CalledProcessError as err:
        details = _format_process_error(err)
        _LOGGER.error("Subprocess execution failed during ZeroTTS build:\n%s", details)
        raise RuntimeError(f"Failed to build ZeroTTS GGML shared library: {details}") from err
    except RuntimeError, FileNotFoundError:
        raise
    except Exception as err:
        raise RuntimeError(f"Failed to build ZeroTTS GGML shared library: {err}") from err
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    candidate = destination_dir / ZeroTtsFile.LIBZEROTTS_SO
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{ZeroTtsFile.LIBZEROTTS_SO} not found after build at: {candidate}"
        )
    return candidate


def _quarantine_stale_variants(
    target_dir: Path, unsupported_variants: list[str]
) -> list[tuple[Path, Path]]:
    """Quarantine existing variant directories that are no longer supported by the toolchain.

    Args:
        target_dir: Destination directory where variant subdirectories live.
        unsupported_variants: Variants unsupported by the current compiler toolchain.

    Returns:
        List of (original_path, quarantine_path) pairs for rollback or cleanup.
    """
    quarantined: list[tuple[Path, Path]] = []
    for var in unsupported_variants:
        stale_dir = target_dir / var
        if stale_dir.is_symlink() or stale_dir.exists():
            quarantine_path = target_dir / f".{var}.stale.{os.getpid()}"
            if quarantine_path.is_dir() and not quarantine_path.is_symlink():
                shutil.rmtree(quarantine_path, ignore_errors=True)
            elif quarantine_path.is_symlink() or quarantine_path.exists():
                quarantine_path.unlink()
            stale_dir.rename(quarantine_path)
            quarantined.append((stale_dir, quarantine_path))
            _LOGGER.info(
                "Quarantined stale variant directory %s to %s",
                stale_dir,
                quarantine_path,
            )
    return quarantined


def _restore_quarantined_variants(quarantined: list[tuple[Path, Path]]) -> None:
    """Restore quarantined variant directories after a build failure.

    Args:
        quarantined: Pairs of original and quarantined directory paths.
    """
    for orig_dir, q_dir in reversed(quarantined):
        if (q_dir.is_symlink() or q_dir.exists()) and not orig_dir.exists():
            with contextlib.suppress(OSError):
                q_dir.rename(orig_dir)


def _cleanup_quarantined_variants(quarantined: list[tuple[Path, Path]]) -> None:
    """Permanently remove quarantined variant directories after successful build.

    Args:
        quarantined: Pairs of original and quarantined directory paths.
    """
    for _, q_dir in quarantined:
        if q_dir.is_dir() and not q_dir.is_symlink():
            shutil.rmtree(q_dir, ignore_errors=True)
        elif q_dir.is_symlink() or q_dir.exists():
            with contextlib.suppress(OSError):
                q_dir.unlink()


def build_all_zerotts_variants(
    target_dir: Path,
    *,
    repo_url: str = ZEROTTS_REPO_URL,
    commit: str = ZEROTTS_SOURCE_COMMIT,
) -> dict[str, Path]:
    """Build all supported ZeroTTS GGML shared library variants for the host architecture.

    Clones the repository once into a shared temporary workspace and compiles
    each variant into its own subdirectory under target_dir (e.g. target_dir/compat,
    target_dir/avx2, target_dir/avx512). A default symlink is also created at
    target_dir/libzerotts.so pointing to the compat variant.

    Args:
        target_dir: Base directory where variant subdirectories will be placed.
        repo_url: Git repository clone URL.
        commit: Pinned commit SHA to checkout.

    Returns:
        Dictionary mapping variant names to their compiled libzerotts.so paths.
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    variants = get_supported_variants_for_host()
    supported_variants = [v for v in variants if _is_toolchain_variant_supported(v)]
    unsupported_variants = [v for v in variants if v not in supported_variants]
    if unsupported_variants:
        _LOGGER.warning(
            "Compiler toolchain lacks flag support for variants: %s; skipping them",
            ", ".join(unsupported_variants),
        )
    if not supported_variants:
        raise RuntimeError(
            f"Compiler toolchain does not support required flags for any variant in {variants}"
        )

    quarantined = _quarantine_stale_variants(target_dir, unsupported_variants)
    build_dir = Path(tempfile.mkdtemp(prefix="_build_all_", dir=target_dir))
    clone_dest = build_dir / "ZeroTTS"
    results: dict[str, Path] = {}

    try:
        _LOGGER.info(
            "Building ZeroTTS GGML variants (%s) into %s...",
            ", ".join(supported_variants),
            target_dir,
        )
        print(f"--> Cloning ZeroTTS once for all variants ({commit})...")
        _checkout_repo(clone_dest, repo_url, commit)

        for var in supported_variants:
            variant_dest_dir = target_dir / var
            variant_dest_dir.mkdir(parents=True, exist_ok=True)
            variant_stage_dir = build_dir / f"stage_{var}"
            variant_stage_dir.mkdir(parents=True, exist_ok=True)

            print(f"--> Compiling ZeroTTS variant: {var}...")
            _compile_libraries(clone_dest, variant_stage_dir, variant=var)
            staged_candidate = variant_stage_dir / ZeroTtsFile.LIBZEROTTS_SO
            if not staged_candidate.is_file():
                raise FileNotFoundError(
                    f"{ZeroTtsFile.LIBZEROTTS_SO} not found after build for {var}"
                )
            _verify_library_loadable(staged_candidate, variant=var)
            _publish_libraries(variant_stage_dir, variant_dest_dir)
            results[var] = variant_dest_dir / ZeroTtsFile.LIBZEROTTS_SO

        if ZeroTtsVariant.COMPAT in results:
            compat_link = target_dir / ZeroTtsFile.LIBZEROTTS_SO
            if compat_link.is_symlink() or compat_link.exists():
                compat_link.unlink()
            compat_link.symlink_to(Path(ZeroTtsVariant.COMPAT) / ZeroTtsFile.LIBZEROTTS_SO)
        _cleanup_quarantined_variants(quarantined)
    except subprocess.CalledProcessError as err:
        _restore_quarantined_variants(quarantined)
        details = _format_process_error(err)
        _LOGGER.error("Subprocess execution failed during multi-variant build:\n%s", details)
        raise RuntimeError(f"Failed to build ZeroTTS GGML shared libraries: {details}") from err
    except RuntimeError, FileNotFoundError:
        _restore_quarantined_variants(quarantined)
        raise
    except Exception as err:
        _restore_quarantined_variants(quarantined)
        raise RuntimeError(f"Failed to build ZeroTTS GGML shared libraries: {err}") from err
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    return results


def main() -> int:
    """CLI entry point for building ZeroTTS GGML shared library."""
    parser = argparse.ArgumentParser(
        description="Build ZeroTTS GGML native shared library from pinned upstream source."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=os.environ.get("ZEROTTS_LIB_DIR", "/app/lib"),
        help="Directory where libzerotts.so and dependencies will be placed (default: /app/lib)",
    )
    parser.add_argument(
        "--commit",
        default=ZEROTTS_SOURCE_COMMIT,
        help=f"Git commit hash to check out (default: {ZEROTTS_SOURCE_COMMIT})",
    )
    parser.add_argument(
        "--repo-url",
        default=ZEROTTS_REPO_URL,
        help=f"Git repository URL (default: {ZEROTTS_REPO_URL})",
    )
    parser.add_argument(
        "--variant",
        choices=[
            ZeroTtsVariant.COMPAT,
            ZeroTtsVariant.SSE4,
            ZeroTtsVariant.AVX,
            ZeroTtsVariant.AVX2,
            ZeroTtsVariant.AVX512,
            ZeroTtsVariant.ARM_DOTPROD,
            ZeroTtsVariant.NATIVE,
        ],
        default=None,
        help="Optimization variant to build (default: compat)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        default=False,
        help="Build all supported architecture variants into subdirectories",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        default=False,
        help="Optimize binary for host processor architecture instead of portable baseline",
    )
    args = parser.parse_args()

    if args.all_variants and (args.variant is not None or args.native):
        parser.error("Cannot combine --all-variants with --variant or --native")
    if args.native and args.variant is not None:
        parser.error("Cannot combine --native with --variant")
    if args.variant is not None and args.variant != ZeroTtsVariant.NATIVE:
        supported = get_supported_variants_for_host()
        if args.variant not in supported:
            machine = platform.machine().lower()
            parser.error(
                f"Variant '{args.variant}' is not supported on host architecture '{machine}' "
                f"(supported: {', '.join(supported)})"
            )

    target_path = Path(args.target_dir).resolve()
    try:
        if args.all_variants:
            results = build_all_zerotts_variants(
                target_path,
                repo_url=args.repo_url,
                commit=args.commit,
            )
            print(f"Successfully built ZeroTTS GGML library variants: {list(results.keys())}")
        else:
            variant = args.variant or ZeroTtsVariant.COMPAT
            output_so = build_zerotts_ggml_lib(
                target_path,
                repo_url=args.repo_url,
                commit=args.commit,
                variant=variant,
                native=args.native,
            )
            print(f"Successfully built ZeroTTS GGML library at: {output_so}")
        return 0
    except Exception as exc:
        print(f"Error building ZeroTTS GGML library: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
