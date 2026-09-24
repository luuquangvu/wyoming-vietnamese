"""Unit and integration tests for ZeroTTS GGML runtime engine."""

from __future__ import annotations

import ctypes
import re
import shutil
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from wyoming_vietnamese.const import (
    ZEROTTS_SOURCE_COMMIT,
    TtsEngine,
    ZeroTtsAudio,
    ZeroTtsFile,
    ZeroTtsVariant,
)
from wyoming_vietnamese.cpu import (
    _AVX2_CPU_FLAGS,
    _AVX512_CPU_FLAGS,
    _AVX_CPU_FLAGS,
    _SSE4_CPU_FLAGS,
    can_host_execute_variant,
    detect_cpu_variant,
    get_host_cpu_flags,
    get_supported_variants_for_host,
)
from wyoming_vietnamese.tts import warm_up_tts
from wyoming_vietnamese.tts_model import ZEROTTS_VOICES
from wyoming_vietnamese.zerotts_engine import (
    _ZEROTTS_RUNTIME_DEPENDENCIES,
    ZeroTtsGgmlEngine,
    _build_zerotts_ggml_lib,
    _load_zerotts_candidate,
    _unload_cdll,
    _ZeroTtsHParams,
    get_zerotts_candidate_lib_paths,
    resolve_zerotts_ggml_lib,
)


@pytest.mark.parametrize("native", [False, True])
def test_build_zerotts_ggml_lib_success(tmp_path: Path, native: bool) -> None:
    """Test _build_zerotts_ggml_lib executes build commands and locates output."""
    target_dir = tmp_path / "lib"
    executed_commands: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        executed_commands.append(cmd)
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"ELF_FAKE")
        return Mock(returncode=0)

    mock_cmake_proc = Mock()
    mock_cmake_proc.communicate.return_value = (b"", b"")
    mock_cmake_proc.returncode = 0
    mock_cmake_proc.pid = 12345

    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc) as mock_popen,
        patch("tools.build_zerotts.ctypes.CDLL") as mock_cdll,
    ):
        result = _build_zerotts_ggml_lib(target_dir, native=native)
        mock_popen.assert_called_once()
        popen_args, popen_kwargs = mock_popen.call_args
        assert popen_args[0][:2] == ["cmake", "--build"]
        assert popen_kwargs.get("start_new_session") is True
        assert result.is_file()
        assert result.name == ZeroTtsFile.LIBZEROTTS_SO
        assert result.parent == target_dir
        assert any("checkout" in cmd and ZEROTTS_SOURCE_COMMIT in cmd for cmd in executed_commands)
        cmake_configure_cmd = next(
            cmd for cmd in executed_commands if "cmake" in cmd[0] and "-B" in cmd
        )
        if native:
            assert "-DGGML_NATIVE=ON" in cmake_configure_cmd
            assert "-DGGML_NATIVE=OFF" not in cmake_configure_cmd
        else:
            assert "-DGGML_NATIVE=OFF" in cmake_configure_cmd
            assert "-DGGML_NATIVE_DEFAULT=OFF" in cmake_configure_cmd
            assert "-DGGML_AVX2=OFF" in cmake_configure_cmd
            assert "-DGGML_AVX512=OFF" in cmake_configure_cmd
        assert "-DCMAKE_BUILD_RPATH=$ORIGIN" in cmake_configure_cmd
        assert "-DCMAKE_INSTALL_RPATH=$ORIGIN" in cmake_configure_cmd
        mock_cdll.assert_called_once()
        verified_path = Path(mock_cdll.call_args[0][0])
        assert verified_path.name == ZeroTtsFile.LIBZEROTTS_SO
        assert "stage" in verified_path.parts


def test_build_zerotts_ggml_lib_import_error_calls_script(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib falls back to subprocess when module import fails."""
    target_dir = tmp_path / "lib"
    fake_lib = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    with (
        patch.dict("sys.modules", {"tools.build_zerotts": None}),
        patch("wyoming_vietnamese.zerotts_engine.Path.is_file", return_value=True),
        patch(
            "wyoming_vietnamese.zerotts_engine._run_zerotts_build_script",
            return_value=fake_lib,
        ) as mock_run_script,
    ):
        result = _build_zerotts_ggml_lib(target_dir, variant=ZeroTtsVariant.AVX2)
        assert result == fake_lib
        mock_run_script.assert_called_once()


def test_build_zerotts_ggml_lib_publish_subdir(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib forwards publish_subdir to build_zerotts_ggml_lib."""
    target_dir = tmp_path / "lib"
    fake_lib = target_dir / ZeroTtsVariant.AVX2 / ZeroTtsFile.LIBZEROTTS_SO
    with patch(
        "tools.build_zerotts.build_zerotts_ggml_lib",
        return_value=fake_lib,
    ) as mock_build:
        result = _build_zerotts_ggml_lib(
            target_dir, variant=ZeroTtsVariant.AVX2, publish_subdir=True
        )
        assert result == fake_lib
        assert mock_build.call_args[1]["publish_subdir"] is True


def test_build_zerotts_ggml_lib_import_error_missing_script(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib raises RuntimeError if fallback build script is missing."""
    target_dir = tmp_path / "lib"
    with (
        patch.dict("sys.modules", {"tools.build_zerotts": None}),
        patch("wyoming_vietnamese.zerotts_engine.Path.is_file", return_value=False),
        pytest.raises(RuntimeError, match=r"tools/build_zerotts\.py not found"),
    ):
        _build_zerotts_ggml_lib(target_dir)


def test_build_zerotts_ggml_lib_loadability_failure(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib raises RuntimeError when runtime loadability check fails."""
    target_dir = tmp_path / "lib"

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"ELF_FAKE")
        return Mock(returncode=0)

    mock_cmake_proc = Mock(communicate=Mock(return_value=(b"", b"")), returncode=0, pid=12345)
    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc),
        patch(
            "tools.build_zerotts.ctypes.CDLL",
            side_effect=OSError("libggml-cpu.so.0: cannot open shared object file"),
        ),
        pytest.raises(RuntimeError, match="failed runtime loadability check"),
    ):
        _build_zerotts_ggml_lib(target_dir)

    assert not (target_dir / ZeroTtsFile.LIBZEROTTS_SO).exists()


def test_build_zerotts_ggml_lib_preserves_existing_artifacts_on_loadability_failure(
    tmp_path: Path,
) -> None:
    """Test existing valid target_dir artifacts are preserved when loadability check fails."""
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_lib = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    existing_lib.write_bytes(b"OLD_VALID_LIB")
    existing_ggml = target_dir / "libggml-cpu.so.0.1.0"
    existing_ggml.write_bytes(b"OLD_VALID_GGML")

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"NEW_BROKEN_LIB")
        return Mock(returncode=0)

    mock_cmake_proc = Mock(communicate=Mock(return_value=(b"", b"")), returncode=0, pid=12345)
    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc),
        patch(
            "tools.build_zerotts.ctypes.CDLL",
            side_effect=OSError("symbol not found"),
        ),
        pytest.raises(RuntimeError, match="failed runtime loadability check"),
    ):
        _build_zerotts_ggml_lib(target_dir)

    assert existing_lib.read_bytes() == b"OLD_VALID_LIB"
    assert existing_ggml.read_bytes() == b"OLD_VALID_GGML"


def test_build_zerotts_ggml_lib_preserves_existing_artifacts_on_compile_failure(
    tmp_path: Path,
) -> None:
    """Test existing valid target_dir artifacts are preserved when compilation fails."""
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_lib = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    existing_lib.write_bytes(b"OLD_VALID_LIB")

    with (
        patch(
            "tools.build_zerotts.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["g++"]),
        ),
        pytest.raises(RuntimeError, match="Failed to build ZeroTTS GGML shared library"),
    ):
        _build_zerotts_ggml_lib(target_dir)

    assert existing_lib.read_bytes() == b"OLD_VALID_LIB"


def test_resolve_zerotts_ggml_lib_cannot_select_failed_build_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib does not find partially built output after build failure."""
    target_dir = tmp_path / "lib"
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"NEW_BROKEN_LIB")
        return Mock(returncode=0)

    mock_cmake_proc = Mock(communicate=Mock(return_value=(b"", b"")), returncode=0, pid=12345)
    with (
        patch(
            "wyoming_vietnamese.zerotts_engine._build_zerotts_ggml_lib",
            side_effect=lambda *_args, **_kwargs: _build_zerotts_ggml_lib(target_dir),
        ),
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc),
        patch(
            "tools.build_zerotts.ctypes.CDLL",
            side_effect=OSError("symbol lookup error"),
        ),
        pytest.raises(RuntimeError, match="failed runtime loadability check"),
    ):
        resolve_zerotts_ggml_lib()

    assert not (target_dir / ZeroTtsFile.LIBZEROTTS_SO).exists()


def test_publish_libraries_dependency_order(tmp_path: Path) -> None:
    """Test _publish_libraries publishes dependencies and relative symlinks cleanly."""
    from tools.build_zerotts import _publish_libraries

    staging_dir = tmp_path / "stage"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "target"

    real_ggml = staging_dir / "libggml-cpu.so.0.1.0"
    real_ggml.write_bytes(b"GGML_BYTES")

    link_ggml = staging_dir / "libggml-cpu.so.0"
    link_ggml.symlink_to("libggml-cpu.so.0.1.0")

    zerotts_lib = staging_dir / ZeroTtsFile.LIBZEROTTS_SO
    zerotts_lib.write_bytes(b"ZEROTTS_BYTES")

    _publish_libraries(staging_dir, target_dir)

    pub_real = target_dir / "libggml-cpu.so.0.1.0"
    pub_link = target_dir / "libggml-cpu.so.0"
    pub_zerotts = target_dir / ZeroTtsFile.LIBZEROTTS_SO

    assert pub_real.is_file()
    assert pub_real.read_bytes() == b"GGML_BYTES"
    assert pub_link.is_symlink()
    assert pub_link.read_bytes() == b"GGML_BYTES"
    assert pub_zerotts.is_file()
    assert pub_zerotts.read_bytes() == b"ZEROTTS_BYTES"


def test_publish_libraries_rollback_on_failure(tmp_path: Path) -> None:
    """Test _publish_libraries restores old files and removes newly published files on failure."""
    from tools.build_zerotts import _publish_libraries

    staging_dir = tmp_path / "stage"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Pre-existing file in target_dir that should be restored
    existing_file = target_dir / "libggml-cpu.so.0.1.0"
    existing_file.write_bytes(b"OLD_GGML_BYTES")

    # Staging files
    staged_ggml = staging_dir / "libggml-cpu.so.0.1.0"
    staged_ggml.write_bytes(b"NEW_GGML_BYTES")

    staged_zerotts = staging_dir / ZeroTtsFile.LIBZEROTTS_SO
    staged_zerotts.write_bytes(b"NEW_ZEROTTS_BYTES")

    from tools.build_zerotts import _publish_single_file as real_publish

    def fake_publish_single_file(src: Path, dst_dir: Path) -> None:
        if src.name == ZeroTtsFile.LIBZEROTTS_SO:
            raise OSError("Simulated disk error during libzerotts publication")
        real_publish(src, dst_dir)

    with (
        patch("tools.build_zerotts._publish_single_file", side_effect=fake_publish_single_file),
        pytest.raises(OSError, match="Simulated disk error"),
    ):
        _publish_libraries(staging_dir, target_dir)

    assert existing_file.is_file()
    assert existing_file.read_bytes() == b"OLD_GGML_BYTES"
    published_zerotts = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    assert not published_zerotts.exists()


def test_publish_libraries_rollback_on_system_exit(tmp_path: Path) -> None:
    """Test _publish_libraries restores old files and cleans published files on SystemExit."""
    from tools.build_zerotts import _publish_libraries

    staging_dir = tmp_path / "stage"
    staging_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_file = target_dir / "libggml-cpu.so.0.1.0"
    existing_file.write_bytes(b"OLD_GGML_BYTES")

    staged_ggml = staging_dir / "libggml-cpu.so.0.1.0"
    staged_ggml.write_bytes(b"NEW_GGML_BYTES")

    staged_zerotts = staging_dir / ZeroTtsFile.LIBZEROTTS_SO
    staged_zerotts.write_bytes(b"NEW_ZEROTTS_BYTES")

    from tools.build_zerotts import _publish_single_file as real_publish

    def fake_publish_single_file(src: Path, dst_dir: Path) -> None:
        if src.name == ZeroTtsFile.LIBZEROTTS_SO:
            raise SystemExit(143)
        real_publish(src, dst_dir)

    with (
        patch("tools.build_zerotts._publish_single_file", side_effect=fake_publish_single_file),
        pytest.raises(SystemExit) as exc_info,
    ):
        _publish_libraries(staging_dir, target_dir)

    assert exc_info.value.code == 143
    assert existing_file.is_file()
    assert existing_file.read_bytes() == b"OLD_GGML_BYTES"
    published_zerotts = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    assert not published_zerotts.exists()


def test_publish_single_file_handles_exdev_fallback(tmp_path: Path) -> None:
    """Test _publish_single_file falls back to copying when EXDEV error is encountered."""
    from tools.build_zerotts import _publish_single_file

    staged_file = tmp_path / "stage" / "libsample.so"
    staged_file.parent.mkdir(parents=True, exist_ok=True)
    staged_file.write_bytes(b"SAMPLE_BYTES")

    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    orig_replace = Path.replace

    def fake_replace(self: Path, target: Path | str) -> Path:
        if self == staged_file:
            err = OSError("Invalid cross-device link")
            err.errno = 18
            raise err
        return orig_replace(self, target)

    with patch.object(Path, "replace", fake_replace):
        _publish_single_file(staged_file, target_dir)

    published = target_dir / "libsample.so"
    assert published.is_file()
    assert published.read_bytes() == b"SAMPLE_BYTES"


def test_build_zerotts_ggml_lib_failure(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib raises RuntimeError when subprocess fails."""
    target_dir = tmp_path / "lib"
    with (
        patch(
            "tools.build_zerotts.subprocess.run",
            side_effect=Exception("compiler not found"),
        ),
        pytest.raises(RuntimeError, match="Failed to build ZeroTTS GGML shared library"),
    ):
        _build_zerotts_ggml_lib(target_dir)


def test_build_zerotts_ggml_lib_called_process_error_includes_output(
    tmp_path: Path,
) -> None:
    """Test _build_zerotts_ggml_lib includes stderr and stdout when CalledProcessError occurs."""
    target_dir = tmp_path / "lib"
    cpe = subprocess.CalledProcessError(
        returncode=2,
        cmd=["g++", "-shared"],
        output=b"compiling zerotts.cpp\n",
        stderr=b"fatal error: ggml.h missing\n",
    )
    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=cpe),
        pytest.raises(RuntimeError) as exc_info,
    ):
        _build_zerotts_ggml_lib(target_dir)

    err_str = str(exc_info.value)
    assert "Failed to build ZeroTTS GGML shared library" in err_str
    assert "compiling zerotts.cpp" in err_str
    assert "fatal error: ggml.h missing" in err_str


def test_format_process_error_variants() -> None:
    """Test _format_process_error formats string and bytes outputs correctly."""
    from tools.build_zerotts import _format_process_error

    err_str = subprocess.CalledProcessError(
        returncode=1,
        cmd="git clone",
        output="cloning output",
        stderr="network timeout",
    )
    formatted = _format_process_error(err_str)
    assert "cloning output" in formatted
    assert "network timeout" in formatted

    err_empty = subprocess.CalledProcessError(
        returncode=1,
        cmd="make",
        output=None,
        stderr=b"",
    )
    formatted_empty = _format_process_error(err_empty)
    assert "Standard output" not in formatted_empty
    assert "Standard error" not in formatted_empty


def test_build_zerotts_ggml_lib_missing_output(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib raises FileNotFoundError when binary is missing."""
    target_dir = tmp_path / "lib"
    mock_cmake_proc = Mock(communicate=Mock(return_value=(b"", b"")), returncode=0, pid=12345)
    with (
        patch("tools.build_zerotts.subprocess.run", return_value=Mock(returncode=0)),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc),
        pytest.raises(FileNotFoundError, match="not found after build"),
    ):
        _build_zerotts_ggml_lib(target_dir)


@pytest.mark.parametrize(
    ("extra_args", "expected_native"),
    [([], False), (["--native"], True)],
)
def test_build_zerotts_cli_main_success(
    tmp_path: Path,
    extra_args: list[str],
    expected_native: bool,
) -> None:
    """Test tools.build_zerotts.main parses arguments and returns 0 on success."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    fake_lib = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir), *extra_args]),
        patch("tools.build_zerotts.build_zerotts_ggml_lib", return_value=fake_lib) as mock_build,
    ):
        exit_code = build_zerotts_main()
        assert exit_code == 0
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("native") is expected_native


def test_build_zerotts_cli_main_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test tools.build_zerotts.main handles build failure and returns 1."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir)]),
        patch(
            "tools.build_zerotts.build_zerotts_ggml_lib",
            side_effect=RuntimeError("build failed"),
        ),
    ):
        exit_code = build_zerotts_main()
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error building ZeroTTS GGML library: build failed" in captured.err


def test_resolve_zerotts_ggml_lib_custom_path(tmp_path: Path) -> None:
    """Test resolve_zerotts_ggml_lib uses custom_lib_path if provided."""
    custom_so = tmp_path / "custom_libzerotts.so"
    custom_so.write_bytes(b"ELF")

    fake_cdll = MagicMock()
    with patch(
        "wyoming_vietnamese.zerotts_engine.ctypes.CDLL", return_value=fake_cdll
    ) as mock_cdll:
        lib = resolve_zerotts_ggml_lib(custom_lib_path=custom_so)
        assert lib is fake_cdll
        mock_cdll.assert_called_once_with(str(custom_so.resolve()))


def test_resolve_zerotts_ggml_lib_env_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_zerotts_ggml_lib uses ZEROTTS_LIB_PATH environment variable."""
    env_so = tmp_path / "env_libzerotts.so"
    env_so.write_bytes(b"ELF")
    monkeypatch.setenv("ZEROTTS_LIB_PATH", str(env_so))

    fake_cdll = MagicMock()
    with patch("wyoming_vietnamese.zerotts_engine.ctypes.CDLL", return_value=fake_cdll):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_cdll


def test_resolve_zerotts_ggml_lib_load_failure(tmp_path: Path) -> None:
    """Test resolve_zerotts_ggml_lib raises RuntimeError when ctypes.CDLL fails."""
    custom_so = tmp_path / "broken.so"
    custom_so.write_bytes(b"ELF")
    fake_built = tmp_path / ZeroTtsFile.LIBZEROTTS_SO
    fake_built.write_bytes(b"ELF")

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine._build_zerotts_ggml_lib",
            return_value=fake_built,
        ),
        patch("wyoming_vietnamese.zerotts_engine.ctypes.CDLL", side_effect=OSError("invalid ELF")),
        pytest.raises(RuntimeError, match=r"Failed to load libzerotts\.so"),
    ):
        resolve_zerotts_ggml_lib(custom_lib_path=custom_so)


def test_zerotts_engine_validation_errors(tmp_path: Path) -> None:
    """Test ZeroTtsGgmlEngine constructor enforces required inputs."""
    gguf_path = tmp_path / ZeroTtsFile.DEFAULT_GGUF_MODEL
    voice = ZEROTTS_VOICES[0]

    with pytest.raises(ValueError, match="At least one ZeroTTS voice must be configured"):
        ZeroTtsGgmlEngine(
            model_dir=tmp_path,
            gguf_path=gguf_path,
            voices=(),
        )

    with pytest.raises(FileNotFoundError, match="Missing ZeroTTS GGUF model weights"):
        ZeroTtsGgmlEngine(
            model_dir=tmp_path,
            gguf_path=gguf_path,
            voices=(voice,),
        )


def test_zerotts_engine_missing_codec_meta(tmp_path: Path) -> None:
    """Test that MossCodecDecoder raises FileNotFoundError when metadata is missing."""
    from zerotts.codec import MossCodecDecoder

    codec_dir = tmp_path / "onnx" / "codec"
    codec_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match=r"codec_browser_onnx_meta\.json"):
        MossCodecDecoder(codec_dir)


def _setup_mock_zerotts_assets(model_dir: Path) -> tuple[Path, Path]:
    """Create directory structure and dummy files for ZeroTTS engine testing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = model_dir / ZeroTtsFile.DEFAULT_GGUF_MODEL
    gguf_path.write_bytes(b"GGUF_TEST")

    config_path = model_dir / "config.json"
    config_path.write_text('{"vocab_size": 1000}', encoding="utf-8")

    for voice_id in ("maichi", "baotrang"):
        voice_dir = model_dir / "voices" / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        emb_data = np.ones((1, 32, 64), dtype=np.float32)
        np.savez(voice_dir / "voice.npz", voice_emb=emb_data)

    codec_dir = model_dir / "onnx" / "codec"
    codec_dir.mkdir(parents=True, exist_ok=True)
    (codec_dir / "codec_browser_onnx_meta.json").write_text("{}", encoding="utf-8")

    return model_dir, gguf_path


def test_zerotts_engine_init_context_failure(tmp_path: Path) -> None:
    """Test ZeroTtsGgmlEngine raises RuntimeError when native init returns null."""
    model_dir, gguf_path = _setup_mock_zerotts_assets(tmp_path)
    fake_cdll = MagicMock()
    fake_cdll.zerotts_init_from_file.return_value = None

    with (
        patch("wyoming_vietnamese.zerotts_engine.resolve_zerotts_ggml_lib", return_value=fake_cdll),
        pytest.raises(RuntimeError, match="Failed to initialize ZeroTTS GGML context"),
    ):
        ZeroTtsGgmlEngine(
            model_dir=model_dir,
            gguf_path=gguf_path,
            voices=(ZEROTTS_VOICES[0],),
        )


def test_zerotts_engine_init_releases_context_on_config_error(tmp_path: Path) -> None:
    """Test native context is freed when config initialization fails."""
    model_dir, gguf_path = _setup_mock_zerotts_assets(tmp_path)
    # Remove config.json to trigger post-context failure
    (model_dir / "config.json").unlink()

    fake_cdll = MagicMock()
    fake_cdll.zerotts_init_from_file.return_value = 999999
    dummy_hp = _ZeroTtsHParams(d_model=64, n_voice_queries=32, num_codebooks=8)
    fake_cdll.zerotts_hparams_of.return_value = ctypes.pointer(dummy_hp)

    with (
        patch("wyoming_vietnamese.zerotts_engine.resolve_zerotts_ggml_lib", return_value=fake_cdll),
        pytest.raises(FileNotFoundError, match=r"Missing ZeroTTS config\.json"),
    ):
        ZeroTtsGgmlEngine(
            model_dir=model_dir,
            gguf_path=gguf_path,
            voices=(ZEROTTS_VOICES[0],),
        )

    fake_cdll.zerotts_free.assert_called_once_with(999999)


def test_zerotts_engine_missing_voice_pack_raises_error(tmp_path: Path) -> None:
    """Test missing voice pack raises FileNotFoundError and frees native context."""
    model_dir, gguf_path = _setup_mock_zerotts_assets(tmp_path)
    # Delete voice pack for baotrang
    baotrang_voice_file = model_dir / "voices" / "baotrang" / "voice.npz"
    baotrang_voice_file.unlink()

    fake_cdll = MagicMock()
    fake_cdll.zerotts_init_from_file.return_value = 888888
    dummy_hp = _ZeroTtsHParams(d_model=64, n_voice_queries=32, num_codebooks=8)
    fake_cdll.zerotts_hparams_of.return_value = ctypes.pointer(dummy_hp)

    fake_tokenizer = Mock(return_value=[1, 2])
    fake_codec = MagicMock()

    with (
        patch("wyoming_vietnamese.zerotts_engine.resolve_zerotts_ggml_lib", return_value=fake_cdll),
        patch("zerotts.tokenizer.load_tokenizer", return_value=fake_tokenizer),
        patch("zerotts.codec.MossCodecDecoder", return_value=fake_codec),
        pytest.raises(FileNotFoundError, match=r"Voice pack file not found for baotrang"),
    ):
        ZeroTtsGgmlEngine(
            model_dir=model_dir,
            gguf_path=gguf_path,
            voices=(ZEROTTS_VOICES[0], ZEROTTS_VOICES[1]),
        )

    fake_cdll.zerotts_free.assert_called_once_with(888888)


def test_zerotts_engine_mismatched_voice_embedding_size_raises(tmp_path: Path) -> None:
    """Test ZeroTtsGgmlEngine rejects voice embeddings with mismatched element count."""
    model_dir, gguf_path = _setup_mock_zerotts_assets(tmp_path)
    maichi_voice_file = model_dir / "voices" / "maichi" / "voice.npz"
    np.savez(maichi_voice_file, voice_emb=np.ones(10, dtype=np.float32))

    fake_cdll = MagicMock()
    fake_cdll.zerotts_init_from_file.return_value = 888888
    dummy_hp = _ZeroTtsHParams(d_model=64, n_voice_queries=32, num_codebooks=8)
    fake_cdll.zerotts_hparams_of.return_value = ctypes.pointer(dummy_hp)

    fake_tokenizer = Mock(return_value=[1, 2])
    fake_codec = MagicMock()

    with (
        patch("wyoming_vietnamese.zerotts_engine.resolve_zerotts_ggml_lib", return_value=fake_cdll),
        patch("zerotts.tokenizer.load_tokenizer", return_value=fake_tokenizer),
        patch("zerotts.codec.MossCodecDecoder", return_value=fake_codec),
        pytest.raises(
            ValueError,
            match=r"Invalid voice embedding size for maichi: expected 2048, got 10",
        ),
    ):
        ZeroTtsGgmlEngine(
            model_dir=model_dir,
            gguf_path=gguf_path,
            voices=(ZEROTTS_VOICES[0],),
        )

    fake_cdll.zerotts_free.assert_called_once_with(888888)


def _setup_engine_fixture(
    tmp_path: Path,
) -> tuple[ZeroTtsGgmlEngine, MagicMock, MagicMock, Mock]:
    """Helper creating a test ZeroTtsGgmlEngine with mocked dependencies."""
    model_dir, gguf_path = _setup_mock_zerotts_assets(tmp_path)
    voices = (ZEROTTS_VOICES[0], ZEROTTS_VOICES[1])

    fake_cdll = MagicMock()
    fake_cdll.zerotts_init_from_file.return_value = 123456

    dummy_hp = _ZeroTtsHParams(d_model=64, n_voice_queries=32, num_codebooks=8)
    fake_cdll.zerotts_hparams_of.return_value = ctypes.pointer(dummy_hp)

    step_counter = {"val": 0}

    def fake_zerotts_begin(
        ctx: object, text_ids: object, text_len: object, voice_emb: object
    ) -> int:
        step_counter["val"] = 0
        return 0

    fake_cdll.zerotts_begin.side_effect = fake_zerotts_begin
    fake_cdll.zerotts_frame.return_value = 0
    fake_cdll.zerotts_advance.return_value = 0

    fake_tokenizer = Mock(return_value=[1, 2, 3])
    fake_stream_decoder = MagicMock()
    fake_stream_decoder.decode_chunk.return_value = np.zeros(960, dtype=np.float32)
    fake_codec = MagicMock()
    fake_codec.streaming_decoder.return_value = fake_stream_decoder

    with (
        patch("wyoming_vietnamese.zerotts_engine.resolve_zerotts_ggml_lib", return_value=fake_cdll),
        patch("zerotts.tokenizer.load_tokenizer", return_value=fake_tokenizer),
        patch("zerotts.codec.MossCodecDecoder", return_value=fake_codec),
    ):
        engine = ZeroTtsGgmlEngine(
            model_dir=model_dir,
            gguf_path=gguf_path,
            voices=voices,
            num_threads=2,
        )

    return engine, fake_cdll, fake_stream_decoder, fake_tokenizer


def test_zerotts_engine_lifecycle_and_inference(tmp_path: Path) -> None:
    """Test full ZeroTtsGgmlEngine lifecycle including voice lookup, inference, and close."""
    engine, fake_cdll, fake_stream_decoder, fake_tokenizer = _setup_engine_fixture(tmp_path)
    voices = (ZEROTTS_VOICES[0], ZEROTTS_VOICES[1])

    assert engine.sample_rate == ZeroTtsAudio.SAMPLE_RATE
    assert engine.chunks_are_sentences is False
    assert "Mai Chi" in engine._preset_voices
    assert "Bảo Trang" in engine._preset_voices

    # Voice embedding resolution
    emb_default = engine._resolve_voice_embedding(None)
    assert emb_default is not None
    assert emb_default.ndim == 1
    emb_by_name = engine._resolve_voice_embedding("Mai Chi")
    assert emb_by_name is not None
    emb_by_id = engine._resolve_voice_embedding("baotrang")
    assert emb_by_id is not None

    with pytest.raises(ValueError, match="Unsupported ZeroTTS voice"):
        engine._resolve_voice_embedding("unknown_voice")

    step_counter = {"val": 0}

    def fake_begin(ctx: object, text_ids: object, text_len: object, voice_emb: object) -> int:
        step_counter["val"] = 0
        return 0

    fake_cdll.zerotts_begin.side_effect = fake_begin

    def fake_frame(
        ctx: int,
        forbid_eoa: bool,
        sp: object,
        cu: float,
        au: object,
        codes: object,
        is_eoa: ctypes.c_void_p,
    ) -> int:
        step_counter["val"] += 1
        if step_counter["val"] >= 3:
            eoa_ptr = ctypes.cast(is_eoa, ctypes.POINTER(ctypes.c_int))
            eoa_ptr.contents.value = 1
        return 0

    fake_cdll.zerotts_frame.side_effect = fake_frame

    # Streaming synthesis
    chunks = list(engine.infer_stream("Xin chào Việt Nam", voice="Mai Chi"))
    assert chunks
    assert fake_cdll.zerotts_begin.called
    assert fake_stream_decoder.close.called

    # Empty tokens early exit
    fake_tokenizer.return_value = []
    empty_chunks = list(engine.infer_stream("", voice="Mai Chi"))
    assert not empty_chunks
    fake_tokenizer.return_value = [1, 2, 3]

    # Multi-voice warmup in ZeroTTS engine
    warm_up_tts(engine, voices=voices, engine=TtsEngine.ZEROTTS)

    # Engine close
    engine.close()
    assert fake_cdll.zerotts_free.called
    assert engine._ctx is None

    # Closed context error
    with pytest.raises(RuntimeError, match="ZeroTTS GGML context is closed"):
        list(engine.infer_stream("Alo"))


def test_zerotts_engine_generation_stops_at_maximum_frames(tmp_path: Path) -> None:
    """Test synthesis stops at maximum frames without EOA and closes decoder."""
    engine, fake_cdll, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    chunks = list(engine.infer_stream("Xin chào Việt Nam", voice="Mai Chi"))
    assert chunks
    assert fake_cdll.zerotts_frame.call_count == 1500
    assert fake_cdll.zerotts_advance.call_count == 1500
    assert fake_stream_decoder.close.called


def test_zerotts_engine_begin_failure(tmp_path: Path) -> None:
    """Test zerotts_begin non-zero return raises RuntimeError."""
    engine, fake_cdll, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    fake_cdll.zerotts_begin.side_effect = None
    fake_cdll.zerotts_begin.return_value = -1

    with pytest.raises(RuntimeError, match=r"zerotts_begin failed with status code -1"):
        list(engine.infer_stream("Xin chào"))

    assert not fake_stream_decoder.close.called


def test_zerotts_engine_frame_failure_closes_decoder(tmp_path: Path) -> None:
    """Test zerotts_frame non-zero return raises RuntimeError and closes decoder."""
    engine, fake_cdll, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    fake_cdll.zerotts_frame.side_effect = None
    fake_cdll.zerotts_frame.return_value = 1

    with pytest.raises(RuntimeError, match=r"zerotts_frame failed at step 0"):
        list(engine.infer_stream("Xin chào"))

    assert fake_stream_decoder.close.called


def test_zerotts_engine_advance_failure_closes_decoder(tmp_path: Path) -> None:
    """Test zerotts_advance non-zero return raises RuntimeError and closes decoder."""
    engine, fake_cdll, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    fake_cdll.zerotts_advance.return_value = 2

    with pytest.raises(RuntimeError, match=r"zerotts_advance failed at step 0"):
        list(engine.infer_stream("Xin chào"))

    assert fake_stream_decoder.close.called


def test_zerotts_engine_decoder_exception_closes_decoder(tmp_path: Path) -> None:
    """Test exception in decode_chunk propagates and closes stream decoder."""
    engine, _, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    fake_stream_decoder.decode_chunk.side_effect = RuntimeError("MOSS codec decode failed")

    with pytest.raises(RuntimeError, match="MOSS codec decode failed"):
        list(engine.infer_stream("Xin chào"))

    assert fake_stream_decoder.close.called


def test_zerotts_engine_close_is_idempotent(tmp_path: Path) -> None:
    """Test calling close multiple times only frees the native context once."""
    engine, fake_cdll, _, _ = _setup_engine_fixture(tmp_path)

    engine.close()
    assert fake_cdll.zerotts_free.call_count == 1
    assert engine._ctx is None

    # Second call should be a no-op
    engine.close()
    assert fake_cdll.zerotts_free.call_count == 1


def test_zerotts_engine_null_voice_embedding_is_flat(tmp_path: Path) -> None:
    """Test fallback null voice embedding is 1D contiguous float32."""
    engine, _, _, _ = _setup_engine_fixture(tmp_path)
    assert engine._null_voice_emb.ndim == 1
    assert engine._null_voice_emb.dtype == np.float32
    assert engine._null_voice_emb.flags.c_contiguous


def test_resolve_zerotts_ggml_lib_excludes_scratch_paths(tmp_path: Path) -> None:
    """Test candidate paths for libzerotts.so do not reference scratch directories."""
    searched_paths: list[str] = []

    def fake_is_file(path_obj: Path) -> bool:
        searched_paths.append(str(path_obj))
        return False

    with (
        patch.object(Path, "is_file", fake_is_file),
        patch("wyoming_vietnamese.zerotts_engine._build_zerotts_ggml_lib") as mock_build,
        patch("wyoming_vietnamese.zerotts_engine.ctypes.CDLL"),
    ):
        mock_build.return_value = tmp_path / ZeroTtsFile.LIBZEROTTS_SO
        resolve_zerotts_ggml_lib()

    assert all("scratch" not in p for p in searched_paths)


def test_zerotts_engine_yields_eoa_frame(tmp_path: Path) -> None:
    """Test that the frame where EOA is signaled is buffered and decoded."""
    engine, fake_cdll, fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)
    step_counter = {"val": 0}

    def fake_begin(ctx: object, text_ids: object, text_len: object, voice_emb: object) -> int:
        step_counter["val"] = 0
        return 0

    fake_cdll.zerotts_begin.side_effect = fake_begin

    def fake_frame(
        ctx: int,
        forbid_eoa: bool,
        sp: object,
        cu: float,
        au: object,
        codes: object,
        is_eoa: ctypes.c_void_p,
    ) -> int:
        step_counter["val"] += 1
        eoa_ptr = ctypes.cast(is_eoa, ctypes.POINTER(ctypes.c_int))
        eoa_ptr.contents.value = 1
        return 0

    fake_cdll.zerotts_frame.side_effect = fake_frame

    chunks = list(engine.infer_stream("Test", voice="Mai Chi"))
    assert len(chunks) == 1
    assert fake_stream_decoder.decode_chunk.call_count == 1
    assert step_counter["val"] == 2


def test_build_zerotts_rejects_command_injection_arguments(tmp_path: Path) -> None:
    """Test build_zerotts rejects arguments that start with a dash or contain unsafe characters."""
    from tools.build_zerotts import build_zerotts_ggml_lib

    target_dir = tmp_path / "lib"
    with pytest.raises(RuntimeError, match="Invalid or unsafe repo_url"):
        build_zerotts_ggml_lib(target_dir, repo_url="--upload-pack=evil")

    with pytest.raises(RuntimeError, match="Invalid or unsafe repo_url"):
        build_zerotts_ggml_lib(target_dir, repo_url="https://evil.com/owner/repo")

    with pytest.raises(RuntimeError, match="Invalid or unsafe commit"):
        build_zerotts_ggml_lib(target_dir, commit="-b evil")

    with pytest.raises(RuntimeError, match="Invalid or unsafe commit"):
        build_zerotts_ggml_lib(target_dir, commit="not-hex-commit")


def test_build_zerotts_unique_temp_build_dir(tmp_path: Path) -> None:
    """Test build_zerotts_ggml_lib uses unique temp dir and preserves sibling dirs."""
    from tools.build_zerotts import build_zerotts_ggml_lib

    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)
    sibling_dir = target_dir / "sibling_build"
    sibling_dir.mkdir(parents=True, exist_ok=True)
    sibling_file = sibling_dir / "keep.txt"
    sibling_file.write_text("important")

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"ELF_FAKE")
        return Mock(returncode=0)

    mock_cmake_proc = Mock(communicate=Mock(return_value=(b"", b"")), returncode=0, pid=12345)
    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_cmake_proc),
        patch("tools.build_zerotts.ctypes.CDLL"),
    ):
        result = build_zerotts_ggml_lib(target_dir)
        assert result.is_file()

    # Sibling directory should still exist untouched
    assert sibling_file.is_file()
    assert sibling_file.read_text() == "important"

    # No leftover temporary build directories
    build_dirs = [p for p in target_dir.iterdir() if p.is_dir() and p.name.startswith("_build_")]
    assert not build_dirs


def test_zerotts_engine_infer_stream_resets_rng_per_request(tmp_path: Path) -> None:
    """Test that infer_stream reseeds the RNG so each request is independent of prior runs."""
    engine, fake_cdll, _fake_stream_decoder, _ = _setup_engine_fixture(tmp_path)

    recorded_cu: list[float] = []

    def fake_frame(
        ctx: int,
        forbid_eoa: bool,
        sp: object,
        cu: float,
        au: object,
        codes: object,
        is_eoa: ctypes.c_void_p,
    ) -> int:
        recorded_cu.append(cu)
        eoa_ptr = ctypes.cast(is_eoa, ctypes.POINTER(ctypes.c_int))
        eoa_ptr.contents.value = 1
        return 0

    fake_cdll.zerotts_frame.side_effect = fake_frame

    # First call
    list(engine.infer_stream("First request", voice="Mai Chi"))
    first_cu = list(recorded_cu)
    recorded_cu.clear()

    # Second call
    list(engine.infer_stream("Second request", voice="Mai Chi"))
    second_cu = list(recorded_cu)

    assert len(first_cu) == 2
    assert first_cu == second_cu


def test_copy_ggml_libraries_preserves_valid_relative_links(tmp_path: Path) -> None:
    """Test _copy_ggml_libraries preserves relative symlinks in target_dir after cleanup."""
    from tools.build_zerotts import _copy_ggml_libraries

    vendor_src_dir = tmp_path / "build" / "vendor-ggml" / "src"
    vendor_src_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "app" / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    real_file = vendor_src_dir / "libggml-cpu.so.0.1.0"
    real_file.write_bytes(b"GGML_CPU_BYTES")

    rel_symlink = vendor_src_dir / "libggml-cpu.so.0"
    rel_symlink.symlink_to("libggml-cpu.so.0.1.0")

    abs_symlink = vendor_src_dir / "libggml-cpu.so"
    abs_symlink.symlink_to(rel_symlink.resolve())

    _copy_ggml_libraries(vendor_src_dir, target_dir)

    # Simulate deleting the build tree
    shutil.rmtree(tmp_path / "build")

    copied_real = target_dir / "libggml-cpu.so.0.1.0"
    copied_rel = target_dir / "libggml-cpu.so.0"
    copied_abs = target_dir / "libggml-cpu.so"

    assert copied_real.is_file()
    assert copied_real.read_bytes() == b"GGML_CPU_BYTES"

    for link in (copied_rel, copied_abs):
        assert link.is_file()
        assert link.read_bytes() == b"GGML_CPU_BYTES"
        assert link.resolve().parent == target_dir.resolve()


def test_copy_ggml_libraries_copies_resolved_target_when_symlink_points_outside(
    tmp_path: Path,
) -> None:
    """Test _copy_ggml_libraries converts external symlinks into standalone resolved files."""
    from tools.build_zerotts import _copy_ggml_libraries

    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    external_file = external_dir / "libggml-external.so"
    external_file.write_bytes(b"EXTERNAL_BYTES")

    vendor_src_dir = tmp_path / "build" / "vendor-ggml" / "src"
    vendor_src_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "app" / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    vendor_symlink = vendor_src_dir / "libggml-other.so"
    vendor_symlink.symlink_to(external_file)

    _copy_ggml_libraries(vendor_src_dir, target_dir)

    # Delete external and build directories
    shutil.rmtree(tmp_path / "external")
    shutil.rmtree(tmp_path / "build")

    copied_file = target_dir / "libggml-other.so"
    assert copied_file.is_file()
    assert copied_file.read_bytes() == b"EXTERNAL_BYTES"
    assert not copied_file.is_symlink() or copied_file.resolve().parent == target_dir.resolve()


def test_copy_single_library_skips_invalid_candidates(tmp_path: Path) -> None:
    """Test _copy_single_library skips unresolvable symlinks and non-files."""
    from tools.build_zerotts import _copy_single_library

    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Broken symlink
    broken_symlink = vendor_dir / "libggml-broken.so"
    broken_symlink.symlink_to(vendor_dir / "nonexistent.so")
    assert _copy_single_library(broken_symlink, vendor_dir, target_dir) is None

    # Directory instead of file
    sub_dir = vendor_dir / "libggml-dir.so"
    sub_dir.mkdir(parents=True, exist_ok=True)
    assert _copy_single_library(sub_dir, vendor_dir, target_dir) is None


@pytest.mark.parametrize(
    ("machine", "cpu_flags", "expected_variant"),
    [
        ("x86_64", _AVX512_CPU_FLAGS | {"sse4_2"}, ZeroTtsVariant.AVX512),
        ("x86_64", _AVX2_CPU_FLAGS | {"avx512f", "sse4_2"}, ZeroTtsVariant.AVX2),
        ("x86_64", _AVX2_CPU_FLAGS | {"sse4_2"}, ZeroTtsVariant.AVX2),
        ("x86_64", _AVX_CPU_FLAGS | {"sse4_2"}, ZeroTtsVariant.AVX),
        ("x86_64", _SSE4_CPU_FLAGS, ZeroTtsVariant.SSE4),
        ("x86_64", {"sse4_1", "sse4_2", "popcnt", "ssse3"}, ZeroTtsVariant.SSE4),
        ("x86_64", {"sse4_2", "ssse3"}, ZeroTtsVariant.COMPAT),
        ("aarch64", {"asimd", "asimddp"}, ZeroTtsVariant.ARM_DOTPROD),
        ("aarch64", {"asimd", "dotprod"}, ZeroTtsVariant.ARM_DOTPROD),
        ("aarch64", {"asimd"}, ZeroTtsVariant.COMPAT),
        ("aarch64", set(), ZeroTtsVariant.COMPAT),
    ],
)
def test_detect_cpu_variant_host_architecture(
    machine: str,
    cpu_flags: set[str],
    expected_variant: str,
) -> None:
    """Test detect_cpu_variant evaluates machine architecture and host CPU flags."""
    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value=machine),
        patch("wyoming_vietnamese.cpu.get_host_cpu_flags", return_value=cpu_flags),
    ):
        assert detect_cpu_variant() == expected_variant


def test_get_proc_cpu_flags_reads_proc(tmp_path: Path) -> None:
    """Test get_host_cpu_flags parses flags and Features lines from /proc/cpuinfo."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nflags : fpu vme avx2 fma\nFeatures : fp asimd\n")
    with patch("wyoming_vietnamese.cpu.Path") as mock_path:
        mock_path.return_value = cpuinfo
        flags = get_host_cpu_flags()
        assert "avx2" in flags
        assert "fma" in flags
        assert "asimd" in flags


def test_get_proc_cpu_flags_heterogeneous_cores(tmp_path: Path) -> None:
    """Test get_host_cpu_flags takes the intersection of flags across all cores."""
    cpuinfo = tmp_path / "cpuinfo"
    # Core 0 has avx512f, but Core 1 only has avx2
    cpuinfo.write_text(
        "processor : 0\nflags : fpu vme avx2 avx512f fma\n\n"
        "processor : 1\nflags : fpu vme avx2 fma\n"
    )
    with patch("wyoming_vietnamese.cpu.Path") as mock_path:
        mock_path.return_value = cpuinfo
        flags = get_host_cpu_flags()
        assert "avx2" in flags
        assert "fma" in flags
        assert "avx512f" not in flags


def test_get_proc_cpu_flags_darwin_sysctl() -> None:
    """Test get_host_cpu_flags queries sysctl on macOS when /proc/cpuinfo is missing."""
    with (
        patch("wyoming_vietnamese.cpu.Path.is_file", return_value=False),
        patch("wyoming_vietnamese.cpu.platform.system", return_value="Darwin"),
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.subprocess.run",
            return_value=Mock(
                returncode=0,
                stdout="AVX1.0 AVX2 FMA F16C BMI2 SSE4.1 SSE4.2\n",
            ),
        ),
    ):
        flags = get_host_cpu_flags()
        assert "avx" in flags
        assert "avx2" in flags
        assert "fma" in flags
        assert "sse4_1" in flags
        assert "sse4_2" in flags


def test_get_proc_cpu_flags_darwin_arm64_sysctl() -> None:
    """Test get_host_cpu_flags queries ARM sysctl on macOS Apple Silicon."""
    with (
        patch("wyoming_vietnamese.cpu.Path.is_file", return_value=False),
        patch("wyoming_vietnamese.cpu.platform.system", return_value="Darwin"),
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="arm64"),
        patch(
            "wyoming_vietnamese.cpu.subprocess.run",
            return_value=Mock(
                returncode=0,
                stdout="1\n",
            ),
        ),
    ):
        flags = get_host_cpu_flags()
        assert "asimd" in flags
        assert "asimddp" in flags
        assert "dotprod" in flags
        assert can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD, flags=flags)
        assert detect_cpu_variant() == ZeroTtsVariant.ARM_DOTPROD


def test_get_proc_cpu_flags_darwin_arm64_sysctl_no_dotprod() -> None:
    """Test get_host_cpu_flags on macOS ARM when FEAT_DotProd is not supported."""
    with (
        patch("wyoming_vietnamese.cpu.Path.is_file", return_value=False),
        patch("wyoming_vietnamese.cpu.platform.system", return_value="Darwin"),
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="arm64"),
        patch(
            "wyoming_vietnamese.cpu.subprocess.run",
            return_value=Mock(
                returncode=0,
                stdout="0\n",
            ),
        ),
    ):
        flags = get_host_cpu_flags()
        assert "asimd" in flags
        assert "asimddp" not in flags
        assert not can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD, flags=flags)
        assert detect_cpu_variant() == ZeroTtsVariant.COMPAT


@pytest.mark.parametrize("with_env", [False, True])
def test_get_zerotts_candidate_lib_paths_priority_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_env: bool
) -> None:
    """Test candidate library path ordering for custom, env, and variants."""
    custom_lib = tmp_path / "custom.so"
    expected_prefix: list[tuple[str, Path]] = [("custom", custom_lib)]
    if with_env:
        env_lib = tmp_path / "env.so"
        monkeypatch.setenv("ZEROTTS_LIB_PATH", str(env_lib))
        expected_prefix.append(("env", env_lib))
    else:
        monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)

    with patch(
        "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
        return_value=ZeroTtsVariant.AVX2,
    ):
        candidates = get_zerotts_candidate_lib_paths(custom_lib)
        assert candidates[: len(expected_prefix)] == expected_prefix
        labels = [var for var, _ in candidates]
        assert "default" not in labels
        assert "avx2" in labels
        assert "avx" in labels
        assert "sse4" in labels
        assert "compat" in labels
        assert (
            labels.index("avx2")
            < labels.index("avx")
            < labels.index("sse4")
            < labels.index("compat")
        )


def test_get_zerotts_candidate_lib_paths_avx512() -> None:
    """Test get_zerotts_candidate_lib_paths includes lower variants when avx512 is optimal."""
    with patch(
        "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
        return_value=ZeroTtsVariant.AVX512,
    ):
        candidates = get_zerotts_candidate_lib_paths()
        var_labels = [var for var, _ in candidates]
        assert "avx512" in var_labels
        assert "avx2" in var_labels
        assert "avx" in var_labels
        assert "sse4" in var_labels
        assert "compat" in var_labels
        assert (
            var_labels.index("avx512")
            < var_labels.index("avx2")
            < var_labels.index("avx")
            < var_labels.index("sse4")
            < var_labels.index("compat")
        )


def test_get_zerotts_candidate_lib_paths_sse4() -> None:
    """Test get_zerotts_candidate_lib_paths includes sse4 and compat when sse4 is optimal."""
    with patch(
        "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
        return_value=ZeroTtsVariant.SSE4,
    ):
        candidates = get_zerotts_candidate_lib_paths()
        var_labels = [var for var, _ in candidates]
        assert "sse4" in var_labels
        assert "compat" in var_labels
        assert var_labels.index("sse4") < var_labels.index("compat")
        assert "avx" not in var_labels
        assert "avx2" not in var_labels
        assert "avx512" not in var_labels


def test_get_zerotts_candidate_lib_paths_arm_dotprod() -> None:
    """Test get_zerotts_candidate_lib_paths includes arm_dotprod and compat on ARM."""
    with patch(
        "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
        return_value=ZeroTtsVariant.ARM_DOTPROD,
    ):
        candidates = get_zerotts_candidate_lib_paths()
        var_labels = [var for var, _ in candidates]
        assert "arm_dotprod" in var_labels
        assert "compat" in var_labels
        assert var_labels.index("arm_dotprod") < var_labels.index("compat")
        assert "avx" not in var_labels
        assert "sse4" not in var_labels


@pytest.mark.parametrize("is_symlink", [True, False])
def test_get_zerotts_candidate_lib_paths_symlink_and_unversioned(is_symlink: bool) -> None:
    """Test candidate paths admits compatible symlinks and rejects unversioned binaries."""
    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.COMPAT,
        ),
        patch.object(Path, "is_symlink", return_value=is_symlink),
        patch("os.readlink", return_value="compat/libzerotts.so"),
    ):
        candidates = get_zerotts_candidate_lib_paths()
        labels = [var for var, _ in candidates]
        assert ("symlink-compat" in labels) is is_symlink
        assert "default" not in labels


def test_get_zerotts_candidate_lib_paths_incompatible_symlink_rejected() -> None:
    """Test get_zerotts_candidate_lib_paths rejects symlinks pointing to unsupported variants."""
    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.COMPAT,
        ),
        patch.object(Path, "is_symlink", return_value=True),
        patch("os.readlink", return_value="avx512/libzerotts.so"),
    ):
        candidates = get_zerotts_candidate_lib_paths()
        labels = [var for var, _ in candidates]
        assert "symlink-avx512" not in labels
        assert "default" not in labels


def test_get_zerotts_candidate_lib_paths_native() -> None:
    """Test get_zerotts_candidate_lib_paths includes native and compat when native is optimal."""
    with patch(
        "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
        return_value=ZeroTtsVariant.NATIVE,
    ):
        candidates = get_zerotts_candidate_lib_paths()
        labels = [var for var, _ in candidates]
        assert "native" in labels
        assert "compat" in labels
        assert labels.index("native") < labels.index("compat")
        assert "avx512" not in labels
        assert "avx2" not in labels
        assert "avx" not in labels
        assert "sse4" not in labels


def test_resolve_zerotts_ggml_lib_selects_optimal_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib selects the detected optimal variant when present."""
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)
    avx2_dir = tmp_path / "lib" / "avx2"
    avx2_dir.mkdir(parents=True, exist_ok=True)
    avx2_so = avx2_dir / ZeroTtsFile.LIBZEROTTS_SO
    avx2_so.write_bytes(b"ELF_AVX2")

    fake_cdll = MagicMock()
    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.AVX2,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.get_zerotts_candidate_lib_paths",
            return_value=[("avx2", avx2_so)],
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            return_value=fake_cdll,
        ) as mock_cdll,
    ):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_cdll
        mock_cdll.assert_called_once_with(str(avx2_so.resolve()))


def test_load_zerotts_candidate_binds_variant_ggml_dependencies(tmp_path: Path) -> None:
    """Test a variant loads its GGML dependencies from the same directory."""
    variant_dir = tmp_path / "avx2"
    variant_dir.mkdir()
    candidate = variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    candidate.write_bytes(b"ELF_AVX2")
    for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (variant_dir / dependency).write_bytes(b"ELF_GGML")

    fake_cdll = MagicMock()
    with patch(
        "wyoming_vietnamese.zerotts_engine.ctypes.CDLL", return_value=fake_cdll
    ) as mock_cdll:
        assert _load_zerotts_candidate(candidate) is fake_cdll

    assert [c.args[0] for c in mock_cdll.call_args_list] == [
        str((variant_dir / dependency).resolve()) for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES
    ] + [str(candidate.resolve())]
    assert not any(
        "mode" in c.kwargs and c.kwargs["mode"] == getattr(ctypes, "RTLD_GLOBAL", -1)
        for c in mock_cdll.call_args_list
    )


def test_load_zerotts_candidate_unloads_dependencies_on_failure(tmp_path: Path) -> None:
    """Test preloaded dependencies are unloaded if candidate library loading fails."""
    variant_dir = tmp_path / "avx2"
    variant_dir.mkdir()
    candidate = variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    candidate.write_bytes(b"ELF_AVX2")
    for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (variant_dir / dependency).write_bytes(b"ELF_GGML")

    fake_deps = [MagicMock() for _ in _ZEROTTS_RUNTIME_DEPENDENCIES]
    dep_map = {
        str((variant_dir / dep).resolve()): fake_dep
        for dep, fake_dep in zip(_ZEROTTS_RUNTIME_DEPENDENCIES, fake_deps, strict=True)
    }

    def fake_loader(path_str: str, **kwargs: object) -> MagicMock:
        if path_str in dep_map:
            return dep_map[path_str]
        raise OSError("Failed to load candidate binary")

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            side_effect=fake_loader,
        ),
        patch("wyoming_vietnamese.zerotts_engine._unload_cdll") as mock_unload,
        pytest.raises(OSError, match="Failed to load candidate binary"),
    ):
        _load_zerotts_candidate(candidate)

    assert mock_unload.call_count == len(_ZEROTTS_RUNTIME_DEPENDENCIES)
    assert mock_unload.call_args_list == [call(dep) for dep in reversed(fake_deps)]


def test_unload_cdll_invokes_dlclose() -> None:
    """Test _unload_cdll closes the underlying handle when handle is an integer."""
    fake_lib = MagicMock()
    fake_lib._handle = 123456
    with patch("_ctypes.dlclose") as mock_dlclose:
        _unload_cdll(fake_lib)
        mock_dlclose.assert_called_once_with(123456)
        assert fake_lib._handle is None


def test_unload_cdll_safely_ignores_invalid_handle() -> None:
    """Test _unload_cdll handles non-integer or missing handles without error."""
    fake_lib = MagicMock()
    fake_lib._handle = "not_an_int"
    with patch("_ctypes.dlclose") as mock_dlclose:
        _unload_cdll(fake_lib)
        mock_dlclose.assert_not_called()

    empty_lib = MagicMock(spec=[])
    _unload_cdll(empty_lib)


def test_load_zerotts_candidate_rejects_incomplete_dependencies(tmp_path: Path) -> None:
    """Test _load_zerotts_candidate raises OSError on partial dependency sets."""
    variant_dir = tmp_path / "avx2"
    variant_dir.mkdir()
    candidate = variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    candidate.write_bytes(b"ELF_AVX2")
    (variant_dir / _ZEROTTS_RUNTIME_DEPENDENCIES[0]).write_bytes(b"ELF_GGML")
    expected_missing = re.escape(", ".join(_ZEROTTS_RUNTIME_DEPENDENCIES[1:]))

    with pytest.raises(
        OSError,
        match=rf"Incomplete ZeroTTS runtime dependencies for .*: missing {expected_missing}",
    ):
        _load_zerotts_candidate(candidate)


def test_resolve_zerotts_ggml_lib_fallback_on_incomplete_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib falls back when candidate has incomplete dependencies."""
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)
    base_dir = tmp_path / "lib"
    avx2_dir = base_dir / "avx2"
    compat_dir = base_dir / "compat"
    avx2_dir.mkdir(parents=True)
    compat_dir.mkdir(parents=True)

    avx2_so = avx2_dir / ZeroTtsFile.LIBZEROTTS_SO
    compat_so = compat_dir / ZeroTtsFile.LIBZEROTTS_SO
    avx2_so.write_bytes(b"ELF_AVX2")
    compat_so.write_bytes(b"ELF_COMPAT")

    (avx2_dir / _ZEROTTS_RUNTIME_DEPENDENCIES[0]).write_bytes(b"ELF_AVX2_DEP")
    for dep in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (compat_dir / dep).write_bytes(b"ELF_COMPAT_DEP")

    fake_compat_cdll = MagicMock()

    def fake_loader(path_str: str, **kwargs: object) -> MagicMock:
        if "compat" in path_str and path_str.endswith(ZeroTtsFile.LIBZEROTTS_SO):
            return fake_compat_cdll
        mock = MagicMock()
        mock._path = path_str
        return mock

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.AVX2,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.get_zerotts_candidate_lib_paths",
            return_value=[("avx2", avx2_so), ("compat", compat_so)],
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            side_effect=fake_loader,
        ),
    ):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_compat_cdll


def test_load_zerotts_candidate_resolves_symlink_for_dependencies(tmp_path: Path) -> None:
    """Test _load_zerotts_candidate resolves symlinks to preload target dependencies."""
    target_dir = tmp_path / "actual" / "avx2"
    target_dir.mkdir(parents=True)
    real_candidate = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    real_candidate.write_bytes(b"ELF_AVX2")

    for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (target_dir / dependency).write_bytes(b"ELF_GGML")

    symlink_dir = tmp_path / "symlinks"
    symlink_dir.mkdir(parents=True)
    symlink_candidate = symlink_dir / "custom_zerotts.so"
    symlink_candidate.symlink_to(real_candidate)

    fake_cdll = MagicMock()
    with patch(
        "wyoming_vietnamese.zerotts_engine.ctypes.CDLL", return_value=fake_cdll
    ) as mock_cdll:
        assert _load_zerotts_candidate(symlink_candidate) is fake_cdll

    assert [c.args[0] for c in mock_cdll.call_args_list] == [
        str((target_dir / dependency).resolve()) for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES
    ] + [str(real_candidate.resolve())]


def test_resolve_zerotts_ggml_lib_symlink_candidate_preloads_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib with symlinked custom/env candidate preloads dependencies."""
    target_dir = tmp_path / "installed" / "avx2"
    target_dir.mkdir(parents=True)
    real_candidate = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    real_candidate.write_bytes(b"ELF_AVX2")

    for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (target_dir / dependency).write_bytes(b"ELF_GGML")

    symlink_path = tmp_path / "env_symlink.so"
    symlink_path.symlink_to(real_candidate)
    monkeypatch.setenv("ZEROTTS_LIB_PATH", str(symlink_path))

    fake_cdll = MagicMock()
    with patch(
        "wyoming_vietnamese.zerotts_engine.ctypes.CDLL", return_value=fake_cdll
    ) as mock_cdll:
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_cdll

    assert [c.args[0] for c in mock_cdll.call_args_list] == [
        str((target_dir / dependency).resolve()) for dependency in _ZEROTTS_RUNTIME_DEPENDENCIES
    ] + [str(real_candidate.resolve())]


def test_resolve_zerotts_ggml_lib_fallback_unloads_failed_variant_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test candidate dependencies are cleaned up before fallback candidate is loaded."""
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)
    base_dir = tmp_path / "lib"
    avx2_dir = base_dir / "avx2"
    compat_dir = base_dir / "compat"
    avx2_dir.mkdir(parents=True)
    compat_dir.mkdir(parents=True)

    avx2_so = avx2_dir / ZeroTtsFile.LIBZEROTTS_SO
    compat_so = compat_dir / ZeroTtsFile.LIBZEROTTS_SO
    avx2_so.write_bytes(b"ELF_AVX2")
    compat_so.write_bytes(b"ELF_COMPAT")

    for dep in _ZEROTTS_RUNTIME_DEPENDENCIES:
        (avx2_dir / dep).write_bytes(b"ELF_AVX2_DEP")
        (compat_dir / dep).write_bytes(b"ELF_COMPAT_DEP")

    fake_compat_cdll = MagicMock()
    unloaded: list[str] = []

    def fake_loader(path_str: str, **kwargs: object) -> MagicMock:
        if "avx2" in path_str and path_str.endswith(ZeroTtsFile.LIBZEROTTS_SO):
            raise OSError("AVX2 candidate invalid")
        mock = MagicMock()
        mock._path = path_str
        if "compat" in path_str and path_str.endswith(ZeroTtsFile.LIBZEROTTS_SO):
            return fake_compat_cdll
        return mock

    def fake_unload(lib: ctypes.CDLL) -> None:
        unloaded.append(getattr(lib, "_path", str(lib)))

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.AVX2,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.get_zerotts_candidate_lib_paths",
            return_value=[("avx2", avx2_so), ("compat", compat_so)],
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            side_effect=fake_loader,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine._unload_cdll",
            side_effect=fake_unload,
        ),
    ):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_compat_cdll
        assert any("avx2" in path for path in unloaded)


def test_resolve_zerotts_ggml_lib_fallback_to_compat_on_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib falls back to compat variant on failure."""
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)
    base_dir = tmp_path / "lib"
    avx2_so = base_dir / "avx2" / ZeroTtsFile.LIBZEROTTS_SO
    compat_so = base_dir / "compat" / ZeroTtsFile.LIBZEROTTS_SO
    avx2_so.parent.mkdir(parents=True, exist_ok=True)
    compat_so.parent.mkdir(parents=True, exist_ok=True)
    avx2_so.write_bytes(b"ELF_AVX2")
    compat_so.write_bytes(b"ELF_COMPAT")

    fake_cdll = MagicMock()

    def fake_cdll_loader(path_str: str) -> MagicMock:
        if "avx2" in path_str:
            raise OSError("Illegal instruction")
        return fake_cdll

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.AVX2,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.get_zerotts_candidate_lib_paths",
            return_value=[("avx2", avx2_so), ("compat", compat_so)],
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            side_effect=fake_cdll_loader,
        ) as mock_cdll,
    ):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_cdll
        assert mock_cdll.call_count == 2
        assert mock_cdll.call_args_list[0][0][0] == str(avx2_so.resolve())
        assert mock_cdll.call_args_list[1][0][0] == str(compat_so.resolve())


def test_resolve_zerotts_ggml_lib_falls_back_to_build_on_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test resolve_zerotts_ggml_lib warns and builds when all candidates fail to load."""
    monkeypatch.delenv("ZEROTTS_LIB_PATH", raising=False)
    base_dir = tmp_path / "lib"
    avx2_so = base_dir / "avx2" / ZeroTtsFile.LIBZEROTTS_SO
    avx2_so.parent.mkdir(parents=True, exist_ok=True)
    avx2_so.write_bytes(b"BROKEN_AVX2")

    fake_built = base_dir / "avx2" / "libzerotts_built.so"
    fake_cdll = MagicMock()

    def fake_cdll_loader(path_str: str) -> MagicMock:
        if path_str == str(avx2_so.resolve()):
            raise OSError("Illegal instruction in candidate")
        return fake_cdll

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine.detect_cpu_variant",
            return_value=ZeroTtsVariant.AVX2,
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine.get_zerotts_candidate_lib_paths",
            return_value=[("avx2", avx2_so)],
        ),
        patch(
            "wyoming_vietnamese.zerotts_engine._build_zerotts_ggml_lib",
            return_value=fake_built,
        ) as mock_build,
        patch(
            "wyoming_vietnamese.zerotts_engine.ctypes.CDLL",
            side_effect=fake_cdll_loader,
        ),
    ):
        lib = resolve_zerotts_ggml_lib()
        assert lib is fake_cdll
        mock_build.assert_called_once()
        assert mock_build.call_args[1]["variant"] == ZeroTtsVariant.AVX2
        assert mock_build.call_args[1]["publish_subdir"] is True


def test_can_host_execute_variant() -> None:
    """Test can_host_execute_variant checks host CPU capabilities."""
    from tools.build_zerotts import _can_host_execute_variant

    assert can_host_execute_variant(ZeroTtsVariant.COMPAT) is True
    assert can_host_execute_variant(ZeroTtsVariant.NATIVE) is True
    assert _can_host_execute_variant(ZeroTtsVariant.COMPAT) is True

    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value=set(_AVX2_CPU_FLAGS),
        ),
    ):
        assert can_host_execute_variant(ZeroTtsVariant.SSE4) is True
        assert can_host_execute_variant(ZeroTtsVariant.AVX) is True
        assert can_host_execute_variant(ZeroTtsVariant.AVX2) is True
        assert can_host_execute_variant(ZeroTtsVariant.AVX512) is False

    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value=set(_SSE4_CPU_FLAGS),
        ),
    ):
        assert can_host_execute_variant(ZeroTtsVariant.SSE4) is True
        assert can_host_execute_variant(ZeroTtsVariant.AVX) is False
        assert can_host_execute_variant(ZeroTtsVariant.AVX2) is False

    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="aarch64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value={"asimd", "asimddp"},
        ),
    ):
        assert can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD) is True
        assert can_host_execute_variant(ZeroTtsVariant.SSE4) is False
        assert can_host_execute_variant(ZeroTtsVariant.AVX) is False
        assert can_host_execute_variant(ZeroTtsVariant.AVX2) is False

    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="aarch64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value={"asimd"},
        ),
    ):
        assert can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD) is False

    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value={"asimddp"},
        ),
    ):
        assert can_host_execute_variant(ZeroTtsVariant.ARM_DOTPROD) is False


def test_can_host_execute_variant_avx512_positive_and_flags_cache() -> None:
    """Test can_host_execute_variant accepts AVX512 and honors flags param without re-reading."""
    avx512_flags = set(_AVX512_CPU_FLAGS)
    with (
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.get_host_cpu_flags",
            return_value=avx512_flags,
        ) as mock_get_flags,
    ):
        assert can_host_execute_variant(ZeroTtsVariant.AVX512) is True
        mock_get_flags.assert_called_once()

        mock_get_flags.reset_mock()
        assert can_host_execute_variant(ZeroTtsVariant.AVX512, flags=avx512_flags) is True
        mock_get_flags.assert_not_called()


@pytest.mark.parametrize(
    ("variant", "flag_set"),
    [
        (ZeroTtsVariant.SSE4, _SSE4_CPU_FLAGS),
        (ZeroTtsVariant.AVX, _AVX_CPU_FLAGS),
        (ZeroTtsVariant.AVX2, _AVX2_CPU_FLAGS),
        (ZeroTtsVariant.AVX512, _AVX512_CPU_FLAGS),
    ],
)
def test_can_host_execute_variant_missing_flag(variant: str, flag_set: frozenset[str]) -> None:
    """Test can_host_execute_variant rejects variants when any required flag is missing."""
    for missing_flag in sorted(flag_set):
        flags = flag_set - {missing_flag}
        with (
            patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
            patch("wyoming_vietnamese.cpu.get_host_cpu_flags", return_value=flags),
        ):
            assert can_host_execute_variant(variant) is False


@pytest.mark.parametrize(
    ("machine", "expected_variants"),
    [
        (
            "x86_64",
            [
                ZeroTtsVariant.COMPAT,
                ZeroTtsVariant.SSE4,
                ZeroTtsVariant.AVX,
                ZeroTtsVariant.AVX2,
                ZeroTtsVariant.AVX512,
            ],
        ),
        ("aarch64", [ZeroTtsVariant.COMPAT, ZeroTtsVariant.ARM_DOTPROD]),
        ("arm64", [ZeroTtsVariant.COMPAT, ZeroTtsVariant.ARM_DOTPROD]),
        ("riscv64", [ZeroTtsVariant.COMPAT]),
    ],
)
def test_get_supported_variants_for_host(machine: str, expected_variants: list[str]) -> None:
    """Test get_supported_variants_for_host returns appropriate variants per arch."""
    with patch("wyoming_vietnamese.cpu.platform.machine", return_value=machine):
        assert get_supported_variants_for_host() == expected_variants


@pytest.mark.parametrize(
    ("machine", "expected_variants"),
    [
        ("x86_64", ["compat", "sse4", "avx", "avx2", "avx512"]),
        ("aarch64", ["compat", "arm_dotprod"]),
    ],
)
def test_build_all_zerotts_variants_success(
    tmp_path: Path, machine: str, expected_variants: list[str]
) -> None:
    """Test build_all_zerotts_variants compiles each supported variant cleanly."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"
    compiled_variants: list[str] = []

    def fake_compile(
        clone_dest: Path,
        staging_dir: Path,
        *,
        variant: str = "compat",
        **kwargs: object,
    ) -> None:
        compiled_variants.append(variant)
        so_file = staging_dir / ZeroTtsFile.LIBZEROTTS_SO
        so_file.write_bytes(f"ELF_{variant}".encode())

    with (
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries", side_effect=fake_compile),
        patch("tools.build_zerotts._verify_library_loadable"),
        patch("tools.build_zerotts._is_toolchain_variant_supported", return_value=True),
        patch("platform.machine", return_value=machine),
    ):
        results = build_all_zerotts_variants(target_dir)
        assert set(results.keys()) == set(expected_variants)
        assert compiled_variants == expected_variants
        default_link = target_dir / ZeroTtsFile.LIBZEROTTS_SO
        assert default_link.is_symlink()
        assert default_link.readlink() == Path(ZeroTtsVariant.COMPAT) / ZeroTtsFile.LIBZEROTTS_SO


def test_build_all_zerotts_variants_compilation_failure(tmp_path: Path) -> None:
    """Test build_all_zerotts_variants cleans up and raises on compilation error."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"
    with (
        patch("tools.build_zerotts._checkout_repo"),
        patch(
            "tools.build_zerotts._compile_libraries",
            side_effect=subprocess.CalledProcessError(
                1, ["cmake"], output=b"", stderr=b"fatal build error"
            ),
        ),
        patch("tools.build_zerotts._is_toolchain_variant_supported", return_value=True),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(RuntimeError, match="Failed to build ZeroTTS GGML shared libraries"),
    ):
        build_all_zerotts_variants(target_dir)

    leftovers = list(target_dir.glob("_build_all_*"))
    assert not leftovers


def test_build_all_zerotts_variants_missing_staged_library(tmp_path: Path) -> None:
    """Test build_all_zerotts_variants raises FileNotFoundError if output .so is missing."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"
    with (
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries"),
        patch("tools.build_zerotts._is_toolchain_variant_supported", return_value=True),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(FileNotFoundError, match="not found after build"),
    ):
        build_all_zerotts_variants(target_dir)

    leftovers = list(target_dir.glob("_build_all_*"))
    assert not leftovers


def test_build_all_zerotts_variants_loadability_failure(tmp_path: Path) -> None:
    """Test build_all_zerotts_variants raises RuntimeError if loadability check fails."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"

    def fake_compile(clone_dest: Path, staging_dir: Path, **kwargs: object) -> None:
        (staging_dir / ZeroTtsFile.LIBZEROTTS_SO).write_bytes(b"FAKE")

    with (
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries", side_effect=fake_compile),
        patch(
            "tools.build_zerotts._verify_library_loadable",
            side_effect=RuntimeError("load check failed"),
        ),
        patch("tools.build_zerotts._is_toolchain_variant_supported", return_value=True),
        patch("platform.machine", return_value="x86_64"),
        pytest.raises(RuntimeError, match="load check failed"),
    ):
        build_all_zerotts_variants(target_dir)

    leftovers = list(target_dir.glob("_build_all_*"))
    assert not leftovers


def test_build_zerotts_cli_main_all_variants(tmp_path: Path) -> None:
    """Test tools.build_zerotts.main handles --all-variants argument."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    fake_results = {
        "compat": target_dir / "compat" / ZeroTtsFile.LIBZEROTTS_SO,
        "avx2": target_dir / "avx2" / ZeroTtsFile.LIBZEROTTS_SO,
    }
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir), "--all-variants"]),
        patch(
            "tools.build_zerotts.build_all_zerotts_variants",
            return_value=fake_results,
        ) as mock_build_all,
    ):
        exit_code = build_zerotts_main()
        assert exit_code == 0
        mock_build_all.assert_called_once()


@pytest.mark.parametrize(
    ("cli_flag", "expected_kwarg", "mock_machine"),
    [
        (["--variant", "sse4"], {"variant": "sse4"}, "x86_64"),
        (["--variant", "avx"], {"variant": "avx"}, "x86_64"),
        (["--variant", "avx2"], {"variant": "avx2"}, "x86_64"),
        (["--variant", "arm_dotprod"], {"variant": "arm_dotprod"}, "aarch64"),
        (["--native"], {"native": True}, "x86_64"),
    ],
)
def test_build_zerotts_cli_main_single_variant_options(
    tmp_path: Path,
    cli_flag: list[str],
    expected_kwarg: dict[str, object],
    mock_machine: str,
) -> None:
    """Test tools.build_zerotts.main handles variant selection CLI options."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    fake_so = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir), *cli_flag]),
        patch("platform.machine", return_value=mock_machine),
        patch(
            "tools.build_zerotts.build_zerotts_ggml_lib",
            return_value=fake_so,
        ) as mock_build,
    ):
        exit_code = build_zerotts_main()
        assert exit_code == 0
        mock_build.assert_called_once()
        for key, val in expected_kwarg.items():
            assert mock_build.call_args[1][key] == val


@pytest.mark.parametrize(
    ("flags", "expected_err"),
    [
        (
            ["--all-variants", "--variant", "avx2"],
            "Cannot combine --all-variants with --variant or --native",
        ),
        (
            ["--all-variants", "--native"],
            "Cannot combine --all-variants with --variant or --native",
        ),
        (
            ["--variant", "avx2", "--native"],
            "Cannot combine --native with --variant",
        ),
    ],
)
def test_build_zerotts_cli_main_conflicting_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str], expected_err: str
) -> None:
    """Test tools.build_zerotts.main rejects conflicting CLI argument combinations."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir), *flags]),
        pytest.raises(SystemExit) as exc_info,
    ):
        build_zerotts_main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert expected_err in captured.err


@pytest.mark.parametrize(
    ("flags", "machine", "expected_err"),
    [
        (
            ["--variant", "arm_dotprod"],
            "x86_64",
            "Variant 'arm_dotprod' is not supported on host architecture 'x86_64'",
        ),
        (
            ["--variant", "avx2"],
            "aarch64",
            "Variant 'avx2' is not supported on host architecture 'aarch64'",
        ),
    ],
)
def test_build_zerotts_cli_main_unsupported_host_variant(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    machine: str,
    expected_err: str,
) -> None:
    """Test tools.build_zerotts.main rejects variants unsupported on host architecture."""
    from tools.build_zerotts import main as build_zerotts_main

    target_dir = tmp_path / "lib"
    with (
        patch("sys.argv", ["build_zerotts.py", str(target_dir), *flags]),
        patch("platform.machine", return_value=machine),
        pytest.raises(SystemExit) as exc_info,
    ):
        build_zerotts_main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert expected_err in captured.err


def test_is_toolchain_variant_supported() -> None:
    """Test _is_toolchain_variant_supported validates compiler flag support."""
    from tools.build_zerotts import _is_toolchain_variant_supported

    assert _is_toolchain_variant_supported(ZeroTtsVariant.COMPAT) is True
    assert _is_toolchain_variant_supported(ZeroTtsVariant.NATIVE) is True

    tested_variants = (
        ZeroTtsVariant.SSE4,
        ZeroTtsVariant.AVX,
        ZeroTtsVariant.AVX512,
        ZeroTtsVariant.ARM_DOTPROD,
    )
    with patch("tools.build_zerotts.subprocess.run", return_value=Mock(returncode=0)):
        for var in tested_variants:
            assert _is_toolchain_variant_supported(var) is True

    with patch("tools.build_zerotts.subprocess.run", return_value=Mock(returncode=1)):
        for var in tested_variants:
            assert _is_toolchain_variant_supported(var) is False

    with patch("tools.build_zerotts.subprocess.run", side_effect=OSError("g++ not found")):
        assert _is_toolchain_variant_supported(ZeroTtsVariant.AVX512) is False


def test_build_zerotts_ggml_lib_unsupported_toolchain_fails_early(tmp_path: Path) -> None:
    """Test build_zerotts_ggml_lib raises RuntimeError early when toolchain lacks flag support."""
    from tools.build_zerotts import build_zerotts_ggml_lib

    target_dir = tmp_path / "lib"
    with (
        patch("tools.build_zerotts._is_toolchain_variant_supported", return_value=False),
        pytest.raises(RuntimeError, match="Compiler toolchain does not support required flags"),
    ):
        build_zerotts_ggml_lib(target_dir, variant=ZeroTtsVariant.AVX512)


def test_build_all_zerotts_variants_skips_unsupported_toolchain_variant(tmp_path: Path) -> None:
    """Test build_all_zerotts_variants skips variants not supported by the toolchain."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"

    def fake_supported(variant: str) -> bool:
        return variant != ZeroTtsVariant.AVX512

    def fake_compile(
        clone_dest: Path, staging_dir: Path, variant: str = ZeroTtsVariant.COMPAT, **kwargs: object
    ) -> None:
        (staging_dir / ZeroTtsFile.LIBZEROTTS_SO).write_bytes(f"FAKE_{variant}".encode())

    with (
        patch("platform.machine", return_value="x86_64"),
        patch("tools.build_zerotts._is_toolchain_variant_supported", side_effect=fake_supported),
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries", side_effect=fake_compile),
        patch("tools.build_zerotts._verify_library_loadable"),
    ):
        results = build_all_zerotts_variants(target_dir)
        assert ZeroTtsVariant.COMPAT in results
        assert ZeroTtsVariant.SSE4 in results
        assert ZeroTtsVariant.AVX in results
        assert ZeroTtsVariant.AVX2 in results
        assert ZeroTtsVariant.AVX512 not in results


@pytest.mark.parametrize(
    ("native", "variant", "expected_flag", "unexpected_flag"),
    [
        (True, ZeroTtsVariant.NATIVE, "--native", "--variant"),
        (False, ZeroTtsVariant.AVX2, "--variant", "--native"),
    ],
)
def test_run_zerotts_build_script_flags(
    tmp_path: Path,
    native: bool,
    variant: str,
    expected_flag: str,
    unexpected_flag: str,
) -> None:
    """Test _run_zerotts_build_script passes expected flags without conflict."""
    from wyoming_vietnamese.zerotts_engine import _run_zerotts_build_script

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)
    fake_lib = target_dir / ZeroTtsFile.LIBZEROTTS_SO
    fake_lib.write_bytes(b"ELF_FAKE")

    recorded_cmd: list[str] = []

    def fake_subprocess_popen(cmd: list[str], **kwargs: object) -> Mock:
        nonlocal recorded_cmd
        recorded_cmd = list(cmd)
        mock_proc = Mock()
        mock_proc.pid = 12345
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        return mock_proc

    with patch(
        "wyoming_vietnamese.zerotts_engine.subprocess.Popen",
        side_effect=fake_subprocess_popen,
    ):
        res = _run_zerotts_build_script(build_script, target_dir, variant=variant, native=native)
        assert res == fake_lib
        assert expected_flag in recorded_cmd
        assert unexpected_flag not in recorded_cmd


def test_run_zerotts_build_script_publish_subdir(tmp_path: Path) -> None:
    """Test _run_zerotts_build_script places output in variant subdir when publish_subdir=True."""
    from wyoming_vietnamese.zerotts_engine import _run_zerotts_build_script

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    variant_dir = target_dir / ZeroTtsVariant.AVX2
    variant_dir.mkdir(parents=True, exist_ok=True)
    fake_lib = variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    fake_lib.write_bytes(b"ELF_VARIANT")

    recorded_cmd: list[str] = []

    def fake_subprocess_popen(cmd: list[str], **kwargs: object) -> Mock:
        nonlocal recorded_cmd
        recorded_cmd = list(cmd)
        mock_proc = Mock()
        mock_proc.pid = 12345
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        return mock_proc

    with patch(
        "wyoming_vietnamese.zerotts_engine.subprocess.Popen",
        side_effect=fake_subprocess_popen,
    ):
        res = _run_zerotts_build_script(
            build_script,
            target_dir,
            variant=ZeroTtsVariant.AVX2,
            native=False,
            publish_subdir=True,
        )
        assert res == fake_lib
        assert str(variant_dir) in recorded_cmd


@pytest.mark.parametrize("should_fail", [False, True])
def test_build_all_zerotts_variants_stale_variant_handling(
    tmp_path: Path, should_fail: bool
) -> None:
    """Test build_all_zerotts_variants cleans up or restores stale variant directories."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"
    stale_variant_dir = target_dir / ZeroTtsVariant.AVX512
    stale_variant_dir.mkdir(parents=True, exist_ok=True)
    stale_lib = stale_variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    stale_lib.write_bytes(b"OLD_STALE_AVX512")

    def fake_supported(variant: str) -> bool:
        return variant != ZeroTtsVariant.AVX512

    def fake_compile(
        clone_dest: Path, staging_dir: Path, variant: str = ZeroTtsVariant.COMPAT, **kwargs: object
    ) -> None:
        if should_fail:
            raise RuntimeError("Compilation crashed")
        (staging_dir / ZeroTtsFile.LIBZEROTTS_SO).write_bytes(f"FAKE_{variant}".encode())

    with (
        patch("platform.machine", return_value="x86_64"),
        patch("tools.build_zerotts._is_toolchain_variant_supported", side_effect=fake_supported),
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries", side_effect=fake_compile),
        patch("tools.build_zerotts._verify_library_loadable"),
    ):
        if should_fail:
            with pytest.raises(RuntimeError, match="Compilation crashed"):
                build_all_zerotts_variants(target_dir)
            assert stale_variant_dir.exists()
            assert stale_lib.exists()
            assert stale_lib.read_bytes() == b"OLD_STALE_AVX512"
        else:
            results = build_all_zerotts_variants(target_dir)
            assert ZeroTtsVariant.AVX512 not in results
            assert not stale_variant_dir.exists()
            assert not stale_lib.exists()


def test_build_all_zerotts_variants_restores_quarantine_on_system_exit(
    tmp_path: Path,
) -> None:
    """Test build_all_zerotts_variants restores quarantined dirs on SystemExit."""
    from tools.build_zerotts import build_all_zerotts_variants

    target_dir = tmp_path / "lib"
    stale_variant_dir = target_dir / ZeroTtsVariant.AVX512
    stale_variant_dir.mkdir(parents=True, exist_ok=True)
    stale_lib = stale_variant_dir / ZeroTtsFile.LIBZEROTTS_SO
    stale_lib.write_bytes(b"OLD_STALE_AVX512")

    def fake_supported(variant: str) -> bool:
        return variant != ZeroTtsVariant.AVX512

    def fake_compile(*args: object, **kwargs: object) -> None:
        raise SystemExit(143)

    with (
        patch("platform.machine", return_value="x86_64"),
        patch("tools.build_zerotts._is_toolchain_variant_supported", side_effect=fake_supported),
        patch("tools.build_zerotts._checkout_repo"),
        patch("tools.build_zerotts._compile_libraries", side_effect=fake_compile),
        pytest.raises(SystemExit) as exc_info,
    ):
        build_all_zerotts_variants(target_dir)

    assert exc_info.value.code == 143
    assert stale_variant_dir.exists()
    assert stale_lib.exists()
    assert stale_lib.read_bytes() == b"OLD_STALE_AVX512"


def test_get_variant_build_flags_definitions_and_compiler_options() -> None:
    """Test _get_variant_build_flags emits expected CMake definitions and g++ flags per variant."""
    from tools.build_zerotts import _get_variant_build_flags

    # COMPAT
    compat_cmake, compat_gxx = _get_variant_build_flags(ZeroTtsVariant.COMPAT)
    assert "-DGGML_SSE42=OFF" in compat_cmake
    assert "-DGGML_AVX=OFF" in compat_cmake
    assert "-DGGML_AVX2=OFF" in compat_cmake
    assert "-DGGML_AVX512=OFF" in compat_cmake
    assert not compat_gxx

    # SSE4
    sse4_cmake, sse4_gxx = _get_variant_build_flags(ZeroTtsVariant.SSE4)
    assert "-DGGML_SSE42=ON" in sse4_cmake
    assert "-DGGML_AVX=OFF" in sse4_cmake
    assert "-DGGML_AVX2=OFF" in sse4_cmake
    assert "-DGGML_AVX512=OFF" in sse4_cmake
    assert "-msse4.2" in sse4_gxx
    assert "-mpopcnt" in sse4_gxx

    # AVX
    avx_cmake, avx_gxx = _get_variant_build_flags(ZeroTtsVariant.AVX)
    assert "-DGGML_SSE42=ON" in avx_cmake
    assert "-DGGML_AVX=ON" in avx_cmake
    assert "-DGGML_AVX2=OFF" in avx_cmake
    assert "-DGGML_AVX512=OFF" in avx_cmake
    assert "-mavx" in avx_gxx

    # AVX2
    avx2_cmake, avx2_gxx = _get_variant_build_flags(ZeroTtsVariant.AVX2)
    assert "-DGGML_SSE42=ON" in avx2_cmake
    assert "-DGGML_AVX=ON" in avx2_cmake
    assert "-DGGML_AVX2=ON" in avx2_cmake
    assert "-DGGML_AVX512=OFF" in avx2_cmake
    assert "-mavx2" in avx2_gxx
    assert "-mfma" in avx2_gxx

    # AVX512
    avx512_cmake, avx512_gxx = _get_variant_build_flags(ZeroTtsVariant.AVX512)
    assert "-DGGML_SSE42=ON" in avx512_cmake
    assert "-DGGML_AVX=ON" in avx512_cmake
    assert "-DGGML_AVX2=ON" in avx512_cmake
    assert "-DGGML_AVX512=ON" in avx512_cmake
    assert "-mavx512f" in avx512_gxx
    assert "-mavx2" in avx512_gxx

    # ARM_DOTPROD
    arm_cmake, arm_gxx = _get_variant_build_flags(ZeroTtsVariant.ARM_DOTPROD)
    assert "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod" in arm_cmake
    assert "-march=armv8.2-a+dotprod" in arm_gxx

    # NATIVE on x86 vs ARM
    with patch("platform.machine", return_value="x86_64"):
        native_cmake, native_gxx = _get_variant_build_flags(ZeroTtsVariant.NATIVE)
        assert "-DGGML_NATIVE=ON" in native_cmake
        assert "-march=native" in native_gxx

    with patch("platform.machine", return_value="aarch64"):
        native_arm_cmake, native_arm_gxx = _get_variant_build_flags(ZeroTtsVariant.NATIVE)
        assert "-DGGML_NATIVE=ON" in native_arm_cmake
        assert "-mcpu=native" in native_arm_gxx


def test_get_darwin_cpu_flags_timeout() -> None:
    """Test get_darwin_cpu_flags suppresses subprocess.TimeoutExpired gracefully."""
    from wyoming_vietnamese.cpu import get_darwin_cpu_flags

    with (
        patch("wyoming_vietnamese.cpu.platform.system", return_value="Darwin"),
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="arm64"),
        patch(
            "wyoming_vietnamese.cpu.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=5.0),
        ),
    ):
        assert get_darwin_cpu_flags() == {"asimd"}

    with (
        patch("wyoming_vietnamese.cpu.platform.system", return_value="Darwin"),
        patch("wyoming_vietnamese.cpu.platform.machine", return_value="x86_64"),
        patch(
            "wyoming_vietnamese.cpu.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["sysctl"], timeout=5.0),
        ),
    ):
        assert get_darwin_cpu_flags() == set()


def test_run_zerotts_build_script_timeout(tmp_path: Path) -> None:
    """Test _run_zerotts_build_script raises RuntimeError on timeout and cleans workspace."""
    from wyoming_vietnamese.zerotts_engine import (
        _ZEROTTS_BUILD_TIMEOUT_SECONDS,
        _run_zerotts_build_script,
    )

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    stale_build_dir = target_dir / "_build_12345_temp123"
    stale_build_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_build_dir / "temp.o"
    stale_file.write_bytes(b"temp")

    unrelated_build_dir = target_dir / "_build_99999_temp"
    unrelated_build_dir.mkdir(parents=True, exist_ok=True)

    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
        cmd=["python"], timeout=_ZEROTTS_BUILD_TIMEOUT_SECONDS
    )
    mock_proc.wait.return_value = 0

    with (
        patch("wyoming_vietnamese.zerotts_engine.subprocess.Popen", return_value=mock_proc),
        patch("wyoming_vietnamese.zerotts_engine.os.killpg") as mock_killpg,
        pytest.raises(
            RuntimeError,
            match=f"ZeroTTS GGML build timed out after {_ZEROTTS_BUILD_TIMEOUT_SECONDS:g} seconds",
        ),
    ):
        _run_zerotts_build_script(
            build_script,
            target_dir,
            variant=ZeroTtsVariant.COMPAT,
            native=False,
        )

    assert call(12345, signal.SIGTERM) in mock_killpg.call_args_list
    assert not stale_build_dir.exists()
    assert unrelated_build_dir.exists()


def test_terminate_process_tree_escalates_to_sigkill() -> None:
    """Test _terminate_process_tree sends SIGTERM first and escalates to SIGKILL on timeout."""
    from wyoming_vietnamese.zerotts_engine import (
        _SIGKILL_GRACE_PERIOD_SECONDS,
        _terminate_process_tree,
    )

    mock_proc = Mock()
    mock_proc.pid = 99999
    mock_proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd=["proc"], timeout=_SIGKILL_GRACE_PERIOD_SECONDS),
        0,
    ]

    with (
        patch("wyoming_vietnamese.zerotts_engine.os.killpg") as mock_killpg,
        patch("wyoming_vietnamese.zerotts_engine._is_process_group_alive", return_value=True),
    ):
        _terminate_process_tree(mock_proc)
        assert mock_killpg.call_args_list == [
            call(99999, signal.SIGTERM),
            call(99999, signal.SIGKILL),
        ]


def test_clean_zerotts_build_workspace(tmp_path: Path) -> None:
    """Test _clean_zerotts_build_workspace removes transient files and restores rollback copies."""
    from wyoming_vietnamese.zerotts_engine import _clean_zerotts_build_workspace

    # Scratch directories for current build PID (123)
    build_dir = tmp_path / "_build_123_temp456"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "obj.o").write_bytes(b"data")

    build_all_dir = tmp_path / "_build_all_123_temp789"
    build_all_dir.mkdir(parents=True, exist_ok=True)

    # Scratch directory for concurrent build PID (999) - must NOT be touched
    unrelated_build_dir = tmp_path / "_build_999_temp"
    unrelated_build_dir.mkdir(parents=True, exist_ok=True)

    # Temporary publish files for PID 123 vs PID 999
    tmp_file_target = tmp_path / ".libzerotts.so.tmp.123"
    tmp_file_target.write_bytes(b"tmp_123")
    tmp_file_other = tmp_path / ".libzerotts.so.tmp.999"
    tmp_file_other.write_bytes(b"tmp_999")

    # Rollback files for PID 123 where destination is missing (should be restored)
    bak_file_missing = tmp_path / ".libzerotts.so.bak.123"
    bak_file_missing.write_bytes(b"PREVIOUS_VALID_LIB")

    # Rollback directory for PID 123 where destination is missing (should be restored)
    stale_dir_missing = tmp_path / ".avx2.stale.123"
    stale_dir_missing.mkdir(parents=True, exist_ok=True)
    (stale_dir_missing / "libzerotts.so").write_bytes(b"AVX2_LIB")

    # Rollback file for PID 999 from another concurrent build (must NOT be touched)
    bak_file_other = tmp_path / ".libggml.so.bak.999"
    bak_file_other.write_bytes(b"OTHER_BAK")

    # Regular unmanaged file (must NOT be touched)
    other_file = tmp_path / "README.txt"
    other_file.write_text("keep me", encoding="utf-8")

    _clean_zerotts_build_workspace(tmp_path, pid=123)

    # Scratch directories for PID 123 deleted, PID 999 preserved
    assert not build_dir.exists()
    assert not build_all_dir.exists()
    assert unrelated_build_dir.exists()

    # PID 123 tmp deleted, PID 999 tmp preserved
    assert not tmp_file_target.exists()
    assert tmp_file_other.exists()

    # PID 123 missing files restored to their original names
    restored_lib = tmp_path / ZeroTtsFile.LIBZEROTTS_SO
    assert restored_lib.exists()
    assert restored_lib.read_bytes() == b"PREVIOUS_VALID_LIB"
    assert not bak_file_missing.exists()

    restored_dir = tmp_path / ZeroTtsVariant.AVX2
    assert restored_dir.exists()
    assert (restored_dir / "libzerotts.so").read_bytes() == b"AVX2_LIB"
    assert not stale_dir_missing.exists()

    # PID 999 rollback and other files preserved
    assert bak_file_other.exists()
    assert other_file.exists()


def test_run_zerotts_build_script_failure(tmp_path: Path) -> None:
    """Test _run_zerotts_build_script raises RuntimeError on non-zero returncode."""
    from wyoming_vietnamese.zerotts_engine import _run_zerotts_build_script

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    mock_proc = Mock()
    mock_proc.pid = 12345
    mock_proc.communicate.return_value = ("", "compiler error: syntax error")
    mock_proc.returncode = 1

    with (
        patch("wyoming_vietnamese.zerotts_engine.subprocess.Popen", return_value=mock_proc),
        pytest.raises(
            RuntimeError,
            match="Failed to build ZeroTTS GGML shared library: compiler error",
        ),
    ):
        _run_zerotts_build_script(
            build_script,
            target_dir,
            variant=ZeroTtsVariant.COMPAT,
            native=False,
        )


def test_run_zerotts_build_script_invalid_variant(tmp_path: Path) -> None:
    """Test _run_zerotts_build_script validates variant before execution."""
    from wyoming_vietnamese.zerotts_engine import _run_zerotts_build_script

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid ZeroTTS variant"):
        _run_zerotts_build_script(
            build_script,
            tmp_path / "lib",
            variant="invalid_variant_123",
            native=False,
        )


def test_checkout_repo_timeout(tmp_path: Path) -> None:
    """Test _checkout_repo passes network and local timeouts to git commands."""
    from tools.build_zerotts import (
        _GIT_LOCAL_TIMEOUT_SECONDS,
        _GIT_NETWORK_TIMEOUT_SECONDS,
        _checkout_repo,
    )

    recorded_timeouts: list[object] = []

    def fake_run(_cmd: list[str], **kwargs: object) -> Mock:
        recorded_timeouts.append(kwargs.get("timeout"))
        return Mock(returncode=0)

    with patch("tools.build_zerotts.subprocess.run", side_effect=fake_run):
        _checkout_repo(
            tmp_path / "dest",
            "https://github.com/zeroweight-ai/ZeroTTS.git",
            "319c3e07c8575c1e38617f07c773ec76a780a1f1",
        )

    assert recorded_timeouts == [
        _GIT_NETWORK_TIMEOUT_SECONDS,
        _GIT_LOCAL_TIMEOUT_SECONDS,
        _GIT_NETWORK_TIMEOUT_SECONDS,
    ]


def test_is_toolchain_variant_supported_timeout() -> None:
    """Test _is_toolchain_variant_supported returns False when g++ times out."""
    from tools.build_zerotts import _is_toolchain_variant_supported

    with patch(
        "tools.build_zerotts.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["g++"], timeout=10.0),
    ):
        assert not _is_toolchain_variant_supported("avx2")


def test_format_process_error_with_timeout_expired() -> None:
    """Test _format_process_error formats subprocess.TimeoutExpired properly."""
    from tools.build_zerotts import _format_process_error

    err = subprocess.TimeoutExpired(
        cmd=["git", "clone"],
        timeout=300.0,
        output=b"cloning stdout\n",
        stderr=b"cloning stderr\n",
    )
    msg = _format_process_error(err)
    assert "timed out after 300" in msg
    assert "cloning stdout" in msg
    assert "cloning stderr" in msg


def test_run_zerotts_build_script_timeout_windows(tmp_path: Path) -> None:
    """Test _run_zerotts_build_script handles timeout on Windows using taskkill."""
    from wyoming_vietnamese.zerotts_engine import (
        _ZEROTTS_BUILD_TIMEOUT_SECONDS,
        _run_zerotts_build_script,
    )

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    stale_build_dir = target_dir / "_build_54321_temp"
    stale_build_dir.mkdir(parents=True, exist_ok=True)
    unrelated_build_dir = target_dir / "_build_99999_temp"
    unrelated_build_dir.mkdir(parents=True, exist_ok=True)

    mock_proc = Mock()
    mock_proc.pid = 54321
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
        cmd=["python"], timeout=_ZEROTTS_BUILD_TIMEOUT_SECONDS
    )
    mock_proc.wait.return_value = 0

    taskkill_calls: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        taskkill_calls.append(cmd)
        return Mock(returncode=0)

    with (
        patch("wyoming_vietnamese.zerotts_engine.platform.system", return_value="Windows"),
        patch(
            "wyoming_vietnamese.zerotts_engine.hasattr",
            side_effect=lambda obj, attr: False if attr == "killpg" else hasattr(obj, attr),
        ),
        patch("wyoming_vietnamese.zerotts_engine.subprocess.run", side_effect=fake_subprocess_run),
        patch("wyoming_vietnamese.zerotts_engine.subprocess.Popen", return_value=mock_proc),
        pytest.raises(
            RuntimeError,
            match=f"ZeroTTS GGML build timed out after {_ZEROTTS_BUILD_TIMEOUT_SECONDS:g} seconds",
        ),
    ):
        _run_zerotts_build_script(
            build_script,
            target_dir,
            variant=ZeroTtsVariant.COMPAT,
            native=False,
        )

    assert any("taskkill" in cmd[0] and "54321" in cmd for cmd in taskkill_calls)
    assert not stale_build_dir.exists()
    assert unrelated_build_dir.exists()


def test_abort_process_termination_error_does_not_skip_cleanup_or_timeout(
    tmp_path: Path,
) -> None:
    """Test process termination errors do not skip workspace cleanup or timeout error."""
    from wyoming_vietnamese.zerotts_engine import (
        _ZEROTTS_BUILD_TIMEOUT_SECONDS,
        _run_zerotts_build_script,
    )

    build_script = tmp_path / "build_zerotts.py"
    build_script.write_text("# dummy", encoding="utf-8")
    target_dir = tmp_path / "lib"
    target_dir.mkdir(parents=True, exist_ok=True)

    stale_build_dir = target_dir / "_build_11111_temp"
    stale_build_dir.mkdir(parents=True, exist_ok=True)

    mock_proc = Mock()
    mock_proc.pid = 11111
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
        cmd=["python"], timeout=_ZEROTTS_BUILD_TIMEOUT_SECONDS
    )

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine._terminate_process_tree",
            side_effect=RuntimeError("kill failed"),
        ),
        patch("wyoming_vietnamese.zerotts_engine.subprocess.Popen", return_value=mock_proc),
        pytest.raises(
            RuntimeError,
            match=f"ZeroTTS GGML build timed out after {_ZEROTTS_BUILD_TIMEOUT_SECONDS:g} seconds",
        ),
    ):
        _run_zerotts_build_script(
            build_script,
            target_dir,
            variant=ZeroTtsVariant.COMPAT,
            native=False,
        )

    assert not stale_build_dir.exists()


def test_terminate_process_tree_descendants_alive_even_if_child_exited() -> None:
    """Test _terminate_process_tree sends SIGKILL when descendants remain even if child exited."""
    from wyoming_vietnamese.zerotts_engine import _terminate_process_tree

    mock_proc = Mock()
    mock_proc.pid = 88888
    mock_proc.wait.return_value = 0

    killpg_calls: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))

    with (
        patch("wyoming_vietnamese.zerotts_engine.os.killpg", side_effect=fake_killpg),
        patch("wyoming_vietnamese.zerotts_engine._is_process_group_alive", return_value=True),
    ):
        _terminate_process_tree(mock_proc)

    assert (88888, signal.SIGTERM) in killpg_calls
    assert (88888, signal.SIGKILL) in killpg_calls


def test_terminate_process_tree_descendants_dead_no_sigkill() -> None:
    """Test _terminate_process_tree does not send SIGKILL if no descendants remain."""
    from wyoming_vietnamese.zerotts_engine import _terminate_process_tree

    mock_proc = Mock()
    mock_proc.pid = 88888
    mock_proc.wait.return_value = 0

    killpg_calls: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))

    with (
        patch("wyoming_vietnamese.zerotts_engine.os.killpg", side_effect=fake_killpg),
        patch("wyoming_vietnamese.zerotts_engine._is_process_group_alive", return_value=False),
    ):
        _terminate_process_tree(mock_proc)

    assert (88888, signal.SIGTERM) in killpg_calls
    assert (88888, signal.SIGKILL) not in killpg_calls


def test_is_process_group_alive() -> None:
    """Test _is_process_group_alive returns True when group exists and False when not found."""
    from wyoming_vietnamese.zerotts_engine import _is_process_group_alive

    with patch("wyoming_vietnamese.zerotts_engine.os.killpg", return_value=None):
        assert _is_process_group_alive(12345) is True

    with patch("wyoming_vietnamese.zerotts_engine.os.killpg", side_effect=ProcessLookupError):
        assert _is_process_group_alive(12345) is False

    with patch("wyoming_vietnamese.zerotts_engine.os.killpg", side_effect=PermissionError):
        assert _is_process_group_alive(12345) is True


def test_run_cmake_build_success() -> None:
    """Test _run_cmake_build starts CMake in new session and waits for completion."""
    from tools.build_zerotts import _run_cmake_build

    mock_proc = Mock()
    mock_proc.communicate.return_value = (b"Built target", b"")
    mock_proc.returncode = 0

    with patch("tools.build_zerotts.subprocess.Popen", return_value=mock_proc) as mock_popen:
        _run_cmake_build(["cmake", "--build", "/tmp/build"], timeout=60.0)
        mock_popen.assert_called_once()
        _, kwargs = mock_popen.call_args
        assert kwargs.get("start_new_session") is True


def test_run_cmake_build_adopts_managed_session() -> None:
    """Test _run_cmake_build starts new session even when running under managed session."""
    from tools.build_zerotts import _run_cmake_build

    mock_proc = Mock()
    mock_proc.communicate.return_value = (b"Built target", b"")
    mock_proc.returncode = 0

    with (
        patch("tools.build_zerotts.os.getpgrp", return_value=12345),
        patch("tools.build_zerotts.os.getpid", return_value=12345),
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        _run_cmake_build(["cmake", "--build", "/tmp/build"], timeout=60.0)
        mock_popen.assert_called_once()
        _, kwargs = mock_popen.call_args
        assert kwargs.get("start_new_session") is True


def test_run_cmake_build_called_process_error() -> None:
    """Test _run_cmake_build raises CalledProcessError when returncode is non-zero."""
    from tools.build_zerotts import _run_cmake_build

    mock_proc = Mock()
    mock_proc.communicate.return_value = (b"", b"Error: build failed")
    mock_proc.returncode = 2

    with (
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_proc),
        pytest.raises(subprocess.CalledProcessError) as exc_info,
    ):
        _run_cmake_build(["cmake", "--build", "/tmp/build"], timeout=60.0)

    assert exc_info.value.returncode == 2
    assert exc_info.value.stderr == b"Error: build failed"


def test_run_cmake_build_timeout_terminates_group() -> None:
    """Test _run_cmake_build terminates process group on timeout and raises TimeoutExpired."""
    from tools.build_zerotts import _run_cmake_build

    mock_proc = Mock()
    mock_proc.pid = 77777
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
        cmd=["cmake", "--build"],
        timeout=10.0,
        output=b"partial build stdout",
        stderr=b"partial build stderr",
    )
    mock_proc.wait.return_value = 0

    killpg_calls: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))

    with (
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_proc),
        patch("tools.build_zerotts.os.killpg", side_effect=fake_killpg),
        patch("tools.build_zerotts._is_cmake_process_group_alive", return_value=True),
        pytest.raises(subprocess.TimeoutExpired) as exc_info,
    ):
        _run_cmake_build(["cmake", "--build", "/tmp/build"], timeout=10.0)

    assert (77777, signal.SIGTERM) in killpg_calls
    assert (77777, signal.SIGKILL) in killpg_calls
    assert exc_info.value.output == b"partial build stdout"
    assert exc_info.value.stderr == b"partial build stderr"


def test_run_cmake_build_timeout_bounded_communicate_captures_trailing_output() -> None:
    """Test _run_cmake_build captures trailing output via bounded communicate on timeout."""
    from tools.build_zerotts import _run_cmake_build

    mock_proc = Mock()
    mock_proc.pid = 88888
    mock_proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["cmake", "--build"], timeout=10.0),
        (b"trailing stdout", b"trailing stderr"),
    ]
    mock_proc.wait.return_value = 0

    with (
        patch("tools.build_zerotts.subprocess.Popen", return_value=mock_proc),
        patch("tools.build_zerotts.os.killpg"),
        patch("tools.build_zerotts._is_cmake_process_group_alive", return_value=False),
        pytest.raises(subprocess.TimeoutExpired) as exc_info,
    ):
        _run_cmake_build(["cmake", "--build", "/tmp/build"], timeout=10.0)

    assert exc_info.value.output == b"trailing stdout"
    assert exc_info.value.stderr == b"trailing stderr"


def test_terminate_cmake_process_tree_windows() -> None:
    """Test _terminate_cmake_process_tree uses taskkill on Windows without calling os.killpg."""
    from tools.build_zerotts import _terminate_cmake_process_tree

    mock_proc = Mock()
    mock_proc.pid = 66666
    mock_proc.wait.return_value = 0

    taskkill_cmds: list[list[str]] = []

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        taskkill_cmds.append(cmd)
        return Mock(returncode=0)

    with (
        patch("tools.build_zerotts.platform.system", return_value="Windows"),
        patch(
            "tools.build_zerotts.hasattr",
            side_effect=lambda obj, attr: False if attr == "killpg" else hasattr(obj, attr),
        ),
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
    ):
        _terminate_cmake_process_tree(mock_proc)

    assert any("taskkill" in cmd[0] and "66666" in cmd for cmd in taskkill_cmds)


def test_is_cmake_process_group_alive() -> None:
    """Test _is_cmake_process_group_alive returns True when group exists and False when absent."""
    from tools.build_zerotts import _is_cmake_process_group_alive

    with patch("tools.build_zerotts.os.killpg", return_value=None):
        assert _is_cmake_process_group_alive(12345) is True

    with patch("tools.build_zerotts.os.killpg", side_effect=ProcessLookupError):
        assert _is_cmake_process_group_alive(12345) is False

    with patch("tools.build_zerotts.os.killpg", side_effect=PermissionError):
        assert _is_cmake_process_group_alive(12345) is True


def test_private_final_timeouts() -> None:
    """Test private final timeouts in build and engine modules."""
    from tools.build_zerotts import (
        _MAX_CMAKE_TEARDOWN_SECONDS as BUILD_MAX_TEARDOWN,
    )
    from tools.build_zerotts import (
        _PROCESS_TERMINATION_TIMEOUT_SECONDS as BUILD_TERM_TIMEOUT,
    )
    from wyoming_vietnamese.zerotts_engine import (
        _PROCESS_TERMINATION_TIMEOUT_SECONDS as ENGINE_TERM_TIMEOUT,
    )
    from wyoming_vietnamese.zerotts_engine import (
        _SIGKILL_GRACE_PERIOD_SECONDS as ENGINE_SIGKILL_GRACE_PERIOD,
    )
    from wyoming_vietnamese.zerotts_engine import (
        _ZEROTTS_BUILD_TIMEOUT_SECONDS as ENGINE_BUILD_TIMEOUT,
    )

    assert BUILD_TERM_TIMEOUT == 5.0
    assert ENGINE_TERM_TIMEOUT == 5.0
    assert ENGINE_BUILD_TIMEOUT == 1800.0
    assert BUILD_MAX_TEARDOWN == 10.0
    assert ENGINE_SIGKILL_GRACE_PERIOD == 15.0
    assert ENGINE_SIGKILL_GRACE_PERIOD > BUILD_MAX_TEARDOWN


def test_build_zerotts_sigterm_handler() -> None:
    """Test _handle_sigterm tears down active CMake process and exits."""
    import tools.build_zerotts as bz

    mock_proc = Mock()
    mock_proc.pid = 98765
    mock_proc.wait.return_value = 0

    with (
        patch.object(bz, "_ACTIVE_CMAKE_PROC", mock_proc),
        patch("tools.build_zerotts._terminate_cmake_process_tree") as mock_term,
        pytest.raises(SystemExit) as exc_info,
    ):
        bz._handle_sigterm(signal.SIGTERM, None)

    mock_term.assert_called_once_with(mock_proc)
    assert exc_info.value.code == 128 + signal.SIGTERM


def test_build_zerotts_sigterm_handler_no_active_proc() -> None:
    """Test _handle_sigterm exits cleanly when no CMake process is active."""
    import tools.build_zerotts as bz

    with (
        patch.object(bz, "_ACTIVE_CMAKE_PROC", None),
        patch("tools.build_zerotts._terminate_cmake_process_tree") as mock_term,
        pytest.raises(SystemExit) as exc_info,
    ):
        bz._handle_sigterm(signal.SIGTERM, None)

    mock_term.assert_not_called()
    assert exc_info.value.code == 128 + signal.SIGTERM


def test_build_zerotts_handle_build_signals_context_manager() -> None:
    """Test _handle_build_signals registers and restores signal handlers."""
    from tools.build_zerotts import _handle_build_signals, _handle_sigterm

    original_handler = signal.getsignal(signal.SIGTERM)
    with _handle_build_signals():
        current_handler = signal.getsignal(signal.SIGTERM)
        assert current_handler is _handle_sigterm

    restored_handler = signal.getsignal(signal.SIGTERM)
    assert restored_handler == original_handler
