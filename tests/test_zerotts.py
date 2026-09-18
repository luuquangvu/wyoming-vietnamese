"""Unit and integration tests for ZeroTTS GGML runtime engine."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from wyoming_vietnamese.const import (
    ZEROTTS_SOURCE_COMMIT,
    TtsEngine,
    ZeroTtsAudio,
    ZeroTtsFile,
)
from wyoming_vietnamese.tts import warm_up_tts
from wyoming_vietnamese.tts_model import ZEROTTS_VOICES
from wyoming_vietnamese.zerotts_engine import (
    ZeroTtsGgmlEngine,
    _build_zerotts_ggml_lib,
    _ZeroTtsHParams,
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

    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
        patch("tools.build_zerotts.ctypes.CDLL") as mock_cdll,
    ):
        result = _build_zerotts_ggml_lib(target_dir, native=native)
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
            assert "-DGGML_AVX512=OFF" in cmake_configure_cmd
        assert "-DCMAKE_BUILD_RPATH=$ORIGIN" in cmake_configure_cmd
        assert "-DCMAKE_INSTALL_RPATH=$ORIGIN" in cmake_configure_cmd
        mock_cdll.assert_called_once()
        verified_path = Path(mock_cdll.call_args[0][0])
        assert verified_path.name == ZeroTtsFile.LIBZEROTTS_SO
        assert "stage" in verified_path.parts


def test_build_zerotts_ggml_lib_loadability_failure(tmp_path: Path) -> None:
    """Test _build_zerotts_ggml_lib raises RuntimeError when runtime loadability check fails."""
    target_dir = tmp_path / "lib"

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> Mock:
        if "g++" in cmd[0]:
            out_file = Path(cmd[cmd.index("-o") + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(b"ELF_FAKE")
        return Mock(returncode=0)

    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
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

    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
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

    with (
        patch(
            "wyoming_vietnamese.zerotts_engine._build_zerotts_ggml_lib",
            side_effect=lambda *_args, **_kwargs: _build_zerotts_ggml_lib(target_dir),
        ),
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
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
    with (
        patch("tools.build_zerotts.subprocess.run", return_value=Mock(returncode=0)),
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


def test_build_zerotts_cli_main_failure(tmp_path: Path) -> None:
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

    with (
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

    with (
        patch("tools.build_zerotts.subprocess.run", side_effect=fake_subprocess_run),
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

    assert copied_rel.is_file()
    assert copied_rel.read_bytes() == b"GGML_CPU_BYTES"
    assert copied_rel.resolve().parent == target_dir.resolve()

    assert copied_abs.is_file()
    assert copied_abs.read_bytes() == b"GGML_CPU_BYTES"
    assert copied_abs.resolve().parent == target_dir.resolve()


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
