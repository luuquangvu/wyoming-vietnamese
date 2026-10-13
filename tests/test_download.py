"""Model synchronization and asset structuring tests."""

from __future__ import annotations

import errno
import json
import os
import struct
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import numpy as np
import pytest

from wyoming_vietnamese import download as download_module
from wyoming_vietnamese.const import (
    DEFAULT_STT_ENGINE,
    STT_DIR,
    TTS_DIR,
    VIETNAMESE_LANGUAGE,
    ZEROTTS_REPO_ID,
    NghiTtsFile,
    SttEngine,
    TtsEngine,
    ZeroTtsDirectory,
    ZeroTtsFile,
    ZeroTtsVariant,
)
from wyoming_vietnamese.download import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUF_MAGIC,
    GGUF_VERSION,
    ModelDownloadResult,
    _copy_or_link,
    _download_verified_file,
    _encode_onnx_metadata,
    _generate_nghitts_tokens,
    _generate_tokens_from_bpe,
    _model_structure_lock,
    _nghitts_sherpa_metadata,
    _read_varint,
    _skip_protobuf_field,
    _structure_nghitts_files,
    _structure_stt_files,
    _structure_zerotts_files,
    _sync_nghitts_model,
    _sync_repo,
    convert_zerotts_q8_to_q4_hybrid,
    download_models,
    is_attention_tensor,
    is_ffn_tensor,
    is_pinned_q8_tensor,
    quantize_row_q8_0_to_q4_0,
    setup_hf_environment,
)
from wyoming_vietnamese.stt_model import (
    GIPFORMER_MODEL,
    ZIPFORMER_MODEL,
    SttArtifact,
    SttModelSpec,
    get_stt_model,
)
from wyoming_vietnamese.tts_model import (
    DEFAULT_NGHITTS_VOICE,
    DEFAULT_ZEROTTS_VOICE,
    NghiTtsArtifact,
    NghiTtsVoiceSpec,
    ZeroTtsArtifact,
    ZeroTtsModelSpec,
    ZeroTtsVoiceSpec,
    get_voice,
    resolve_zerotts_model,
)


def _varint(value: int) -> bytes:
    """Build test support for  varint."""
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _sentencepiece_model(*pieces: str) -> bytes:
    """Build test support for  sentencepiece model."""
    model = bytearray(b"\x10\x01")
    for piece in pieces:
        text = piece.encode()
        entry = b"\x0a" + _varint(len(text)) + text + b"\x15\0\0\0\0"
        model.extend(b"\x0a" + _varint(len(entry)) + entry)
    return bytes(model)


def _make_stt_snapshot(directory: Path) -> SttModelSpec:
    """Build an STT snapshot and the pinned spec that exactly describes it."""
    directory.mkdir(parents=True, exist_ok=True)
    graphs: list[SttArtifact] = []
    for part in ("encoder", "decoder", "joiner"):
        content = part.encode()
        remote_name = f"{part}-epoch-1-avg-1.int8.onnx"
        (directory / remote_name).write_bytes(content)
        graphs.append(
            SttArtifact(
                remote_name,
                f"{part}.onnx",
                sha256(content).hexdigest(),
            )
        )
    bpe = _sentencepiece_model("<unk>", "xin")
    (directory / "bpe.model").write_bytes(bpe)
    return SttModelSpec(
        repo="owner/stt",
        revision="0123456789abcdef0123456789abcdef01234567",
        graphs=tuple(graphs),
        tokenizer=SttArtifact("bpe.model", "bpe.model", sha256(bpe).hexdigest()),
    )


def _pin_stt_model(monkeypatch: pytest.MonkeyPatch, spec: SttModelSpec) -> None:
    """Point the downloader at a test model spec instead of the shipped one."""
    monkeypatch.setattr("wyoming_vietnamese.download.DEFAULT_STT_MODEL", spec)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.get_stt_model",
        lambda engine=DEFAULT_STT_ENGINE: (
            spec if engine in (SttEngine.ZIPFORMER, DEFAULT_STT_ENGINE) else get_stt_model(engine)
        ),
    )


def _make_nghitts_snapshot(directory: Path) -> None:
    """Build a minimal raw NghiTTS voice snapshot."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / NghiTtsFile.MODEL).write_bytes(b"onnx")
    (directory / NghiTtsFile.CONFIG).write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": VIETNAMESE_LANGUAGE},
                "num_speakers": 1,
                "phoneme_id_map": {"_": [0], " ": [1], "a": [2]},
                "phoneme_type": "espeak",
            }
        ),
        encoding="utf-8",
    )


def test_setup_hf_environment_uses_hf_home_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test setup uses HF_HOME and enables offline mode without a Hub cache override."""
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(name, raising=False)

    setup_hf_environment(tmp_path, offline=True)
    resolved = str(tmp_path.resolve())
    assert os.environ["HF_HOME"] == resolved
    assert "HF_HUB_CACHE" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_sync_repo_offline_and_online_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sync repo offline and online paths."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    snapshot = tmp_path / "snapshot"
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert _sync_repo("owner/model", True, revision="rev") == snapshot
    assert download.call_args.kwargs["local_files_only"] is True
    assert "cache_dir" not in download.call_args.kwargs

    download.reset_mock(return_value=True)
    download.return_value = str(snapshot)
    assert _sync_repo("owner/model", False) == snapshot
    assert download.call_args.kwargs["local_files_only"] is False


def test_sync_repo_falls_back_to_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sync repo falls back to cache."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    snapshot = tmp_path / "snapshot"
    download = Mock(side_effect=[OSError("network"), str(snapshot)])
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert _sync_repo("owner/model", False) == snapshot
    assert download.call_args.kwargs["local_files_only"] is True


def test_download_verified_file_checks_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test direct model downloads are checksum verified."""
    content = b"verified model"
    monkeypatch.setattr(
        "wyoming_vietnamese.download.urlopen",
        lambda *_args, **_kwargs: BytesIO(content),
    )
    destination = tmp_path / NghiTtsFile.MODEL
    _download_verified_file(
        "https://models.example/model.onnx",
        destination,
        expected_sha256=sha256(content).hexdigest(),
    )
    assert destination.read_bytes() == content

    bad_sha = sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        _download_verified_file(
            "https://models.example/model.onnx",
            destination,
            expected_sha256=bad_sha,
        )


def test_download_verified_file_uses_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _download_verified_file uses _DOWNLOAD_TIMEOUT_SECONDS."""
    from wyoming_vietnamese.download import _DOWNLOAD_TIMEOUT_SECONDS

    content = b"verified model"
    recorded_kwargs: dict[str, object] = {}

    def fake_urlopen(_request: object, **kwargs: object) -> BytesIO:
        recorded_kwargs.update(kwargs)
        return BytesIO(content)

    monkeypatch.setattr("wyoming_vietnamese.download.urlopen", fake_urlopen)
    destination = tmp_path / NghiTtsFile.MODEL
    _download_verified_file(
        "https://models.example/model.onnx",
        destination,
        expected_sha256=sha256(content).hexdigest(),
    )
    assert recorded_kwargs.get("timeout") == _DOWNLOAD_TIMEOUT_SECONDS
    assert _DOWNLOAD_TIMEOUT_SECONDS == 60.0


@pytest.mark.parametrize("offline", [True, False])
def test_sync_repo_reports_unavailable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    offline: bool,
) -> None:
    """Test sync repo reports unavailable snapshot."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        Mock(side_effect=OSError("missing")),
    )
    expected_message = "repository" if offline else "Repository"
    with pytest.raises(RuntimeError, match=expected_message):
        _sync_repo("owner/model", offline)


def test_model_structure_lock_creates_lock_file(tmp_path: Path) -> None:
    """Test model structure lock creates lock file."""
    with _model_structure_lock(tmp_path):
        assert (tmp_path / ".structure.lock").is_file()


def test_bounded_varint_and_field_skipping() -> None:
    """Test bounded varint and field skipping."""
    assert _read_varint(b"\xac\x02", 0, 2) == (300, 2)
    assert _skip_protobuf_field(b"\x01", 0, 1, 0) == 1
    assert _skip_protobuf_field(b"12345678", 0, 8, 1) == 8
    assert _skip_protobuf_field(b"\x02ab", 0, 3, 2) == 3
    assert _skip_protobuf_field(b"1234", 0, 4, 5) == 4
    with pytest.raises(ValueError, match="varint"):
        _read_varint(b"\x80", 0, 1)
    with pytest.raises(ValueError, match="wire type"):
        _skip_protobuf_field(b"", 0, 0, 3)
    with pytest.raises(ValueError, match="exceeds"):
        _skip_protobuf_field(b"12", 0, 2, 1)


def test_generate_tokens_from_sentencepiece(tmp_path: Path) -> None:
    """Test generate tokens from sentencepiece."""
    model = tmp_path / "bpe.model"
    tokens = tmp_path / "tokens.txt"
    model.write_bytes(_sentencepiece_model("<unk>", "xin chào"))
    _generate_tokens_from_bpe(model, tokens)
    assert tokens.read_text(encoding="utf-8") == "<unk> 0\nxin chào 1\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_generate_nghitts_tokens_from_model_configuration(tmp_path: Path) -> None:
    """Test NghiTTS phoneme metadata is converted to Sherpa's ordered token table."""
    config = tmp_path / NghiTtsFile.CONFIG
    tokens = tmp_path / NghiTtsFile.TOKENS
    config.write_text(
        json.dumps({"phoneme_id_map": {"a": [2], "_": [0], " ": [1]}}),
        encoding="utf-8",
    )
    _generate_nghitts_tokens(config, tokens)
    assert tokens.read_text(encoding="utf-8") == "_ 0\n  1\na 2\n"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("phoneme_map", "message"),
    [
        ({}, "no phoneme_id_map"),
        ({"a": []}, "no token IDs"),
        ({"a": [True]}, "invalid token ID"),
        ({"a": [1]}, "contiguous"),
        ({"a": [0], "b": [0]}, "duplicated"),
    ],
)
def test_generate_nghitts_tokens_rejects_invalid_maps(
    tmp_path: Path,
    phoneme_map: dict[str, list[object]],
    message: str,
) -> None:
    """Test malformed NghiTTS metadata cannot create a partial token table."""
    config = tmp_path / NghiTtsFile.CONFIG
    config.write_text(json.dumps({"phoneme_id_map": phoneme_map}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _generate_nghitts_tokens(config, tmp_path / NghiTtsFile.TOKENS)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "no token pieces"),
        (b"\x80", "varint"),
        (b"\x0a\x05\x0a\x01a", "exceeds the model size"),
        (b"\x0a\x02\x10\x01", "no piece text"),
        (b"\x1b", "wire type"),
    ],
)
def test_generate_tokens_rejects_corrupt_models(
    tmp_path: Path, content: bytes, message: str
) -> None:
    """Test generate tokens rejects corrupt models."""
    model = tmp_path / "bpe.model"
    model.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        _generate_tokens_from_bpe(model, tmp_path / "tokens.txt")


def _write_source(tmp_path: Path, content: bytes = b"content") -> Path:
    """Create a source file with the given content for copy-or-link tests."""
    source = tmp_path / "source"
    source.write_bytes(content)
    return source


def test_copy_or_link_is_atomic_and_idempotent(tmp_path: Path) -> None:
    """Test copy or link is atomic and idempotent."""
    source = _write_source(tmp_path, b"one")
    destination = tmp_path / "nested" / "destination"
    _copy_or_link(source, destination)
    assert destination.read_bytes() == b"one"
    assert os.path.samefile(source, destination)
    _copy_or_link(source, destination)
    assert os.path.samefile(source, destination)

    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"different")
    _copy_or_link(replacement, destination)
    assert destination.read_bytes() == b"different"


def test_copy_or_link_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test copy or link falls back to copy."""
    source = _write_source(tmp_path)
    destination = tmp_path / "destination"
    monkeypatch.setattr(
        "wyoming_vietnamese.download.os.link",
        Mock(side_effect=OSError(errno.EXDEV, "cross-device")),
    )
    _copy_or_link(source, destination)
    assert destination.read_bytes() == b"content"
    assert not os.path.samefile(source, destination)


def test_copy_or_link_replaces_stale_directory(tmp_path: Path) -> None:
    """Test copy or link replaces stale directory."""
    source = _write_source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "stale").write_bytes(b"stale")
    _copy_or_link(source, destination)
    assert destination.is_file()


def test_copy_or_link_propagates_unexpected_link_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test copy or link propagates unexpected link error."""
    source = _write_source(tmp_path)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.os.link",
        Mock(side_effect=OSError(errno.EIO, "disk")),
    )
    with pytest.raises(OSError, match="disk"):
        _copy_or_link(source, tmp_path / "destination")


def test_structure_stt_copies_pinned_graphs_and_generates_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the pinned artifacts become stable local filenames plus generated tokens."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _pin_stt_model(monkeypatch, _make_stt_snapshot(snapshot))
    destination.mkdir()

    paths = _structure_stt_files(snapshot, destination)

    assert (destination / "encoder.onnx").read_bytes() == b"encoder"
    assert (destination / "decoder.onnx").read_bytes() == b"decoder"
    assert (destination / "joiner.onnx").read_bytes() == b"joiner"
    assert (destination / "tokens.txt").read_text(encoding="utf-8").startswith("<unk> 0")
    assert len(paths) == 4


def test_structure_stt_rejects_a_missing_pinned_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a snapshot missing any pinned file stops startup."""
    snapshot = tmp_path / "snapshot"
    spec = _make_stt_snapshot(snapshot)
    _pin_stt_model(monkeypatch, spec)
    (snapshot / spec.graphs[0].remote_name).unlink()

    with pytest.raises(FileNotFoundError, match="Pinned STT artifact is missing"):
        _structure_stt_files(snapshot, tmp_path / "model")


@pytest.mark.parametrize("tampered", [b"replaced-content", b"encode"])
def test_structure_stt_rejects_content_that_is_not_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered: bytes
) -> None:
    """Test changed upstream content is rejected instead of silently loaded."""
    snapshot = tmp_path / "snapshot"
    spec = _make_stt_snapshot(snapshot)
    _pin_stt_model(monkeypatch, spec)
    (snapshot / spec.graphs[0].remote_name).write_bytes(tampered)

    with pytest.raises(RuntimeError, match="does not match its recorded"):
        _structure_stt_files(snapshot, tmp_path / "model")


def test_structure_stt_rejects_a_tampered_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the SentencePiece source is verified before tokens are generated."""
    snapshot = tmp_path / "snapshot"
    spec = _make_stt_snapshot(snapshot)
    _pin_stt_model(monkeypatch, spec)
    (snapshot / spec.tokenizer.remote_name).write_bytes(b"not-a-sentencepiece-model")

    with pytest.raises(RuntimeError, match="does not match its recorded"):
        _structure_stt_files(snapshot, tmp_path / "model")


def test_structure_nghitts_files_converts_raw_nghitts_model(tmp_path: Path) -> None:
    """Test raw NghiTTS assets become a complete Sherpa-compatible local model."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _make_nghitts_snapshot(snapshot)
    paths = _structure_nghitts_files(snapshot, destination)
    converted_model = destination / NghiTtsFile.MODEL
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / NghiTtsFile.CONFIG))
    assert converted_model.read_bytes() == b"onnx" + metadata
    assert (destination / NghiTtsFile.CONFIG).is_file()
    assert (destination / NghiTtsFile.TOKENS).read_text(encoding="utf-8") == ("_ 0\n  1\na 2\n")
    assert len(paths) == 3

    (snapshot / NghiTtsFile.MODEL).unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete NghiTTS"):
        _structure_nghitts_files(snapshot, tmp_path / "incomplete")


def test_structure_nghitts_files_skips_rehashing_a_prepared_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a recorded preparation avoids re-digesting an unchanged model graph."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _make_nghitts_snapshot(snapshot)
    _structure_nghitts_files(snapshot, destination)
    prepared = (destination / NghiTtsFile.MODEL).read_bytes()

    digests: list[Path] = []
    original_digest = download_module._file_sha256

    def record_digest(path: Path) -> str:
        """Record every full-file digest requested during preparation."""
        digests.append(path)
        return original_digest(path)

    monkeypatch.setattr("wyoming_vietnamese.download._file_sha256", record_digest)

    _structure_nghitts_files(snapshot, destination)
    assert not digests
    assert (destination / NghiTtsFile.MODEL).read_bytes() == prepared

    (snapshot / NghiTtsFile.MODEL).write_bytes(b"onnx-v2")
    _structure_nghitts_files(snapshot, destination)
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / NghiTtsFile.CONFIG))
    assert (destination / NghiTtsFile.MODEL).read_bytes() == b"onnx-v2" + metadata


def test_structure_nghitts_files_rebuilds_after_a_corrupt_marker(tmp_path: Path) -> None:
    """Test an unreadable preparation marker falls back to digest verification."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _make_nghitts_snapshot(snapshot)
    _structure_nghitts_files(snapshot, destination)

    marker = download_module._prepared_marker_path(destination / NghiTtsFile.MODEL)
    marker.write_text("not json", encoding="utf-8")
    (destination / NghiTtsFile.MODEL).write_bytes(b"corrupted")

    _structure_nghitts_files(snapshot, destination)
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / NghiTtsFile.CONFIG))
    assert (destination / NghiTtsFile.MODEL).read_bytes() == b"onnx" + metadata
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["metadata_length"] == len(metadata)
    assert recorded["metadata_sha256"] == sha256(metadata).hexdigest()


def test_structure_nghitts_files_rebuilds_after_same_length_metadata_change(
    tmp_path: Path,
) -> None:
    """Test config metadata identity invalidates a prepared model even at equal length."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _make_nghitts_snapshot(snapshot)
    _structure_nghitts_files(snapshot, destination)
    prepared_model = destination / NghiTtsFile.MODEL
    original = prepared_model.read_bytes()

    config = snapshot / NghiTtsFile.CONFIG
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            f'"voice": "{VIETNAMESE_LANGUAGE}"', '"voice": "en"'
        ),
        encoding="utf-8",
    )
    _structure_nghitts_files(snapshot, destination)

    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / NghiTtsFile.CONFIG))
    assert len(prepared_model.read_bytes()) == len(original)
    assert prepared_model.read_bytes() == b"onnx" + metadata
    assert prepared_model.read_bytes() != original


def test_sync_nghitts_model_caches_pinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test encoded API names and verified offline reuse for a selected voice."""
    model = b"model"
    config = b"config"
    voice = NghiTtsVoiceSpec(
        "test-voice",
        "Ngọc (mới)",
        NghiTtsArtifact("Ngọc (mới).onnx", NghiTtsFile.MODEL, sha256(model).hexdigest()),
        NghiTtsArtifact(
            "Ngọc (mới).onnx.json",
            NghiTtsFile.CONFIG,
            sha256(config).hexdigest(),
        ),
    )
    download = Mock(side_effect=[BytesIO(model), BytesIO(config)])
    monkeypatch.setattr("wyoming_vietnamese.download.urlopen", download)

    snapshot = _sync_nghitts_model(
        tmp_path,
        False,
        voice,
        base_url="https://models.example/api/model",
    )
    assert (snapshot / NghiTtsFile.MODEL).read_bytes() == model
    assert (snapshot / NghiTtsFile.CONFIG).read_bytes() == config
    assert "%28m%E1%BB%9Bi%29.onnx" in download.call_args_list[0].args[0].full_url
    assert (
        _sync_nghitts_model(
            tmp_path,
            True,
            voice,
            base_url="https://models.example/api/model",
        )
        == snapshot
    )
    assert download.call_count == 2


def test_sync_nghitts_model_rejects_invalid_source_and_offline_miss(tmp_path: Path) -> None:
    """Test model synchronization rejects insecure sources and missing cache files."""
    with pytest.raises(ValueError, match="HTTPS"):
        _sync_nghitts_model(
            tmp_path, False, DEFAULT_NGHITTS_VOICE, base_url="http://models.example"
        )
    with pytest.raises(RuntimeError, match="Offline startup failed"):
        _sync_nghitts_model(
            tmp_path,
            True,
            DEFAULT_NGHITTS_VOICE,
            base_url="https://models.example",
        )


def test_download_models_structures_selected_nghitts_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download builds STT and one voice-specific NghiTTS directory."""
    stt = tmp_path / "stt-snapshot"
    tts = tmp_path / "tts-snapshot"
    spec = _make_stt_snapshot(stt)
    _pin_stt_model(monkeypatch, spec)
    _make_nghitts_snapshot(tts)
    sync = Mock(return_value=stt)
    sync_tts = Mock(return_value=tts)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", sync)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_nghitts_model", sync_tts)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())

    second_voice = get_voice("chieu-thanh")
    paths = download_models(
        tmp_path / "cache",
        tmp_path / "models",
        (DEFAULT_NGHITTS_VOICE, second_voice),
    )

    assert paths[STT_DIR] == tmp_path / "models" / STT_DIR / SttEngine.ZIPFORMER
    assert (paths[STT_DIR] / "encoder.onnx").is_file()
    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.NGHITTS
    assert (paths[TTS_DIR] / DEFAULT_NGHITTS_VOICE.id / NghiTtsFile.MODEL).is_file()
    assert (paths[TTS_DIR] / DEFAULT_NGHITTS_VOICE.id / NghiTtsFile.TOKENS).is_file()
    assert (paths[TTS_DIR] / second_voice.id / NghiTtsFile.MODEL).is_file()
    assert set(paths) == {STT_DIR, TTS_DIR}
    assert sync.call_args.args == (spec.repo, False)
    assert sync.call_args.kwargs["revision"] == spec.revision
    assert sync.call_args.kwargs["allow_patterns"] == spec.allow_patterns
    assert [item.args for item in sync_tts.call_args_list] == [
        (tmp_path / "cache", False, DEFAULT_NGHITTS_VOICE),
        (tmp_path / "cache", False, second_voice),
    ]


def _make_zerotts_snapshot(directory: Path) -> tuple[ZeroTtsModelSpec, ZeroTtsVoiceSpec]:
    """Build a ZeroTTS snapshot and pinned specs that match its files."""
    directory.mkdir(parents=True, exist_ok=True)
    gguf_content = b"GGUF_TEST_BYTES"
    gguf_dir = directory / ZeroTtsDirectory.GGUF
    gguf_dir.mkdir(parents=True, exist_ok=True)
    (gguf_dir / ZeroTtsFile.DEFAULT_GGUF_MODEL).write_bytes(gguf_content)

    config_content = b"{}"
    (directory / "config.json").write_bytes(config_content)

    tok_content = b'{"vocab": {}}'
    (directory / "tokenizer.json").write_bytes(tok_content)

    null_emb_content = b"EMB"
    (directory / "null_voice_emb.npy").write_bytes(null_emb_content)

    codec_dir = directory / "onnx" / "codec"
    codec_dir.mkdir(parents=True, exist_ok=True)
    codec_files = []
    for name in (
        "codec_browser_onnx_meta.json",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
        "moss_audio_tokenizer_decode_step.onnx",
    ):
        content = f"CODEC_{name}".encode()
        (codec_dir / name).write_bytes(content)
        codec_files.append(
            ZeroTtsArtifact(
                f"onnx/codec/{name}",
                f"onnx/codec/{name}",
                sha256(content).hexdigest(),
            )
        )

    voice_dir = directory / "voices" / DEFAULT_ZEROTTS_VOICE.id
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_content = b"VOICE_NPZ_TEST"
    (voice_dir / "voice.npz").write_bytes(voice_content)
    test_voice = ZeroTtsVoiceSpec(
        id=DEFAULT_ZEROTTS_VOICE.id,
        name=DEFAULT_ZEROTTS_VOICE.name,
        gender="nữ",
        description="test",
        artifact=ZeroTtsArtifact(
            f"voices/{DEFAULT_ZEROTTS_VOICE.id}/voice.npz",
            "voice.npz",
            sha256(voice_content).hexdigest(),
        ),
    )

    spec = ZeroTtsModelSpec(
        repo=ZEROTTS_REPO_ID,
        revision="92ca8651645d4733df56620b3aadc768a76f7c46",
        config=ZeroTtsArtifact("config.json", "config.json", sha256(config_content).hexdigest()),
        tokenizer=ZeroTtsArtifact(
            "tokenizer.json", "tokenizer.json", sha256(tok_content).hexdigest()
        ),
        null_voice_emb=ZeroTtsArtifact(
            "null_voice_emb.npy", "null_voice_emb.npy", sha256(null_emb_content).hexdigest()
        ),
        gguf_model=ZeroTtsArtifact(
            f"{ZeroTtsDirectory.GGUF}/{ZeroTtsFile.DEFAULT_GGUF_MODEL}",
            f"{ZeroTtsDirectory.GGUF}/{ZeroTtsFile.DEFAULT_GGUF_MODEL}",
            sha256(gguf_content).hexdigest(),
        ),
        codec_files=tuple(codec_files),
    )
    return spec, test_voice


def test_structure_zerotts_files_missing_gguf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises FileNotFoundError when GGUF is missing."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, _ = _make_zerotts_snapshot(snapshot_dir)
    gguf_file = snapshot_dir / ZeroTtsDirectory.GGUF / ZeroTtsFile.DEFAULT_GGUF_MODEL
    if gguf_file.is_file():
        gguf_file.unlink()
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    target_dir = tmp_path / TTS_DIR

    with pytest.raises(FileNotFoundError, match="Missing ZeroTTS GGUF model weights"):
        _structure_zerotts_files(snapshot_dir, target_dir)


def test_structure_zerotts_files_missing_artifact(tmp_path: Path) -> None:
    """Test _structure_zerotts_files raises FileNotFoundError when artifact is missing."""
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / TTS_DIR

    with pytest.raises(FileNotFoundError, match="Pinned ZeroTTS artifact is missing"):
        _structure_zerotts_files(snapshot_dir, target_dir)


def test_structure_zerotts_files_corrupted_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises RuntimeError when artifact digest fails verification."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, _ = _make_zerotts_snapshot(snapshot_dir)
    (snapshot_dir / "config.json").write_bytes(b"CORRUPTED_CONFIG")
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    target_dir = tmp_path / TTS_DIR
    with pytest.raises(RuntimeError, match="does not match its recorded SHA-256 digest"):
        _structure_zerotts_files(snapshot_dir, target_dir)


def test_download_models_structures_zerotts_quality_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures ZeroTTS GGUF weights and voice latents in quality mode."""
    stt = tmp_path / "stt-snapshot"
    spec = _make_stt_snapshot(stt)
    _pin_stt_model(monkeypatch, spec)

    tts_snapshot = tmp_path / "zerotts-snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(tts_snapshot)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))

    sync = Mock(side_effect=[stt, tts_snapshot])
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", sync)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())
    monkeypatch.setattr(
        "wyoming_vietnamese.tts_model.detect_cpu_variant",
        lambda: ZeroTtsVariant.AVX2,
    )

    paths = download_models(
        tmp_path / "cache",
        tmp_path / "models",
        (test_voice,),
        tts_engine=TtsEngine.ZEROTTS,
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.DEFAULT_GGUF_MODEL).is_file()
    assert (paths[TTS_DIR] / "config.json").is_file()
    assert (paths[TTS_DIR] / "tokenizer.json").is_file()
    assert (paths[TTS_DIR] / "null_voice_emb.npy").is_file()
    assert (paths[TTS_DIR] / "onnx" / "codec" / "codec_browser_onnx_meta.json").is_file()
    assert (paths[TTS_DIR] / "onnx" / "codec" / "moss_audio_tokenizer_decode_full.onnx").is_file()
    assert (paths[TTS_DIR] / "voices" / test_voice.id / "voice.npz").is_file()

    assert sync.call_count == 2
    assert sync.call_args_list[1].args[0] == ZEROTTS_REPO_ID
    assert sync.call_args_list[1].kwargs["revision"] == tts_spec.revision
    assert sync.call_args_list[1].kwargs["allow_patterns"] == tts_spec.allow_patterns


def _make_hybrid_model_spec(base_spec: ZeroTtsModelSpec, sha256_digest: str) -> ZeroTtsModelSpec:
    """Create a hybrid ZeroTtsModelSpec from a base spec and custom sha256."""
    return ZeroTtsModelSpec(
        repo=base_spec.repo,
        revision=base_spec.revision,
        config=base_spec.config,
        tokenizer=base_spec.tokenizer,
        null_voice_emb=base_spec.null_voice_emb,
        gguf_model=ZeroTtsArtifact(
            f"gguf/{ZeroTtsFile.Q4_HYBRID_GGUF_MODEL}",
            f"gguf/{ZeroTtsFile.Q4_HYBRID_GGUF_MODEL}",
            sha256_digest,
        ),
        codec_files=base_spec.codec_files,
    )


def _make_spec_without_gguf(base_spec: ZeroTtsModelSpec) -> ZeroTtsModelSpec:
    """Create a ZeroTtsModelSpec without a gguf_model artifact."""
    return ZeroTtsModelSpec(
        repo=base_spec.repo,
        revision=base_spec.revision,
        config=base_spec.config,
        tokenizer=base_spec.tokenizer,
        null_voice_emb=base_spec.null_voice_emb,
        gguf_model=None,
        codec_files=base_spec.codec_files,
    )


def _setup_zerotts_download_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, ZeroTtsModelSpec, ZeroTtsVoiceSpec]:
    """Prepare STT and ZeroTTS snapshot directories and pinned models."""
    stt = tmp_path / "stt-snapshot"
    spec = _make_stt_snapshot(stt)
    _pin_stt_model(monkeypatch, spec)
    tts_snapshot = tmp_path / "zerotts-snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(tts_snapshot)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))
    return stt, tts_snapshot, tts_spec, test_voice


def _write_q4_hybrid_snapshot_artifact(
    tts_snapshot: Path, tts_spec: ZeroTtsModelSpec, content: bytes = b"GGUF_Q4_TEST_BYTES"
) -> ZeroTtsModelSpec:
    """Write Q4-Hybrid model to snapshot and return the corresponding spec."""
    gguf_dir = tts_snapshot / ZeroTtsDirectory.GGUF
    (gguf_dir / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL).write_bytes(content)
    return _make_hybrid_model_spec(tts_spec, sha256(content).hexdigest())


def _execute_zerotts_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stt: Path,
    tts_snapshot: Path,
    test_voice: ZeroTtsVoiceSpec,
    *,
    zerotts_quant: str = "auto",
) -> tuple[ModelDownloadResult, Mock]:
    """Execute download_models for ZeroTTS and return result paths and sync mock."""
    sync = Mock(side_effect=[stt, tts_snapshot])
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", sync)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())
    paths = download_models(
        tmp_path / "cache",
        tmp_path / "models",
        (test_voice,),
        tts_engine=TtsEngine.ZEROTTS,
        zerotts_quant=zerotts_quant,
    )
    return paths, sync


def test_download_models_structures_zerotts_q4_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures ZeroTTS Q4 quantized GGUF weights when requested."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    q4_spec = _write_q4_hybrid_snapshot_artifact(tts_snapshot, tts_spec)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant: q4_spec if quant and ("q4" in quant or "hybrid" in quant) else tts_spec,
    )
    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL).is_file()
    assert sync.call_count == 2
    assert sync.call_args_list[1].kwargs["allow_patterns"] == q4_spec.allow_patterns
    assert paths.zerotts_resolution is not None
    assert paths.zerotts_resolution.spec == q4_spec
    assert (
        paths.zerotts_resolution.gguf_path
        == paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )


def test_download_models_structures_zerotts_q8_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures ZeroTTS Q8 GGUF weights when explicitly requested."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: tts_spec,
    )
    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q8_0"
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.DEFAULT_GGUF_MODEL).is_file()
    assert sync.call_count == 2
    assert sync.call_args_list[1].kwargs["allow_patterns"] == tts_spec.allow_patterns
    assert paths.zerotts_resolution is not None
    assert paths.zerotts_resolution.spec == tts_spec
    assert (
        paths.zerotts_resolution.gguf_path
        == paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.DEFAULT_GGUF_MODEL
    )


def test_download_models_structures_zerotts_auto_edge_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures Q4-Hybrid model when auto is selected on edge CPU variants."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    q4_spec = _write_q4_hybrid_snapshot_artifact(tts_snapshot, tts_spec)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec if quant and ("q4" in quant or "hybrid" in quant) else tts_spec,
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.tts_model.detect_cpu_variant",
        lambda: ZeroTtsVariant.SSE4,
    )
    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="auto"
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL).is_file()
    assert sync.call_count == 2
    assert sync.call_args_list[1].kwargs["allow_patterns"] == q4_spec.allow_patterns


def test_download_models_structures_zerotts_auto_avx_ignores_existing_local_q8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download on AVX edge variant structures Q4-Hybrid ignoring existing local Q8 file."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    q4_spec = _write_q4_hybrid_snapshot_artifact(tts_snapshot, tts_spec, b"GGUF_Q4_AVX_TEST_BYTES")
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec if quant and ("q4" in quant or "hybrid" in quant) else tts_spec,
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.tts_model.detect_cpu_variant",
        lambda: ZeroTtsVariant.AVX,
    )

    dest_gguf_dir = tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS / ZeroTtsDirectory.GGUF
    dest_gguf_dir.mkdir(parents=True, exist_ok=True)
    existing_q8 = dest_gguf_dir / ZeroTtsFile.DEFAULT_GGUF_MODEL
    existing_q8.write_bytes(b"EXISTING_LOCAL_Q8_MODEL")

    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="auto"
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL).is_file()
    assert sync.call_count == 2
    assert sync.call_args_list[1].kwargs["allow_patterns"] == q4_spec.allow_patterns


def test_download_models_structures_zerotts_q4_converts_from_q8_when_hybrid_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download converts Q8 weights to Q4-Hybrid when hybrid model is absent from snapshot."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    converted_content = b"GGUF_CONVERTED_Q4_HYBRID_BYTES"
    q4_spec = _make_hybrid_model_spec(tts_spec, sha256(converted_content).hexdigest())

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    def fake_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(converted_content)
        return dst

    monkeypatch.setattr("wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", fake_convert)

    paths, _sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
    )

    dst_hybrid = paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert dst_hybrid.is_file()
    assert dst_hybrid.read_bytes() == converted_content


def test_download_models_clears_stale_zerotts_hybrid_cache_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download clears stale or mismatched hybrid model cache and rebuilds from Q8."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    converted_content = b"FRESH_REBUILT_Q4_HYBRID_BYTES"
    q4_spec = _make_hybrid_model_spec(tts_spec, sha256(converted_content).hexdigest())

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    def fake_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(converted_content)
        return dst

    monkeypatch.setattr("wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", fake_convert)

    stale_dst = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    stale_dst.parent.mkdir(parents=True, exist_ok=True)
    stale_dst.write_bytes(b"STALE_CORRUPTED_CACHE_BYTES")

    paths, _sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
    )

    dst_hybrid = paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert dst_hybrid.is_file()
    assert dst_hybrid.read_bytes() == converted_content


def test_download_models_clears_stale_zerotts_hybrid_directory_cache_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download removes stale directory located at hybrid path and rebuilds as file."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    converted_content = b"FRESH_REBUILT_FROM_DIR_CACHE"
    q4_spec = _make_hybrid_model_spec(tts_spec, sha256(converted_content).hexdigest())

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    def fake_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(converted_content)
        return dst

    monkeypatch.setattr("wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", fake_convert)

    stale_dir = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "nested_dummy.txt").write_text("STALE_NESTED_CONTENT", encoding="utf-8")

    paths, _sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
    )

    dst_hybrid = paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert dst_hybrid.is_file()
    assert not dst_hybrid.is_dir()
    assert dst_hybrid.read_bytes() == converted_content


@pytest.mark.parametrize("target_is_directory", [True, False])
def test_download_models_clears_stale_zerotts_hybrid_symlink_cache_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_is_directory: bool
) -> None:
    """Test download unlinks symlinks (directory or verified file) and rebuilds as file."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    converted_content = b"FRESH_REBUILT_FROM_SYMLINK"
    q4_spec = _make_hybrid_model_spec(tts_spec, sha256(converted_content).hexdigest())

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    def fake_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(converted_content)
        return dst

    monkeypatch.setattr("wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", fake_convert)

    external_target = tmp_path / (
        "external_target_dir" if target_is_directory else "external_verified.gguf"
    )
    if target_is_directory:
        external_target.mkdir(parents=True, exist_ok=True)
        (external_target / "protected.txt").write_text("PROTECTED", encoding="utf-8")
    else:
        external_target.write_bytes(converted_content)

    stale_symlink = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    stale_symlink.parent.mkdir(parents=True, exist_ok=True)
    stale_symlink.symlink_to(external_target, target_is_directory=target_is_directory)

    paths, _sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
    )

    dst_hybrid = paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert dst_hybrid.is_file()
    assert not dst_hybrid.is_symlink()
    assert dst_hybrid.read_bytes() == converted_content
    # Ensure external target was unlinked, not deleted or corrupted
    if target_is_directory:
        assert external_target.is_dir()
        assert (external_target / "protected.txt").is_file()
    else:
        assert external_target.is_file()
        assert external_target.read_bytes() == converted_content


def test_download_models_clears_invalid_generated_hybrid_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download removes invalid generated hybrid model file if digest check fails."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    q4_spec = _make_hybrid_model_spec(
        tts_spec, "0000000000000000000000000000000000000000000000000000000000000000"
    )

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    def fake_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"BAD_CONVERTED_BYTES")
        return dst

    monkeypatch.setattr("wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", fake_convert)

    with pytest.raises(RuntimeError, match="does not match its recorded SHA-256 digest"):
        _execute_zerotts_download(
            tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
        )

    dst_hybrid = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    assert not dst_hybrid.exists()
    assert not dst_hybrid.is_symlink()


@pytest.mark.parametrize("artifact_kind", ["file", "directory", "dangling_symlink"])
def test_download_models_clears_partial_hybrid_on_conversion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_kind: str
) -> None:
    """Test download removes partial destination and propagates conversion error."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    converted_content = b"FRESH_Q4_HYBRID"
    q4_spec = _make_hybrid_model_spec(tts_spec, sha256(converted_content).hexdigest())

    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: q4_spec,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    stale_dst = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    stale_dst.parent.mkdir(parents=True, exist_ok=True)
    stale_dst.write_bytes(b"STALE_OLD_CONTENT")

    def failing_convert(src: Path, dst: Path) -> Path:
        assert not dst.exists()
        assert not dst.is_symlink()
        if artifact_kind == "directory":
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "partial.bin").write_bytes(b"NESTED_PARTIAL_DATA")
        elif artifact_kind == "dangling_symlink":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(tmp_path / "nonexistent_target")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"PARTIAL_MALFORMED_GGUF_DATA")
        raise RuntimeError("GGUF conversion failed during quantization")

    monkeypatch.setattr(
        "wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid", failing_convert
    )

    with pytest.raises(RuntimeError, match="GGUF conversion failed during quantization"):
        _execute_zerotts_download(
            tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="q4_hybrid"
        )

    dst_hybrid = (
        tmp_path
        / "models"
        / TTS_DIR
        / TtsEngine.ZEROTTS
        / ZeroTtsDirectory.GGUF
        / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    )
    assert not dst_hybrid.exists()
    assert not dst_hybrid.is_symlink()


def test_download_models_structures_zerotts_auto_higher_capability_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures Q8 model when auto is selected on higher capability CPUs."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: tts_spec,
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.tts_model.detect_cpu_variant",
        lambda: ZeroTtsVariant.AVX2,
    )
    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant="auto"
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    assert (paths[TTS_DIR] / ZeroTtsDirectory.GGUF / ZeroTtsFile.DEFAULT_GGUF_MODEL).is_file()
    assert sync.call_count == 2
    assert sync.call_args_list[1].kwargs["allow_patterns"] == tts_spec.allow_patterns


def test_download_models_zerotts_custom_absolute_path_skips_gguf_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download_models with an absolute custom GGUF model skips downloading GGUF from HF."""
    stt, tts_snapshot, tts_spec, test_voice = _setup_zerotts_download_test(tmp_path, monkeypatch)
    custom_gguf = tmp_path / "external-custom.gguf"
    custom_gguf.write_bytes(b"CUSTOM_GGUF_DATA")

    base_spec = _make_spec_without_gguf(tts_spec)
    monkeypatch.setattr(
        "wyoming_vietnamese.download.resolve_zerotts_model_spec",
        lambda quant, **kw: base_spec,
    )
    paths, sync = _execute_zerotts_download(
        tmp_path, monkeypatch, stt, tts_snapshot, test_voice, zerotts_quant=str(custom_gguf)
    )

    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.ZEROTTS
    allow_patterns = sync.call_args_list[1].kwargs["allow_patterns"]
    assert all("gguf" not in p for p in allow_patterns)
    assert not (paths[TTS_DIR] / ZeroTtsDirectory.GGUF).exists()
    assert not list(paths[TTS_DIR].rglob("*.gguf"))

    resolution = resolve_zerotts_model(paths[TTS_DIR], str(custom_gguf))
    assert resolution.gguf_path == custom_gguf
    assert resolution.spec.gguf_model is None
    assert custom_gguf.is_file()
    assert custom_gguf.read_bytes() == b"CUSTOM_GGUF_DATA"


def test_structure_zerotts_files_corrupted_destination_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises RuntimeError when destination voice is corrupted."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(snapshot_dir)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))

    target_dir = tmp_path / TTS_DIR
    corrupted_dst = target_dir / "voices" / test_voice.id / "voice.npz"
    corrupted_dst.parent.mkdir(parents=True, exist_ok=True)
    corrupted_dst.write_bytes(b"CORRUPTED_VOICE_DATA")

    snapshot_voice = snapshot_dir / test_voice.artifact.remote_name
    if snapshot_voice.is_file():
        snapshot_voice.unlink()
    flat_voice = snapshot_dir / "voices" / f"{test_voice.id}.npz"
    if flat_voice.is_file():
        flat_voice.unlink()

    with pytest.raises(RuntimeError, match="does not match its recorded SHA-256 digest"):
        _structure_zerotts_files(snapshot_dir, target_dir, voices=(test_voice,))


def test_structure_zerotts_files_missing_configured_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises FileNotFoundError when configured voice is missing."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(snapshot_dir)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", ())

    target_dir = tmp_path / TTS_DIR
    with pytest.raises(FileNotFoundError, match="is missing from"):
        _structure_zerotts_files(snapshot_dir, target_dir, voices=(test_voice,))


def test_structure_zerotts_files_corrupted_destination_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises RuntimeError when destination artifact is corrupted."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(snapshot_dir)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))

    target_dir = tmp_path / TTS_DIR
    _structure_zerotts_files(snapshot_dir, target_dir, voices=(test_voice,))

    corrupted_artifact = target_dir / tts_spec.artifacts[0].local_name
    source_artifact = snapshot_dir / tts_spec.artifacts[0].remote_name
    if not source_artifact.is_file():
        source_artifact = (
            snapshot_dir / ZeroTtsDirectory.GGUF / Path(tts_spec.artifacts[0].remote_name).name
        )
    source_stat = source_artifact.stat()

    corrupted_artifact.write_bytes(b"X" * source_stat.st_size)
    os.utime(corrupted_artifact, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))

    with pytest.raises(RuntimeError, match="does not match its recorded SHA-256 digest"):
        _structure_zerotts_files(snapshot_dir, target_dir, voices=(test_voice,))


def test_download_models_rejects_unknown_tts_engine(tmp_path: Path) -> None:
    """Test download_models rejects unknown TTS engine before any synchronization."""
    with pytest.raises(ValueError, match="Unknown TTS engine"):
        download_models(
            tmp_path / "cache",
            tmp_path / "models",
            (DEFAULT_NGHITTS_VOICE,),
            tts_engine="unknown_engine",
        )
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "models").exists()


def test_download_models_rejects_mismatched_voice_spec(tmp_path: Path) -> None:
    """Test download_models rejects voice specifications mismatched to the engine."""
    with pytest.raises(TypeError, match="requires ZeroTtsVoiceSpec"):
        download_models(
            tmp_path / "cache",
            tmp_path / "models",
            (DEFAULT_NGHITTS_VOICE,),
            tts_engine=TtsEngine.ZEROTTS,
        )
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "models").exists()


def test_download_models_structures_gipformer_stt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download structures STT using Gipformer engine."""
    stt = tmp_path / "stt-snapshot"
    tts = tmp_path / "tts-snapshot"
    stt.mkdir(parents=True, exist_ok=True)
    for part in ("encoder", "decoder", "joiner"):
        remote_name = f"{part}.int8.onnx"
        (stt / remote_name).write_bytes(part.encode())
    bpe = _sentencepiece_model("<unk>", "xin")
    (stt / "bpe.model").write_bytes(bpe)
    spec = SttModelSpec(
        repo=GIPFORMER_MODEL.repo,
        revision=GIPFORMER_MODEL.revision,
        graphs=(
            SttArtifact("encoder.int8.onnx", "encoder.onnx", sha256(b"encoder").hexdigest()),
            SttArtifact("decoder.int8.onnx", "decoder.onnx", sha256(b"decoder").hexdigest()),
            SttArtifact("joiner.int8.onnx", "joiner.onnx", sha256(b"joiner").hexdigest()),
        ),
        tokenizer=SttArtifact("bpe.model", "bpe.model", sha256(bpe).hexdigest()),
        description=GIPFORMER_MODEL.description,
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.download.get_stt_model",
        lambda engine=DEFAULT_STT_ENGINE: (
            spec if engine == SttEngine.GIPFORMER else get_stt_model(engine)
        ),
    )
    _make_nghitts_snapshot(tts)
    sync = Mock(return_value=stt)
    sync_tts = Mock(return_value=tts)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", sync)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_nghitts_model", sync_tts)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())

    paths = download_models(
        tmp_path / "cache",
        tmp_path / "models",
        (DEFAULT_NGHITTS_VOICE,),
        stt_engine=SttEngine.GIPFORMER,
    )

    assert paths[STT_DIR] == tmp_path / "models" / STT_DIR / SttEngine.GIPFORMER
    assert paths[TTS_DIR] == tmp_path / "models" / TTS_DIR / TtsEngine.NGHITTS
    assert (paths[STT_DIR] / "encoder.onnx").is_file()
    assert sync.call_args.args == (spec.repo, False)
    assert sync.call_args.kwargs["revision"] == spec.revision
    assert sync.call_args.kwargs["allow_patterns"] == spec.allow_patterns


def test_download_models_rejects_unknown_stt_engine(tmp_path: Path) -> None:
    """Test download_models rejects unknown STT engine before any synchronization."""
    with pytest.raises(ValueError, match="STT_ENGINE must be one of"):
        download_models(
            tmp_path / "cache",
            tmp_path / "models",
            (DEFAULT_NGHITTS_VOICE,),
            stt_engine="unknown_stt",
        )
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "models").exists()


def test_structure_stt_files_with_explicit_model_spec(tmp_path: Path) -> None:
    """Test _structure_stt_files with an explicitly provided SttModelSpec."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    spec = _make_stt_snapshot(snapshot)
    destination.mkdir()

    paths = _structure_stt_files(snapshot, destination, stt_model=spec)

    assert (destination / "encoder.onnx").read_bytes() == b"encoder"
    assert (destination / "decoder.onnx").read_bytes() == b"decoder"
    assert (destination / "joiner.onnx").read_bytes() == b"joiner"
    assert (destination / "tokens.txt").read_text(encoding="utf-8").startswith("<unk> 0")
    assert len(paths) == 4


def test_download_models_structures_different_stt_engines_in_separate_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that two different STT engines structure into separate directories."""
    stt_zipformer = tmp_path / "zipformer-snapshot"
    stt_gipformer = tmp_path / "gipformer-snapshot"
    tts = tmp_path / "tts-snapshot"
    stt_zipformer.mkdir(parents=True, exist_ok=True)
    stt_gipformer.mkdir(parents=True, exist_ok=True)

    for part in ("encoder", "decoder", "joiner"):
        (stt_zipformer / f"{part}.onnx").write_bytes(f"zipformer-{part}".encode())
        (stt_gipformer / f"{part}.int8.onnx").write_bytes(f"gipformer-{part}".encode())
    bpe_zip = _sentencepiece_model("<unk>", "zip")
    bpe_gip = _sentencepiece_model("<unk>", "gip")
    (stt_zipformer / "bpe.model").write_bytes(bpe_zip)
    (stt_gipformer / "bpe.model").write_bytes(bpe_gip)

    zip_spec = SttModelSpec(
        repo=ZIPFORMER_MODEL.repo,
        revision=ZIPFORMER_MODEL.revision,
        graphs=(
            SttArtifact("encoder.onnx", "encoder.onnx", sha256(b"zipformer-encoder").hexdigest()),
            SttArtifact("decoder.onnx", "decoder.onnx", sha256(b"zipformer-decoder").hexdigest()),
            SttArtifact("joiner.onnx", "joiner.onnx", sha256(b"zipformer-joiner").hexdigest()),
        ),
        tokenizer=SttArtifact("bpe.model", "bpe.model", sha256(bpe_zip).hexdigest()),
        description=ZIPFORMER_MODEL.description,
    )
    gip_spec = SttModelSpec(
        repo=GIPFORMER_MODEL.repo,
        revision=GIPFORMER_MODEL.revision,
        graphs=(
            SttArtifact(
                "encoder.int8.onnx", "encoder.onnx", sha256(b"gipformer-encoder").hexdigest()
            ),
            SttArtifact(
                "decoder.int8.onnx", "decoder.onnx", sha256(b"gipformer-decoder").hexdigest()
            ),
            SttArtifact("joiner.int8.onnx", "joiner.onnx", sha256(b"gipformer-joiner").hexdigest()),
        ),
        tokenizer=SttArtifact("bpe.model", "bpe.model", sha256(bpe_gip).hexdigest()),
        description=GIPFORMER_MODEL.description,
    )

    def mock_get_stt_model(engine: str = DEFAULT_STT_ENGINE) -> SttModelSpec:
        return gip_spec if engine == SttEngine.GIPFORMER else zip_spec

    def mock_sync_repo(repo: str, offline: bool, **kwargs: object) -> Path:
        return stt_gipformer if repo == GIPFORMER_MODEL.repo else stt_zipformer

    _make_nghitts_snapshot(tts)
    sync_tts = Mock(return_value=tts)
    monkeypatch.setattr("wyoming_vietnamese.download.get_stt_model", mock_get_stt_model)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", mock_sync_repo)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_nghitts_model", sync_tts)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())

    models_dir = tmp_path / "models"
    zip_paths = download_models(
        tmp_path / "cache",
        models_dir,
        (DEFAULT_NGHITTS_VOICE,),
        stt_engine=SttEngine.ZIPFORMER,
    )
    gip_paths = download_models(
        tmp_path / "cache",
        models_dir,
        (DEFAULT_NGHITTS_VOICE,),
        stt_engine=SttEngine.GIPFORMER,
    )

    assert zip_paths[STT_DIR] == models_dir / STT_DIR / SttEngine.ZIPFORMER
    assert gip_paths[STT_DIR] == models_dir / STT_DIR / SttEngine.GIPFORMER
    assert (zip_paths[STT_DIR] / "encoder.onnx").read_bytes() == b"zipformer-encoder"
    assert (gip_paths[STT_DIR] / "encoder.onnx").read_bytes() == b"gipformer-encoder"


def test_download_models_structures_different_tts_engines_in_separate_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that two different TTS engines structure into separate directories."""
    stt = tmp_path / "stt-snapshot"
    stt_spec = _make_stt_snapshot(stt)
    _pin_stt_model(monkeypatch, stt_spec)

    nghitts_snapshot = tmp_path / "nghitts-snapshot"
    _make_nghitts_snapshot(nghitts_snapshot)
    zerotts_snapshot = tmp_path / "zerotts-snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(zerotts_snapshot)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))

    def mock_sync_repo(repo: str, offline: bool, **kwargs: object) -> Path:
        return zerotts_snapshot if repo == ZEROTTS_REPO_ID else stt

    sync_nghitts = Mock(return_value=nghitts_snapshot)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_repo", mock_sync_repo)
    monkeypatch.setattr("wyoming_vietnamese.download._sync_nghitts_model", sync_nghitts)
    monkeypatch.setattr("wyoming_vietnamese.download.setup_hf_environment", Mock())

    models_dir = tmp_path / "models"
    nghitts_paths = download_models(
        tmp_path / "cache",
        models_dir,
        (DEFAULT_NGHITTS_VOICE,),
        tts_engine=TtsEngine.NGHITTS,
    )
    zerotts_paths = download_models(
        tmp_path / "cache",
        models_dir,
        (test_voice,),
        tts_engine=TtsEngine.ZEROTTS,
    )

    assert nghitts_paths[TTS_DIR] == models_dir / TTS_DIR / TtsEngine.NGHITTS
    assert zerotts_paths[TTS_DIR] == models_dir / TTS_DIR / TtsEngine.ZEROTTS
    assert (nghitts_paths[TTS_DIR] / DEFAULT_NGHITTS_VOICE.id / NghiTtsFile.MODEL).is_file()
    assert (zerotts_paths[TTS_DIR] / "config.json").is_file()


@pytest.mark.parametrize(
    ("tensor_name", "expected"),
    [
        ("model.layers.0.mlp.down_proj.weight", True),
        ("model.layers.0.ffn.up_proj.weight", True),
        ("model.layers.0.ffn_up.weight", True),
        ("model.layers.0.ffn_down.weight", True),
        ("model.layers.0.ffn_gate.weight", True),
        ("model.layers.0.fc1.weight", True),
        ("model.layers.0.fc2.weight", True),
        ("model.layers.0.w1.weight", True),
        ("model.layers.0.w2.weight", True),
        ("model.layers.0.w3.weight", True),
        ("model.layers.0.mlp.norm.weight", False),
        ("model.layers.0.mlp.bias", False),
        ("model.embed_tokens.weight", False),
        ("lm_head.weight", False),
        ("model.text_encoder.ffn.weight", False),
        ("model.layers.0.self_attn.q_proj.weight", False),
        ("codebook.w1.weight", False),
        ("flow.speaker_embed.w10.weight", False),
        ("raw1_weight", False),
        ("sw2_weight", False),
        ("flow1_weight", False),
    ],
)
def test_is_ffn_tensor_patterns(tensor_name: str, expected: bool) -> None:
    """Test is_ffn_tensor correctly classifies weight tensors."""
    assert is_ffn_tensor(tensor_name) is expected


@pytest.mark.parametrize(
    ("qs_values", "expected_scale_val", "expected_payload"),
    [
        (np.arange(1, 33, dtype=np.int8), None, None),
        (np.zeros(32, dtype=np.int8), 0.0, b"\x88" * 16),
        (np.array([8] * 16 + [-8] * 16, dtype=np.int8), -1.0, b"\xf0" * 16),
    ],
)
def test_quantize_row_q8_0_to_q4_0_cases(
    qs_values: np.ndarray,
    expected_scale_val: float | None,
    expected_payload: bytes | None,
) -> None:
    """Test quantize_row_q8_0_to_q4_0 produces expected bytes and scale payload."""
    delta = np.float16(1.0).tobytes()
    q8_bytes = delta + qs_values.tobytes()
    q4_bytes = quantize_row_q8_0_to_q4_0(q8_bytes, 32)
    assert len(q4_bytes) == 18
    if expected_scale_val is not None and expected_payload is not None:
        assert q4_bytes == np.float16(expected_scale_val).tobytes() + expected_payload


def test_quantize_row_q8_0_to_q4_0_size_mismatch() -> None:
    """Test quantize_row_q8_0_to_q4_0 raises ValueError on invalid elements or payloads."""
    with pytest.raises(ValueError, match="Invalid n_elements: 0"):
        quantize_row_q8_0_to_q4_0(b"\x00" * 34, 0)

    with pytest.raises(ValueError, match="Invalid n_elements: -32"):
        quantize_row_q8_0_to_q4_0(b"\x00" * 34, -32)

    with pytest.raises(ValueError, match="Invalid n_elements: 33"):
        quantize_row_q8_0_to_q4_0(b"\x00" * 34, 33)

    with pytest.raises(ValueError, match="Unexpected Q8_0 size"):
        quantize_row_q8_0_to_q4_0(b"\x00" * 20, 32)

    with pytest.raises(ValueError, match="Unexpected Q8_0 size"):
        quantize_row_q8_0_to_q4_0(b"\x00" * 50, 32)


@pytest.mark.parametrize(
    ("tensor_name", "expected"),
    [
        ("model.layers.0.self_attn.weight", True),
        ("dec.0.self_q.weight", True),
        ("dec.0.self_k.weight", True),
        ("dec.0.self_v.weight", True),
        ("dec.0.self_o.weight", True),
        ("dec.0.cross_q.weight", True),
        ("loc.0.attn_q.weight", True),
        ("loc.0.attn_k.weight", True),
        ("loc.0.attn_v.weight", True),
        ("loc.0.attn_o.weight", True),
        ("attention.q_proj.weight", True),
        ("attention.query.weight", True),
        ("attention.q_proj", True),
        ("attention.query", True),
        ("transformer.layers.0.attention.wq", True),
        ("transformer.layers.0.attention.wk", True),
        ("transformer.layers.0.attention.wv", True),
        ("transformer.layers.0.attention.wo", True),
        ("model.layers.0.self_attention.weight", True),
        ("model.layers.0.cross_attention.weight", True),
        ("model.layers.0.q_proj.weight", True),
        ("model.layers.0.k_proj.weight", True),
        ("model.layers.0.v_proj.weight", True),
        ("model.layers.0.o_proj.weight", True),
        ("model.layers.0.out_proj.weight", True),
        ("ahead_attention.q_proj.weight", True),
        ("member_attn.k_proj.weight", True),
        ("assembly_attn.v_proj.weight", True),
        ("model.layers.0.attn.norm.weight", False),
        ("model.layers.0.norm_attn.weight", False),
        ("model.layers.0.attn.bias", False),
        ("audio_head.0.weight", False),
        ("audio_embed.0.weight", False),
        ("codebook.attn.weight", False),
    ],
)
def test_is_attention_tensor_patterns(tensor_name: str, expected: bool) -> None:
    """Test is_attention_tensor correctly classifies attention projection weights."""
    assert is_attention_tensor(tensor_name) is expected


@pytest.mark.parametrize(
    ("tensor_name", "expected"),
    [
        ("audio_head.0.weight", True),
        ("audio_embed.15.weight", True),
        ("codebook.weight", True),
        ("text_encoder.weight", True),
        ("txt.0.attn_q.weight", True),
        ("dec.0.self_q.weight", False),
        ("dec.0.ffn_up.weight", False),
        ("loc.0.ffn_down.weight", False),
    ],
)
def test_is_pinned_q8_tensor_patterns(tensor_name: str, expected: bool) -> None:
    """Test is_pinned_q8_tensor correctly identifies audio heads and codebooks."""
    assert is_pinned_q8_tensor(tensor_name) is expected


def _pack_tensor_meta(name: bytes, dims: tuple[int, ...], ttype: int, offset: int = 0) -> bytes:
    """Pack GGUF tensor header metadata into bytes."""
    header = struct.pack("<Q", len(name)) + name + struct.pack("<I", len(dims))
    for d in dims:
        header += struct.pack("<Q", d)
    return header + struct.pack("<IQ", ttype, offset)


def _write_synthetic_gguf(path: Path) -> None:
    """Write a synthetic GGUF model exercising all KV and tensor parsing branches."""
    with path.open("wb") as fout:
        fout.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 6, 8))

        def write_kv(key: bytes, vtype: int, raw_bytes: bytes) -> None:
            fout.write(struct.pack("<Q", len(key)) + key + struct.pack("<I", vtype) + raw_bytes)

        write_kv(b"k0", 0, b"\x01")
        write_kv(b"k1", 1, b"\x01")
        write_kv(b"k2", 2, b"\x01\x00")
        write_kv(b"k4", 4, b"\x01\x00\x00\x00")
        write_kv(b"k10", 10, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        write_kv(b"k7", 7, b"\x01")
        write_kv(b"k8", 8, struct.pack("<Q", 4) + b"test")
        write_kv(b"k9", 9, struct.pack("<IQ", 4, 2) + b"\x01\x00\x00\x00\x02\x00\x00\x00")

        tensors = (
            (b"model.layers.0.mlp.weight", (32, 1), GGML_TYPE_Q8_0, 0),
            (b"model.layers.0.self_attn.weight", (32, 1), GGML_TYPE_Q8_0, 64),
            (b"model.layers.0.norm.weight", (32,), GGML_TYPE_F32, 128),
            (b"model.layers.0.bias", (32,), GGML_TYPE_F16, 256),
            (b"model.layers.0.existing_q4", (32,), GGML_TYPE_Q4_0, 320),
            (b"audio_head.weight", (32, 1), GGML_TYPE_Q8_0, 352),
        )
        for name, dims, ttype, offset in tensors:
            fout.write(_pack_tensor_meta(name, dims, ttype, offset))

        header_len = fout.tell()
        pad = (32 - (header_len % 32)) % 32
        fout.write(b"\x00" * pad)

        t0_data = np.float16(1.0).tobytes() + (b"\x01" * 32)
        fout.write(t0_data + (b"\x00" * 30))

        t1_data = np.float16(1.0).tobytes() + (b"\x02" * 32)
        fout.write(t1_data + (b"\x00" * 30))

        fout.write(b"\x00" * 128)
        fout.write(b"\x00" * 64)
        fout.write(b"\x00" * 18 + (b"\x00" * 14))

        t5_data = np.float16(2.0).tobytes() + (b"\x05" * 32)
        fout.write(t5_data + (b"\x00" * 30))


def _skip_inspect_kv_value(fin: BinaryIO, vt: int) -> None:
    """Skip an uninspected GGUF KV payload based on its type."""
    if vt in {0, 1, 7}:
        fin.seek(1, 1)
    elif vt in {2, 3}:
        fin.seek(2, 1)
    elif vt in {4, 5, 6}:
        fin.seek(4, 1)
    elif vt in {10, 11, 12}:
        fin.seek(8, 1)
    elif vt == 8:
        s_len = struct.unpack("<Q", fin.read(8))[0]
        fin.seek(s_len, 1)
    elif vt == 9:
        ayt, cnt = struct.unpack("<IQ", fin.read(12))
        step = 4 if ayt in {4, 5, 6} else 8
        fin.seek(step * cnt, 1)


def _parse_inspect_alignment(fin: BinaryIO, n_kv: int) -> int:
    """Scan GGUF KV headers and return alignment value (default 32)."""
    alignment = 32
    for _ in range(n_kv):
        k_len = struct.unpack("<Q", fin.read(8))[0]
        k_name = fin.read(k_len).decode("utf-8", errors="replace")
        vt = struct.unpack("<I", fin.read(4))[0]
        if k_name == "general.alignment":
            if vt in {4, 5}:
                fmt = "<I" if vt == 4 else "<i"
                alignment = struct.unpack(fmt, fin.read(4))[0]
            elif vt == 10:
                alignment = struct.unpack("<Q", fin.read(8))[0]
            else:
                fin.seek(4, 1)
        else:
            _skip_inspect_kv_value(fin, vt)
    return alignment


def _parse_inspect_tensors(fin: BinaryIO, n_tensors: int) -> dict[str, tuple[int, int]]:
    """Parse GGUF tensor names, types, and offsets."""
    tensors: dict[str, tuple[int, int]] = {}
    for _ in range(n_tensors):
        n_len = struct.unpack("<Q", fin.read(8))[0]
        name = fin.read(n_len).decode("utf-8")
        n_dims = struct.unpack("<I", fin.read(4))[0]
        fin.seek(n_dims * 8, 1)
        ttype, offset = struct.unpack("<IQ", fin.read(12))
        tensors[name] = (ttype, offset)
    return tensors


def _inspect_gguf_tensors(
    path: Path,
) -> tuple[dict[str, tuple[int, int]], bytes, int]:
    """Parse GGUF tensors and return metadata, mlp Q4 payload, and data start offset."""
    with path.open("rb") as fin:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", fin.read(24))
        if magic != GGUF_MAGIC or version != GGUF_VERSION:
            raise ValueError("Corrupt test GGUF file")

        alignment = _parse_inspect_alignment(fin, n_kv)
        tensors = _parse_inspect_tensors(fin, n_tensors)

        header_end = fin.tell()
        pad = (alignment - (header_end % alignment)) % alignment
        data_start = header_end + pad

        q4_payload = b""
        if "model.layers.0.mlp.weight" in tensors:
            mlp_offset = tensors["model.layers.0.mlp.weight"][1]
            fin.seek(data_start + mlp_offset)
            q4_payload = fin.read(18)
        return tensors, q4_payload, data_start


def _assert_synthetic_hybrid_tensors(
    tensors: dict[str, tuple[int, int]],
    expected_attn_type: int = GGML_TYPE_Q8_0,
) -> None:
    """Assert common synthetic GGUF tensor types and count."""
    assert len(tensors) == 6
    assert tensors["model.layers.0.mlp.weight"][0] == GGML_TYPE_Q4_0
    assert tensors["model.layers.0.self_attn.weight"][0] == expected_attn_type
    assert tensors["model.layers.0.norm.weight"][0] == GGML_TYPE_F32
    assert tensors["model.layers.0.bias"][0] == GGML_TYPE_F16
    assert tensors["model.layers.0.existing_q4"][0] == GGML_TYPE_Q4_0
    assert tensors["audio_head.weight"][0] == GGML_TYPE_Q8_0


def _assert_tensor_payload(
    fin: BinaryIO,
    data_start: int,
    offset: int,
    expected: bytes,
) -> None:
    """Assert a tensor payload at data_start + offset matches expected bytes."""
    fin.seek(data_start + offset)
    payload = fin.read(len(expected))
    assert len(payload) == len(expected)
    assert payload == expected


def _assert_hybrid_attention_and_head_payloads(
    dst: Path,
    tensors: dict[str, tuple[int, int]],
    data_start: int,
    expected_attn: bytes,
) -> None:
    """Assert attention tensor and audio head payloads match expected byte content."""
    attn_offset = tensors["model.layers.0.self_attn.weight"][1]
    head_offset = tensors["audio_head.weight"][1]
    expected_head = np.float16(2.0).tobytes() + (b"\x05" * 32)
    with dst.open("rb") as fin:
        _assert_tensor_payload(fin, data_start, attn_offset, expected_attn)
        _assert_tensor_payload(fin, data_start, head_offset, expected_head)


def test_convert_zerotts_q8_to_q4_hybrid_success(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid converts GGUF tensors correctly."""
    src = tmp_path / "model-q8_0.gguf"
    dst = tmp_path / "model-q4-hybrid.gguf"
    _write_synthetic_gguf(src)

    result = convert_zerotts_q8_to_q4_hybrid(src, dst)
    assert result == dst
    assert dst.is_file()

    tensors, q4_payload, data_start = _inspect_gguf_tensors(dst)
    assert data_start % 32 == 0
    _assert_synthetic_hybrid_tensors(tensors, GGML_TYPE_Q8_0)

    assert len(q4_payload) == 18
    expected_q4 = quantize_row_q8_0_to_q4_0(np.float16(1.0).tobytes() + (b"\x01" * 32), 32)
    assert q4_payload == expected_q4

    expected_q8 = np.float16(1.0).tobytes() + (b"\x02" * 32)
    _assert_hybrid_attention_and_head_payloads(dst, tensors, data_start, expected_q8)


def _write_gguf_with_alignment(path: Path, alignment_val: int, *, n_tensors: int = 0) -> None:
    """Write a minimal GGUF model file with general.alignment metadata."""
    with path.open("wb") as fout:
        fout.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, n_tensors, 1))
        align_key = b"general.alignment"
        fout.write(
            struct.pack("<Q", len(align_key))
            + align_key
            + struct.pack("<I", 4)
            + struct.pack("<I", alignment_val)
        )
        if n_tensors > 0:
            fout.write(_pack_tensor_meta(b"model.layers.0.mlp.weight", (32,), GGML_TYPE_Q8_0, 0))

            header_len = fout.tell()
            pad = (alignment_val - (header_len % alignment_val)) % alignment_val
            fout.write(b"\x00" * pad)
            fout.write(np.float16(1.0).tobytes() + (b"\x01" * 32) + (b"\x00" * 30))


def test_convert_zerotts_q8_to_q4_hybrid_custom_alignment(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid honors general.alignment metadata."""
    src = tmp_path / "aligned-q8_0.gguf"
    dst = tmp_path / "aligned-q4-hybrid.gguf"
    _write_gguf_with_alignment(src, 64, n_tensors=1)

    result = convert_zerotts_q8_to_q4_hybrid(src, dst)
    assert result == dst
    assert dst.is_file()

    tensors, q4_payload, data_start = _inspect_gguf_tensors(dst)
    assert data_start % 64 == 0
    assert "model.layers.0.mlp.weight" in tensors
    ttype, offset = tensors["model.layers.0.mlp.weight"]
    assert ttype == GGML_TYPE_Q4_0
    assert offset % 64 == 0
    assert (data_start + offset) % 64 == 0

    assert len(q4_payload) == 18
    expected_q4 = quantize_row_q8_0_to_q4_0(np.float16(1.0).tobytes() + (b"\x01" * 32), 32)
    assert q4_payload == expected_q4


def _assert_convert_raises(tmp_path: Path, filename: str, content: bytes, error_match: str) -> None:
    """Write invalid GGUF model content and assert convert raises matching ValueError."""
    src = tmp_path / filename
    src.write_bytes(content)
    with pytest.raises(ValueError, match=error_match):
        convert_zerotts_q8_to_q4_hybrid(src, tmp_path / "out.gguf")


def test_convert_zerotts_q8_to_q4_hybrid_invalid_alignment(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects invalid alignment metadata."""
    src = tmp_path / "bad-align.gguf"
    _write_gguf_with_alignment(src, 33, n_tensors=0)
    with pytest.raises(ValueError, match="Invalid GGUF alignment: 33"):
        convert_zerotts_q8_to_q4_hybrid(src, tmp_path / "out.gguf")


def test_convert_zerotts_q8_to_q4_hybrid_invalid_tensor_dims(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects invalid tensor dimensions."""
    content = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 1, 0) + _pack_tensor_meta(
        b"bad_dim_tensor", (33,), GGML_TYPE_Q8_0, 0
    )
    _assert_convert_raises(tmp_path, "bad-dims.gguf", content, "must be a positive multiple of 32")


def test_convert_zerotts_q8_to_q4_hybrid_invalid_magic(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects files with invalid magic."""
    _assert_convert_raises(
        tmp_path, "invalid.gguf", b"BAD_MAGIC_HEADER_1234567890123456", "Not a valid GGUF file"
    )


def test_convert_zerotts_q8_to_q4_hybrid_unknown_kv_type(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects unknown KV types."""
    key = b"bad_key"
    content = (
        struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 999)
    )
    _assert_convert_raises(tmp_path, "bad_kv.gguf", content, "Unknown GGUF KV type 999")


def test_convert_zerotts_q8_to_q4_hybrid_unsupported_tensor_type(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects unsupported tensor types."""
    content = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 1, 0) + _pack_tensor_meta(
        b"bad_tensor", (32,), 999, 0
    )
    _assert_convert_raises(tmp_path, "bad_ttype.gguf", content, "Unsupported tensor type 999")


def test_convert_zerotts_q8_to_q4_hybrid_file_not_found(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid raises FileNotFoundError for missing input."""
    src = tmp_path / "nonexistent.gguf"
    dst = tmp_path / "out.gguf"
    with pytest.raises(FileNotFoundError, match="Source GGUF model file not found"):
        convert_zerotts_q8_to_q4_hybrid(src, dst)


def test_convert_zerotts_q8_to_q4_hybrid_file_too_small(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects files smaller than GGUF header."""
    _assert_convert_raises(
        tmp_path, "tiny.gguf", b"short", "File too small to be a valid GGUF file"
    )


def test_convert_zerotts_q8_to_q4_hybrid_invalid_version(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects unsupported GGUF versions."""
    content = struct.pack("<IIQQ", GGUF_MAGIC, 99, 0, 0)
    _assert_convert_raises(tmp_path, "bad_ver.gguf", content, "Unsupported GGUF version: 99")


def test_convert_zerotts_q8_to_q4_hybrid_excessive_dimensions(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects tensor dimension count > 8."""
    t_name = b"high_dims"
    content = (
        struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 1, 0)
        + struct.pack("<Q", len(t_name))
        + t_name
        + struct.pack("<I", 9)
    )
    _assert_convert_raises(
        tmp_path, "bad_dims.gguf", content, "Invalid GGUF tensor dimension count: 9"
    )


def test_convert_zerotts_q8_to_q4_hybrid_payload_out_of_bounds(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects tensor payload extending past EOF."""
    content = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 1, 0) + _pack_tensor_meta(
        b"model.layers.0.mlp.weight", (32,), GGML_TYPE_Q8_0, 0
    )
    _assert_convert_raises(tmp_path, "truncated.gguf", content, "payload extends beyond EOF")


@pytest.mark.parametrize("scalar_type", [0, 2, 4, 10])
def test_convert_zerotts_q8_to_q4_hybrid_truncated_kv_scalar(
    tmp_path: Path, scalar_type: int
) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects truncated KV scalar values."""
    key = b"dummy_key"
    content = (
        struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", scalar_type)
    )
    _assert_convert_raises(
        tmp_path,
        f"truncated_scalar_{scalar_type}.gguf",
        content,
        "Truncated GGUF file reading scalar value",
    )


def test_convert_zerotts_q8_to_q4_hybrid_truncated_alignment(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects truncated alignment value."""
    key = b"general.alignment"
    content = (
        struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 4)
    )
    _assert_convert_raises(
        tmp_path, "truncated_align.gguf", content, "Truncated GGUF file reading alignment value"
    )


def test_convert_zerotts_q8_to_q4_hybrid_unsupported_alignment_type(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid rejects unsupported alignment type."""
    key = b"general.alignment"
    content = (
        struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 8)
    )
    _assert_convert_raises(
        tmp_path, "bad_align_type.gguf", content, r"Unsupported type for general\.alignment"
    )


def test_convert_zerotts_q8_to_q4_hybrid_uint64_alignment(tmp_path: Path) -> None:
    """Test convert_zerotts_q8_to_q4_hybrid supports uint64 general.alignment metadata."""
    src = tmp_path / "u64_align.gguf"
    with src.open("wb") as fout:
        fout.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 1))
        key = b"general.alignment"
        fout.write(
            struct.pack("<Q", len(key)) + key + struct.pack("<I", 10) + struct.pack("<Q", 64)
        )
    dst = tmp_path / "out.gguf"
    convert_zerotts_q8_to_q4_hybrid(src, dst)
    assert dst.is_file()


def test_model_download_result_properties(tmp_path: Path) -> None:
    """Test ModelDownloadResult exposes stt_dir, tts_dir, and zerotts_resolution."""
    stt_dir = tmp_path / "stt"
    tts_dir = tmp_path / "tts"
    res = ModelDownloadResult(stt_dir, tts_dir, zerotts_resolution=None)
    assert res.stt_dir == stt_dir
    assert res.tts_dir == tts_dir
    assert res.zerotts_resolution is None
    assert res["stt"] == stt_dir
    assert res["tts"] == tts_dir


def _setup_structure_test_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, ZeroTtsModelSpec, ZeroTtsVoiceSpec]:
    """Prepare snapshot and destination directories for structure verification tests."""
    snapshot_dir = tmp_path / "snapshot"
    tts_spec, test_voice = _make_zerotts_snapshot(snapshot_dir)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_VOICES", (test_voice,))
    dest_dir = tmp_path / TTS_DIR
    return snapshot_dir, dest_dir, tts_spec, test_voice


def test_structure_zerotts_files_hybrid_already_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files reuses already verified hybrid GGUF file."""
    snapshot_dir, dest_dir, tts_spec, test_voice = _setup_structure_test_dirs(tmp_path, monkeypatch)

    hybrid_content = b"PRE_EXISTING_VERIFIED_HYBRID"
    hybrid_dst = dest_dir / ZeroTtsDirectory.GGUF / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    hybrid_dst.parent.mkdir(parents=True, exist_ok=True)
    hybrid_dst.write_bytes(hybrid_content)

    hybrid_model_spec = _make_hybrid_model_spec(tts_spec, sha256(hybrid_content).hexdigest())

    copied = _structure_zerotts_files(
        snapshot_dir,
        dest_dir,
        voices=(test_voice,),
        zerotts_model=hybrid_model_spec,
    )
    assert hybrid_dst in copied


def test_structure_zerotts_files_hybrid_converts_and_fails_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises RuntimeError if converted hybrid fails checksum."""
    snapshot_dir, dest_dir, tts_spec, test_voice = _setup_structure_test_dirs(tmp_path, monkeypatch)

    hybrid_model_spec = _make_hybrid_model_spec(tts_spec, "expected_sha256_that_will_not_match")

    def dummy_convert(src: Path, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"WRONG_CHECKSUM_DATA")
        return dst

    monkeypatch.setattr(
        "wyoming_vietnamese.download.convert_zerotts_q8_to_q4_hybrid",
        dummy_convert,
    )
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", tts_spec)

    with pytest.raises(RuntimeError, match="does not match its recorded SHA-256 digest"):
        _structure_zerotts_files(
            snapshot_dir,
            dest_dir,
            voices=(test_voice,),
            zerotts_model=hybrid_model_spec,
        )


def test_structure_zerotts_files_hybrid_missing_base_q8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _structure_zerotts_files raises RuntimeError when base Q8 artifact is missing."""
    snapshot_dir, dest_dir, tts_spec, test_voice = _setup_structure_test_dirs(tmp_path, monkeypatch)

    hybrid_model_spec = _make_hybrid_model_spec(tts_spec, "dummy_hash")
    base_spec_without_gguf = _make_spec_without_gguf(tts_spec)
    monkeypatch.setattr("wyoming_vietnamese.download.ZEROTTS_MODEL", base_spec_without_gguf)

    with pytest.raises(RuntimeError, match="Base ZeroTTS Q8 GGUF artifact is missing"):
        _structure_zerotts_files(
            snapshot_dir,
            dest_dir,
            voices=(test_voice,),
            zerotts_model=hybrid_model_spec,
        )


def test_create_q4_hybrid_cli_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test tools/create_q4_hybrid.py CLI commands."""
    import importlib.util
    import sys

    tool_path = Path(__file__).resolve().parent.parent / "tools" / "create_q4_hybrid.py"
    spec = importlib.util.spec_from_file_location("create_q4_hybrid", tool_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = tmp_path / "model-q8_0.gguf"
    dst = tmp_path / "model-q4-hybrid.gguf"
    _write_synthetic_gguf(src)

    mod.print_q4_hybrid_summary(src, dst)

    monkeypatch.setattr(
        sys, "argv", ["create_q4_hybrid.py", "--input", str(tmp_path / "missing.gguf")]
    )
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["create_q4_hybrid.py", "--input", str(src), "--output", str(dst)],
    )
    mod.main()
    assert dst.is_file()

    tensors_default, _, _ = _inspect_gguf_tensors(dst)
    assert tensors_default["model.layers.0.self_attn.weight"][0] == GGML_TYPE_Q8_0
    assert tensors_default["model.layers.0.mlp.weight"][0] == GGML_TYPE_Q4_0
    assert tensors_default["audio_head.weight"][0] == GGML_TYPE_Q8_0

    default_out = src.parent / "zerotts-q4-hybrid.gguf"
    monkeypatch.setattr(
        sys,
        "argv",
        ["create_q4_hybrid.py", "--input", str(src)],
    )
    mod.main()
    assert default_out.is_file()
    tensors_q8, _, _ = _inspect_gguf_tensors(default_out)
    assert tensors_q8["model.layers.0.self_attn.weight"][0] == GGML_TYPE_Q8_0
    assert tensors_q8["model.layers.0.mlp.weight"][0] == GGML_TYPE_Q4_0
    assert tensors_q8["audio_head.weight"][0] == GGML_TYPE_Q8_0
