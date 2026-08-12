"""Model synchronization and asset structuring tests."""

from __future__ import annotations

import errno
import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from wyoming_vietnamese import download as download_module
from wyoming_vietnamese.const import (
    TTS_CONFIG_FILE,
    TTS_MODEL_FILE,
    TTS_TOKENS_FILE,
    VIETNAMESE_LANGUAGE,
)
from wyoming_vietnamese.download import (
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
    _sync_nghitts_model,
    _sync_repo,
    download_models,
    setup_hf_environment,
)
from wyoming_vietnamese.stt_model import SttArtifact, SttModelSpec
from wyoming_vietnamese.tts_model import (
    DEFAULT_TTS_VOICE,
    TtsArtifact,
    TtsVoiceSpec,
    get_voice,
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
    monkeypatch.setattr("wyoming_vietnamese.download.STT_MODEL", spec)


def _make_nghitts_snapshot(directory: Path) -> None:
    """Build a minimal raw NghiTTS voice snapshot."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TTS_MODEL_FILE).write_bytes(b"onnx")
    (directory / TTS_CONFIG_FILE).write_text(
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
    destination = tmp_path / "model.onnx"
    _download_verified_file(
        "https://models.example/model.onnx",
        destination,
        expected_sha256=sha256(content).hexdigest(),
    )
    assert destination.read_bytes() == content

    with pytest.raises(ValueError, match="SHA-256"):
        _download_verified_file(
            "https://models.example/model.onnx",
            destination,
            expected_sha256=sha256(b"different").hexdigest(),
        )


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
    config = tmp_path / "model.onnx.json"
    tokens = tmp_path / "tokens.txt"
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
    config = tmp_path / "model.onnx.json"
    config.write_text(json.dumps({"phoneme_id_map": phoneme_map}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _generate_nghitts_tokens(config, tmp_path / "tokens.txt")


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
    converted_model = destination / TTS_MODEL_FILE
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / TTS_CONFIG_FILE))
    assert converted_model.read_bytes() == b"onnx" + metadata
    assert (destination / TTS_CONFIG_FILE).is_file()
    assert (destination / TTS_TOKENS_FILE).read_text(encoding="utf-8") == ("_ 0\n  1\na 2\n")
    assert len(paths) == 3

    (snapshot / TTS_MODEL_FILE).unlink()
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
    prepared = (destination / TTS_MODEL_FILE).read_bytes()

    digests: list[Path] = []
    original_digest = download_module._file_sha256

    def record_digest(path: Path) -> str:
        """Record every full-file digest requested during preparation."""
        digests.append(path)
        return original_digest(path)

    monkeypatch.setattr("wyoming_vietnamese.download._file_sha256", record_digest)

    _structure_nghitts_files(snapshot, destination)
    assert not digests
    assert (destination / TTS_MODEL_FILE).read_bytes() == prepared

    (snapshot / TTS_MODEL_FILE).write_bytes(b"onnx-v2")
    _structure_nghitts_files(snapshot, destination)
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / TTS_CONFIG_FILE))
    assert (destination / TTS_MODEL_FILE).read_bytes() == b"onnx-v2" + metadata


def test_structure_nghitts_files_rebuilds_after_a_corrupt_marker(tmp_path: Path) -> None:
    """Test an unreadable preparation marker falls back to digest verification."""
    snapshot = tmp_path / "snapshot"
    destination = tmp_path / "model"
    _make_nghitts_snapshot(snapshot)
    _structure_nghitts_files(snapshot, destination)

    marker = download_module._prepared_marker_path(destination / TTS_MODEL_FILE)
    marker.write_text("not json", encoding="utf-8")
    (destination / TTS_MODEL_FILE).write_bytes(b"corrupted")

    _structure_nghitts_files(snapshot, destination)
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / TTS_CONFIG_FILE))
    assert (destination / TTS_MODEL_FILE).read_bytes() == b"onnx" + metadata
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
    prepared_model = destination / TTS_MODEL_FILE
    original = prepared_model.read_bytes()

    config = snapshot / TTS_CONFIG_FILE
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            f'"voice": "{VIETNAMESE_LANGUAGE}"', '"voice": "en"'
        ),
        encoding="utf-8",
    )
    _structure_nghitts_files(snapshot, destination)

    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(destination / TTS_CONFIG_FILE))
    assert len(prepared_model.read_bytes()) == len(original)
    assert prepared_model.read_bytes() == b"onnx" + metadata
    assert prepared_model.read_bytes() != original


def test_sync_nghitts_model_caches_pinned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test encoded API names and verified offline reuse for a selected voice."""
    model = b"model"
    config = b"config"
    voice = TtsVoiceSpec(
        "test-voice",
        "Ngọc (mới)",
        TtsArtifact("Ngọc (mới).onnx", TTS_MODEL_FILE, sha256(model).hexdigest()),
        TtsArtifact(
            "Ngọc (mới).onnx.json",
            TTS_CONFIG_FILE,
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
    assert (snapshot / TTS_MODEL_FILE).read_bytes() == model
    assert (snapshot / TTS_CONFIG_FILE).read_bytes() == config
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
        _sync_nghitts_model(tmp_path, False, DEFAULT_TTS_VOICE, base_url="http://models.example")
    with pytest.raises(RuntimeError, match="Offline startup failed"):
        _sync_nghitts_model(
            tmp_path,
            True,
            DEFAULT_TTS_VOICE,
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
        (DEFAULT_TTS_VOICE, second_voice),
    )

    assert (paths["stt"] / "encoder.onnx").is_file()
    assert paths["tts"] == tmp_path / "models" / "tts"
    assert (paths["tts"] / DEFAULT_TTS_VOICE.id / TTS_MODEL_FILE).is_file()
    assert (paths["tts"] / DEFAULT_TTS_VOICE.id / TTS_TOKENS_FILE).is_file()
    assert (paths["tts"] / second_voice.id / TTS_MODEL_FILE).is_file()
    assert set(paths) == {"stt", "tts"}
    assert sync.call_args.args == (spec.repo, False)
    assert sync.call_args.kwargs["revision"] == spec.revision
    assert sync.call_args.kwargs["allow_patterns"] == spec.allow_patterns
    assert [item.args for item in sync_tts.call_args_list] == [
        (tmp_path / "cache", False, DEFAULT_TTS_VOICE),
        (tmp_path / "cache", False, second_voice),
    ]
