"""Checksum-pinned NghiTTS and ZeroTTS voice catalogue tests."""

import pytest

from wyoming_vietnamese.const import (
    NghiTtsFile,
    TtsEngine,
)
from wyoming_vietnamese.tts_model import (
    DEFAULT_NGHITTS_VOICE,
    DEFAULT_NGHITTS_VOICE_ID,
    DEFAULT_ZEROTTS_VOICE,
    DEFAULT_ZEROTTS_VOICE_ID,
    NGHITTS_VOICES,
    NGHITTS_VOICES_BY_ID,
    ZEROTTS_MODEL,
    ZEROTTS_VOICES,
    ZEROTTS_VOICES_BY_ID,
    get_voice,
    get_zerotts_voice,
)


def test_voice_catalog_has_unique_valid_entries() -> None:
    """Test every selectable voice has a unique ID and pinned artifacts."""
    assert len(NGHITTS_VOICES) == 20
    assert len(NGHITTS_VOICES_BY_ID) == len(NGHITTS_VOICES)
    assert len({voice.name for voice in NGHITTS_VOICES}) == len(NGHITTS_VOICES)
    assert all(voice.id and voice.name for voice in NGHITTS_VOICES)


def test_voice_artifacts_are_fully_pinned() -> None:
    """Test each voice pins a SHA-256 digest for every file it downloads."""
    for voice in NGHITTS_VOICES:
        assert voice.artifacts == (voice.model, voice.config)
        assert voice.model.local_name == NghiTtsFile.MODEL
        assert voice.config.local_name == NghiTtsFile.CONFIG
        assert voice.model.remote_name == f"{voice.name}.onnx"
        assert voice.config.remote_name == f"{voice.name}.onnx.json"
        for artifact in voice.artifacts:
            assert len(artifact.sha256) == 64
            assert artifact.sha256 == artifact.sha256.lower()
            assert int(artifact.sha256, 16) >= 0


def test_get_voice_resolves_default_and_rejects_unknown_id() -> None:
    """Test catalog lookup is deterministic and reports supported choices."""
    assert get_voice(DEFAULT_NGHITTS_VOICE_ID) is DEFAULT_NGHITTS_VOICE
    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("missing")


def test_zerotts_voice_catalog() -> None:
    """Test ZeroTTS voice catalogue entries and helper lookups."""
    assert len(ZEROTTS_VOICES) == 8
    assert len(ZEROTTS_VOICES_BY_ID) == 8
    assert DEFAULT_ZEROTTS_VOICE_ID == "maichi"
    assert DEFAULT_ZEROTTS_VOICE.id == "maichi"
    assert DEFAULT_NGHITTS_VOICE_ID == "ngoc-huyen-moi"

    for voice in ZEROTTS_VOICES:
        assert voice.id in ZEROTTS_VOICES_BY_ID
        assert voice.name
        assert voice.gender in {"nữ", "nam"}
        assert voice.description
        assert voice.artifacts == (voice.artifact,)
        assert voice.artifact.remote_name == f"voices/{voice.id}/voice.npz"
        assert voice.artifact.local_name == "voice.npz"
        assert len(voice.artifact.sha256) == 64
        assert voice.artifact.sha256 == voice.artifact.sha256.lower()
        assert int(voice.artifact.sha256, 16) >= 0

    mai_chi = get_zerotts_voice("maichi")
    assert mai_chi.name == "Mai Chi"
    assert mai_chi.gender == "nữ"

    assert get_voice("maichi", engine=TtsEngine.ZEROTTS) is mai_chi
    assert get_voice("ngoc-huyen-moi", engine=TtsEngine.NGHITTS) is DEFAULT_NGHITTS_VOICE

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_zerotts_voice("nonexistent")

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("ngoc-huyen-moi", engine=TtsEngine.ZEROTTS)

    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("maichi", engine=TtsEngine.NGHITTS)

    with pytest.raises(ValueError, match="Unknown TTS engine"):
        get_voice("maichi", engine="invalid")


def test_zerotts_model_specification_is_fully_pinned() -> None:
    """Test the ZeroTTS model specification and all its artifacts are pinned."""
    assert ZEROTTS_MODEL.repo == "zeroweight-ai/ZeroTTS-GGUF"
    assert len(ZEROTTS_MODEL.revision) == 40
    assert len(ZEROTTS_MODEL.core_artifacts) == 4
    assert len(ZEROTTS_MODEL.codec_files) == 4
    assert len(ZEROTTS_MODEL.artifacts) == 8
    assert len(ZEROTTS_MODEL.allow_patterns) == 6

    for artifact in ZEROTTS_MODEL.artifacts:
        assert artifact.remote_name
        assert artifact.local_name
        assert len(artifact.sha256) == 64
        assert artifact.sha256 == artifact.sha256.lower()
        assert int(artifact.sha256, 16) >= 0
