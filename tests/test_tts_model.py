"""Checksum-pinned NghiTTS voice catalogue tests."""

import pytest

from wyoming_vietnamese.const import TTS_CONFIG_FILE, TTS_MODEL_FILE
from wyoming_vietnamese.tts_model import (
    DEFAULT_TTS_VOICE,
    DEFAULT_TTS_VOICE_ID,
    TTS_VOICES,
    TTS_VOICES_BY_ID,
    get_voice,
)


def test_voice_catalog_has_unique_valid_entries() -> None:
    """Test every selectable voice has a unique ID and pinned artifacts."""
    assert len(TTS_VOICES) == 20
    assert len(TTS_VOICES_BY_ID) == len(TTS_VOICES)
    assert len({voice.name for voice in TTS_VOICES}) == len(TTS_VOICES)
    assert all(voice.id and voice.name for voice in TTS_VOICES)


def test_voice_artifacts_are_fully_pinned() -> None:
    """Test each voice pins a SHA-256 digest for every file it downloads."""
    for voice in TTS_VOICES:
        assert voice.artifacts == (voice.model, voice.config)
        assert voice.model.local_name == TTS_MODEL_FILE
        assert voice.config.local_name == TTS_CONFIG_FILE
        assert voice.model.remote_name == f"{voice.name}.onnx"
        assert voice.config.remote_name == f"{voice.name}.onnx.json"
        for artifact in voice.artifacts:
            assert len(artifact.sha256) == 64
            assert artifact.sha256 == artifact.sha256.lower()
            assert int(artifact.sha256, 16) >= 0


def test_get_voice_resolves_default_and_rejects_unknown_id() -> None:
    """Test catalog lookup is deterministic and reports supported choices."""
    assert get_voice(DEFAULT_TTS_VOICE_ID) is DEFAULT_TTS_VOICE
    with pytest.raises(ValueError, match="TTS_VOICE must be one of"):
        get_voice("missing")
