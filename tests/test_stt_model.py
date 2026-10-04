"""Checksum-pinned Sherpa-ONNX speech-to-text model tests."""

import pytest

from wyoming_vietnamese.const import SttEngine
from wyoming_vietnamese.stt_model import (
    DEFAULT_STT_MODEL,
    GIPFORMER_MODEL,
    STT_MODELS,
    ZIPFORMER_MODEL,
    get_stt_model,
)


def test_default_stt_model_alias() -> None:
    """Test default STT model aliases point to the default engine model."""
    assert DEFAULT_STT_MODEL is ZIPFORMER_MODEL


def test_zipformer_model_identity_is_fully_pinned() -> None:
    """Test the Zipformer model is pinned to one repository at one immutable commit."""
    assert ZIPFORMER_MODEL.repo == "hynt/Zipformer-30M-RNNT-6000h"
    assert len(ZIPFORMER_MODEL.revision) == 40
    assert int(ZIPFORMER_MODEL.revision, 16) >= 0
    assert ZIPFORMER_MODEL.description == "Zipformer RNNT STT model"


def test_gipformer_model_identity_is_fully_pinned() -> None:
    """Test the Gipformer model is pinned to its repository at one immutable commit."""
    assert GIPFORMER_MODEL.repo == "g-group-ai-lab/gipformer1.5-68M-rnnt"
    assert len(GIPFORMER_MODEL.revision) == 40
    assert int(GIPFORMER_MODEL.revision, 16) >= 0
    assert GIPFORMER_MODEL.description == "Gipformer 1.5 68M RNNT STT model"


def test_stt_artifacts_are_unique_and_fully_pinned() -> None:
    """Test every downloaded file pins a SHA-256 digest exactly once."""
    for model in (ZIPFORMER_MODEL, GIPFORMER_MODEL):
        artifacts = model.artifacts
        assert artifacts == (*model.graphs, model.tokenizer)
        assert len({artifact.remote_name for artifact in artifacts}) == len(artifacts)
        for artifact in artifacts:
            assert artifact.remote_name
            assert artifact.local_name
            assert len(artifact.sha256) == 64
            assert artifact.sha256 == artifact.sha256.lower()
            assert int(artifact.sha256, 16) >= 0


def test_stt_graphs_cover_the_transducer_and_download_list_matches() -> None:
    """Test the pinned set is a complete transducer plus its tokenizer source."""
    for model in (ZIPFORMER_MODEL, GIPFORMER_MODEL):
        assert [artifact.local_name for artifact in model.graphs] == [
            "encoder.onnx",
            "decoder.onnx",
            "joiner.onnx",
        ]
        assert model.tokenizer.remote_name == "bpe.model"
        assert model.allow_patterns == tuple(artifact.remote_name for artifact in model.artifacts)


def test_get_stt_model_resolves_supported_engines() -> None:
    """Test resolving models by engine name with case insensitivity."""
    assert get_stt_model() is ZIPFORMER_MODEL
    assert get_stt_model(SttEngine.ZIPFORMER) is ZIPFORMER_MODEL
    assert get_stt_model("zipformer") is ZIPFORMER_MODEL
    assert get_stt_model("Zipformer") is ZIPFORMER_MODEL
    assert get_stt_model(SttEngine.GIPFORMER) is GIPFORMER_MODEL
    assert get_stt_model("gipformer") is GIPFORMER_MODEL
    assert get_stt_model("GIPFORMER") is GIPFORMER_MODEL


def test_get_stt_model_rejects_unknown_engine() -> None:
    """Test get_stt_model raises ValueError for unknown engine."""
    with pytest.raises(ValueError, match="STT_ENGINE must be one of: zipformer, gipformer"):
        get_stt_model("whisper")


def test_stt_models_catalogue_contains_expected_engines() -> None:
    """Test STT_MODELS maps all supported engine enums to model specs."""
    assert set(STT_MODELS) == {SttEngine.ZIPFORMER, SttEngine.GIPFORMER}
    assert STT_MODELS[SttEngine.ZIPFORMER] is ZIPFORMER_MODEL
    assert STT_MODELS[SttEngine.GIPFORMER] is GIPFORMER_MODEL
