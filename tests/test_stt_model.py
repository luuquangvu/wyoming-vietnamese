"""Checksum-pinned Sherpa-ONNX speech-to-text model tests."""

from wyoming_vietnamese.stt_model import STT_MODEL


def test_stt_model_identity_is_fully_pinned() -> None:
    """Test the model is pinned to one repository at one immutable commit."""
    assert STT_MODEL.repo == "hynt/Zipformer-30M-RNNT-6000h"
    assert len(STT_MODEL.revision) == 40
    assert int(STT_MODEL.revision, 16) >= 0


def test_stt_artifacts_are_unique_and_fully_pinned() -> None:
    """Test every downloaded file pins a SHA-256 digest exactly once."""
    artifacts = STT_MODEL.artifacts
    assert artifacts == (*STT_MODEL.graphs, STT_MODEL.tokenizer)
    assert len({artifact.remote_name for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        assert artifact.remote_name
        assert artifact.local_name
        assert len(artifact.sha256) == 64
        assert artifact.sha256 == artifact.sha256.lower()
        assert int(artifact.sha256, 16) >= 0


def test_stt_graphs_cover_the_transducer_and_download_list_matches() -> None:
    """Test the pinned set is a complete transducer plus its tokenizer source."""
    assert [artifact.local_name for artifact in STT_MODEL.graphs] == [
        "encoder.onnx",
        "decoder.onnx",
        "joiner.onnx",
    ]
    assert STT_MODEL.tokenizer.remote_name == "bpe.model"
    assert STT_MODEL.allow_patterns == tuple(
        artifact.remote_name for artifact in STT_MODEL.artifacts
    )
