"""Tests for the remote model comparison tool."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, closing
from email.message import Message
from hashlib import sha256
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request

import pytest

from tools import check_models
from wyoming_vietnamese.stt_model import STT_MODEL
from wyoming_vietnamese.tts_model import TTS_VOICES


def _response(payload: object) -> AbstractContextManager[BytesIO]:
    """Build a response-like byte stream for the HTTP test double."""
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return closing(BytesIO(data))


class _ShortReadResponse:
    """Provide a response stream that returns only a few bytes per read."""

    def __init__(self, payload: bytes, chunk_size: int) -> None:
        """Initialize the stream and its maximum bytes-per-read size."""
        self._stream = BytesIO(payload)
        self._chunk_size = chunk_size

    def read(self, _size: int = -1) -> bytes:
        """Return at most the configured short chunk."""
        return self._stream.read(self._chunk_size)

    def close(self) -> None:
        """Close the underlying response stream."""
        self._stream.close()


def _fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
    """Serve deterministic catalogue, tree, and artifact responses."""
    url = request.full_url
    if url.endswith("/api/models"):
        return _response({"models": [voice.name for voice in TTS_VOICES] + ["test-voice"]})
    if "/tree/" in url:
        return _response(
            [
                {"path": artifact.remote_name, "lfs": {"oid": artifact.sha256}}
                for artifact in STT_MODEL.artifacts
            ]
        )
    if url.endswith(f"/api/models/{STT_MODEL.repo}"):
        return _response({"sha": STT_MODEL.revision})

    artifact_name = unquote(url.rsplit("/", 1)[-1])
    if artifact_name == "test-voice.onnx":
        return _response(b"test-voice-model")
    if artifact_name == "test-voice.onnx.json":
        return _response(b"test-voice-config")
    for voice in TTS_VOICES:
        for artifact in voice.artifacts:
            if artifact.remote_name == artifact_name:
                return _response(artifact.sha256.encode())
    raise AssertionError(f"Unexpected URL: {url}")


def test_compare_models_reports_new_voice_and_matching_stt(
    monkeypatch,
) -> None:
    """Test the report detects a remote voice while matching the STT pin."""
    monkeypatch.setattr(check_models, "urlopen", _fake_urlopen)
    monkeypatch.setattr(
        check_models,
        "_request_sha256",
        lambda url, _timeout: (
            next(
                artifact.sha256
                for voice in TTS_VOICES
                for artifact in voice.artifacts
                if artifact.remote_name == unquote(url.rsplit("/", 1)[-1])
            )
            if unquote(url.rsplit("/", 1)[-1]) not in {"test-voice.onnx", "test-voice.onnx.json"}
            else sha256(
                b"test-voice-model" if url.endswith("test-voice.onnx") else b"test-voice-config"
            ).hexdigest()
        ),
    )

    report = check_models.compare_models(timeout=1)

    assert report.tts.added_names == ("test-voice",)
    assert report.tts.removed_names == ()
    test_voice = next(voice for voice in report.tts.voices if voice.name == "test-voice")
    assert test_voice.status == "NEW"
    assert all(voice.status == "MATCH" for voice in report.tts.voices if voice.name != "test-voice")
    assert report.stt.revision_changed is False
    assert all(artifact.status == "MATCH" for artifact in report.stt.artifacts)
    assert report.needs_update is True


def test_skip_tts_hashes_still_compares_catalogue_and_stt(monkeypatch) -> None:
    """Test the low-bandwidth mode leaves artifact hashes unknown."""
    monkeypatch.setattr(check_models, "urlopen", _fake_urlopen)

    report = check_models.compare_models(timeout=1, hash_tts=False)

    assert report.tts.voices[0].artifacts[0].status == "UNKNOWN"
    assert report.tts.added_names == ("test-voice",)
    assert report.stt.needs_update is False


def test_compare_tts_includes_removed_local_voices(monkeypatch) -> None:
    """Test removed local voices expose removed artifact statuses."""
    monkeypatch.setattr(check_models, "_remote_voice_names", lambda _timeout: ("Ban Mai",))

    comparison = check_models._compare_tts(timeout=1, hash_artifacts=False)

    removed_voice = next(voice for voice in comparison.voices if voice.name == "Chiếu Thành")
    assert removed_voice.status == "REMOVED"
    assert all(artifact.status == "REMOVED" for artifact in removed_voice.artifacts)


def test_slugify_voice_name_matches_local_id_convention() -> None:
    """Test display names become usable suggested voice IDs."""
    assert check_models._slugify_voice_name("Ngọc Huyền (mới)") == "ngoc-huyen-moi"
    assert check_models._slugify_voice_name("Mỹ Tâm Real") == "my-tam-real"


def test_request_sha256_streams_remote_content(monkeypatch) -> None:
    """Test remote artifact hashing uses the complete response body."""
    payload = b"payload requiring several reads"
    monkeypatch.setattr(
        check_models,
        "urlopen",
        lambda *_args, **_kwargs: closing(_ShortReadResponse(payload, chunk_size=3)),
    )

    assert (
        check_models._request_sha256("https://example.test/model", 1) == sha256(payload).hexdigest()
    )


def test_request_json_rejects_oversized_response(monkeypatch) -> None:
    """Test JSON responses exceeding the safety limit are rejected."""
    monkeypatch.setattr(check_models, "_MAX_JSON_RESPONSE_BYTES", 3)
    monkeypatch.setattr(check_models, "urlopen", lambda *_args, **_kwargs: _response(b"{}{}"))

    with pytest.raises(check_models.ModelCheckError, match="exceeds"):
        check_models._request_json("https://example.test/catalogue", 1)


def test_request_sha256_treats_http_404_as_missing(monkeypatch) -> None:
    """Test a missing remote artifact is distinct from other request failures."""

    def raise_not_found(*_args, **_kwargs):
        raise HTTPError("https://example.test/model", 404, "Not Found", Message(), None)

    monkeypatch.setattr(check_models, "urlopen", raise_not_found)

    assert check_models._request_sha256("https://example.test/model", 1) is None


def test_compare_tts_reports_missing_remote_artifact(monkeypatch) -> None:
    """Test a missing TTS file becomes a removed voice and requires an update."""
    monkeypatch.setattr(check_models, "_remote_voice_names", lambda _timeout: ("Ban Mai",))

    def fake_hash(url: str, _timeout: int) -> str | None:
        return None if url.endswith("Ban%20Mai.onnx") else TTS_VOICES[0].config.sha256

    monkeypatch.setattr(check_models, "_request_sha256", fake_hash)

    comparison = check_models._compare_tts(timeout=1, hash_artifacts=True)
    ban_mai = next(voice for voice in comparison.voices if voice.name == "Ban Mai")

    assert ban_mai.status == "REMOVED"
    assert ban_mai.artifacts[0].remote_present is False
    assert comparison.needs_update is True


def test_remote_stt_artifacts_normalizes_sha256_prefix(monkeypatch) -> None:
    """Test Hugging Face LFS digests may include the sha256 prefix."""
    payload = [
        {"path": artifact.remote_name, "lfs": {"oid": f"sha256:{artifact.sha256}"}}
        for artifact in STT_MODEL.artifacts
    ]
    monkeypatch.setattr(check_models, "_request_json", lambda _url, _timeout: payload)

    artifacts = check_models._remote_stt_artifacts("revision", timeout=1)

    assert artifacts == {artifact.remote_name: artifact.sha256 for artifact in STT_MODEL.artifacts}
