"""Tests for the remote model comparison tool."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager, closing
from email.message import Message
from hashlib import sha256
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import quote, unquote
from urllib.request import Request

import pytest

from tools import check_models
from wyoming_vietnamese.const import ZEROTTS_REPO_ID
from wyoming_vietnamese.stt_model import STT_MODEL
from wyoming_vietnamese.tts_model import NGHITTS_VOICES, ZEROTTS_MODEL, ZEROTTS_VOICES


class _ResponseStream(BytesIO):
    """Provide a response byte stream that can hold optional response headers."""

    headers: Message | None = None
    url: str | None = None


def _response(
    payload: object,
    headers: Message | dict[str, str] | None = None,
    url: str | None = None,
) -> AbstractContextManager[_ResponseStream]:
    """Build a response-like byte stream for the HTTP test double."""
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    stream = _ResponseStream(data)
    stream.url = url
    if headers is not None:
        if isinstance(headers, Message):
            stream.headers = headers
        else:
            msg = Message()
            for key, value in headers.items():
                msg.add_header(key, value)
            stream.headers = msg
    return closing(stream)


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
        return _response({"models": [voice.name for voice in NGHITTS_VOICES] + ["test-voice"]})
    if f"/api/models/{ZEROTTS_REPO_ID}/tree/" in url:
        model_files = [
            {"type": "file", "path": artifact.remote_name, "lfs": {"oid": artifact.sha256}}
            for artifact in ZEROTTS_MODEL.artifacts
        ]
        voice_files = [
            {
                "type": "file",
                "path": voice.artifact.remote_name,
                "lfs": {"oid": voice.artifact.sha256},
            }
            for voice in ZEROTTS_VOICES
        ]
        return _response(model_files + voice_files)
    if url.endswith(f"/api/models/{ZEROTTS_REPO_ID}"):
        return _response({"sha": ZEROTTS_MODEL.revision})
    if f"/api/models/{STT_MODEL.repo}/tree/" in url or ("/tree/" in url and STT_MODEL.repo in url):
        return _response(
            [
                {"type": "file", "path": artifact.remote_name, "lfs": {"oid": artifact.sha256}}
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
    for voice in NGHITTS_VOICES:
        for artifact in voice.artifacts:
            if artifact.remote_name == artifact_name:
                return _response(artifact.sha256.encode())
    raise AssertionError(f"Unexpected URL: {url}")


def test_compare_models_reports_new_voice_and_matching_stt(
    monkeypatch,
) -> None:
    """Test the report detects a remote voice while matching the STT pin."""
    monkeypatch.setattr(check_models, "urlopen", _fake_urlopen)

    def _fake_sha(url: str, _timeout: int) -> str:
        name = unquote(url.rsplit("/", 1)[-1])
        if name == "test-voice.onnx":
            return sha256(b"test-voice-model").hexdigest()
        if name == "test-voice.onnx.json":
            return sha256(b"test-voice-config").hexdigest()
        for artifact in ZEROTTS_MODEL.artifacts:
            if artifact.remote_name == name:
                return artifact.sha256
        for voice in NGHITTS_VOICES:
            for artifact in voice.artifacts:
                if artifact.remote_name == name:
                    return artifact.sha256
        return "fake_sha"

    monkeypatch.setattr(check_models, "_request_sha256", _fake_sha)

    report = check_models.compare_models(timeout=1)

    assert report.tts.added_names == ("test-voice",)
    assert report.tts.removed_names == ()
    test_voice = next(voice for voice in report.tts.voices if voice.name == "test-voice")
    assert test_voice.status == "NEW"
    assert all(voice.status == "MATCH" for voice in report.tts.voices if voice.name != "test-voice")
    assert report.stt.revision_changed is False
    assert all(artifact.status == "MATCH" for artifact in report.stt.artifacts)
    assert report.zerotts.repo == ZEROTTS_REPO_ID
    assert report.zerotts.remote_revision == ZEROTTS_MODEL.revision
    assert report.zerotts.added_voice_ids == ()
    assert report.zerotts.removed_voice_ids == ()
    assert report.zerotts.needs_update is False
    assert all(artifact.remote_present for artifact in report.zerotts.artifacts)
    assert report.nghitts == report.tts
    assert report.needs_update is True


def test_skip_tts_hashes_still_compares_catalogue_and_stt(monkeypatch) -> None:
    """Test the low-bandwidth mode leaves artifact hashes unknown."""
    monkeypatch.setattr(check_models, "urlopen", _fake_urlopen)

    report = check_models.compare_models(timeout=1, hash_tts=False)

    assert report.tts.voices[0].artifacts[0].status == "UNKNOWN"
    assert report.tts.added_names == ("test-voice",)
    assert report.stt.needs_update is False
    assert report.zerotts.needs_update is False


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
        return None if url.endswith("Ban%20Mai.onnx") else NGHITTS_VOICES[0].config.sha256

    monkeypatch.setattr(check_models, "_request_sha256", fake_hash)

    comparison = check_models._compare_tts(timeout=1, hash_artifacts=True)
    ban_mai = next(voice for voice in comparison.voices if voice.name == "Ban Mai")

    assert ban_mai.status == "REMOVED"
    assert ban_mai.artifacts[0].remote_present is False
    assert comparison.needs_update is True


def test_remote_stt_artifacts_normalizes_sha256_prefix(monkeypatch) -> None:
    """Test Hugging Face LFS digests may include the sha256 prefix."""
    payload = [
        {"type": "file", "path": artifact.remote_name, "lfs": {"oid": f"sha256:{artifact.sha256}"}}
        for artifact in STT_MODEL.artifacts
    ]
    monkeypatch.setattr(check_models, "_request_json", lambda _url, _timeout: payload)

    artifacts = check_models._remote_stt_artifacts("revision", timeout=1)

    assert artifacts == {artifact.remote_name: artifact.sha256 for artifact in STT_MODEL.artifacts}


def test_extract_zerotts_voice_ids_handles_nested_and_flat_layouts() -> None:
    """Test voice ID extraction handles both flat npz files and directory layouts."""
    tree_paths = [
        "voices/voice_a/voice.npz",
        "voices/voice_b.npz",
        "voices/voice_c/nested/extra.wav",
        "voices/voice_a/other.json",
        "other_dir/voice_d.npz",
        "voices/",
        "voices/README.txt",
    ]
    assert check_models._extract_zerotts_voice_ids(tree_paths) == (
        "voice_a",
        "voice_b",
    )


def test_compare_zerotts_detects_added_and_removed_voices_and_missing_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ZeroTTS comparison flags catalogue drift and missing model files."""
    monkeypatch.setattr(check_models, "_remote_hf_revision", lambda _repo, _timeout: "new_rev")
    tree_entries = {
        "config.json": "sha_config",
        "voices/brand_new_voice.npz": None,
    }
    monkeypatch.setattr(
        check_models,
        "_remote_hf_tree_entries",
        lambda _repo, _rev, _timeout: tree_entries,
    )

    comparison = check_models._compare_zerotts(timeout=1)

    assert comparison.repo == ZEROTTS_REPO_ID
    assert comparison.remote_revision == "new_rev"
    assert comparison.added_voice_ids == ("brand_new_voice",)
    assert set(comparison.removed_voice_ids) == {voice.id for voice in ZEROTTS_VOICES}
    assert comparison.needs_update is True

    config_art = next(art for art in comparison.artifacts if art.remote_name == "config.json")
    assert config_art.remote_present is True

    missing_arts = [art for art in comparison.artifacts if not art.remote_present]
    assert len(missing_arts) == len(check_models._ZEROTTS_CORE_FILES) - 1


def test_compare_zerotts_fetches_voice_and_codec_hashes_when_missing_from_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ZeroTTS comparison fetches missing raw hashes for voice and codec files."""
    monkeypatch.setattr(
        check_models, "_remote_hf_revision", lambda _repo, _timeout: ZEROTTS_MODEL.revision
    )
    first_voice = ZEROTTS_VOICES[0]
    first_codec = ZEROTTS_MODEL.codec_files[0]
    tree_entries = {
        first_voice.artifact.remote_name: None,
        first_codec.remote_name: None,
        "config.json": "sha_config",
    }
    monkeypatch.setattr(
        check_models,
        "_remote_hf_tree_entries",
        lambda _repo, _rev, _timeout: tree_entries,
    )
    requested_urls: list[str] = []

    def fake_request_sha256(url: str, timeout: int) -> str:
        requested_urls.append(url)
        return "fetched_sha"

    monkeypatch.setattr(check_models, "_request_sha256", fake_request_sha256)

    comparison = check_models._compare_zerotts(timeout=1, hash_artifacts=True)

    assert any(first_codec.remote_name in url for url in requested_urls)
    assert any(first_voice.artifact.remote_name in url for url in requested_urls)
    assert all("%2F" not in url for url in requested_urls)

    voice_comp = next(v for v in comparison.voices if v.voice_id == first_voice.id)
    assert voice_comp.artifacts[0].remote_sha256 == "fetched_sha"


def test_compare_zerotts_marks_artifact_removed_when_fallback_hash_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ZeroTTS marks artifact as REMOVED when fallback hash returns None."""
    monkeypatch.setattr(
        check_models, "_remote_hf_revision", lambda _repo, _timeout: ZEROTTS_MODEL.revision
    )
    first_voice = ZEROTTS_VOICES[0]
    first_core = ZEROTTS_MODEL.artifacts[0]
    tree_entries = {
        first_voice.artifact.remote_name: None,
        first_core.remote_name: None,
    }
    monkeypatch.setattr(
        check_models,
        "_remote_hf_tree_entries",
        lambda _repo, _rev, _timeout: tree_entries,
    )
    monkeypatch.setattr(check_models, "_request_sha256", lambda _url, _timeout: None)

    comparison = check_models._compare_zerotts(timeout=1, hash_artifacts=True)

    core_comp = next(a for a in comparison.artifacts if a.remote_name == first_core.remote_name)
    assert core_comp.remote_present is False
    assert core_comp.status == "REMOVED"

    voice_comp = next(v for v in comparison.voices if v.voice_id == first_voice.id)
    assert voice_comp.artifacts[0].remote_present is False
    assert voice_comp.artifacts[0].status == "REMOVED"
    assert voice_comp.status == "REMOVED"
    assert comparison.needs_update is True

    comparison_no_hash = check_models._compare_zerotts(timeout=1, hash_artifacts=False)
    core_comp_no_hash = next(
        a for a in comparison_no_hash.artifacts if a.remote_name == first_core.remote_name
    )
    assert core_comp_no_hash.remote_present is True
    assert core_comp_no_hash.status == "UNKNOWN"

    voice_comp_no_hash = next(v for v in comparison_no_hash.voices if v.voice_id == first_voice.id)
    assert voice_comp_no_hash.artifacts[0].remote_present is True
    assert voice_comp_no_hash.artifacts[0].status == "UNKNOWN"


def test_report_to_dict_includes_zerotts_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test report serialization converts ZeroTTS comparison accurately."""
    monkeypatch.setattr(check_models, "urlopen", _fake_urlopen)
    monkeypatch.setattr(check_models, "_request_sha256", lambda _url, _timeout: "sha")

    report = check_models.compare_models(timeout=1, hash_tts=False)
    data = check_models._report_to_dict(report)

    assert "zerotts" in data
    zerotts_data = data["zerotts"]
    assert isinstance(zerotts_data, dict)
    assert zerotts_data["repo"] == ZEROTTS_REPO_ID
    assert zerotts_data["remote_revision"] == ZEROTTS_MODEL.revision
    assert zerotts_data["added_voice_ids"] == ()
    assert zerotts_data["removed_voice_ids"] == ()
    assert zerotts_data["needs_update"] is False
    artifacts_data = zerotts_data["artifacts"]
    assert isinstance(artifacts_data, list)
    assert len(artifacts_data) == len(check_models._ZEROTTS_CORE_FILES)


def test_print_zerotts_report_formats_sections(capsys: pytest.CaptureFixture[str]) -> None:
    """Test console printing formats ZeroTTS sections and markers."""
    comparison = check_models.ZeroTtsComparison(
        repo=ZEROTTS_REPO_ID,
        remote_revision="test_rev",
        remote_voice_ids=("voice1", "voice2"),
        local_voice_ids=("voice1", "voice3"),
        artifacts=(
            check_models.ArtifactComparison("config.json", "sha1", "sha1", True),
            check_models.ArtifactComparison("missing.bin", "sha2", None, False),
        ),
    )
    check_models._print_zerotts_report(comparison)
    captured = capsys.readouterr().out
    assert "ZeroTTS" in captured
    assert f"Repository:    {ZEROTTS_REPO_ID}" in captured
    assert "ADD     voice2" in captured
    assert "REMOVE  voice3" in captured
    assert "REMOVED missing.bin" in captured


def test_remote_hf_tree_entries_paginates_link_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HTTP Link rel=next header fetches subsequent pages affecting ZeroTTS comparison."""
    requested_urls: list[str] = []
    base_tree_url = (
        f"{check_models._HF_API_BASE_URL}/{quote(ZEROTTS_REPO_ID, safe='/')}/tree/"
        f"{quote(ZEROTTS_MODEL.revision, safe='')}?recursive=true"
    )
    page2_url = f"{base_tree_url}&cursor=page2"

    def fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
        url = request.full_url
        requested_urls.append(url)
        if url.endswith(f"/api/models/{ZEROTTS_REPO_ID}"):
            return _response({"sha": ZEROTTS_MODEL.revision})
        if url == base_tree_url:
            page1_files = [
                {"type": "file", "path": artifact.remote_name, "lfs": {"oid": artifact.sha256}}
                for artifact in ZEROTTS_MODEL.artifacts
            ]
            headers = Message()
            headers.add_header("Link", f'<{page2_url}>; rel="next"')
            return _response(page1_files, headers=headers)
        if url == page2_url:
            page2_files = [
                {
                    "type": "file",
                    "path": voice.artifact.remote_name,
                    "lfs": {"oid": voice.artifact.sha256},
                }
                for voice in ZEROTTS_VOICES
            ] + [
                {
                    "type": "file",
                    "path": "voices/new_voice.npz",
                    "lfs": {"oid": f"sha256:{'1' * 64}"},
                }
            ]
            return _response(page2_files)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)

    comparison = check_models._compare_zerotts(timeout=1, hash_artifacts=False)

    assert page2_url in requested_urls
    assert "new_voice" in comparison.added_voice_ids
    assert all(voice_comp.artifacts[0].remote_present for voice_comp in comparison.voices)


def test_remote_hf_tree_entries_rejects_external_next_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test remote tree fetch rejects next Link headers outside the Hugging Face tree endpoint."""
    base_tree_url = (
        f"{check_models._HF_API_BASE_URL}/{quote(ZEROTTS_REPO_ID, safe='/')}/tree/"
        f"{quote(ZEROTTS_MODEL.revision, safe='')}?recursive=true"
    )

    def fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
        url = request.full_url
        if url == base_tree_url:
            headers = Message()
            headers.add_header("Link", '<https://evil.test/other/tree>; rel="next"')
            return _response([], headers=headers)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)

    with pytest.raises(
        check_models.ModelCheckError, match="outside expected Hugging Face tree endpoint"
    ):
        check_models._remote_hf_tree_entries(ZEROTTS_REPO_ID, ZEROTTS_MODEL.revision, timeout=1)


def test_remote_hf_tree_entries_rejects_redirect_leaving_tree_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test remote tree fetch rejects redirects leaving the Hugging Face tree endpoint."""
    base_tree_url = (
        f"{check_models._HF_API_BASE_URL}/{quote(ZEROTTS_REPO_ID, safe='/')}/tree/"
        f"{quote(ZEROTTS_MODEL.revision, safe='')}?recursive=true"
    )

    def fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
        url = request.full_url
        if url == base_tree_url:
            return _response([], url="https://evil.test/redirected")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)

    with pytest.raises(
        check_models.ModelCheckError, match="outside expected Hugging Face tree endpoint"
    ):
        check_models._remote_hf_tree_entries(ZEROTTS_REPO_ID, ZEROTTS_MODEL.revision, timeout=1)


def test_remote_hf_tree_entries_rejects_repeated_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test remote tree fetch raises ModelCheckError when a pagination URL repeats."""
    base_tree_url = (
        f"{check_models._HF_API_BASE_URL}/{quote(ZEROTTS_REPO_ID, safe='/')}/tree/"
        f"{quote(ZEROTTS_MODEL.revision, safe='')}?recursive=true"
    )
    page2_url = f"{base_tree_url}&cursor=page2"

    def fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
        url = request.full_url
        if url == base_tree_url:
            headers = Message()
            headers.add_header("Link", f'<{page2_url}>; rel="next"')
            return _response([], headers=headers)
        if url == page2_url:
            headers = Message()
            headers.add_header("Link", f'<{base_tree_url}>; rel="next"')
            return _response([], headers=headers)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)

    with pytest.raises(check_models.ModelCheckError, match="Repeated URL"):
        check_models._remote_hf_tree_entries(ZEROTTS_REPO_ID, ZEROTTS_MODEL.revision, timeout=1)


def test_remote_hf_tree_entries_rejects_exceeding_max_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test remote tree fetch raises ModelCheckError when page count exceeds the limit."""
    base_tree_url = (
        f"{check_models._HF_API_BASE_URL}/{quote(ZEROTTS_REPO_ID, safe='/')}/tree/"
        f"{quote(ZEROTTS_MODEL.revision, safe='')}?recursive=true"
    )
    monkeypatch.setattr(check_models, "_MAX_TREE_PAGES", 2)

    def fake_urlopen(request: Request, **_kwargs: object) -> AbstractContextManager[BytesIO]:
        url = request.full_url
        if url == base_tree_url:
            headers = Message()
            headers.add_header("Link", f'<{base_tree_url}&cursor=page2>; rel="next"')
            return _response([], headers=headers)
        if url == f"{base_tree_url}&cursor=page2":
            headers = Message()
            headers.add_header("Link", f'<{base_tree_url}&cursor=page3>; rel="next"')
            return _response([], headers=headers)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(check_models, "urlopen", fake_urlopen)

    with pytest.raises(check_models.ModelCheckError, match="Exceeded maximum tree page count"):
        check_models._remote_hf_tree_entries(ZEROTTS_REPO_ID, ZEROTTS_MODEL.revision, timeout=1)


def test_remote_hf_tree_entries_skips_directory_entries_with_npz_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test remote tree fetch skips directory entries with .npz paths."""
    payload = [
        {
            "type": "file",
            "path": "voices/valid_voice/voice.npz",
            "lfs": {"oid": "sha256:" + "a" * 64},
        },
        {"type": "directory", "path": "voices/fake_voice.npz"},
        {"type": "directory", "path": "voices/another_voice/voice.npz"},
    ]
    monkeypatch.setattr(check_models, "_request_json", lambda _url, _timeout: payload)

    entries = check_models._remote_hf_tree_entries("test-repo", "main", timeout=1)

    assert "voices/valid_voice/voice.npz" in entries
    assert "voices/fake_voice.npz" not in entries
    assert "voices/another_voice/voice.npz" not in entries

    voice_ids = check_models._extract_zerotts_voice_ids(entries)
    assert voice_ids == ("valid_voice",)
    assert "fake_voice" not in voice_ids
    assert "another_voice" not in voice_ids
