"""Tests for the remote model comparison tool."""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, closing
from dataclasses import replace
from email.message import Message
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote
from urllib.request import Request

import pytest

from tools import check_models
from wyoming_vietnamese.const import ZEROTTS_REPO_ID, ZeroTtsFile
from wyoming_vietnamese.stt_model import GIPFORMER_MODEL, ZIPFORMER_MODEL
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
    for stt_spec in (ZIPFORMER_MODEL, GIPFORMER_MODEL):
        if f"/api/models/{stt_spec.repo}/tree/" in url or (
            "/tree/" in url and stt_spec.repo in url
        ):
            return _response(
                [
                    {"type": "file", "path": artifact.remote_name, "lfs": {"oid": artifact.sha256}}
                    for artifact in stt_spec.artifacts
                ]
            )
        if url.endswith(f"/api/models/{stt_spec.repo}"):
            return _response({"sha": stt_spec.revision})

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
        for artifact in ZIPFORMER_MODEL.artifacts
    ]
    monkeypatch.setattr(check_models, "_request_json", lambda _url, _timeout: payload)

    artifacts = check_models._remote_stt_artifacts("revision", timeout=1, spec=ZIPFORMER_MODEL)

    expected = {artifact.remote_name: artifact.sha256 for artifact in ZIPFORMER_MODEL.artifacts}
    assert artifacts == expected


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
    assert len(missing_arts) == len(check_models._ZEROTTS_CORE_FILES) - 2


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

    assert "stt" in data
    stt_data = data["stt"]
    assert isinstance(stt_data, dict)
    assert stt_data["repo"] == ZIPFORMER_MODEL.repo
    assert stt_data["local_revision"] == ZIPFORMER_MODEL.revision
    assert "models" in stt_data
    models_list = stt_data["models"]
    assert isinstance(models_list, list)
    assert len(models_list) == 2

    for entry, (expected_engine, expected_spec) in zip(
        models_list,
        [("zipformer", ZIPFORMER_MODEL), ("gipformer", GIPFORMER_MODEL)],
        strict=True,
    ):
        assert entry["engine"] == expected_engine
        assert entry["repo"] == expected_spec.repo
        assert entry["local_revision"] == expected_spec.revision
        assert entry["remote_revision"] == expected_spec.revision
        assert entry["needs_update"] is False
        assert len(entry["artifacts"]) == len(expected_spec.artifacts)
        for art_dict, art_spec in zip(entry["artifacts"], expected_spec.artifacts, strict=True):
            assert art_dict["remote_name"] == art_spec.remote_name
            assert art_dict["local_sha256"] == art_spec.sha256
            assert art_dict["remote_sha256"] == art_spec.sha256
            assert art_dict["remote_present"] is True
            assert art_dict["status"] == "MATCH"


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


def test_print_stt_report_formats_all_models(capsys: pytest.CaptureFixture[str]) -> None:
    """Test console printing formats all STT models."""
    comparison = check_models.SttComparison(
        models=(
            check_models.SttModelComparison(
                engine="zipformer",
                repo="hynt/Zipformer",
                local_revision="rev1",
                remote_revision="rev1",
                artifacts=(check_models.ArtifactComparison("encoder.onnx", "sha1", "sha1", True),),
            ),
            check_models.SttModelComparison(
                engine="gipformer",
                repo="g-group-ai-lab/gipformer",
                local_revision="rev2",
                remote_revision="rev3",
                artifacts=(check_models.ArtifactComparison("decoder.onnx", "sha2", "sha3", True),),
            ),
        )
    )
    check_models._print_stt_report(comparison)
    captured = capsys.readouterr().out
    assert "STT" in captured
    assert "[zipformer] Repository: hynt/Zipformer" in captured
    assert "[gipformer] Repository: g-group-ai-lab/gipformer" in captured
    assert "CHANGED revision local=rev2 remote=rev3" in captured


def test_stt_comparison_properties() -> None:
    """Test SttComparison properties compute aggregate statuses correctly."""
    m1 = check_models.SttModelComparison(
        engine="zipformer",
        repo="repo1",
        local_revision="rev1",
        remote_revision="rev1",
        artifacts=(check_models.ArtifactComparison("a1", "s1", "s1", True),),
    )
    m2 = check_models.SttModelComparison(
        engine="gipformer",
        repo="repo2",
        local_revision="rev2",
        remote_revision="rev2",
        artifacts=(check_models.ArtifactComparison("a2", "s2", "s2", True),),
    )
    comp = check_models.SttComparison((m1, m2))
    assert comp.repo == "repo1"
    assert comp.local_revision == "rev1"
    assert comp.remote_revision == "rev1"
    assert comp.revision_changed is False
    assert comp.needs_update is False
    assert len(comp.artifacts) == 2

    comp_reordered = check_models.SttComparison((m2, m1))
    assert comp_reordered.repo == "repo1"
    assert comp_reordered.local_revision == "rev1"
    assert comp_reordered.remote_revision == "rev1"

    m2_changed = check_models.SttModelComparison(
        engine="gipformer",
        repo="repo2",
        local_revision="rev2",
        remote_revision="rev_changed",
        artifacts=(check_models.ArtifactComparison("a2", "s2", "s2", True),),
    )
    comp_changed = check_models.SttComparison((m1, m2_changed))
    assert comp_changed.revision_changed is True
    assert comp_changed.needs_update is True

    for art in (
        check_models.ArtifactComparison("a2", "s2", "s_other", True),  # CHANGED
        check_models.ArtifactComparison("a2", None, "s_remote", True),  # NEW
        check_models.ArtifactComparison("a2", "s2", None, False),  # REMOVED
    ):
        m2_variant = check_models.SttModelComparison(
            engine="gipformer",
            repo="repo2",
            local_revision="rev2",
            remote_revision="rev2",
            artifacts=(art,),
        )
        comp_variant = check_models.SttComparison((m1, m2_variant))
        assert comp_variant.revision_changed is False
        assert comp_variant.needs_update is True

    m2_unknown = check_models.SttModelComparison(
        engine="gipformer",
        repo="repo2",
        local_revision="rev2",
        remote_revision="rev2",
        artifacts=(check_models.ArtifactComparison("a2", "s2", None, True),),  # UNKNOWN
    )
    comp_unknown = check_models.SttComparison((m1, m2_unknown))
    assert comp_unknown.revision_changed is False
    assert comp_unknown.needs_update is False


def test_remote_hf_revision_rejects_non_hex_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test remote revision metadata cannot inject arbitrary text into source pins."""
    monkeypatch.setattr(
        check_models,
        "_request_json",
        lambda _url, _timeout: {"sha": 'bad"; import os; #'},
    )
    with pytest.raises(check_models.ModelCheckError, match="invalid revision"):
        check_models._remote_hf_revision("owner/model", timeout=1)


def _mock_zerotts_remote_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock remote revision and tree endpoints for ZeroTTS repo."""
    monkeypatch.setattr(
        check_models, "_remote_hf_revision", lambda _repo, _timeout: ZEROTTS_MODEL.revision
    )
    monkeypatch.setattr(check_models, "_remote_hf_tree_entries", lambda _repo, _rev, _timeout: {})


def _find_int8_artifact(
    comparison: check_models.ZeroTtsComparison,
) -> check_models.ArtifactComparison:
    """Find the INT8 codec artifact comparison within a ZeroTTS comparison."""
    return next(
        a
        for a in comparison.artifacts
        if a.remote_name == f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}"
    )


def test_compare_zerotts_handles_local_bundled_asset_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test ZeroTTS comparison accepts bundled assets that match their pinned hash."""
    _mock_zerotts_remote_repo(monkeypatch)
    fake_assets = tmp_path / "assets"
    int8_file = fake_assets / "onnx" / "codec" / ZeroTtsFile.DECODE_STEP_INT8_ONNX
    int8_file.parent.mkdir(parents=True, exist_ok=True)
    int8_file.write_bytes(b"int8_model_content")
    int8_sha = sha256(b"int8_model_content").hexdigest()
    matched_model = replace(
        ZEROTTS_MODEL,
        codec_files=tuple(
            replace(artifact, sha256=int8_sha)
            if artifact.remote_name == f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}"
            else artifact
            for artifact in ZEROTTS_MODEL.codec_files
        ),
    )
    monkeypatch.setattr(check_models, "ZEROTTS_MODEL", matched_model)

    comparison = check_models._compare_zerotts(
        timeout=1,
        hash_artifacts=True,
        bundled_assets_dir=fake_assets,
    )
    int8_comp = _find_int8_artifact(comparison)
    assert int8_comp.remote_present is True
    assert int8_comp.remote_sha256 == int8_sha
    assert int8_comp.status == "MATCH"


def test_compare_zerotts_handles_local_bundled_asset_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test ZeroTTS comparison marks local bundled assets REMOVED when file is missing."""
    _mock_zerotts_remote_repo(monkeypatch)
    empty_assets = tmp_path / "empty_assets"
    empty_assets.mkdir(parents=True, exist_ok=True)

    comparison = check_models._compare_zerotts(
        timeout=1,
        hash_artifacts=False,
        bundled_assets_dir=empty_assets,
    )
    int8_comp = _find_int8_artifact(comparison)
    assert int8_comp.remote_present is False
    assert int8_comp.status == "REMOVED"


def test_compare_zerotts_rejects_tampered_local_bundled_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test ZeroTTS comparison refuses to self-certify a modified bundled asset."""
    _mock_zerotts_remote_repo(monkeypatch)
    fake_assets = tmp_path / "assets"
    flat_int8 = fake_assets / ZeroTtsFile.DECODE_STEP_INT8_ONNX
    flat_int8.parent.mkdir(parents=True, exist_ok=True)
    flat_int8.write_bytes(b"different_content")

    with pytest.raises(check_models.ModelCheckError, match="self-certify"):
        check_models._compare_zerotts(
            timeout=1,
            hash_artifacts=True,
            bundled_assets_dir=fake_assets,
        )


def test_compare_zerotts_handles_local_bundled_asset_unhashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test ZeroTTS comparison skips local bundled asset hashing when hash_artifacts is False."""
    _mock_zerotts_remote_repo(monkeypatch)
    fake_assets = tmp_path / "assets"
    int8_file = fake_assets / "onnx" / "codec" / ZeroTtsFile.DECODE_STEP_INT8_ONNX
    int8_file.parent.mkdir(parents=True, exist_ok=True)
    int8_file.write_bytes(b"int8_model_content")

    comparison = check_models._compare_zerotts(
        timeout=1,
        hash_artifacts=False,
        bundled_assets_dir=fake_assets,
    )
    int8_comp = _find_int8_artifact(comparison)
    assert int8_comp.remote_present is True
    assert int8_comp.remote_sha256 is None
    assert int8_comp.status == "UNKNOWN"


def _assert_file_contains(path: Path, *expected_substrings: str) -> None:
    """Assert that the specified file contains all expected substrings."""
    content = path.read_text(encoding="utf-8")
    for substring in expected_substrings:
        assert substring in content


def _make_test_report(
    *,
    stt_models: tuple[check_models.SttModelComparison, ...] = (),
    zerotts_revision: str = ZEROTTS_MODEL.revision,
    zerotts_remote_voice_ids: tuple[str, ...] = (),
    zerotts_artifacts: tuple[check_models.ArtifactComparison, ...] = (),
    zerotts_voices: tuple[check_models.VoiceComparison, ...] = (),
    tts_voices: tuple[check_models.VoiceComparison, ...] = (),
    tts_remote_names: tuple[str, ...] = (),
) -> check_models.ComparisonReport:
    """Build a ComparisonReport fixture for tests with convenient defaults."""
    return check_models.ComparisonReport(
        stt=check_models.SttComparison(models=stt_models),
        zerotts=check_models.ZeroTtsComparison(
            repo=ZEROTTS_REPO_ID,
            remote_revision=zerotts_revision,
            remote_voice_ids=zerotts_remote_voice_ids,
            local_voice_ids=(),
            artifacts=zerotts_artifacts,
            voices=zerotts_voices,
        ),
        tts=check_models.TtsComparison(
            remote_names=tts_remote_names,
            local_names=(),
            voices=tts_voices,
            hashes_checked=True,
        ),
    )


def _setup_two_transaction_files(
    tmp_path: Path,
) -> tuple[Path, Path, list[tuple[Path, str, str]]]:
    """Create two text files and return them alongside standard update tuples."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("orig1", encoding="utf-8")
    file2.write_text("orig2", encoding="utf-8")
    return file1, file2, [(file1, "new1", "orig1"), (file2, "new2", "orig2")]


def _setup_zerotts_model_files(
    tmp_path: Path,
    revision: str = "old_rev",
) -> tuple[Path, Path]:
    """Create stub stt_model.py and tts_model.py with specified ZeroTTS revision."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        "ZEROTTS_MODEL = ZeroTtsModelSpec(\n"
        "    repo=ZEROTTS_REPO_ID,\n"
        f'    revision="{revision}",\n'
        ")\n",
        encoding="utf-8",
    )
    return stt_file, tts_file


def test_update_model_pins_updates_stt_and_tts_files(tmp_path: Path) -> None:
    """Test update_model_pins modifies pinned hashes and revisions in source files."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text(
        "ZIPFORMER_MODEL = SttModelSpec(\n"
        '    repo="hynt/Zipformer-30M-RNNT-6000h",\n'
        '    revision="old_stt_rev",\n'
        "    graphs=(\n"
        "        SttArtifact(\n"
        '            "encoder-epoch-20-avg-10.int8.onnx",\n'
        '            "encoder.onnx",\n'
        '            "1111111111111111111111111111111111111111111111111111111111111111",\n'
        "        ),\n"
        "    ),\n"
        ")\n",
        encoding="utf-8",
    )

    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        'ZEROTTS_CODEC_INT8_SHA256 = "'
        '2222222222222222222222222222222222222222222222222222222222222222"\n'
        "ZEROTTS_MODEL = ZeroTtsModelSpec(\n"
        "    repo=ZEROTTS_REPO_ID,\n"
        '    revision="old_tts_rev",\n'
        ")\n",
        encoding="utf-8",
    )

    new_stt_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    new_int8_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    stt_comp = check_models.SttComparison(
        models=(
            check_models.SttModelComparison(
                engine="zipformer",
                repo="hynt/Zipformer-30M-RNNT-6000h",
                local_revision="old_stt_rev",
                remote_revision="new_stt_rev",
                artifacts=(
                    check_models.ArtifactComparison(
                        "encoder-epoch-20-avg-10.int8.onnx",
                        "1111111111111111111111111111111111111111111111111111111111111111",
                        new_stt_sha,
                        True,
                    ),
                ),
            ),
        )
    )

    zerotts_comp = check_models.ZeroTtsComparison(
        repo=ZEROTTS_REPO_ID,
        remote_revision="new_tts_rev",
        remote_voice_ids=(),
        local_voice_ids=(),
        artifacts=(
            check_models.ArtifactComparison(
                f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}",
                "2222222222222222222222222222222222222222222222222222222222222222",
                new_int8_sha,
                True,
            ),
        ),
    )

    tts_comp = check_models.TtsComparison(
        remote_names=(),
        local_names=(),
        voices=(),
        hashes_checked=True,
    )

    report = check_models.ComparisonReport(tts=tts_comp, stt=stt_comp, zerotts=zerotts_comp)
    changes = check_models.update_model_pins(report, models_dir=tmp_path)

    assert changes == [
        "STT [zipformer] revision: old_stt_rev -> new_stt_rev",
        (
            f"STT [zipformer] encoder-epoch-20-avg-10.int8.onnx: "
            f"1111111111111111111111111111111111111111111111111111111111111111 -> {new_stt_sha}"
        ),
        "ZeroTTS revision: old_tts_rev -> new_tts_rev",
        (
            f"ZeroTTS local codec onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}: "
            f"2222222222222222222222222222222222222222222222222222222222222222 -> {new_int8_sha}"
        ),
    ]
    _assert_file_contains(stt_file, 'revision="new_stt_rev"', new_stt_sha)
    _assert_file_contains(tts_file, 'revision="new_tts_rev"', new_int8_sha)


def test_update_model_pins_targets_specific_artifact_declarations(tmp_path: Path) -> None:
    """Test updating pins targets specific artifact declarations even if hashes are identical."""
    shared_stt_sha = "1111111111111111111111111111111111111111111111111111111111111111"
    shared_voice_sha = "2222222222222222222222222222222222222222222222222222222222222222"
    shared_core_sha = "3333333333333333333333333333333333333333333333333333333333333333"
    shared_nghitts_sha = "4444444444444444444444444444444444444444444444444444444444444444"

    new_stt_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    new_voice_sha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    new_core_sha = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    new_nghitts_sha = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text(
        "ZIPFORMER_MODEL = SttModelSpec(\n"
        '    repo="hynt/Zipformer-30M-RNNT-6000h",\n'
        '    revision="rev",\n'
        "    graphs=(\n"
        f'        SttArtifact("encoder.onnx", "enc", "{shared_stt_sha}"),\n'
        f'        SttArtifact("decoder.onnx", "dec", "{shared_stt_sha}"),\n'
        "    ),\n"
        ")\n",
        encoding="utf-8",
    )

    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        "ZEROTTS_MODEL = ZeroTtsModelSpec(\n"
        "    repo=ZEROTTS_REPO_ID,\n"
        '    revision="rev",\n'
        f'    config=ZeroTtsArtifact(ZeroTtsFile.CONFIG, "cfg", "{shared_core_sha}"),\n'
        f'    tokenizer=ZeroTtsArtifact(ZeroTtsFile.TOKENIZER, "tok", "{shared_core_sha}"),\n'
        ")\n"
        "ZEROTTS_VOICES = (\n"
        '    _zerotts_voice("maichi", "Mai Chi", "nu", "Nữ miền Bắc (nhẹ nhàng)",\n'
        f'        "{shared_voice_sha}"),\n'
        f'    _zerotts_voice("baotrang", "Bao Trang", "nu", "desc", "{shared_voice_sha}"),\n'
        ")\n"
        "NGHITTS_VOICES = (\n"
        f'    _voice("duy-onyx-moi", "Duy Onyx (mới)", "{shared_nghitts_sha}"),\n'
        f'    _voice("duy-oryx", "Duy Oryx", "{shared_nghitts_sha}"),\n'
        ")\n",
        encoding="utf-8",
    )

    stt_comp = check_models.SttComparison(
        models=(
            check_models.SttModelComparison(
                engine="zipformer",
                repo="hynt/Zipformer-30M-RNNT-6000h",
                local_revision="rev",
                remote_revision="rev",
                artifacts=(
                    check_models.ArtifactComparison(
                        "encoder.onnx",
                        shared_stt_sha,
                        new_stt_sha,
                        True,
                    ),
                    check_models.ArtifactComparison(
                        "decoder.onnx",
                        shared_stt_sha,
                        shared_stt_sha,
                        True,
                    ),
                ),
            ),
        )
    )

    zerotts_comp = check_models.ZeroTtsComparison(
        repo=ZEROTTS_REPO_ID,
        remote_revision="rev",
        remote_voice_ids=(),
        local_voice_ids=(),
        artifacts=(
            check_models.ArtifactComparison(
                ZeroTtsFile.CONFIG,
                shared_core_sha,
                new_core_sha,
                True,
            ),
            check_models.ArtifactComparison(
                ZeroTtsFile.TOKENIZER,
                shared_core_sha,
                shared_core_sha,
                True,
            ),
        ),
        voices=(
            check_models.VoiceComparison(
                name="Mai Chi",
                voice_id="maichi",
                artifacts=(
                    check_models.ArtifactComparison(
                        "voices/maichi/voice.npz",
                        shared_voice_sha,
                        new_voice_sha,
                        True,
                    ),
                ),
            ),
            check_models.VoiceComparison(
                name="Bao Trang",
                voice_id="baotrang",
                artifacts=(
                    check_models.ArtifactComparison(
                        "voices/baotrang/voice.npz",
                        shared_voice_sha,
                        shared_voice_sha,
                        True,
                    ),
                ),
            ),
        ),
    )

    tts_comp = check_models.TtsComparison(
        remote_names=(),
        local_names=(),
        voices=(
            check_models.VoiceComparison(
                name="Duy Onyx (mới)",
                voice_id="duy-onyx-moi",
                artifacts=(
                    check_models.ArtifactComparison(
                        "Duy Onyx (mới).onnx",
                        shared_nghitts_sha,
                        new_nghitts_sha,
                        True,
                    ),
                ),
            ),
            check_models.VoiceComparison(
                name="Duy Oryx",
                voice_id="duy-oryx",
                artifacts=(
                    check_models.ArtifactComparison(
                        "Duy Onyx.onnx",
                        shared_nghitts_sha,
                        shared_nghitts_sha,
                        True,
                    ),
                ),
            ),
        ),
        hashes_checked=True,
    )

    report = check_models.ComparisonReport(tts=tts_comp, stt=stt_comp, zerotts=zerotts_comp)
    changes = check_models.update_model_pins(report, models_dir=tmp_path)

    assert changes == [
        f"STT [zipformer] encoder.onnx: {shared_stt_sha} -> {new_stt_sha}",
        f"ZeroTTS core {ZeroTtsFile.CONFIG}: {shared_core_sha} -> {new_core_sha}",
        f"ZeroTTS voice [Mai Chi] voices/maichi/voice.npz: {shared_voice_sha} -> {new_voice_sha}",
        (
            f"NghiTTS voice [Duy Onyx (mới)] Duy Onyx (mới).onnx: "
            f"{shared_nghitts_sha} -> {new_nghitts_sha}"
        ),
    ]

    stt_text = stt_file.read_text(encoding="utf-8")
    assert f'SttArtifact("encoder.onnx", "enc", "{new_stt_sha}")' in stt_text
    assert f'SttArtifact("decoder.onnx", "dec", "{shared_stt_sha}")' in stt_text

    tts_text = tts_file.read_text(encoding="utf-8")
    assert f'config=ZeroTtsArtifact(ZeroTtsFile.CONFIG, "cfg", "{new_core_sha}")' in tts_text
    assert (
        f'tokenizer=ZeroTtsArtifact(ZeroTtsFile.TOKENIZER, "tok", "{shared_core_sha}")' in tts_text
    )
    assert (
        '_zerotts_voice("maichi", "Mai Chi", "nu", "Nữ miền Bắc (nhẹ nhàng)",\n'
        f'        "{new_voice_sha}")' in tts_text
    )
    assert (
        f'_zerotts_voice("baotrang", "Bao Trang", "nu", "desc", "{shared_voice_sha}")' in tts_text
    )
    assert f'_voice("duy-onyx-moi", "Duy Onyx (mới)", "{new_nghitts_sha}")' in tts_text
    assert f'_voice("duy-oryx", "Duy Oryx", "{shared_nghitts_sha}")' in tts_text


def test_main_update_flag_outputs_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main with --update flag applies updates and returns 0."""
    report = _make_test_report()

    monkeypatch.setattr(check_models, "compare_models", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(check_models, "update_model_pins", lambda _r, **_kw: ["Updated pin foo"])

    rc = check_models.main(["--update"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "UPDATED MODEL PINS:" in captured
    assert "Updated pin foo" in captured

    rc_json = check_models.main(["--update", "--json"])
    assert rc_json == 0
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["updated"] == ["Updated pin foo"]


def test_main_update_flag_noop_outputs_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main with --update flag outputs all up to date when no changes occur."""
    report = _make_test_report()

    monkeypatch.setattr(check_models, "compare_models", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(check_models, "update_model_pins", lambda _r, **_kw: [])

    rc = check_models.main(["--update"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "All model pins are already up to date." in captured
    assert "UPDATED MODEL PINS:" not in captured


def test_main_update_flag_manual_review_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main with --update flag notes manual review when voices are added or removed."""
    report = _make_test_report(
        zerotts_remote_voice_ids=("new_voice_id",),
        tts_remote_names=("new_tts_name",),
    )

    monkeypatch.setattr(check_models, "compare_models", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(check_models, "update_model_pins", lambda _r, **_kw: ["Updated pin foo"])

    rc = check_models.main(["--update"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "NOTE: Voice additions or removals require manual review." in captured


@pytest.mark.parametrize(
    ("error_cls", "error_msg"),
    [
        (check_models.ModelCheckError, "Simulated update failure"),
        (OSError, "Disk write failure"),
    ],
)
def test_main_update_flag_error_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_cls: type[Exception],
    error_msg: str,
) -> None:
    """Test main with --update flag prints error and returns 2 when update fails."""
    report = _make_test_report()

    def _raise_error(*_args: object, **_kwargs: object) -> list[str]:
        raise error_cls(error_msg)

    monkeypatch.setattr(check_models, "compare_models", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(check_models, "update_model_pins", _raise_error)

    rc = check_models.main(["--update"])
    assert rc == 2
    captured = capsys.readouterr()
    assert f"ERROR: {error_msg}" in captured.err


def test_update_model_pins_raises_model_check_error_on_missing_stt_artifact(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when STT artifact cannot be matched."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text(
        'SttModelSpec(engine="zipformer", repo="dummy/repo", revision="rev1", artifacts=())\n',
        encoding="utf-8",
    )
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("", encoding="utf-8")

    stt_comp = check_models.SttModelComparison(
        engine="zipformer",
        repo="dummy/repo",
        local_revision="rev1",
        remote_revision="rev1",
        artifacts=(check_models.ArtifactComparison("missing.onnx", "a" * 64, "b" * 64, True),),
    )
    report = _make_test_report(stt_models=(stt_comp,))

    with pytest.raises(check_models.ModelCheckError, match="Failed to update pin for STT artifact"):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_raises_model_check_error_on_missing_stt_revision(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when STT revision cannot be matched."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text(
        "# Empty stt file without repo or revision\n",
        encoding="utf-8",
    )
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("", encoding="utf-8")

    stt_comp = check_models.SttModelComparison(
        engine="zipformer",
        repo="dummy/repo",
        local_revision="rev1",
        remote_revision="rev2",
        artifacts=(),
    )
    report = _make_test_report(stt_models=(stt_comp,))

    with pytest.raises(
        check_models.ModelCheckError, match="Failed to find revision pin for STT model"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_raises_model_check_error_on_missing_tts_artifact(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when TTS artifact cannot be matched."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("# Empty tts file\n", encoding="utf-8")

    art = check_models.ArtifactComparison("config.json", "a" * 64, "b" * 64, True)
    report = _make_test_report(zerotts_artifacts=(art,))

    with pytest.raises(
        check_models.ModelCheckError, match="Failed to update ZeroTTS core artifact pin"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_atomic_write_text_preserves_file_permissions(tmp_path: Path) -> None:
    """Test _atomic_write_text preserves existing file mode permissions."""
    target = tmp_path / "target_file.py"
    target.write_text("initial content\n", encoding="utf-8")
    target.chmod(0o644)
    expected_mode = target.stat().st_mode

    check_models._atomic_write_text(target, "new content\n")
    assert target.read_text(encoding="utf-8") == "new content\n"
    assert target.stat().st_mode == expected_mode


def test_atomic_write_text_flushes_and_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _atomic_write_text flushes and fsyncs temporary file and parent directory."""
    target = tmp_path / "sync_target.txt"
    target.write_text("initial", encoding="utf-8")

    fsynced_fds: list[int] = []
    orig_fsync = os.fsync

    def mock_fsync(fd: int) -> None:
        fsynced_fds.append(fd)
        orig_fsync(fd)

    monkeypatch.setattr(check_models.os, "fsync", mock_fsync)
    check_models._atomic_write_text(target, "persisted data")

    assert target.read_text(encoding="utf-8") == "persisted data"
    assert len(fsynced_fds) >= 2


def test_sync_dir_handles_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _sync_dir gracefully ignores OSError when directory sync is unsupported."""

    def mock_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("Directory sync not supported")

    monkeypatch.setattr(check_models.os, "open", mock_open)
    check_models._sync_dir(tmp_path)


def test_update_model_pins_transactional_rollback_on_tts_failure(
    tmp_path: Path,
) -> None:
    """Test update_model_pins leaves stt_model.py untouched when TTS matching fails."""
    stt_file = tmp_path / "stt_model.py"
    original_stt = (
        "ZIPFORMER_MODEL = SttModelSpec(\n"
        '    repo="hynt/Zipformer-30M-RNNT-6000h",\n'
        '    revision="old_stt_rev",\n'
        "    graphs=(),\n"
        ")\n"
    )
    stt_file.write_text(original_stt, encoding="utf-8")

    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("# Empty tts file\n", encoding="utf-8")

    stt_comp = check_models.SttModelComparison(
        engine="zipformer",
        repo="hynt/Zipformer-30M-RNNT-6000h",
        local_revision="old_stt_rev",
        remote_revision="new_stt_rev",
        artifacts=(),
    )
    art = check_models.ArtifactComparison("config.json", "a" * 64, "b" * 64, True)
    report = _make_test_report(stt_models=(stt_comp,), zerotts_artifacts=(art,))

    with pytest.raises(
        check_models.ModelCheckError, match="Failed to update ZeroTTS core artifact pin"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)

    assert stt_file.read_text(encoding="utf-8") == original_stt


def test_update_model_pins_raises_when_stt_file_missing_with_changes(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when stt_model.py is missing."""
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("", encoding="utf-8")

    stt_comp = check_models.SttModelComparison(
        engine="zipformer",
        repo="hynt/Zipformer-30M-RNNT-6000h",
        local_revision="old_stt_rev",
        remote_revision="new_stt_rev",
        artifacts=(),
    )
    report = _make_test_report(stt_models=(stt_comp,))

    with pytest.raises(
        check_models.ModelCheckError, match=r"Cannot update STT pins: '.*' does not exist"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_raises_when_tts_file_missing_with_changes(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when tts_model.py is missing."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")

    report = _make_test_report(zerotts_revision="new_tts_rev")

    with pytest.raises(
        check_models.ModelCheckError, match=r"Cannot update TTS pins: '.*' does not exist"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_apply_transactional_writes_rolls_back_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _apply_transactional_writes restores written files when a write fails."""
    file1, file2, updates = _setup_two_transaction_files(tmp_path)
    call_count = 0
    orig_atomic = check_models._atomic_write_text

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("Disk write failed")
        orig_atomic(target, content, encoding)

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    with pytest.raises(
        check_models.ModelCheckError, match="Failed to write model pin updates: Disk write failed"
    ) as exc_info:
        check_models._apply_transactional_writes(updates)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert file1.read_text(encoding="utf-8") == "orig1"
    assert file2.read_text(encoding="utf-8") == "orig2"


def test_apply_transactional_writes_reports_rollback_failures_without_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _apply_transactional_writes preserves original error when rollback also fails."""
    _file1, _file2, updates = _setup_two_transaction_files(tmp_path)
    call_count = 0
    orig_atomic = check_models._atomic_write_text

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("Write failed on file2")
        if call_count == 3:
            raise OSError("Rollback failed on file1")
        orig_atomic(target, content, encoding)

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    expected_pattern = (
        r"Failed to write model pin updates: Write failed on file2\."
        r" Additionally, rollback failed for: file1\.txt \(Rollback failed on file1\)"
    )
    with pytest.raises(check_models.ModelCheckError, match=expected_pattern) as exc_info:
        check_models._apply_transactional_writes(updates)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "Write failed on file2"


def test_apply_transactional_writes_rolls_back_failing_file_if_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _apply_transactional_writes restores the failing file if modified before error."""
    file1, file2, updates = _setup_two_transaction_files(tmp_path)
    orig_atomic = check_models._atomic_write_text

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        if target == file2 and content == "new2":
            orig_atomic(target, content, encoding)
            raise OSError("Write failed after replacement on file2")
        orig_atomic(target, content, encoding)

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    with pytest.raises(
        check_models.ModelCheckError,
        match="Failed to write model pin updates: Write failed after replacement on file2",
    ):
        check_models._apply_transactional_writes(updates)

    assert file1.read_text(encoding="utf-8") == "orig1"
    assert file2.read_text(encoding="utf-8") == "orig2"


def test_replace_exact_match_raises_on_ambiguous_matches() -> None:
    """Test _replace_exact_match raises ModelCheckError when multiple matches exist."""
    content = 'foo="old"\nfoo="old"\n'
    with pytest.raises(
        check_models.ModelCheckError, match="Ambiguous matches found while updating test pin"
    ):
        check_models._replace_exact_match(
            r'foo="old"',
            'foo="new"',
            content,
            "test pin",
        )


def test_update_model_pins_raises_on_ambiguous_voice_matches(tmp_path: Path) -> None:
    """Test update_model_pins raises ModelCheckError when duplicate voice pins exist."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")

    shared_sha = "1111111111111111111111111111111111111111111111111111111111111111"
    new_sha = "2222222222222222222222222222222222222222222222222222222222222222"
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        f'_voice("duy-onyx", "Duy Onyx", "{shared_sha}")\n'
        f'_voice("duy-onyx", "Duy Onyx", "{shared_sha}")\n',
        encoding="utf-8",
    )

    voice_comp = check_models.VoiceComparison(
        name="Duy Onyx",
        voice_id="duy-onyx",
        artifacts=(check_models.ArtifactComparison("Duy Onyx.onnx", shared_sha, new_sha, True),),
    )
    report = _make_test_report(tts_voices=(voice_comp,))

    with pytest.raises(
        check_models.ModelCheckError, match="Ambiguous matches found while updating"
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_captures_arbitrary_tts_file_revision(
    tmp_path: Path,
) -> None:
    """Test update_model_pins derives the ZeroTTS old revision directly from tts_model.py."""
    _stt_file, tts_file = _setup_zerotts_model_files(tmp_path, revision="arbitrary_local_rev")

    report = _make_test_report(zerotts_revision="new_hf_rev")
    changes = check_models.update_model_pins(report, models_dir=tmp_path)

    assert changes == ["ZeroTTS revision: arbitrary_local_rev -> new_hf_rev"]
    _assert_file_contains(tts_file, 'revision="new_hf_rev"')


def test_update_model_pins_raises_model_check_error_on_missing_zerotts_revision(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when ZeroTTS revision cannot be matched."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")

    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("# No ZEROTTS_MODEL revision pin\n", encoding="utf-8")

    report = _make_test_report(zerotts_revision="new_hf_rev")

    with pytest.raises(check_models.ModelCheckError, match="Failed to update ZeroTTS revision pin"):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_raises_model_check_error_on_ambiguous_zerotts_revision(
    tmp_path: Path,
) -> None:
    """Test update_model_pins raises ModelCheckError when duplicate ZeroTTS revision pins exist."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")

    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        "ZEROTTS_MODEL = ZeroTtsModelSpec(\n"
        "    repo=ZEROTTS_REPO_ID,\n"
        '    revision="rev_one",\n'
        ")\n"
        "ZEROTTS_MODEL_DUP = ZeroTtsModelSpec(\n"
        "    repo=ZEROTTS_REPO_ID,\n"
        '    revision="rev_two",\n'
        ")\n",
        encoding="utf-8",
    )

    report = _make_test_report(zerotts_revision="new_hf_rev")

    with pytest.raises(
        check_models.ModelCheckError,
        match="Ambiguous matches found while updating ZeroTTS revision pin",
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_updates_shared_nghitts_config_once(tmp_path: Path) -> None:
    """Test update_model_pins updates shared NghiTTS config constants once for multiple voices."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text("", encoding="utf-8")

    shared_cfg_old = "1111111111111111111111111111111111111111111111111111111111111111"
    shared_cfg_new = "2222222222222222222222222222222222222222222222222222222222222222"
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text(
        f'_NGHITTS_COMMON_CONFIG_SHA256 = "{shared_cfg_old}"\n'
        "NGHITTS_VOICES = (\n"
        '    _voice("v1", "Voice One",\n'
        '           "3333333333333333333333333333333333333333333333333333333333333333"),\n'
        '    _voice("v2", "Voice Two",\n'
        '           "4444444444444444444444444444444444444444444444444444444444444444"),\n'
        ")\n",
        encoding="utf-8",
    )

    voices = (
        check_models.VoiceComparison(
            name="Voice One",
            voice_id="v1",
            artifacts=(
                check_models.ArtifactComparison(
                    "Voice One.onnx.json",
                    shared_cfg_old,
                    shared_cfg_new,
                    True,
                ),
            ),
        ),
        check_models.VoiceComparison(
            name="Voice Two",
            voice_id="v2",
            artifacts=(
                check_models.ArtifactComparison(
                    "Voice Two.onnx.json",
                    shared_cfg_old,
                    shared_cfg_new,
                    True,
                ),
            ),
        ),
    )
    report = _make_test_report(tts_voices=voices)
    changes = check_models.update_model_pins(report, models_dir=tmp_path)

    assert changes == [
        f"NghiTTS config [_NGHITTS_COMMON_CONFIG_SHA256]: {shared_cfg_old} -> {shared_cfg_new}"
    ]
    assert f'_NGHITTS_COMMON_CONFIG_SHA256 = "{shared_cfg_new}"' in tts_file.read_text(
        encoding="utf-8"
    )


def test_main_update_flag_json_includes_voice_review_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test main with --update --json includes review note when voice additions exist."""
    report = _make_test_report(tts_remote_names=("extra_voice",))

    monkeypatch.setattr(check_models, "compare_models", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(check_models, "update_model_pins", lambda _r, **_kw: ["Updated pin foo"])

    rc = check_models.main(["--update", "--json"])
    assert rc == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["updated"] == ["Updated pin foo"]
    assert data["note"] == "Voice additions or removals require manual review."


def test_apply_transactional_writes_rolls_back_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _apply_transactional_writes restores original file timestamps and mode on rollback."""
    file1, file2, updates = _setup_two_transaction_files(tmp_path)

    past_atime_ns = 1_000_000_000_000_000_000
    past_mtime_ns = 1_200_000_000_000_000_000
    os.utime(file1, ns=(past_atime_ns, past_mtime_ns))
    orig_stat = file1.stat()

    orig_atomic = check_models._atomic_write_text

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        if target == file2:
            raise OSError("Disk write failed on file2")
        orig_atomic(target, content, encoding)

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    with pytest.raises(
        check_models.ModelCheckError,
        match="Failed to write model pin updates: Disk write failed on file2",
    ):
        check_models._apply_transactional_writes(updates)

    assert file1.read_text(encoding="utf-8") == "orig1"
    post_rollback_stat = file1.stat()
    assert post_rollback_stat.st_mtime_ns == orig_stat.st_mtime_ns
    assert post_rollback_stat.st_mode == orig_stat.st_mode


def test_apply_transactional_writes_cleans_up_previously_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test rollback unlinks previously absent paths without leaving empty files."""
    file_absent = tmp_path / "absent.txt"
    file2 = tmp_path / "file2.txt"
    file2.write_text("orig2", encoding="utf-8")

    orig_atomic = check_models._atomic_write_text

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        if target == file2:
            raise OSError("Disk write failed on file2")
        orig_atomic(target, content, encoding)

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    updates = [(file_absent, "new_created", ""), (file2, "new2", "orig2")]
    with pytest.raises(
        check_models.ModelCheckError,
        match="Failed to write model pin updates: Disk write failed on file2",
    ):
        check_models._apply_transactional_writes(updates)

    assert not file_absent.exists()
    assert file2.read_text(encoding="utf-8") == "orig2"


def test_apply_transactional_writes_failed_write_on_absent_file_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test failed write to previously absent path does not leave an empty file."""
    file_absent = tmp_path / "absent.txt"

    def mock_atomic(target: Path, content: str, encoding: str = "utf-8") -> None:
        raise OSError("Permission denied on absent file")

    monkeypatch.setattr(check_models, "_atomic_write_text", mock_atomic)

    updates = [(file_absent, "new_created", "")]
    with pytest.raises(
        check_models.ModelCheckError,
        match="Failed to write model pin updates: Permission denied on absent file",
    ):
        check_models._apply_transactional_writes(updates)

    assert not file_absent.exists()


def test_update_model_pins_blocks_stt_revision_when_artifact_unresolved(
    tmp_path: Path,
) -> None:
    """Test update_model_pins blocks STT revision bump when artifact is unresolved."""
    stt_file = tmp_path / "stt_model.py"
    stt_file.write_text(
        "ZIPFORMER_MODEL = SttModelSpec(\n"
        '    repo="hynt/Zipformer-30M-RNNT-6000h",\n'
        '    revision="old_rev",\n'
        ")\n",
        encoding="utf-8",
    )
    tts_file = tmp_path / "tts_model.py"
    tts_file.write_text("", encoding="utf-8")

    stt_model = check_models.SttModelComparison(
        engine="zipformer",
        repo="hynt/Zipformer-30M-RNNT-6000h",
        local_revision="old_rev",
        remote_revision="new_rev",
        artifacts=(check_models.ArtifactComparison("encoder.onnx", "a" * 64, None, True),),
    )
    report = _make_test_report(stt_models=(stt_model,))
    with pytest.raises(
        check_models.ModelCheckError,
        match=(
            r"Cannot update revision for STT model 'zipformer'"
            r".*unresolved artifacts: encoder\.onnx"
        ),
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_blocks_zerotts_revision_when_core_artifact_unresolved(
    tmp_path: Path,
) -> None:
    """Test update_model_pins blocks ZeroTTS revision bump when core artifact is REMOVED."""
    _setup_zerotts_model_files(tmp_path)

    art = check_models.ArtifactComparison("config.json", "a" * 64, None, False)
    report = _make_test_report(zerotts_revision="new_rev", zerotts_artifacts=(art,))
    with pytest.raises(
        check_models.ModelCheckError,
        match=r"Cannot update ZeroTTS revision: unresolved artifacts: config\.json",
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_blocks_zerotts_revision_when_voice_artifact_unresolved(
    tmp_path: Path,
) -> None:
    """Test update_model_pins blocks ZeroTTS revision bump when voice artifact is UNKNOWN."""
    _setup_zerotts_model_files(tmp_path)

    voice = check_models.VoiceComparison(
        name="Mai Chi",
        voice_id="maichi",
        artifacts=(
            check_models.ArtifactComparison("voices/maichi/voice.npz", "a" * 64, None, True),
        ),
    )
    report = _make_test_report(zerotts_revision="new_rev", zerotts_voices=(voice,))
    with pytest.raises(
        check_models.ModelCheckError,
        match=r"Cannot update ZeroTTS revision: unresolved artifacts: voices/maichi/voice\.npz",
    ):
        check_models.update_model_pins(report, models_dir=tmp_path)


def test_update_model_pins_allows_zerotts_revision_when_local_artifact_unresolved(
    tmp_path: Path,
) -> None:
    """Test update_model_pins allows ZeroTTS revision bump when only local artifact is UNKNOWN."""
    _stt_file, tts_file = _setup_zerotts_model_files(tmp_path)

    local_art = check_models.ArtifactComparison(
        f"onnx/codec/{ZeroTtsFile.DECODE_STEP_INT8_ONNX}",
        "a" * 64,
        None,
        False,
    )
    report = _make_test_report(zerotts_revision="new_rev", zerotts_artifacts=(local_art,))
    changes = check_models.update_model_pins(report, models_dir=tmp_path)
    assert changes == ["ZeroTTS revision: old_rev -> new_rev"]
    _assert_file_contains(tts_file, 'revision="new_rev"')
