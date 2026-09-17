"""Compare the remote speech model catalogues with the local pins."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from email.message import Message
from hashlib import sha256
from pathlib import Path
from typing import Final, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, urlopen

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyoming_vietnamese.const import (
    HUGGINGFACE_API_MODELS_URL,
    HUGGINGFACE_BASE_URL,
    ZEROTTS_REPO_ID,
    NghiTtsFile,
)
from wyoming_vietnamese.const import (
    NGHITTS_MODEL_BASE_URL as _NGHITTS_MODEL_BASE_URL_SOURCE,
)
from wyoming_vietnamese.stt_model import STT_MODEL
from wyoming_vietnamese.tts_model import (
    NGHITTS_VOICES,
    ZEROTTS_MODEL,
    ZEROTTS_VOICES,
    NghiTtsArtifact,
)

_USER_AGENT: Final = "wyoming-vietnamese-model-check/1.0"
_DEFAULT_TIMEOUT_SECONDS: Final = 60
_HASH_WORKERS: Final = 4
_HF_API_BASE_URL: Final = HUGGINGFACE_API_MODELS_URL
_NGHITTS_MODEL_BASE_URL: Final = _NGHITTS_MODEL_BASE_URL_SOURCE.rstrip("/")
_MAX_JSON_RESPONSE_BYTES: Final = 4 * 1024 * 1024
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_ZEROTTS_CORE_FILES: Final = tuple(artifact.remote_name for artifact in ZEROTTS_MODEL.artifacts)
_LINK_HEADER_PATTERN: Final = re.compile(r"<([^>]+)>([^,]*)")
_REL_PARAM_PATTERN: Final = re.compile(r"""\brel\s*=\s*(?:"([^"]*)"|([^"\s,;]+))""", re.IGNORECASE)
_MAX_TREE_PAGES: Final = 1000


class ModelCheckError(RuntimeError):
    """Report malformed or unavailable remote model metadata."""


@dataclass(frozen=True, slots=True)
class ArtifactComparison:
    """Compare one local artifact pin with its remote digest."""

    remote_name: str
    local_sha256: str | None
    remote_sha256: str | None
    remote_present: bool = True

    @property
    def status(self) -> str:
        """Return the comparison status for this artifact."""
        if not self.remote_present:
            return "REMOVED"
        if self.remote_sha256 is None:
            return "UNKNOWN"
        if self.local_sha256 is None:
            return "NEW"
        return "MATCH" if self.local_sha256 == self.remote_sha256 else "CHANGED"


@dataclass(frozen=True, slots=True)
class VoiceComparison:
    """Compare one local TTS voice with a remote voice entry."""

    name: str
    voice_id: str | None
    artifacts: tuple[ArtifactComparison, ...]

    @property
    def status(self) -> str:
        """Return the aggregate comparison status for this voice."""
        statuses = {artifact.status for artifact in self.artifacts}
        if "CHANGED" in statuses:
            return "CHANGED"
        if "REMOVED" in statuses:
            return "REMOVED"
        if "UNKNOWN" in statuses:
            return "UNKNOWN"
        return "NEW" if "NEW" in statuses else "MATCH"


@dataclass(frozen=True, slots=True)
class TtsComparison:
    """Compare the complete local and remote TTS catalogues."""

    remote_names: tuple[str, ...]
    local_names: tuple[str, ...]
    voices: tuple[VoiceComparison, ...]
    hashes_checked: bool

    @property
    def added_names(self) -> tuple[str, ...]:
        """Return remote voices not yet present in the local catalogue."""
        local_names = set(self.local_names)
        return tuple(name for name in self.remote_names if name not in local_names)

    @property
    def removed_names(self) -> tuple[str, ...]:
        """Return local voices no longer present in the remote catalogue."""
        remote_names = set(self.remote_names)
        return tuple(name for name in self.local_names if name not in remote_names)

    @property
    def needs_update(self) -> bool:
        """Return whether the local TTS model definitions need review."""
        return bool(
            self.added_names
            or self.removed_names
            or any(voice.status in {"CHANGED", "NEW", "REMOVED"} for voice in self.voices)
        )


@dataclass(frozen=True, slots=True)
class SttComparison:
    """Compare the local STT revision and required files with Hugging Face."""

    remote_revision: str
    artifacts: tuple[ArtifactComparison, ...]

    @property
    def revision_changed(self) -> bool:
        """Return whether the remote repository head moved past the local pin."""
        return self.remote_revision != STT_MODEL.revision

    @property
    def needs_update(self) -> bool:
        """Return whether the local STT model definition needs review."""
        return self.revision_changed or any(
            artifact.status in {"CHANGED", "NEW", "REMOVED"} for artifact in self.artifacts
        )


@dataclass(frozen=True, slots=True)
class ZeroTtsComparison:
    """Compare the local ZeroTTS voice catalogue and required files with Hugging Face."""

    repo: str
    remote_revision: str
    remote_voice_ids: tuple[str, ...]
    local_voice_ids: tuple[str, ...]
    artifacts: tuple[ArtifactComparison, ...]
    voices: tuple[VoiceComparison, ...] = ()
    hashes_checked: bool = True

    @property
    def revision_changed(self) -> bool:
        """Return whether the remote repository head moved past the local pin."""
        return self.remote_revision != ZEROTTS_MODEL.revision

    @property
    def added_voice_ids(self) -> tuple[str, ...]:
        """Return remote voice IDs not present in the local catalogue."""
        local = set(self.local_voice_ids)
        return tuple(v for v in self.remote_voice_ids if v not in local)

    @property
    def removed_voice_ids(self) -> tuple[str, ...]:
        """Return local voice IDs no longer present in the remote repository."""
        remote = set(self.remote_voice_ids)
        return tuple(v for v in self.local_voice_ids if v not in remote)

    @property
    def needs_update(self) -> bool:
        """Return whether the local ZeroTTS model or voice catalogue needs review."""
        return bool(
            self.revision_changed
            or self.added_voice_ids
            or self.removed_voice_ids
            or any(artifact.status in {"CHANGED", "NEW", "REMOVED"} for artifact in self.artifacts)
            or any(voice.status in {"CHANGED", "NEW", "REMOVED"} for voice in self.voices)
        )


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Hold the complete remote-versus-local comparison result."""

    tts: TtsComparison
    stt: SttComparison
    zerotts: ZeroTtsComparison

    @property
    def nghitts(self) -> TtsComparison:
        """Return the NghiTTS comparison."""
        return self.tts

    @property
    def needs_update(self) -> bool:
        """Return whether any local model definition needs review."""
        return self.tts.needs_update or self.stt.needs_update or self.zerotts.needs_update


class _JsonResponseList(list[object]):
    """JSON list response with optional attached HTTP response headers."""

    headers: Message | None = None
    url: str | None = None


def _iter_link_headers(headers: object) -> list[str]:
    """Collect raw Link header strings from response headers."""
    if isinstance(headers, Message):
        vals = headers.get_all("Link")
        return list(vals) if vals else []
    if isinstance(headers, Mapping):
        val = headers.get("Link")
        if isinstance(val, str):
            return [val]
        if isinstance(val, (list, tuple)):
            return [str(item) for item in val if item is not None]
    return []


def _extract_next_link(headers: object, current_url: str) -> str | None:
    """Extract the next page URL from HTTP Link headers with rel="next"."""
    for link_header in _iter_link_headers(headers):
        for match in _LINK_HEADER_PATTERN.finditer(link_header):
            target = match.group(1).strip()
            params = match.group(2)
            rel_match = _REL_PARAM_PATTERN.search(params)
            if not rel_match:
                continue
            rel_val = (rel_match.group(1) or rel_match.group(2) or "").lower()
            if "next" in rel_val.split():
                return urljoin(current_url, target)
    return None


def _request_json(url: str, timeout: int) -> object:
    """Fetch and decode one JSON API response."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return _handle_json_response(response)
    except Exception as err:
        raise ModelCheckError(f"Could not fetch JSON from {url}: {err}") from err


class _ReadableResponse(Protocol):
    """Minimal protocol for readable HTTP response streams."""

    def read(self, size: int = -1, /) -> bytes:
        """Read up to size bytes from the response stream."""
        ...


def _extract_response_url(response: _ReadableResponse) -> str | None:
    """Extract a response URL from an HTTP response or test double."""
    url_attr = getattr(response, "url", None)
    if isinstance(url_attr, str):
        return url_attr
    geturl = getattr(response, "geturl", None)
    if isinstance(geturl, Callable):
        result = geturl()
        if isinstance(result, str):
            return result
    return None


def _handle_json_response(response: _ReadableResponse) -> object:
    """Decode a size-bounded JSON response, preserving headers on list results."""
    body = bytearray()
    while True:
        read_size = min(64 * 1024, _MAX_JSON_RESPONSE_BYTES - len(body) + 1)
        chunk = response.read(read_size)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > _MAX_JSON_RESPONSE_BYTES:
            raise ValueError(f"JSON response exceeds {_MAX_JSON_RESPONSE_BYTES} bytes")
    data = json.loads(body)
    headers = getattr(response, "headers", None)
    response_url = _extract_response_url(response)
    if isinstance(data, list):
        result = _JsonResponseList(data)
        result.headers = headers
        result.url = response_url
        return result
    return data


def _request_sha256(url: str, timeout: int) -> str | None:
    """Stream one remote artifact and return its digest, or None for HTTP 404."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    digest = sha256()
    try:
        with urlopen(request, timeout=timeout) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
    except HTTPError as err:
        if err.code == 404:
            err.close()
            return None
        raise ModelCheckError(f"Could not hash remote artifact {url}: {err}") from err
    except Exception as err:
        raise ModelCheckError(f"Could not hash remote artifact {url}: {err}") from err
    return digest.hexdigest()


def _require_dict(value: object, context: str) -> dict[str, object]:
    """Validate one decoded JSON object."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ModelCheckError(f"Unexpected JSON object from {context}")
    return value


def _require_string(value: object, context: str) -> str:
    """Validate one decoded JSON string."""
    if not isinstance(value, str) or not value:
        raise ModelCheckError(f"Missing string field {context}")
    return value


def _require_string_list(value: object, context: str) -> tuple[str, ...]:
    """Validate one decoded JSON string list."""
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ModelCheckError(f"Missing string list {context}")
    return tuple(value)


def _normalize_sha256_digest(value: object) -> str | None:
    """Return a validated SHA-256 digest from a Hugging Face LFS OID."""
    if not isinstance(value, str):
        return None
    digest = value.removeprefix("sha256:")
    return digest if _SHA256_PATTERN.fullmatch(digest) else None


def _remote_voice_names(timeout: int) -> tuple[str, ...]:
    """Read the NghiTTS voice catalogue."""
    payload = _require_dict(
        _request_json(f"{_NGHITTS_MODEL_BASE_URL.rsplit('/', 1)[0]}/models", timeout),
        "NghiTTS model catalogue",
    )
    return _require_string_list(payload.get("models"), "NghiTTS models")


def _artifact_url(artifact: NghiTtsArtifact) -> str:
    """Build the URL for one NghiTTS voice artifact."""
    return f"{_NGHITTS_MODEL_BASE_URL}/{quote(artifact.remote_name, safe='')}"


def _remote_voice_artifacts(
    artifacts: tuple[NghiTtsArtifact, ...],
    futures: tuple[Future[str | None], ...],
) -> tuple[ArtifactComparison, ...]:
    """Build artifact comparisons from futures submitted to the shared pool."""
    comparisons: list[ArtifactComparison] = []
    for artifact, future in zip(artifacts, futures, strict=True):
        remote_sha256 = future.result()
        comparisons.append(
            ArtifactComparison(
                artifact.remote_name,
                artifact.sha256 or None,
                remote_sha256,
                remote_sha256 is not None,
            )
        )
    return tuple(comparisons)


def _slugify_voice_name(name: str) -> str:
    """Return the repository's conventional ID form for a display name."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def _voice_artifacts_for_name(name: str) -> tuple[NghiTtsArtifact, ...]:
    """Build the standard NghiTTS artifact names for a display name."""
    return tuple(
        NghiTtsArtifact(remote_name, local_name, "")
        for remote_name, local_name in (
            (f"{name}.onnx", NghiTtsFile.MODEL),
            (f"{name}.onnx.json", NghiTtsFile.CONFIG),
        )
    )


def _compare_tts(timeout: int, *, hash_artifacts: bool) -> TtsComparison:
    """Compare the NghiTTS remote catalogue and artifacts with local pins."""
    remote_names = _remote_voice_names(timeout)
    local_by_name = {voice.name: voice for voice in NGHITTS_VOICES}
    voices: list[VoiceComparison] = []
    if hash_artifacts:
        remote_jobs: list[
            tuple[
                str,
                str | None,
                tuple[NghiTtsArtifact, ...],
                tuple[Future[str | None], ...],
            ]
        ] = []
        executor = ThreadPoolExecutor(max_workers=_HASH_WORKERS)
        try:
            for name in remote_names:
                local_voice = local_by_name.get(name)
                artifacts = (
                    local_voice.artifacts if local_voice else _voice_artifacts_for_name(name)
                )
                futures = tuple(
                    executor.submit(_request_sha256, _artifact_url(artifact), timeout)
                    for artifact in artifacts
                )
                remote_jobs.append(
                    (name, local_voice.id if local_voice else None, artifacts, futures)
                )
            voices.extend(
                VoiceComparison(name, voice_id, _remote_voice_artifacts(artifacts, futures))
                for name, voice_id, artifacts, futures in remote_jobs
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    else:
        for name in remote_names:
            local_voice = local_by_name.get(name)
            artifacts = local_voice.artifacts if local_voice else _voice_artifacts_for_name(name)
            comparisons = tuple(
                ArtifactComparison(artifact.remote_name, artifact.sha256 or None, None)
                for artifact in artifacts
            )
            voices.append(
                VoiceComparison(name, local_voice.id if local_voice else None, comparisons)
            )

    remote_name_set = set(remote_names)
    for local_voice in NGHITTS_VOICES:
        if local_voice.name in remote_name_set:
            continue
        comparisons = tuple(
            ArtifactComparison(artifact.remote_name, artifact.sha256, None, False)
            for artifact in local_voice.artifacts
        )
        voices.append(VoiceComparison(local_voice.name, local_voice.id, comparisons))

    return TtsComparison(
        remote_names,
        tuple(voice.name for voice in NGHITTS_VOICES),
        tuple(voices),
        hash_artifacts,
    )


def _remote_hf_revision(repo: str, timeout: int) -> str:
    """Read the current revision for a Hugging Face repository."""
    url = f"{_HF_API_BASE_URL}/{quote(repo, safe='/')}"
    payload = _require_dict(_request_json(url, timeout), f"Hugging Face model {repo}")
    return _require_string(payload.get("sha"), "Hugging Face model sha")


def _validate_tree_url(candidate_url: str, expected_tree_url: str) -> None:
    """Validate that candidate_url matches the scheme, host, and tree endpoint path."""
    expected = urlsplit(expected_tree_url)
    candidate = urlsplit(candidate_url)
    if (
        candidate.scheme.lower() != expected.scheme.lower()
        or candidate.netloc.lower() != expected.netloc.lower()
        or candidate.path != expected.path
    ):
        raise ModelCheckError(
            f"URL {candidate_url!r} is outside expected Hugging Face tree endpoint "
            f"{expected_tree_url!r}"
        )


def _parse_hf_tree_entry(raw_entry: object, repo: str) -> tuple[str, str | None] | None:
    """Extract the file path and SHA-256 digest from one Hugging Face tree entry.

    Returns None if the entry type is not 'file'.
    """
    entry = _require_dict(raw_entry, f"Hugging Face model tree for {repo}")
    if entry.get("type") != "file":
        return None
    path = _require_string(entry.get("path"), "Hugging Face tree path")
    lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        return path, _normalize_sha256_digest(lfs.get("oid"))
    # Non-LFS entries expose git blob OIDs, not comparable SHA-256 digests; keep them
    # UNKNOWN without making the comparison require an update.
    return path, None


def _collect_tree_page_entries(
    payload: list[object],
    repo: str,
    entries: dict[str, str | None],
) -> None:
    """Parse valid file entries from a Hugging Face tree payload page into the entries map.

    Args:
        payload: List of raw tree entry objects from the API response.
        repo: Hugging Face repository identifier.
        entries: Target dictionary mapping file path to SHA-256 digest or None.
    """
    for raw_entry in payload:
        parsed = _parse_hf_tree_entry(raw_entry, repo)
        if parsed is not None:
            path, digest = parsed
            entries[path] = digest


def _remote_hf_tree_entries(repo: str, revision: str, timeout: int) -> dict[str, str | None]:
    """Read remote file paths and SHA-256 digests from a Hugging Face repository tree."""
    expected_tree_url = (
        f"{_HF_API_BASE_URL}/{quote(repo, safe='/')}/tree/{quote(revision, safe='')}"
    )
    url: str | None = f"{expected_tree_url}?recursive=true"
    entries: dict[str, str | None] = {}
    visited_urls: set[str] = set()

    while url is not None:
        if url in visited_urls:
            raise ModelCheckError(
                f"Repeated URL in Hugging Face model tree pagination for {repo}: {url}"
            )
        if len(visited_urls) >= _MAX_TREE_PAGES:
            raise ModelCheckError(
                f"Exceeded maximum tree page count of {_MAX_TREE_PAGES} for {repo}"
            )
        visited_urls.add(url)
        payload = _request_json(url, timeout)
        if not isinstance(payload, list):
            raise ModelCheckError(f"Unexpected Hugging Face model tree response for {repo}")

        if response_url := getattr(payload, "url", None):
            _validate_tree_url(response_url, expected_tree_url)

        _collect_tree_page_entries(payload, repo, entries)

        headers = getattr(payload, "headers", None)
        next_url = _extract_next_link(headers, url)
        if next_url is not None:
            _validate_tree_url(next_url, expected_tree_url)
        url = next_url

    return entries


def _remote_stt_revision(timeout: int) -> str:
    """Read the current Hugging Face repository revision for STT."""
    return _remote_hf_revision(STT_MODEL.repo, timeout)


def _remote_stt_artifacts(revision: str, timeout: int) -> dict[str, str | None]:
    """Read remote SHA-256 digests for the pinned STT file names."""
    return _remote_hf_tree_entries(STT_MODEL.repo, revision, timeout)


def _compare_stt(timeout: int) -> SttComparison:
    """Compare the pinned STT files with the current Hugging Face head."""
    remote_revision = _remote_stt_revision(timeout)
    remote_artifacts = _remote_stt_artifacts(remote_revision, timeout)
    comparisons = tuple(
        ArtifactComparison(
            artifact.remote_name,
            artifact.sha256,
            remote_artifacts.get(artifact.remote_name),
            artifact.remote_name in remote_artifacts,
        )
        for artifact in STT_MODEL.artifacts
    )
    return SttComparison(remote_revision, comparisons)


def _parse_zerotts_voice_id(path: str) -> str | None:
    """Extract a valid ZeroTTS voice ID from one remote path, or return None."""
    if not path.startswith("voices/"):
        return None
    rel = path.removeprefix("voices/").strip("/")
    if rel.endswith("/voice.npz"):
        voice_id = rel.removesuffix("/voice.npz").strip()
    elif rel.endswith(".npz"):
        voice_id = rel.removesuffix(".npz").strip()
    else:
        return None
    return voice_id if voice_id and "/" not in voice_id else None


def _extract_zerotts_voice_ids(tree_paths: Iterable[str]) -> tuple[str, ...]:
    """Extract unique voice IDs from remote ZeroTTS voice paths."""
    voice_ids = {
        voice_id for path in tree_paths if (voice_id := _parse_zerotts_voice_id(path)) is not None
    }
    return tuple(sorted(voice_ids))


def _build_zerotts_artifact_comparison(
    remote_name: str,
    local_sha256: str | None,
    remote_entries: Mapping[str, str | None],
    fetched_hashes: Mapping[str, str | None],
) -> ArtifactComparison:
    """Build one artifact comparison for ZeroTTS, resolving remote presence and hashes."""
    present = remote_name in remote_entries
    remote_sha = remote_entries.get(remote_name)
    if remote_sha is None and remote_name in fetched_hashes:
        remote_sha = fetched_hashes[remote_name]
        if remote_sha is None:
            present = False
    return ArtifactComparison(
        remote_name=remote_name,
        local_sha256=local_sha256,
        remote_sha256=remote_sha,
        remote_present=present,
    )


def _compare_zerotts(timeout: int, *, hash_artifacts: bool = True) -> ZeroTtsComparison:
    """Compare the ZeroTTS voice catalogue and required files with Hugging Face."""
    remote_revision = _remote_hf_revision(ZEROTTS_REPO_ID, timeout)
    remote_entries = _remote_hf_tree_entries(ZEROTTS_REPO_ID, remote_revision, timeout)
    remote_voices = _extract_zerotts_voice_ids(remote_entries)
    local_voices = tuple(voice.id for voice in ZEROTTS_VOICES)

    fetched_hashes: dict[str, str | None] = {}
    if hash_artifacts:
        declared_artifacts = (
            *ZEROTTS_MODEL.artifacts,
            *(voice.artifact for voice in ZEROTTS_VOICES),
        )
        if missing_hashes := [
            art.remote_name
            for art in declared_artifacts
            if art.remote_name in remote_entries and remote_entries.get(art.remote_name) is None
        ]:
            with ThreadPoolExecutor(max_workers=min(len(missing_hashes), _HASH_WORKERS)) as pool:
                futures = {
                    name: pool.submit(
                        _request_sha256,
                        (
                            f"{HUGGINGFACE_BASE_URL}/{ZEROTTS_REPO_ID}/raw/"
                            f"{quote(remote_revision, safe='')}/{quote(name, safe='/')}"
                        ),
                        timeout,
                    )
                    for name in missing_hashes
                }
                fetched_hashes = {name: future.result() for name, future in futures.items()}

    artifact_comparisons = tuple(
        _build_zerotts_artifact_comparison(
            artifact.remote_name,
            artifact.sha256,
            remote_entries,
            fetched_hashes,
        )
        for artifact in ZEROTTS_MODEL.artifacts
    )

    voice_comparisons = tuple(
        VoiceComparison(
            voice.name,
            voice.id,
            (
                _build_zerotts_artifact_comparison(
                    voice.artifact.remote_name,
                    voice.artifact.sha256,
                    remote_entries,
                    fetched_hashes,
                ),
            ),
        )
        for voice in ZEROTTS_VOICES
    )

    return ZeroTtsComparison(
        repo=ZEROTTS_REPO_ID,
        remote_revision=remote_revision,
        remote_voice_ids=remote_voices,
        local_voice_ids=local_voices,
        artifacts=artifact_comparisons,
        voices=voice_comparisons,
        hashes_checked=hash_artifacts,
    )


def compare_models(
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    *,
    hash_tts: bool = True,
) -> ComparisonReport:
    """Compare remote metadata with local speech model definitions.

    ``hash_tts`` controls artifact-body hashing for both NghiTTS and ZeroTTS.
    """
    return ComparisonReport(
        _compare_tts(timeout, hash_artifacts=hash_tts),
        _compare_stt(timeout),
        _compare_zerotts(timeout, hash_artifacts=hash_tts),
    )


def _format_artifact(artifact: ArtifactComparison) -> str:
    """Format one artifact comparison for terminal output."""
    remote_sha = artifact.remote_sha256 or "unavailable"
    local_sha = artifact.local_sha256 or "none"
    return f"    {artifact.status:<7} {artifact.remote_name} local={local_sha} remote={remote_sha}"


def _print_tts_report(tts: TtsComparison) -> None:
    """Print the NghiTTS comparison results."""
    print("TTS (NghiTTS)")
    print(f"  Remote voices: {len(tts.remote_names)}")
    print(f"  Local voices:  {len(tts.local_names)}")
    if not tts.hashes_checked:
        print("  Artifact hashes: SKIPPED (use the default mode for integrity checks)")
    for name in tts.added_names:
        print(f"  ADD     {name} (suggested id: {_slugify_voice_name(name)})")
    for name in tts.removed_names:
        print(f"  REMOVE  {name}")
    for voice in tts.voices:
        status = voice.status
        if status in {"MATCH", "UNKNOWN"}:
            continue
        if status == "REMOVED" and voice.name in tts.removed_names:
            continue
        print(f"  {status:<7} {voice.name}")
        for artifact in voice.artifacts:
            print(_format_artifact(artifact))


def _print_zerotts_report(zerotts: ZeroTtsComparison) -> None:
    """Print the ZeroTTS comparison results."""
    revision_status = "CHANGED" if zerotts.revision_changed else "MATCH"
    print("ZeroTTS")
    print(f"  Repository:    {zerotts.repo}")
    print(
        f"  {revision_status:<7} revision local={ZEROTTS_MODEL.revision} "
        f"remote={zerotts.remote_revision}"
    )
    print(f"  Remote voices: {len(zerotts.remote_voice_ids)}")
    print(f"  Local voices:  {len(zerotts.local_voice_ids)}")
    if not zerotts.hashes_checked:
        print("  Artifact hashes: SKIPPED (use the default mode for integrity checks)")
    for voice_id in zerotts.added_voice_ids:
        print(f"  ADD     {voice_id}")
    for voice_id in zerotts.removed_voice_ids:
        print(f"  REMOVE  {voice_id}")
    for artifact in zerotts.artifacts:
        if artifact.status in {"MATCH", "UNKNOWN"}:
            continue
        print(_format_artifact(artifact))
    for voice in zerotts.voices:
        if voice.status in {"MATCH", "UNKNOWN"}:
            continue
        print(f"  {voice.status:<7} {voice.name}")
        for artifact in voice.artifacts:
            print(_format_artifact(artifact))


def _print_stt_report(stt: SttComparison) -> None:
    """Print the STT comparison results."""
    revision_status = "CHANGED" if stt.revision_changed else "MATCH"
    print("STT")
    print(f"  Repository: {STT_MODEL.repo}")
    print(
        f"  {revision_status:<7} revision local={STT_MODEL.revision} remote={stt.remote_revision}"
    )
    for artifact in stt.artifacts:
        print(_format_artifact(artifact))


def _print_report(report: ComparisonReport) -> None:
    """Print a concise human-readable model comparison."""
    _print_tts_report(report.tts)
    _print_zerotts_report(report.zerotts)
    _print_stt_report(report.stt)
    print("RESULT: UPDATE_REQUIRED" if report.needs_update else "RESULT: UP_TO_DATE")


def _artifact_to_dict(artifact: ArtifactComparison) -> dict[str, object]:
    """Serialize one artifact comparison for JSON output."""
    return {
        "remote_name": artifact.remote_name,
        "local_sha256": artifact.local_sha256,
        "remote_sha256": artifact.remote_sha256,
        "remote_present": artifact.remote_present,
        "status": artifact.status,
    }


def _report_to_dict(report: ComparisonReport) -> dict[str, object]:
    """Serialize a comparison report for machine-readable output."""
    return {
        "needs_update": report.needs_update,
        "tts": {
            "remote_names": report.tts.remote_names,
            "local_names": report.tts.local_names,
            "added_names": report.tts.added_names,
            "removed_names": report.tts.removed_names,
            "needs_update": report.tts.needs_update,
            "hashes_checked": report.tts.hashes_checked,
            "voices": [
                {
                    "name": voice.name,
                    "id": voice.voice_id,
                    "status": voice.status,
                    "artifacts": [_artifact_to_dict(artifact) for artifact in voice.artifacts],
                }
                for voice in report.tts.voices
            ],
        },
        "zerotts": {
            "repo": report.zerotts.repo,
            "local_revision": ZEROTTS_MODEL.revision,
            "remote_revision": report.zerotts.remote_revision,
            "revision_changed": report.zerotts.revision_changed,
            "remote_voice_ids": report.zerotts.remote_voice_ids,
            "local_voice_ids": report.zerotts.local_voice_ids,
            "added_voice_ids": report.zerotts.added_voice_ids,
            "removed_voice_ids": report.zerotts.removed_voice_ids,
            "needs_update": report.zerotts.needs_update,
            "hashes_checked": report.zerotts.hashes_checked,
            "artifacts": [_artifact_to_dict(artifact) for artifact in report.zerotts.artifacts],
            "voices": [
                {
                    "name": voice.name,
                    "id": voice.voice_id,
                    "status": voice.status,
                    "artifacts": [_artifact_to_dict(artifact) for artifact in voice.artifacts],
                }
                for voice in report.zerotts.voices
            ],
        },
        "stt": {
            "repo": STT_MODEL.repo,
            "local_revision": STT_MODEL.revision,
            "remote_revision": report.stt.remote_revision,
            "revision_changed": report.stt.revision_changed,
            "needs_update": report.stt.needs_update,
            "artifacts": [_artifact_to_dict(artifact) for artifact in report.stt.artifacts],
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tts-hashes",
        action="store_true",
        help="do not stream remote TTS voice files; still check the STT metadata",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {_DEFAULT_TIMEOUT_SECONDS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the remote model comparison tool."""
    args = _parse_args(argv)
    if args.timeout <= 0:
        print("ERROR: --timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        report = compare_models(args.timeout, hash_tts=not args.skip_tts_hashes)
    except ModelCheckError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 1 if report.needs_update else 0


if __name__ == "__main__":
    raise SystemExit(main())
