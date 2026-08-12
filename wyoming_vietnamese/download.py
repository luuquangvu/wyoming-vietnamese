"""Hugging Face model synchronization and local asset structuring."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from hashlib import file_digest, sha256
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .config import get_env_bool
from .const import (
    NGHITTS_MODEL_BASE_URL,
    TTS_CONFIG_FILE,
    TTS_MODEL_FILE,
    TTS_SAMPLE_RATE,
    TTS_TOKENS_FILE,
)
from .stt_model import STT_MODEL, SttArtifact
from .tts_model import TtsVoiceSpec

_LOGGER = logging.getLogger(__name__)


def setup_hf_environment(cache_dir: Path, offline: bool = False) -> None:
    """Point Hugging Face home and its default Hub cache at persistent storage."""
    cache_str = str(cache_dir.expanduser().resolve())
    os.environ["HF_HOME"] = cache_str
    local_only = offline or get_env_bool("HF_HUB_OFFLINE", False)
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"


def download_models(
    cache_dir: Path,
    download_dir: Path,
    tts_voices: tuple[TtsVoiceSpec, ...],
    offline: bool = False,
) -> dict[str, Path]:
    """Synchronize required assets and build stable local model paths."""
    if not tts_voices:
        raise ValueError("At least one TTS voice must be configured")
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    setup_hf_environment(cache_dir, offline)

    _LOGGER.info(
        "Synchronizing model assets: cache_dir=%s, download_dir=%s",
        cache_dir,
        download_dir,
    )
    _LOGGER.info(
        "STT model: repo=%s revision=%s files=%d",
        STT_MODEL.repo,
        STT_MODEL.revision,
        len(STT_MODEL.artifacts),
    )
    stt_snapshot = _sync_repo(
        STT_MODEL.repo,
        offline,
        revision=STT_MODEL.revision,
        allow_patterns=STT_MODEL.allow_patterns,
    )

    tts_snapshots = {
        voice.id: _sync_nghitts_model(cache_dir, offline, voice) for voice in tts_voices
    }

    stt_dest = download_dir / "stt"
    tts_dest = download_dir / "tts"
    with _model_structure_lock(download_dir):
        stt_dest.mkdir(parents=True, exist_ok=True)
        _structure_stt_files(stt_snapshot, stt_dest)
        for voice in tts_voices:
            voice_dest = tts_dest / voice.id
            voice_dest.mkdir(parents=True, exist_ok=True)
            _structure_nghitts_files(tts_snapshots[voice.id], voice_dest)

    _LOGGER.info("Model verification and structure synchronization completed")
    return {"stt": stt_dest, "tts": tts_dest}


def _sync_repo(
    repo_id: str,
    offline: bool,
    *,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download a filtered snapshot or load a complete local snapshot."""
    from huggingface_hub import snapshot_download

    pattern_list = list(allow_patterns) if allow_patterns is not None else None
    if _local_only := offline or get_env_bool("HF_HUB_OFFLINE", False):
        _LOGGER.info("Loading %s from the local Hugging Face cache", repo_id)
        try:
            return Path(
                snapshot_download(
                    repo_id,
                    revision=revision,
                    local_files_only=True,
                    allow_patterns=pattern_list,
                )
            )
        except Exception as err:
            raise RuntimeError(
                f"Offline startup failed: repository {repo_id!r} at revision "
                f"{revision or 'main'!r} is not fully cached"
            ) from err

    _LOGGER.info("Downloading or checking repository %s", repo_id)
    try:
        return Path(
            snapshot_download(
                repo_id,
                revision=revision,
                local_files_only=False,
                allow_patterns=pattern_list,
            )
        )
    except Exception as remote_err:
        _LOGGER.warning(
            "Hugging Face check failed for %s; attempting the local cache: %s",
            repo_id,
            remote_err,
        )
        try:
            return Path(
                snapshot_download(
                    repo_id,
                    revision=revision,
                    local_files_only=True,
                    allow_patterns=pattern_list,
                )
            )
        except Exception as local_err:
            raise RuntimeError(
                f"Repository {repo_id!r} could not be downloaded and no complete local "
                "snapshot exists"
            ) from local_err


def _file_sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    with path.open("rb") as input_file:
        return file_digest(input_file, "sha256").hexdigest()


def _is_verified_file(path: Path, expected_sha256: str) -> bool:
    """Return whether a cached artifact exactly matches its pinned identity."""
    return path.is_file() and _file_sha256(path) == expected_sha256


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Download one artifact and atomically publish it after digest verification."""
    temporary_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    digest = sha256()
    try:
        request = Request(url, headers={"User-Agent": "wyoming-vietnamese/0.1"})
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as output_file:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())

        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Download from {url} has SHA-256 {actual_sha256}; expected {expected_sha256}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sync_nghitts_model(
    cache_dir: Path,
    offline: bool,
    voice: TtsVoiceSpec,
    *,
    base_url: str = NGHITTS_MODEL_BASE_URL,
) -> Path:
    """Synchronize the selected NghiTTS voice from its checksum-pinned API files."""
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("The NghiTTS model source must be an absolute HTTPS URL")

    snapshot_path = cache_dir / "nghitts" / voice.id
    snapshot_path.mkdir(parents=True, exist_ok=True)
    lock_path = snapshot_path / ".download.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            for artifact in voice.artifacts:
                destination = snapshot_path / artifact.local_name
                if _is_verified_file(destination, artifact.sha256):
                    continue
                if offline:
                    raise RuntimeError(
                        "Offline startup failed: checksum-pinned NghiTTS artifact is not "
                        f"cached: {destination!s}"
                    )
                source_url = f"{base_url.rstrip('/')}/{quote(artifact.remote_name, safe='')}"
                _LOGGER.info("Downloading NghiTTS artifact %s", artifact.remote_name)
                try:
                    _download_verified_file(
                        source_url,
                        destination,
                        expected_sha256=artifact.sha256,
                    )
                except Exception as err:
                    raise RuntimeError(
                        f"NghiTTS artifact {artifact.remote_name!r} failed checksum-pinned download"
                    ) from err
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return snapshot_path


@contextmanager
def _model_structure_lock(download_dir: Path) -> Iterator[None]:
    """Serialize generated model-directory updates across server processes."""
    lock_path = download_dir / ".structure.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_varint(data: bytes, offset: int, limit: int) -> tuple[int, int]:
    """Read one bounded protobuf varint."""
    result = 0
    shift = 0
    while offset < limit and shift < 70:
        value = data[offset]
        offset += 1
        result |= (value & 0x7F) << shift
        if not value & 0x80:
            return result, offset
        shift += 7
    raise ValueError("Truncated or oversized protobuf varint")


def _skip_protobuf_field(data: bytes, offset: int, limit: int, wire_type: int) -> int:
    """Skip one bounded protobuf field value."""
    if wire_type == 0:
        _, offset = _read_varint(data, offset, limit)
    elif wire_type == 1:
        offset += 8
    elif wire_type == 2:
        length, offset = _read_varint(data, offset, limit)
        offset += length
    elif wire_type == 5:
        offset += 4
    else:
        raise ValueError(f"Unsupported protobuf wire type: {wire_type}")
    if offset > limit:
        raise ValueError("Protobuf field exceeds its containing message")
    return offset


def _generate_tokens_from_bpe(bpe_model_path: Path, output_tokens_path: Path) -> None:
    """Generate Sherpa tokens by safely parsing SentencePiece protobuf pieces."""
    data = bpe_model_path.read_bytes()
    offset = 0
    size = len(data)
    pieces: list[str] = []

    while offset < size:
        tag, offset = _read_varint(data, offset, size)
        field_number = tag >> 3
        wire_type = tag & 7
        if field_number != 1 or wire_type != 2:
            offset = _skip_protobuf_field(data, offset, size, wire_type)
            continue

        message_length, offset = _read_varint(data, offset, size)
        message_end = offset + message_length
        if message_end > size:
            raise ValueError("SentencePiece entry exceeds the model size")

        piece: str | None = None
        while offset < message_end:
            sub_tag, offset = _read_varint(data, offset, message_end)
            sub_field = sub_tag >> 3
            sub_wire = sub_tag & 7
            if sub_field == 1 and sub_wire == 2:
                piece_length, offset = _read_varint(data, offset, message_end)
                piece_end = offset + piece_length
                if piece_end > message_end:
                    raise ValueError("SentencePiece text exceeds its entry")
                piece = data[offset:piece_end].decode("utf-8")
                offset = piece_end
            else:
                offset = _skip_protobuf_field(data, offset, message_end, sub_wire)

        if piece is None:
            raise ValueError("SentencePiece entry has no piece text")
        pieces.append(piece)

    if not pieces:
        raise ValueError("SentencePiece model contains no token pieces")

    output_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_tokens_path.with_name(f".{output_tokens_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as tokens_file:
            for index, piece in enumerate(pieces):
                tokens_file.write(f"{piece} {index}\n")
            tokens_file.flush()
            os.fsync(tokens_file.fileno())
        os.replace(temporary_path, output_tokens_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_nghitts_configuration(config_path: Path) -> dict[str, object]:
    """Load a NghiTTS JSON object without allowing non-string top-level keys."""
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("NghiTTS configuration root must be an object")
    config: dict[str, object] = {}
    for key, value in raw_config.items():
        if not isinstance(key, str):
            raise ValueError("NghiTTS configuration keys must be strings")
        config[key] = value
    return config


def _generate_nghitts_tokens(config_path: Path, output_tokens_path: Path) -> None:
    """Generate Sherpa token IDs from a validated NghiTTS configuration."""
    config = _load_nghitts_configuration(config_path)
    phoneme_id_map = config.get("phoneme_id_map")
    if not isinstance(phoneme_id_map, dict) or not phoneme_id_map:
        raise ValueError("NghiTTS configuration has no phoneme_id_map")

    tokens_by_id: dict[int, str] = {}
    for token, raw_ids in phoneme_id_map.items():
        if not isinstance(token, str) or "\n" in token or "\r" in token:
            raise ValueError("NghiTTS phoneme tokens must be single-line strings")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"NghiTTS phoneme {token!r} has no token IDs")
        for token_id in raw_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"NghiTTS phoneme {token!r} has an invalid token ID")
            if token_id in tokens_by_id:
                raise ValueError(f"NghiTTS token ID {token_id} is duplicated")
            tokens_by_id[token_id] = token

    expected_ids = list(range(len(tokens_by_id)))
    if sorted(tokens_by_id) != expected_ids:
        raise ValueError("NghiTTS token IDs must be contiguous and start at zero")

    output_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_tokens_path.with_name(f".{output_tokens_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as tokens_file:
            for token_id in expected_ids:
                tokens_file.write(f"{tokens_by_id[token_id]} {token_id}\n")
            tokens_file.flush()
            os.fsync(tokens_file.fileno())
        os.replace(temporary_path, output_tokens_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _encode_protobuf_varint(value: int) -> bytes:
    """Encode one non-negative protobuf varint."""
    if value < 0:
        raise ValueError("Protobuf varints must not be negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _encode_protobuf_text_field(field_number: int, value: str) -> bytes:
    """Encode one length-delimited UTF-8 protobuf field."""
    encoded_value = value.encode("utf-8")
    return (
        _encode_protobuf_varint((field_number << 3) | 2)
        + _encode_protobuf_varint(len(encoded_value))
        + encoded_value
    )


def _nghitts_sherpa_metadata(config_path: Path) -> tuple[tuple[str, str], ...]:
    """Build the NghiTTS metadata fields required by Sherpa-ONNX."""
    config = _load_nghitts_configuration(config_path)
    audio = config.get("audio")
    if not isinstance(audio, dict) or audio.get("sample_rate") != TTS_SAMPLE_RATE:
        raise ValueError(f"NghiTTS configuration must use {TTS_SAMPLE_RATE} Hz audio")
    if config.get("num_speakers") != 1:
        raise ValueError("NghiTTS configuration must contain exactly one speaker")
    if config.get("phoneme_type") != "espeak":
        raise ValueError("NghiTTS configuration must use eSpeak phonemes")
    espeak = config.get("espeak")
    voice = espeak.get("voice") if isinstance(espeak, dict) else None
    if not isinstance(voice, str) or not voice:
        raise ValueError("NghiTTS configuration has no eSpeak voice")
    return (
        ("sample_rate", str(TTS_SAMPLE_RATE)),
        ("n_speakers", "1"),
        ("model_type", "vits"),
        ("comment", "piper"),
        ("language", "Vietnamese"),
        ("voice", voice),
        ("has_espeak", "1"),
    )


def _encode_onnx_metadata(entries: Sequence[tuple[str, str]]) -> bytes:
    """Encode ONNX ModelProto metadata_props entries without a runtime ONNX dependency."""
    encoded = bytearray()
    metadata_tag = _encode_protobuf_varint((14 << 3) | 2)
    for key, value in entries:
        entry = _encode_protobuf_text_field(1, key) + _encode_protobuf_text_field(2, value)
        encoded.extend(metadata_tag)
        encoded.extend(_encode_protobuf_varint(len(entry)))
        encoded.extend(entry)
    return bytes(encoded)


def _prepared_marker_path(destination_path: Path) -> Path:
    """Return the sidecar that records one completed Sherpa preparation."""
    return destination_path.with_name(f".{destination_path.name}.prepared.json")


def _preparation_identity(
    source_path: Path,
    destination_path: Path,
    metadata: bytes,
) -> dict[str, int | str]:
    """Describe the source, output, and metadata recorded for one preparation."""
    source_stat = source_path.stat()
    destination_stat = destination_path.stat()
    return {
        "metadata_length": len(metadata),
        "metadata_sha256": sha256(metadata).hexdigest(),
        "output_mtime_ns": destination_stat.st_mtime_ns,
        "output_size": destination_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_size": source_stat.st_size,
    }


def _is_prepared_for_sherpa(
    source_path: Path,
    destination_path: Path,
    metadata: bytes,
) -> bool:
    """Return whether a recorded preparation still matches its file metadata."""
    marker_path = _prepared_marker_path(destination_path)
    if not destination_path.is_file() or not marker_path.is_file():
        return False
    try:
        recorded = json.loads(marker_path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return False
    return recorded == _preparation_identity(source_path, destination_path, metadata)


def _record_sherpa_preparation(
    source_path: Path,
    destination_path: Path,
    metadata: bytes,
) -> None:
    """Publish the preparation marker atomically once the output is in place."""
    marker_path = _prepared_marker_path(destination_path)
    payload = json.dumps(
        _preparation_identity(source_path, destination_path, metadata),
        sort_keys=True,
    )
    temporary_path = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as marker_file:
            marker_file.write(payload)
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(temporary_path, marker_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _convert_nghitts_model_for_sherpa(
    source_path: Path,
    config_path: Path,
    destination_path: Path,
) -> None:
    """Atomically append metadata missing from raw NghiTTS ONNX exports.

    The pinned source digest is already verified during synchronization, so a matching
    marker skips two further full-file hashes of a 60 MB graph on every start. A missing
    or stale marker falls back to complete digest verification.
    """
    metadata = _encode_onnx_metadata(_nghitts_sherpa_metadata(config_path))
    if _is_prepared_for_sherpa(source_path, destination_path, metadata):
        return

    expected_digest = sha256()
    with source_path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            expected_digest.update(chunk)
    expected_digest.update(metadata)
    if not (
        destination_path.is_file()
        and destination_path.stat().st_size == source_path.stat().st_size + len(metadata)
        and _file_sha256(destination_path) == expected_digest.hexdigest()
    ):
        temporary_path = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
        try:
            with source_path.open("rb") as source_file, temporary_path.open("wb") as output_file:
                shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
                output_file.write(metadata)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    _record_sherpa_preparation(source_path, destination_path, metadata)


def _verified_stt_source(snapshot_path: Path, artifact: SttArtifact) -> Path:
    """Return one pinned snapshot file, rejecting any content that is not identical."""
    source = snapshot_path / artifact.remote_name
    if not source.is_file():
        raise FileNotFoundError(
            f"Pinned STT artifact is missing from {snapshot_path!s}: {artifact.remote_name}"
        )
    if not _is_verified_file(source, artifact.sha256):
        raise RuntimeError(
            f"Pinned STT artifact {artifact.remote_name!r} does not match its recorded "
            f"SHA-256 digest"
        )
    return source


def _structure_stt_files(snapshot_path: Path, dest_dir: Path) -> list[Path]:
    """Verify every pinned STT artifact and expose stable local filenames."""
    sources = {
        artifact.remote_name: _verified_stt_source(snapshot_path, artifact)
        for artifact in STT_MODEL.artifacts
    }

    copied_paths: list[Path] = []
    for artifact in STT_MODEL.graphs:
        destination = dest_dir / artifact.local_name
        _copy_or_link(sources[artifact.remote_name], destination)
        copied_paths.append(destination)

    generated_tokens = dest_dir / "tokens.txt"
    _generate_tokens_from_bpe(sources[STT_MODEL.tokenizer.remote_name], generated_tokens)
    copied_paths.append(generated_tokens)
    return copied_paths


def _structure_nghitts_files(snapshot_path: Path, dest_dir: Path) -> list[Path]:
    """Build Sherpa's local layout from a synchronized NghiTTS voice snapshot."""
    model_source = snapshot_path / TTS_MODEL_FILE
    config_source = snapshot_path / TTS_CONFIG_FILE
    if missing := [path.name for path in (model_source, config_source) if not path.is_file()]:
        raise FileNotFoundError(
            f"Incomplete NghiTTS snapshot {snapshot_path!s}; missing: {', '.join(missing)}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    config_destination = dest_dir / TTS_CONFIG_FILE
    tokens_destination = dest_dir / TTS_TOKENS_FILE
    model_destination = dest_dir / TTS_MODEL_FILE
    _copy_or_link(config_source, config_destination)
    _generate_nghitts_tokens(config_destination, tokens_destination)
    _convert_nghitts_model_for_sherpa(
        model_source,
        config_destination,
        model_destination,
    )
    return [model_destination, config_destination, tokens_destination]


def _copy_or_link(src: Path, dst: Path) -> None:
    """Atomically hard-link an asset, falling back to a metadata-preserving copy."""
    source = src.resolve(strict=True)
    source_stat = source.stat()
    if dst.is_file():
        with suppress(OSError):
            if os.path.samefile(source, dst):
                return
        destination_stat = dst.stat()
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return
    elif dst.exists():
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary_path)
        except OSError as err:
            if err.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            shutil.copy2(source, temporary_path)
        os.replace(temporary_path, dst)
    finally:
        temporary_path.unlink(missing_ok=True)
