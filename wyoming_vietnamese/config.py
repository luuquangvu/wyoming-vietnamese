"""Validated environment configuration for the server process."""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .const import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_EVENT_TIMEOUT,
    DEFAULT_INFERENCE_QUEUE_TIMEOUT,
    DEFAULT_MAX_ACTIVE_CONNECTIONS,
    DEFAULT_MAX_STT_AUDIO_SECONDS,
    DEFAULT_MAX_STT_BUFFER_MB,
    DEFAULT_MAX_TTS_TEXT_CHARS,
    DEFAULT_PORT,
    DEFAULT_TTS_CACHE_IDLE_SECONDS,
    DEFAULT_TTS_CACHE_MAX_ENTRIES,
    DEFAULT_TTS_CACHE_MAX_ITEM_MB,
    DEFAULT_TTS_CACHE_MAX_MB,
    DEFAULT_TTS_CLAUSE_SILENCE_MS,
    DEFAULT_TTS_SENTENCE_SILENCE_MS,
    DEFAULT_TTS_SILENCE_JITTER_PERCENT,
    DEFAULT_WRITE_TIMEOUT,
    MAX_TTS_SILENCE_JITTER_PERCENT,
    MAX_TTS_SILENCE_MS,
)
from .tts_model import DEFAULT_TTS_VOICE_ID, TtsVoiceSpec, get_voice


def get_env_bool(
    name: str,
    default: bool,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read a strict boolean environment variable."""
    env = os.environ if environ is None else environ
    raw_value = env.get(name)
    if raw_value is None or not raw_value.strip():
        return default

    value = raw_value.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


def resolve_cpu_threads(configured_threads: int) -> int:
    """Resolve auto mode and cap inference threads to process-available CPUs."""
    if configured_threads < 0:
        raise ValueError("configured threads must not be negative")
    available_threads = max(os.process_cpu_count() or os.cpu_count() or 1, 1)
    return min(configured_threads, available_threads) if configured_threads else available_threads


def _get_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Read and range-check an integer environment variable."""
    raw_value = environ.get(name)
    try:
        value = default if raw_value is None or not raw_value.strip() else int(raw_value)
    except ValueError as err:
        raise ValueError(f"{name} must be an integer") from err

    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be at least {minimum}{suffix}")
    return value


def _get_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum_exclusive: float,
) -> float:
    """Read and range-check a floating-point environment variable."""
    raw_value = environ.get(name)
    try:
        value = default if raw_value is None or not raw_value.strip() else float(raw_value)
    except ValueError as err:
        raise ValueError(f"{name} must be a number") from err

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= minimum_exclusive:
        raise ValueError(f"{name} must be greater than {minimum_exclusive}")
    return value


def _get_required_text(environ: Mapping[str, str], name: str, default: str) -> str:
    """Read a non-empty text environment variable."""
    if value := environ.get(name, default).strip():
        return value
    raise ValueError(f"{name} must not be empty")


def _get_tts_voices(environ: Mapping[str, str]) -> tuple[TtsVoiceSpec, ...]:
    """Read unique TTS voice IDs separated by commas and/or whitespace."""
    raw_value = _get_required_text(environ, "TTS_VOICE", DEFAULT_TTS_VOICE_ID)
    voice_ids = [voice_id.lower() for voice_id in re.split(r"[,\s]+", raw_value) if voice_id]
    if not voice_ids:
        raise ValueError("TTS_VOICE must contain at least one voice ID")
    if len(set(voice_ids)) != len(voice_ids):
        raise ValueError("TTS_VOICE must not contain duplicate voice IDs")
    return tuple(get_voice(voice_id) for voice_id in voice_ids)


def _get_log_level(environ: Mapping[str, str]) -> str:
    """Read a logging level accepted by the standard library."""
    value = environ.get("LOG_LEVEL", "INFO").strip().upper()
    if value not in logging.getLevelNamesMapping():
        raise ValueError(f"LOG_LEVEL is invalid: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Validated process configuration derived from environment variables."""

    port: int
    tts_voices: tuple[TtsVoiceSpec, ...]
    cache_dir: Path
    download_dir: Path
    cpu_threads: int
    offline: bool
    log_level: str
    max_stt_audio_seconds: float
    max_tts_text_chars: int
    tts_sentence_silence_ms: int
    tts_clause_silence_ms: int
    tts_silence_jitter_percent: int
    inference_queue_timeout: float
    event_timeout: float
    write_timeout: float
    max_active_connections: int
    max_stt_buffer_bytes: int
    tts_cache_idle_seconds: float
    tts_cache_max_entries: int
    tts_cache_max_bytes: int
    tts_cache_max_item_bytes: int

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerConfig:
        """Build configuration from an environment mapping."""
        env = os.environ if environ is None else environ

        tts_cache_max_mb = _get_int(
            env,
            "TTS_CACHE_MAX_MB",
            DEFAULT_TTS_CACHE_MAX_MB,
            minimum=0,
        )
        tts_cache_max_item_mb = _get_int(
            env,
            "TTS_CACHE_MAX_ITEM_MB",
            DEFAULT_TTS_CACHE_MAX_ITEM_MB,
            minimum=0,
        )
        if tts_cache_max_mb and tts_cache_max_item_mb > tts_cache_max_mb:
            raise ValueError("TTS_CACHE_MAX_ITEM_MB must not exceed TTS_CACHE_MAX_MB")

        return cls(
            port=_get_int(env, "WYOMING_PORT", DEFAULT_PORT, minimum=1, maximum=65535),
            tts_voices=_get_tts_voices(env),
            cache_dir=Path(_get_required_text(env, "CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser(),
            download_dir=Path(
                _get_required_text(env, "DOWNLOAD_DIR", DEFAULT_DOWNLOAD_DIR)
            ).expanduser(),
            cpu_threads=_get_int(env, "CPU_THREADS", 0, minimum=0),
            offline=get_env_bool("OFFLINE", False, env),
            log_level=_get_log_level(env),
            max_stt_audio_seconds=_get_float(
                env,
                "MAX_STT_AUDIO_SECONDS",
                DEFAULT_MAX_STT_AUDIO_SECONDS,
                minimum_exclusive=0,
            ),
            max_tts_text_chars=_get_int(
                env,
                "MAX_TTS_TEXT_CHARS",
                DEFAULT_MAX_TTS_TEXT_CHARS,
                minimum=1,
            ),
            tts_sentence_silence_ms=_get_int(
                env,
                "TTS_SENTENCE_SILENCE_MS",
                DEFAULT_TTS_SENTENCE_SILENCE_MS,
                minimum=0,
                maximum=MAX_TTS_SILENCE_MS,
            ),
            tts_clause_silence_ms=_get_int(
                env,
                "TTS_CLAUSE_SILENCE_MS",
                DEFAULT_TTS_CLAUSE_SILENCE_MS,
                minimum=0,
                maximum=MAX_TTS_SILENCE_MS,
            ),
            tts_silence_jitter_percent=_get_int(
                env,
                "TTS_SILENCE_JITTER_PERCENT",
                DEFAULT_TTS_SILENCE_JITTER_PERCENT,
                minimum=0,
                maximum=MAX_TTS_SILENCE_JITTER_PERCENT,
            ),
            inference_queue_timeout=_get_float(
                env,
                "INFERENCE_QUEUE_TIMEOUT",
                DEFAULT_INFERENCE_QUEUE_TIMEOUT,
                minimum_exclusive=0,
            ),
            event_timeout=_get_float(
                env,
                "WYOMING_EVENT_TIMEOUT",
                DEFAULT_EVENT_TIMEOUT,
                minimum_exclusive=0,
            ),
            write_timeout=_get_float(
                env,
                "WYOMING_WRITE_TIMEOUT",
                DEFAULT_WRITE_TIMEOUT,
                minimum_exclusive=0,
            ),
            max_active_connections=_get_int(
                env,
                "MAX_ACTIVE_CONNECTIONS",
                DEFAULT_MAX_ACTIVE_CONNECTIONS,
                minimum=1,
            ),
            max_stt_buffer_bytes=(
                _get_int(
                    env,
                    "MAX_STT_BUFFER_MB",
                    DEFAULT_MAX_STT_BUFFER_MB,
                    minimum=1,
                )
                * 1024
                * 1024
            ),
            tts_cache_idle_seconds=_get_float(
                env,
                "TTS_CACHE_IDLE_SECONDS",
                DEFAULT_TTS_CACHE_IDLE_SECONDS,
                minimum_exclusive=0,
            ),
            tts_cache_max_entries=_get_int(
                env,
                "TTS_CACHE_MAX_ENTRIES",
                DEFAULT_TTS_CACHE_MAX_ENTRIES,
                minimum=0,
            ),
            tts_cache_max_bytes=tts_cache_max_mb * 1024 * 1024,
            tts_cache_max_item_bytes=tts_cache_max_item_mb * 1024 * 1024,
        )
