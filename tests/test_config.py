"""Configuration parsing tests."""

from pathlib import Path

import pytest

from wyoming_vietnamese.config import (
    ServerConfig,
    _get_float,
    get_env_bool,
    resolve_cpu_threads,
)
from wyoming_vietnamese.const import DEFAULT_PORT
from wyoming_vietnamese.tts_model import DEFAULT_TTS_VOICE_ID


def test_get_float_conflicting_bounds() -> None:
    """Test _get_float rejects contradictory bound definitions."""
    with pytest.raises(ValueError, match="both minimum_exclusive and minimum_inclusive"):
        _get_float({}, "TEST", 1.0, minimum_exclusive=0.0, minimum_inclusive=0.0)
    with pytest.raises(ValueError, match="maximum_inclusive cannot be less"):
        _get_float({}, "TEST", 1.0, minimum_inclusive=5.0, maximum_inclusive=4.0)
    with pytest.raises(ValueError, match="maximum_inclusive must be greater"):
        _get_float({}, "TEST", 1.0, minimum_exclusive=5.0, maximum_inclusive=5.0)


def test_get_float_bounds() -> None:
    """Test _get_float enforces inclusive and exclusive bounds."""
    assert (
        _get_float({"TEST": "5.0"}, "TEST", 1.0, minimum_inclusive=5.0, maximum_inclusive=10.0)
        == 5.0
    )
    assert (
        _get_float({"TEST": "10.0"}, "TEST", 1.0, minimum_inclusive=5.0, maximum_inclusive=10.0)
        == 10.0
    )
    with pytest.raises(ValueError, match=r"must be at least 5\.0"):
        _get_float({"TEST": "4.9"}, "TEST", 1.0, minimum_inclusive=5.0)
    with pytest.raises(ValueError, match=r"must be at most 10\.0"):
        _get_float({"TEST": "10.1"}, "TEST", 1.0, maximum_inclusive=10.0)


@pytest.mark.parametrize("value", ["true", "1", "YES", " on "])
def test_get_env_bool_true(value: str) -> None:
    """Test get env bool true."""
    assert get_env_bool("FLAG", False, {"FLAG": value}) is True


@pytest.mark.parametrize("value", ["false", "0", "NO", " off "])
def test_get_env_bool_false(value: str) -> None:
    """Test get env bool false."""
    assert get_env_bool("FLAG", True, {"FLAG": value}) is False


def test_get_env_bool_default_and_invalid() -> None:
    """Test get env bool default and invalid."""
    assert get_env_bool("FLAG", True, {}) is True
    assert get_env_bool("FLAG", False, {"FLAG": " "}) is False
    with pytest.raises(ValueError, match="FLAG must be"):
        get_env_bool("FLAG", False, {"FLAG": "sometimes"})


@pytest.mark.parametrize(
    ("configured_threads", "expected_threads"),
    [(0, 4), (2, 2), (8, 4)],
)
def test_resolve_cpu_threads_caps_to_available_cpus(
    configured_threads: int,
    expected_threads: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test auto resolution and explicit thread capping."""
    monkeypatch.setattr("wyoming_vietnamese.config.os.process_cpu_count", lambda: 4)
    assert resolve_cpu_threads(configured_threads) == expected_threads


def test_resolve_cpu_threads_rejects_negative_values() -> None:
    """Test direct callers cannot bypass non-negative thread validation."""
    with pytest.raises(ValueError, match="must not be negative"):
        resolve_cpu_threads(-1)


def test_resolve_cpu_threads_uses_cpu_count_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CPU detection remains positive when affinity data is unavailable."""
    monkeypatch.setattr("wyoming_vietnamese.config.os.process_cpu_count", lambda: None)
    monkeypatch.setattr("wyoming_vietnamese.config.os.cpu_count", lambda: 2)
    assert resolve_cpu_threads(0) == 2
    monkeypatch.setattr("wyoming_vietnamese.config.os.cpu_count", lambda: None)
    assert resolve_cpu_threads(0) == 1


def test_server_config_defaults() -> None:
    """Test server config defaults."""
    config = ServerConfig.from_env({})
    assert config.port == DEFAULT_PORT
    assert config.cpu_threads == 0
    assert config.offline is False
    assert [voice.id for voice in config.tts_voices] == [DEFAULT_TTS_VOICE_ID]
    assert config.event_timeout == 60
    assert config.write_timeout == 5
    assert config.max_active_connections == 64
    assert config.max_stt_buffer_bytes == 256 * 1024 * 1024
    assert config.tts_cache_idle_seconds == 86400
    assert config.tts_cache_max_entries == 128
    assert config.tts_cache_max_bytes == 64 * 1024 * 1024
    assert config.tts_cache_max_item_bytes == 4 * 1024 * 1024


def test_server_config_custom_values() -> None:
    """Test server config custom values."""
    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "12000",
            "TTS_VOICE": "chieu-thanh, ngoc-huyen-moi",
            "CACHE_DIR": "~/cache",
            "DOWNLOAD_DIR": "~/models",
            "CPU_THREADS": "4",
            "OFFLINE": "yes",
            "LOG_LEVEL": "debug",
            "MAX_STT_AUDIO_SECONDS": "2.5",
            "MAX_TTS_TEXT_CHARS": "50",
            "TTS_SENTENCE_SILENCE_MS": "450",
            "TTS_CLAUSE_SILENCE_MS": "0",
            "TTS_SILENCE_JITTER_PERCENT": "40",
            "INFERENCE_QUEUE_TIMEOUT": "1.25",
            "WYOMING_EVENT_TIMEOUT": "2.5",
            "WYOMING_WRITE_TIMEOUT": "0.75",
            "MAX_ACTIVE_CONNECTIONS": "12",
            "MAX_STT_BUFFER_MB": "34",
            "TTS_CACHE_IDLE_SECONDS": "90",
            "TTS_CACHE_MAX_ENTRIES": "8",
            "TTS_CACHE_MAX_MB": "12",
            "TTS_CACHE_MAX_ITEM_MB": "3",
        }
    )
    assert config.port == 12000
    assert [voice.id for voice in config.tts_voices] == [
        "chieu-thanh",
        "ngoc-huyen-moi",
    ]
    assert [voice.name for voice in config.tts_voices] == [
        "Chiếu Thành",
        "Ngọc Huyền (mới)",
    ]
    assert config.cache_dir == Path("~/cache").expanduser()
    assert config.download_dir == Path("~/models").expanduser()
    assert config.cpu_threads == 4
    assert config.offline is True
    assert config.log_level == "DEBUG"
    assert config.max_stt_audio_seconds == 2.5
    assert config.max_tts_text_chars == 50
    assert config.tts_sentence_silence_ms == 450
    assert config.tts_clause_silence_ms == 0
    assert config.tts_silence_jitter_percent == 40
    assert config.inference_queue_timeout == 1.25
    assert config.event_timeout == 2.5
    assert config.write_timeout == 0.75
    assert config.max_active_connections == 12
    assert config.max_stt_buffer_bytes == 34 * 1024 * 1024
    assert config.tts_cache_idle_seconds == 90
    assert config.tts_cache_max_entries == 8
    assert config.tts_cache_max_bytes == 12 * 1024 * 1024
    assert config.tts_cache_max_item_bytes == 3 * 1024 * 1024


@pytest.mark.parametrize(
    "value",
    [
        "chieu-thanh,ngoc-huyen-moi",
        "chieu-thanh ngoc-huyen-moi",
        "chieu-thanh, ngoc-huyen-moi",
        "chieu-thanh,\tngoc-huyen-moi",
    ],
)
def test_server_config_accepts_tts_voice_separators(value: str) -> None:
    """Test comma and whitespace voice separators can be mixed."""
    config = ServerConfig.from_env({"TTS_VOICE": value})
    assert [voice.id for voice in config.tts_voices] == [
        "chieu-thanh",
        "ngoc-huyen-moi",
    ]


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"WYOMING_PORT": "bad"}, "must be an integer"),
        ({"WYOMING_PORT": "0"}, "must be at least"),
        ({"WYOMING_PORT": "70000"}, "at most"),
        ({"CPU_THREADS": "-1"}, "must be at least"),
        ({"MAX_STT_AUDIO_SECONDS": "none"}, "must be a number"),
        ({"MAX_STT_AUDIO_SECONDS": "0"}, "must be greater"),
        ({"MAX_STT_AUDIO_SECONDS": "nan"}, "must be finite"),
        ({"INFERENCE_QUEUE_TIMEOUT": "inf"}, "must be finite"),
        ({"MAX_TTS_TEXT_CHARS": "0"}, "must be at least"),
        ({"TTS_SENTENCE_SILENCE_MS": "-1"}, "must be at least"),
        ({"TTS_CLAUSE_SILENCE_MS": "3001"}, "at most"),
        ({"TTS_SILENCE_JITTER_PERCENT": "-1"}, "must be at least"),
        ({"TTS_SILENCE_JITTER_PERCENT": "101"}, "at most"),
        ({"INFERENCE_QUEUE_TIMEOUT": "0"}, "must be greater"),
        ({"WYOMING_EVENT_TIMEOUT": "0"}, "must be greater"),
        ({"WYOMING_WRITE_TIMEOUT": "-1"}, "must be greater"),
        ({"MAX_ACTIVE_CONNECTIONS": "0"}, "must be at least"),
        ({"MAX_STT_BUFFER_MB": "0"}, "must be at least"),
        ({"TTS_CACHE_IDLE_SECONDS": "0"}, "must be greater"),
        ({"TTS_CACHE_MAX_ENTRIES": "-1"}, "must be at least"),
        ({"TTS_CACHE_MAX_MB": "-1"}, "must be at least"),
        ({"TTS_CACHE_MAX_ITEM_MB": "65"}, "must not exceed"),
        ({"TTS_VOICE": "unknown"}, "must be one of"),
        ({"TTS_VOICE": "ngoc-huyen-moi;ngoc-ngan"}, "must be one of"),
        ({"TTS_VOICE": "ngoc-huyen-moi,ngoc-huyen-moi"}, "duplicate"),
        ({"LOG_LEVEL": "verbose"}, "LOG_LEVEL is invalid"),
    ],
)
def test_server_config_rejects_invalid_values(environment: dict[str, str], message: str) -> None:
    """Test server config rejects invalid values."""
    with pytest.raises(ValueError, match=message):
        ServerConfig.from_env(environment)
