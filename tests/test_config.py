"""Configuration parsing tests."""

from pathlib import Path

import pytest

from wyoming_vietnamese.config import (
    ServerConfig,
    _get_float,
    get_env_bool,
)
from wyoming_vietnamese.const import (
    DEFAULT_PORT,
    DEFAULT_STT_ENGINE,
    DEFAULT_TTS_ENGINE,
    SttEngine,
    TtsEngine,
    TtsSilenceMs,
    ZeroTtsCodebook,
    ZeroTtsEnv,
    ZeroTtsFile,
    ZeroTtsQuality,
)
from wyoming_vietnamese.tts_model import DEFAULT_NGHITTS_VOICE_ID


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


def test_server_config_defaults() -> None:
    """Test server config defaults."""
    config = ServerConfig.from_env({})
    assert config.port == DEFAULT_PORT
    assert config.cpu_threads == 0
    assert config.offline is False
    assert config.stt_engine == DEFAULT_STT_ENGINE
    assert config.tts_engine == DEFAULT_TTS_ENGINE
    assert [voice.id for voice in config.tts_voices] == [DEFAULT_NGHITTS_VOICE_ID]
    assert config.event_timeout == 60
    assert config.write_timeout == 5
    assert config.max_active_connections == 64
    assert config.max_stt_buffer_bytes == 256 * 1024 * 1024
    assert config.tts_sentence_silence_ms == TtsSilenceMs.SENTENCE
    assert config.tts_clause_silence_ms == TtsSilenceMs.CLAUSE
    assert config.tts_paragraph_silence_ms == TtsSilenceMs.PARAGRAPH
    assert config.zerotts_model == "auto"
    assert config.zerotts_opportunistic is True


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
            "MAX_TTS_STREAM_SECONDS": "45",
            "TTS_SENTENCE_SILENCE_MS": "450",
            "TTS_CLAUSE_SILENCE_MS": "0",
            "TTS_PARAGRAPH_SILENCE_MS": "800",
            "INFERENCE_QUEUE_TIMEOUT": "1.25",
            "WYOMING_EVENT_TIMEOUT": "2.5",
            "WYOMING_WRITE_TIMEOUT": "0.75",
            "MAX_ACTIVE_CONNECTIONS": "12",
            "MAX_STT_BUFFER_MB": "34",
            "ZEROTTS_OPPORTUNISTIC": "false",
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
    assert config.max_tts_stream_seconds == 45
    assert config.tts_sentence_silence_ms == 450
    assert config.tts_clause_silence_ms == 0
    assert config.tts_paragraph_silence_ms == 800
    assert config.inference_queue_timeout == 1.25
    assert config.event_timeout == 2.5
    assert config.write_timeout == 0.75
    assert config.max_active_connections == 12
    assert config.max_stt_buffer_bytes == 34 * 1024 * 1024
    assert config.zerotts_opportunistic is False


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
        ({"MAX_TTS_STREAM_SECONDS": "0"}, "must be greater"),
        ({"TTS_SENTENCE_SILENCE_MS": "-1"}, "must be at least"),
        ({"TTS_CLAUSE_SILENCE_MS": "3001"}, "at most"),
        ({"TTS_PARAGRAPH_SILENCE_MS": "-1"}, "must be at least"),
        ({"TTS_PARAGRAPH_SILENCE_MS": "3001"}, "at most"),
        ({"INFERENCE_QUEUE_TIMEOUT": "0"}, "must be greater"),
        ({"WYOMING_EVENT_TIMEOUT": "0"}, "must be greater"),
        ({"WYOMING_WRITE_TIMEOUT": "-1"}, "must be greater"),
        ({"MAX_ACTIVE_CONNECTIONS": "0"}, "must be at least"),
        ({"MAX_STT_BUFFER_MB": "0"}, "must be at least"),
        ({"TTS_VOICE": "unknown"}, "must be one of"),
        ({"TTS_VOICE": "ngoc-huyen-moi;ngoc-ngan"}, "must be one of"),
        ({"TTS_VOICE": "ngoc-huyen-moi,ngoc-huyen-moi"}, "duplicate"),
        ({"LOG_LEVEL": "verbose"}, "LOG_LEVEL is invalid"),
        ({"TTS_ENGINE": "invalid"}, "TTS_ENGINE must be one of"),
        ({"STT_ENGINE": "invalid"}, "STT_ENGINE must be one of"),
        (
            {"TTS_ENGINE": TtsEngine.ZEROTTS, "TTS_VOICE": "ngoc-huyen-moi"},
            "TTS_VOICE must be one of",
        ),
    ],
)
def test_server_config_rejects_invalid_values(environment: dict[str, str], message: str) -> None:
    """Test server config rejects invalid values."""
    with pytest.raises(ValueError, match=message):
        ServerConfig.from_env(environment)


@pytest.mark.parametrize(
    "setting",
    [
        "TTS_SILENCE_JITTER_PERCENT",
        "TTS_CACHE_IDLE_SECONDS",
        "TTS_CACHE_MAX_ENTRIES",
        "TTS_CACHE_MAX_MB",
        "TTS_CACHE_MAX_ITEM_MB",
    ],
)
def test_server_config_warns_on_deprecated_settings(
    setting: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn on deprecated settings and continue with default configuration."""
    with caplog.at_level("WARNING"):
        config = ServerConfig.from_env({setting: "1"})
    assert "Deprecated environment settings are ignored" in caplog.text
    assert setting in caplog.text
    assert config == ServerConfig.from_env({})


def test_server_config_tts_engine_nghitts() -> None:
    """Test server config parses NghiTTS engine explicitly."""
    config = ServerConfig.from_env({"TTS_ENGINE": TtsEngine.NGHITTS})
    assert config.tts_engine == TtsEngine.NGHITTS


def test_server_config_tts_engine_zerotts() -> None:
    """Test server config parses ZeroTTS engine and resolves ZeroTTS voices."""
    config = ServerConfig.from_env({"TTS_ENGINE": TtsEngine.ZEROTTS})
    assert config.tts_engine == TtsEngine.ZEROTTS
    assert [voice.id for voice in config.tts_voices] == ["maichi"]
    assert [voice.name for voice in config.tts_voices] == ["Mai Chi"]

    config_custom = ServerConfig.from_env(
        {
            "TTS_ENGINE": TtsEngine.ZEROTTS,
            "TTS_VOICE": "baotrang, giahuy",
        }
    )
    assert config_custom.tts_engine == TtsEngine.ZEROTTS
    assert [voice.id for voice in config_custom.tts_voices] == ["baotrang", "giahuy"]
    assert [voice.name for voice in config_custom.tts_voices] == ["Bảo Trang", "Gia Huy"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_server_config_tts_cache_enabled(value: str, expected: bool) -> None:
    """Test server config parses TTS_CACHE_ENABLED flag."""
    config = ServerConfig.from_env({"TTS_CACHE_ENABLED": value})
    assert config.tts_cache_enabled is expected


def test_server_config_tts_cache_enabled_default() -> None:
    """Test server config enables TTS cache by default."""
    config = ServerConfig.from_env({})
    assert config.tts_cache_enabled is True


def test_server_config_stt_engine_zipformer() -> None:
    """Test server config parses Zipformer STT engine explicitly."""
    config = ServerConfig.from_env({"STT_ENGINE": SttEngine.ZIPFORMER})
    assert config.stt_engine == SttEngine.ZIPFORMER


def test_server_config_stt_engine_gipformer() -> None:
    """Test server config parses Gipformer STT engine explicitly."""
    config = ServerConfig.from_env({"STT_ENGINE": SttEngine.GIPFORMER})
    assert config.stt_engine == SttEngine.GIPFORMER


def test_server_config_zerotts_model_default() -> None:
    """Test server config sets zerotts_model to 'auto' by default."""
    config = ServerConfig.from_env({})
    assert config.zerotts_model == "auto"


def test_server_config_zerotts_model_explicit() -> None:
    """Test server config parses explicit ZEROTTS_MODEL filename."""
    config = ServerConfig.from_env({ZeroTtsEnv.MODEL: ZeroTtsFile.Q4_HYBRID_GGUF_MODEL})
    assert config.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL


def test_server_config_zerotts_model_auto_preserved() -> None:
    """Test server config preserves auto and default selectors without early resolution."""
    config_auto = ServerConfig.from_env({ZeroTtsEnv.MODEL: "auto"})
    assert config_auto.zerotts_model == "auto"

    config_default = ServerConfig.from_env({ZeroTtsEnv.MODEL: "default"})
    assert config_default.zerotts_model == "auto"


def test_server_config_zerotts_model_shorthands() -> None:
    """Test server config normalizes shorthand names for hybrid, q4, and q8 models."""
    config_hybrid = ServerConfig.from_env({ZeroTtsEnv.MODEL: "q4_hybrid"})
    assert config_hybrid.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL

    config_q4 = ServerConfig.from_env({ZeroTtsEnv.MODEL: "q4"})
    assert config_q4.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL

    config_q8 = ServerConfig.from_env({ZeroTtsEnv.MODEL: "q8"})
    assert config_q8.zerotts_model == ZeroTtsFile.Q8_0_GGUF_MODEL


def test_server_config_zerotts_model_invalid() -> None:
    """Test server config rejects unsupported ZeroTTS model names."""
    with pytest.raises(ValueError, match=r"ZEROTTS_MODEL is invalid: 'missing\.gguf'"):
        ServerConfig.from_env({ZeroTtsEnv.MODEL: "missing.gguf"})


def test_server_config_zerotts_model_absolute_path() -> None:
    """Test server config accepts valid absolute path for custom model."""
    config = ServerConfig.from_env({ZeroTtsEnv.MODEL: "/models/custom.gguf"})
    assert config.zerotts_model == "/models/custom.gguf"


def test_server_config_zerotts_max_codebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test server config parses and bounds ZEROTTS_MAX_CODEBOOKS."""
    # Standard desktop/server architecture defaults to 16
    monkeypatch.setattr("wyoming_vietnamese.config.is_zerotts_edge_variant", lambda: False)
    config_default = ServerConfig.from_env({})
    assert config_default.zerotts_max_codebooks == ZeroTtsCodebook.DEFAULT

    # Edge CPU architecture defaults to 12
    monkeypatch.setattr("wyoming_vietnamese.config.is_zerotts_edge_variant", lambda: True)
    config_edge = ServerConfig.from_env({})
    assert config_edge.zerotts_max_codebooks == ZeroTtsCodebook.MEDIUM

    # Explicit override takes precedence
    config_custom = ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "10"})
    assert config_custom.zerotts_max_codebooks == 10

    # Whitespace-only string uses default
    config_ws = ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "   "})
    assert config_ws.zerotts_max_codebooks == ZeroTtsCodebook.MEDIUM

    monkeypatch.setattr("wyoming_vietnamese.config.is_zerotts_edge_variant", lambda: False)
    config_ws_default = ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "   "})
    assert config_ws_default.zerotts_max_codebooks == ZeroTtsCodebook.DEFAULT

    # Positive signed integer is accepted
    config_signed_pos = ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "+8"})
    assert config_signed_pos.zerotts_max_codebooks == 8

    # Non-numeric string is rejected
    with pytest.raises(ValueError, match="ZEROTTS_MAX_CODEBOOKS must be an integer"):
        ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "abc"})

    # Negative signed integer below minimum is rejected
    with pytest.raises(ValueError, match="ZEROTTS_MAX_CODEBOOKS must be at least 4 and at most 16"):
        ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "-5"})

    with pytest.raises(ValueError, match="ZEROTTS_MAX_CODEBOOKS must be at least 4 and at most 16"):
        ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "2"})

    with pytest.raises(ValueError, match="ZEROTTS_MAX_CODEBOOKS must be at least 4 and at most 16"):
        ServerConfig.from_env({ZeroTtsEnv.MAX_CODEBOOKS: "32"})


def test_server_config_zerotts_quality() -> None:
    """Test server config parses ZEROTTS_QUALITY and configures model and codebooks."""
    # Default (unset) resolves to auto
    config_default = ServerConfig.from_env({})
    assert config_default.zerotts_quality == ZeroTtsQuality.AUTO

    # Explicit auto
    config_auto = ServerConfig.from_env({ZeroTtsEnv.QUALITY: "auto"})
    assert config_auto.zerotts_quality == ZeroTtsQuality.AUTO

    # High quality: q8_0 model and 16 codebooks
    config_high = ServerConfig.from_env({ZeroTtsEnv.QUALITY: "high"})
    assert config_high.zerotts_quality == ZeroTtsQuality.HIGH
    assert config_high.zerotts_model == ZeroTtsFile.Q8_0_GGUF_MODEL
    assert config_high.zerotts_max_codebooks == ZeroTtsCodebook.HIGH

    # Medium quality: q4-hybrid model and 12 codebooks
    config_med = ServerConfig.from_env({ZeroTtsEnv.QUALITY: "medium"})
    assert config_med.zerotts_quality == ZeroTtsQuality.MEDIUM
    assert config_med.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert config_med.zerotts_max_codebooks == ZeroTtsCodebook.MEDIUM

    # Low quality: q4-hybrid model and 10 codebooks
    config_low = ServerConfig.from_env({ZeroTtsEnv.QUALITY: "low"})
    assert config_low.zerotts_quality == ZeroTtsQuality.LOW
    assert config_low.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert config_low.zerotts_max_codebooks == ZeroTtsCodebook.LOW

    # Granular overrides take precedence over quality tier
    config_override = ServerConfig.from_env(
        {
            ZeroTtsEnv.QUALITY: "high",
            ZeroTtsEnv.MODEL: "q4_hybrid",
            ZeroTtsEnv.MAX_CODEBOOKS: "8",
        }
    )
    assert config_override.zerotts_quality == ZeroTtsQuality.HIGH
    assert config_override.zerotts_model == ZeroTtsFile.Q4_HYBRID_GGUF_MODEL
    assert config_override.zerotts_max_codebooks == 8

    # Strictly reject non-standard aliases and invalid tiers
    for invalid in ("studio", "med", "balanced", "fast", "edge", "ultra", "invalid"):
        with pytest.raises(ValueError, match="ZEROTTS_QUALITY is invalid"):
            ServerConfig.from_env({ZeroTtsEnv.QUALITY: invalid})
