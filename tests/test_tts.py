"""Text-to-speech handler and initialization tests."""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import threading
import types
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from wyoming.audio import AudioChunk
from wyoming.event import Event
from wyoming.info import Describe
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from tests.helpers import MemoryWriter, make_reader, stream_writer, written_events
from wyoming_vietnamese.cache import BoundedLruCache
from wyoming_vietnamese.const import (
    MAX_TTS_SILENCE_MS,
    TTS_CONFIG_FILE,
    TTS_MODEL_FILE,
    TTS_SAMPLE_RATE,
    TTS_SAMPLE_WIDTH,
    TTS_TOKENS_FILE,
    VIETNAMESE_LANGUAGE,
)
from wyoming_vietnamese.tts import (
    MultiVoiceTTSEngine,
    NghiTTSEngine,
    StreamClauseDetector,
    TTSEngine,
    TTSEventHandler,
    _audio_to_pcm_bytes,
    _boundary_kind,
    _jittered_silence_ms,
    _next_pcm_chunk,
    _resolve_nghitts_espeak_data,
    _silence_pcm,
    get_tts_info,
    get_tts_sample_rate,
    initialize_tts,
    initialize_tts_voices,
    warm_up_tts,
)
from wyoming_vietnamese.tts_model import DEFAULT_TTS_VOICE, get_voice


class FakeTTS:
    """Provide a test double for FakeTTS."""

    def __init__(
        self,
        chunks: list[np.ndarray] | None = None,
        fail: bool = False,
        sample_rate: int = 48000,
    ) -> None:
        """Initialize the test double."""
        self._preset_voices = {"voice-a": {"description": "Voice A"}}
        self.chunks = chunks if chunks is not None else [np.array([0.0], dtype=np.float32)]
        self.fail = fail
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str | None]] = []
        self.options: list[dict[str, object]] = []
        self.closed = False

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        **kwargs: object,
    ) -> Iterator[np.ndarray]:
        """Build test support for infer stream."""
        self.calls.append((text, voice))
        self.options.append(kwargs)
        if self.fail:
            raise RuntimeError("inference failed")
        yield from self.chunks

    def close(self) -> None:
        """Build test support for close."""
        self.closed = True


class FakeGeneratedAudio:
    """Provide the native generated-audio shape used by the NghiTTS adapter."""

    def __init__(self, samples: np.ndarray) -> None:
        """Store generated samples and their fixed test sample rate."""
        self.samples = samples
        self.sample_rate = TTS_SAMPLE_RATE


class FakeSherpaTTS:
    """Provide a callback-capable Sherpa OfflineTts test double."""

    def __init__(
        self,
        *,
        sample_rate: int = TTS_SAMPLE_RATE,
        num_speakers: int = 1,
    ) -> None:
        """Initialize native metadata and callback tracking."""
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.calls: list[tuple[str, int, float]] = []
        self.callback_results: list[int] = []

    def generate(
        self,
        text: str,
        sid: int = 0,
        speed: float = 1.0,
        callback: Callable[[np.ndarray, float], int] | None = None,
    ) -> FakeGeneratedAudio:
        """Return complete audio and emit two native sentence chunks when requested."""
        self.calls.append((text, sid, speed))
        chunks = (
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        )
        if callback is not None:
            self.callback_results.extend([callback(chunks[0], 0.5), callback(chunks[1], 1.0)])
        return FakeGeneratedAudio(np.concatenate(chunks))


def make_handler(
    engine: object | None = None,
    *,
    max_text_chars: int = 100,
    queue_timeout: float = 0.05,
    lock: asyncio.Lock | None = None,
    cache: BoundedLruCache[bytes, bytes] | None = None,
    writer: MemoryWriter | None = None,
    write_timeout: float = 5,
    sentence_silence_ms: int = 0,
    clause_silence_ms: int = 0,
    silence_jitter_percent: int = 0,
    rng: random.Random | None = None,
) -> tuple[TTSEventHandler, asyncio.StreamWriter]:
    """Build test support for make handler."""
    tts = cast(TTSEngine, engine or FakeTTS())
    stream = stream_writer(writer)
    handler = TTSEventHandler(
        tts,
        get_tts_info(DEFAULT_TTS_VOICE).event(),
        lock or asyncio.Lock(),
        cache
        if cache is not None
        else BoundedLruCache(
            max_entries=10,
            max_bytes=1024 * 1024,
            max_item_bytes=1024 * 1024,
            max_idle_seconds=60,
        ),
        max_text_chars,
        queue_timeout,
        make_reader(),
        stream,
        write_timeout=write_timeout,
        sentence_silence_ms=sentence_silence_ms,
        clause_silence_ms=clause_silence_ms,
        silence_jitter_percent=silence_jitter_percent,
        rng=rng,
    )
    return handler, stream


class GatedWriter(MemoryWriter):
    """Provide a writer whose drains wait for an explicit release."""

    def __init__(self) -> None:
        """Initialize a closed gate for transport backpressure tests."""
        super().__init__()
        self.gate = asyncio.Event()

    async def drain(self) -> None:
        """Wait until the test releases the simulated network."""
        await self.gate.wait()


def test_audio_to_pcm_clips_and_sanitizes() -> None:
    """Test audio to pcm clips and sanitizes."""
    pcm = np.frombuffer(
        _audio_to_pcm_bytes(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan, np.inf])),
        dtype="<i2",
    )
    assert pcm.tolist() == [-32767, -32767, 0, 32767, 32767, 0, 32767]
    assert _audio_to_pcm_bytes(np.array([], dtype=np.float32)) == b""


def test_next_pcm_chunk_converts_audio_and_uses_completion_marker() -> None:
    """Test next PCM chunk converts audio and marks iterator completion."""
    iterator = iter([np.array([1], dtype=np.float32)])
    has_chunk, pcm, engine_seconds, pcm_seconds = _next_pcm_chunk(iterator)
    assert has_chunk is True
    assert pcm == np.array([32767], dtype="<i2").tobytes()
    assert engine_seconds >= 0
    assert pcm_seconds >= 0
    assert _next_pcm_chunk(iterator)[:2] == (False, None)


async def test_tts_describe_and_unknown_event() -> None:
    """Test tts describe and unknown event."""
    handler, writer = make_handler()
    assert await handler.handle_event(Describe().event()) is True
    assert await handler.handle_event(Event(type="select-program")) is True
    assert written_events(writer)[0].type == "info"


@pytest.mark.parametrize(
    "event",
    [
        Event(type="synthesize", data={}),
        Event(type="synthesize", data={"text": 1}),
        Synthesize(text=" ").event(),
        Synthesize(text="plain", text_format=SynthesizeTextFormat.SSML).event(),
        Event(type="synthesize", data={"text": "plain", "voice": {"name": 1}}),
    ],
)
async def test_tts_rejects_invalid_requests(event: Event) -> None:
    """Test tts rejects invalid requests."""
    handler, writer = make_handler()
    assert await handler.handle_event(event) is False
    assert [item.type for item in written_events(writer)] == ["error", "audio-start", "audio-stop"]


async def test_tts_enforces_text_limit() -> None:
    """Test tts enforces text limit."""
    handler, writer = make_handler(max_text_chars=3)
    assert await handler.handle_event(Synthesize(text="long").event()) is False
    assert written_events(writer)[0].data["code"] == "text-too-long"


async def test_tts_passes_wyoming_voice_name_to_engine() -> None:
    """Test tts passes wyoming voice name to engine."""
    engine = FakeTTS()
    handler, writer = make_handler(engine)
    request = Synthesize(text="Xin chào", voice=SynthesizeVoice(name="voice-a")).event()
    assert await handler.handle_event(request) is False
    assert engine.calls == [("Xin chào", "voice-a")]
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]


async def test_tts_resolves_voice_by_catalog_id() -> None:
    """Test a catalog voice ID resolves to its display name before dispatch."""
    engine = FakeTTS()
    engine._preset_voices = {DEFAULT_TTS_VOICE.name: {"description": "NghiTTS voice"}}
    handler, writer = make_handler(engine)
    request = Synthesize(text="Xin chào", voice=SynthesizeVoice(name=DEFAULT_TTS_VOICE.id)).event()
    assert await handler.handle_event(request) is False
    assert engine.calls == [("Xin chào", DEFAULT_TTS_VOICE.name)]
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]


async def test_tts_rejects_a_catalog_id_not_configured_on_this_engine() -> None:
    """Test a real catalog voice ID that isn't one of the configured voices still fails."""
    unconfigured_voice = get_voice("chieu-thanh")
    engine = FakeTTS()
    handler, writer = make_handler(engine)
    request = Synthesize(text="Xin chào", voice=SynthesizeVoice(name=unconfigured_voice.id)).event()
    assert await handler.handle_event(request) is False
    assert engine.calls == []
    assert written_events(writer)[0].data["code"] == "invalid-request"


async def test_tts_logs_real_time_factor_for_one_shot_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test TTS logs the real-time factor for a one-shot request."""
    handler, _ = make_handler(FakeTTS([np.zeros(48000, dtype=np.float32)]))

    with caplog.at_level(logging.INFO, logger="wyoming_vietnamese.tts"):
        await handler.handle_event(Synthesize(text="Xin chào").event())

    assert any(
        "Completed TTS request:" in message
        and "audio_seconds=1.000" in message
        and "real_time_factor=" in message
        for message in caplog.messages
    )


async def test_tts_rejects_unknown_voice() -> None:
    """Test unavailable voices fail instead of silently changing the speaker."""
    engine = FakeTTS()
    handler, writer = make_handler(engine)
    request = Synthesize(text="Xin chào", voice=SynthesizeVoice(name="missing")).event()
    assert await handler.handle_event(request) is False
    assert engine.calls == []
    assert written_events(writer)[0].data["code"] == "invalid-request"


async def test_tts_validates_language_and_speaker_selectors() -> None:
    """Test advertised languages work and unsupported selectors never silently fall back."""
    engine = FakeTTS()
    handler, _ = make_handler(engine)
    vietnamese = Synthesize(
        text="Xin chào", voice=SynthesizeVoice(language=VIETNAMESE_LANGUAGE)
    ).event()
    assert await handler.handle_event(vietnamese) is False
    assert engine.calls == [("Xin chào", None)]

    engine = FakeTTS()
    handler, writer = make_handler(engine)
    regional = Synthesize(text="Xin chào", voice=SynthesizeVoice(language="vi-VN")).event()
    assert await handler.handle_event(regional) is False
    assert engine.calls == []
    assert written_events(writer)[0].data["code"] == "invalid-request"

    engine = FakeTTS()
    handler, writer = make_handler(engine)
    english = Synthesize(text="Hello", voice=SynthesizeVoice(language="en")).event()
    assert await handler.handle_event(english) is False
    assert engine.calls == []
    assert written_events(writer)[0].data["code"] == "invalid-request"

    engine = FakeTTS()
    handler, writer = make_handler(engine)
    speaker = Synthesize(
        text="Xin chào",
        voice=SynthesizeVoice(name="voice-a", speaker="speaker-a"),
    ).event()
    assert await handler.handle_event(speaker) is False
    assert engine.calls == []
    assert written_events(writer)[0].data["code"] == "invalid-request"


async def test_tts_streams_text_by_sentence_and_ignores_compatibility_request() -> None:
    """Test tts streams text by sentence and ignores compatibility request."""
    engine = FakeTTS()
    handler, writer = make_handler(engine)
    start = SynthesizeStart(voice=SynthesizeVoice(name="voice-a")).event()
    assert await handler.handle_event(start) is True

    assert await handler.handle_event(SynthesizeChunk(text="Xin ch").event()) is True
    assert engine.calls == []
    assert await handler.handle_event(SynthesizeChunk(text="ào. Tiếp").event()) is True
    assert engine.calls == [("Xin chào.", "voice-a")]

    compatibility = Synthesize(text="Xin chào. Tiếp").event()
    assert await handler.handle_event(compatibility) is True
    assert engine.calls == [("Xin chào.", "voice-a")]

    assert await handler.handle_event(SynthesizeStop().event()) is True
    assert engine.calls == [
        ("Xin chào.", "voice-a"),
        ("Tiếp", "voice-a"),
    ]
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]


def written_audio(writer: asyncio.StreamWriter) -> bytes:
    """Concatenate the PCM payload of every written audio chunk."""
    return b"".join(
        AudioChunk.from_event(event).audio
        for event in written_events(writer)
        if AudioChunk.is_type(event.type)
    )


def test_boundary_kind_classifies_trailing_punctuation() -> None:
    """Test segment boundaries are classified by their final punctuation."""
    assert _boundary_kind("Xin chào.") == "sentence"
    assert _boundary_kind("Thật sao?!  ") == "sentence"
    assert _boundary_kind('Anh ấy nói "vâng."') == "sentence"
    assert _boundary_kind("Ngày mai,") == "clause"
    assert _boundary_kind("một; hai") == "none"
    assert _boundary_kind("   ") == "none"


def test_silence_pcm_matches_requested_duration() -> None:
    """Test silence padding is sized from the engine sample rate."""
    assert _silence_pcm(0, 22050) == b""
    assert _silence_pcm(100, 22050) == bytes(2205 * TTS_SAMPLE_WIDTH)
    with pytest.raises(ValueError, match="must not be negative"):
        _silence_pcm(-1, 22050)


def test_jittered_silence_stays_inside_the_configured_spread() -> None:
    """Test randomized pauses vary around the base duration without leaving its bounds."""
    rng = random.Random(1234)
    draws = {_jittered_silence_ms(200, 25, rng) for _ in range(200)}
    assert len(draws) > 1
    assert min(draws) >= 150
    assert max(draws) <= 250

    assert _jittered_silence_ms(200, 0, rng) == 200
    assert _jittered_silence_ms(0, 50, rng) == 0
    assert _jittered_silence_ms(MAX_TTS_SILENCE_MS, 100, random.Random(0)) <= MAX_TTS_SILENCE_MS
    with pytest.raises(ValueError, match="Silence duration must not be negative"):
        _jittered_silence_ms(-1, 25, rng)
    with pytest.raises(ValueError, match="Silence jitter must not be negative"):
        _jittered_silence_ms(200, -1, rng)


async def test_tts_randomizes_each_boundary_pause() -> None:
    """Test consecutive sentence boundaries receive independently drawn pauses."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(
        engine,
        sentence_silence_ms=200,
        silence_jitter_percent=25,
        rng=random.Random(7),
    )
    assert await handler.handle_event(SynthesizeStart().event()) is True
    text = "Một hai ba. Bốn năm sáu. Bảy tám chín. Mười mười một"
    assert await handler.handle_event(SynthesizeChunk(text=text).event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True

    speech = _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
    pause_lengths = [len(part) for part in written_audio(writer).split(speech) if part]
    assert len(pause_lengths) == 3
    assert len(set(pause_lengths)) > 1
    minimum = round(engine.sample_rate * 0.150) * TTS_SAMPLE_WIDTH
    maximum = round(engine.sample_rate * 0.250) * TTS_SAMPLE_WIDTH
    assert all(minimum <= length <= maximum for length in pause_lengths)


async def test_tts_pads_line_breaks_like_sentence_boundaries() -> None:
    """Test unpunctuated lines are separated instead of being concatenated bare."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(engine, sentence_silence_ms=100, clause_silence_ms=20)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="Một\nHai\nBa").event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True

    speech = _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
    silence = bytes(round(engine.sample_rate * 0.1) * TTS_SAMPLE_WIDTH)
    assert engine.calls == [("Một", None), ("Hai", None), ("Ba", None)]
    assert written_audio(writer) == speech + silence + speech + silence + speech


async def test_tts_pads_sentence_pause_between_streamed_segments() -> None:
    """Test a sentence pause separates streamed segments but never trails the reply."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(engine, sentence_silence_ms=100)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="Xin chào. Tạm biệt").event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True

    speech = _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
    silence = bytes(round(engine.sample_rate * 0.1) * TTS_SAMPLE_WIDTH)
    assert engine.calls == [("Xin chào.", None), ("Tạm biệt", None)]
    assert written_audio(writer) == speech + silence + speech


async def test_tts_pads_shorter_pause_after_clause_punctuation() -> None:
    """Test clause boundaries use their own shorter pause."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(engine, sentence_silence_ms=100, clause_silence_ms=20)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    chunk = SynthesizeChunk(text="Ngày mai trời đẹp, chúng ta đi").event()
    assert await handler.handle_event(chunk) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True

    speech = _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
    silence = bytes(round(engine.sample_rate * 0.02) * TTS_SAMPLE_WIDTH)
    assert engine.calls == [("Ngày mai trời đẹp,", None), ("chúng ta đi", None)]
    assert written_audio(writer) == speech + silence + speech


async def test_tts_pads_between_engine_sentence_chunks() -> None:
    """Test one-shot synthesis separates the sentences the engine returns."""
    engine = FakeTTS(
        [
            np.array([0.5], dtype=np.float32),
            np.array([0.25], dtype=np.float32),
        ]
    )
    handler, writer = make_handler(engine, sentence_silence_ms=100)
    assert await handler.handle_event(Synthesize(text="Xin chào. Tạm biệt.").event()) is False

    silence = bytes(round(engine.sample_rate * 0.1) * TTS_SAMPLE_WIDTH)
    assert written_audio(writer) == (
        _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
        + silence
        + _audio_to_pcm_bytes(np.array([0.25], dtype=np.float32))
    )


async def test_tts_omits_padding_when_silence_is_disabled() -> None:
    """Test zero-length pauses keep the audio stream free of padding chunks."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(engine, sentence_silence_ms=0)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="Xin chào. Tạm biệt").event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True

    speech = _audio_to_pcm_bytes(np.array([0.5], dtype=np.float32))
    assert written_audio(writer) == speech + speech


async def test_tts_logs_aggregate_timings_without_chunk_noise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test tts logs aggregate timings without chunk noise."""
    handler, _ = make_handler()
    with caplog.at_level(logging.DEBUG, logger="wyoming_vietnamese.tts"):
        await handler.handle_event(SynthesizeStart().event())
        await handler.handle_event(SynthesizeChunk(text="Xin chào.").event())
        await handler.handle_event(SynthesizeStop().event())

    messages = caplog.messages
    assert any("TTS step=text-stream-start" in message for message in messages)
    assert all("TTS step=text-chunk" not in message for message in messages)
    assert any("TTS step=sentence-ready" in message for message in messages)
    assert any(
        "TTS step=queue-acquired" in message and "wait_ms=" in message for message in messages
    )
    assert all("TTS step=engine-chunk" not in message for message in messages)
    assert any(
        "TTS step=segment-complete" in message and "first_audio_ms=" in message
        for message in messages
    )
    assert any(
        "Completed streaming TTS request:" in message
        and "receive_ms=" in message
        and "handle_ms=" in message
        and "real_time_factor=" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "event",
    [
        SynthesizeChunk(text="orphan").event(),
        SynthesizeStop().event(),
        Event(type="synthesize-start", data={"voice": {"name": 1}}),
        SynthesizeStart(voice=SynthesizeVoice(language="en")).event(),
        SynthesizeStart(voice=SynthesizeVoice(name="voice-a", speaker="speaker-a")).event(),
    ],
)
async def test_tts_rejects_invalid_stream_order_or_start(event: Event) -> None:
    """Test tts rejects invalid stream order or start."""
    handler, writer = make_handler()
    assert await handler.handle_event(event) is False
    assert [item.type for item in written_events(writer)] == [
        "error",
        "audio-start",
        "audio-stop",
        "synthesize-stopped",
    ]


async def test_tts_rejects_invalid_stream_format_and_chunk() -> None:
    """Test tts rejects invalid stream format and chunk."""
    handler, writer = make_handler()
    start = SynthesizeStart(text_format=SynthesizeTextFormat.SSML).event()
    assert await handler.handle_event(start) is False
    assert written_events(writer)[0].data["code"] == "unsupported-text-format"

    handler, writer = make_handler()
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(Event(type="synthesize-chunk", data={})) is False
    assert written_events(writer)[0].data["code"] == "invalid-request"


async def test_tts_rejects_duplicate_or_empty_stream() -> None:
    """Test tts rejects duplicate or empty stream."""
    handler, writer = make_handler()
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeStart().event()) is False
    assert written_events(writer)[0].data["code"] == "invalid-event-order"

    handler, writer = make_handler()
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is False
    assert written_events(writer)[0].data["code"] == "empty-text"


async def test_tts_enforces_stream_text_limit() -> None:
    """Test tts enforces stream text limit."""
    handler, writer = make_handler(max_text_chars=5)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="123").event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="456").event()) is False
    assert written_events(writer)[0].data["code"] == "text-too-long"


async def test_tts_contains_streaming_inference_failure() -> None:
    """Test tts contains streaming inference failure."""
    handler, writer = make_handler(FakeTTS(fail=True))
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="Đây là lỗi. Sau").event()) is False
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "error",
        "audio-stop",
        "synthesize-stopped",
    ]


async def test_tts_disconnect_releases_stream_buffer() -> None:
    """Test tts disconnect releases stream buffer."""
    handler, _ = make_handler()
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="unfinished").event()) is True
    await handler.disconnect()
    assert handler.is_streaming is False
    assert handler.stream_text_chars == 0
    assert handler.sentence_boundary.finish() == ""


async def test_tts_streams_and_splits_large_pcm_chunks() -> None:
    """Test tts streams and splits large pcm chunks."""
    engine = FakeTTS(
        [
            np.array([np.nan, -2.0, 2.0], dtype=np.float32),
            np.zeros(3000, dtype=np.float32),
        ]
    )
    handler, writer = make_handler(engine)
    assert await handler.handle_event(Synthesize(text="Audio").event()) is False
    events = written_events(writer)
    chunks = [event for event in events if event.type == "audio-chunk"]
    assert events[0].type == "audio-start"
    assert events[-1].type == "audio-stop"
    assert len(chunks) == 3
    assert sum(len(event.payload or b"") for event in chunks) == 6006


async def test_tts_pcm_conversion_runs_outside_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test TTS PCM conversion does not stall transport event processing."""
    conversion_threads: list[int] = []

    def record_conversion(audio: np.ndarray) -> bytes:
        """Record the conversion worker before delegating to the real converter."""
        conversion_threads.append(threading.get_ident())
        return _audio_to_pcm_bytes(audio)

    monkeypatch.setattr("wyoming_vietnamese.tts._audio_to_pcm_bytes", record_conversion)
    event_loop_thread = threading.get_ident()
    handler, _ = make_handler(FakeTTS([np.zeros(3000, dtype=np.float32)]))
    assert await handler.handle_event(Synthesize(text="Audio").event()) is False
    assert conversion_threads
    assert all(thread_id != event_loop_thread for thread_id in conversion_threads)


@pytest.mark.parametrize("engine", [FakeTTS(chunks=[]), FakeTTS(fail=True)])
async def test_tts_contains_empty_or_failed_inference(engine: FakeTTS) -> None:
    """Test tts contains empty or failed inference."""
    handler, writer = make_handler(engine)
    assert await handler.handle_event(Synthesize(text="Failure").event()) is False
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "error",
        "audio-stop",
    ]


async def test_tts_reports_busy_queue() -> None:
    """Test tts reports busy queue."""
    lock = asyncio.Lock()
    await lock.acquire()
    handler, writer = make_handler(lock=lock, queue_timeout=0.001)
    assert await handler.handle_event(Synthesize(text="Busy").event()) is False
    lock.release()
    assert written_events(writer)[0].data["code"] == "server-busy"


async def test_tts_slow_client_does_not_retain_inference_lock() -> None:
    """Test tts slow client does not retain inference lock."""
    lock = asyncio.Lock()
    engine = FakeTTS([np.zeros(512, dtype=np.float32)])
    gated_writer = GatedWriter()
    first, _ = make_handler(
        engine,
        lock=lock,
        writer=gated_writer,
        write_timeout=1,
    )
    first_request = asyncio.create_task(first.handle_event(Synthesize(text="First").event()))

    for _ in range(100):
        if engine.calls and not lock.locked():
            break
        await asyncio.sleep(0.001)
    assert engine.calls == [("First", None)]
    assert lock.locked() is False

    second, second_writer = make_handler(engine, lock=lock)
    await asyncio.wait_for(
        second.handle_event(Synthesize(text="Second").event()),
        timeout=0.2,
    )
    assert engine.calls == [("First", None), ("Second", None)]
    assert [event.type for event in written_events(second_writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]

    gated_writer.gate.set()
    await first_request


async def test_tts_ignores_a_compatibility_request_after_the_stream_ended() -> None:
    """Test a trailing compatibility synthesize does not replay the whole reply."""
    engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    handler, writer = make_handler(engine)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert await handler.handle_event(SynthesizeChunk(text="Xin chào. Tạm biệt.").event()) is True
    assert await handler.handle_event(SynthesizeStop().event()) is True
    served = list(engine.calls)

    trailing = Synthesize(text="Xin chào. Tạm biệt.").event()
    assert await handler.handle_event(trailing) is True
    assert engine.calls == served
    assert [event.type for event in written_events(writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]

    await handler.disconnect()
    assert await handler.handle_event(trailing) is False
    assert len(engine.calls) == len(served) + 1


async def test_tts_stalled_client_releases_the_lock_for_a_long_segment() -> None:
    """Test inference stops waiting on a slow client once a full utterance is queued."""
    lock = asyncio.Lock()
    engine = FakeTTS([np.zeros(60000, dtype=np.float32)])
    gated_writer = GatedWriter()
    stalled, _ = make_handler(engine, lock=lock, writer=gated_writer, write_timeout=5)
    stalled_request = asyncio.create_task(stalled.handle_event(Synthesize(text="Stalled").event()))

    for _ in range(500):
        if engine.calls and not lock.locked():
            break
        await asyncio.sleep(0.001)
    assert engine.calls == [("Stalled", None)]
    assert lock.locked() is False

    other_engine = FakeTTS([np.array([0.5], dtype=np.float32)])
    other, other_writer = make_handler(other_engine, lock=lock)
    await asyncio.wait_for(other.handle_event(Synthesize(text="Other").event()), timeout=1)
    assert [event.type for event in written_events(other_writer)] == [
        "audio-start",
        "audio-chunk",
        "audio-stop",
    ]

    gated_writer.gate.set()
    await stalled_request


async def test_tts_producer_cancellation_frees_the_lock_and_the_consumer() -> None:
    """Test cancelling inference never strands the shared lock or its waiting consumer."""
    lock = asyncio.Lock()
    engine = FakeTTS([np.zeros(2048, dtype=np.float32) for _ in range(6)])
    handler, _ = make_handler(engine, lock=lock)
    request = asyncio.create_task(handler.handle_event(Synthesize(text="Cancel").event()))

    producer: asyncio.Task[None] | None = None
    for _ in range(500):
        await asyncio.sleep(0)
        producer = next(
            (task for task in asyncio.all_tasks() if task.get_name() == "TTS inference producer"),
            None,
        )
        if producer is not None:
            break
    assert producer is not None

    # Cancel repeatedly: a second cancellation lands while cleanup is already running.
    producer.cancel()
    producer.cancel()

    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=5)
    assert lock.locked() is False


async def test_tts_skips_inference_for_closed_client() -> None:
    """Test tts skips inference for closed client."""
    engine = FakeTTS()
    handler, writer = make_handler(engine)
    writer.close()
    assert await handler.handle_event(Synthesize(text="Gone").event()) is False
    assert engine.calls == []


async def test_tts_replays_cached_pcm_across_clients() -> None:
    """Test tts replays cached pcm across clients."""
    cache = BoundedLruCache[bytes, bytes](
        max_entries=10,
        max_bytes=1024,
        max_item_bytes=1024,
        max_idle_seconds=60,
    )
    first_engine = FakeTTS([np.array([-0.5, 0.5], dtype=np.float32)])
    first_handler, first_writer = make_handler(first_engine, cache=cache)
    await first_handler.handle_event(Synthesize(text="Xin chào").event())
    first_events = written_events(first_writer)
    assert first_engine.calls == [("Xin chào", None)]

    second_engine = FakeTTS([np.array([0.0], dtype=np.float32)])
    second_handler, second_writer = make_handler(second_engine, cache=cache)
    await second_handler.handle_event(Synthesize(text="Xin chào").event())
    second_events = written_events(second_writer)
    assert second_engine.calls == []
    assert [event.type for event in second_events] == [event.type for event in first_events]
    assert [event.payload for event in second_events] == [event.payload for event in first_events]
    assert len(cache) == 1


async def test_tts_cache_key_includes_voice() -> None:
    """Test tts cache key includes voice."""
    cache = BoundedLruCache[bytes, bytes](
        max_entries=10,
        max_bytes=1024,
        max_item_bytes=1024,
        max_idle_seconds=60,
    )
    engine = FakeTTS()
    default_handler, _ = make_handler(engine, cache=cache)
    await default_handler.handle_event(Synthesize(text="Voice test").event())

    voice_handler, _ = make_handler(engine, cache=cache)
    request = Synthesize(text="Voice test", voice=SynthesizeVoice(name="voice-a")).event()
    await voice_handler.handle_event(request)
    assert engine.calls == [("Voice test", None), ("Voice test", "voice-a")]
    assert len(cache) == 2


async def test_tts_does_not_cache_oversized_pcm() -> None:
    """Test tts does not cache oversized pcm."""
    cache = BoundedLruCache[bytes, bytes](
        max_entries=10,
        max_bytes=100,
        max_item_bytes=33,
        max_idle_seconds=60,
    )
    engine = FakeTTS()
    first_handler, _ = make_handler(engine, cache=cache)
    await first_handler.handle_event(Synthesize(text="Too large").event())
    second_handler, _ = make_handler(engine, cache=cache)
    await second_handler.handle_event(Synthesize(text="Too large").event())
    assert engine.calls == [("Too large", None), ("Too large", None)]
    assert len(cache) == 0


def test_tts_info_lists_configured_voices() -> None:
    """Test Wyoming discovery advertises all configured models."""
    second_voice = get_voice("chieu-thanh")
    info = get_tts_info((DEFAULT_TTS_VOICE, second_voice))
    program = info.tts[0]
    assert program.name == "wyoming_vietnamese"
    assert program.description == "NghiTTS Vietnamese (Sherpa-ONNX)"
    assert program.supports_synthesize_streaming is True
    assert [voice.name for voice in program.voices] == [DEFAULT_TTS_VOICE.name, second_voice.name]
    assert program.voices[0].languages == [VIETNAMESE_LANGUAGE]


def test_get_tts_sample_rate_validates_engine_metadata() -> None:
    """Test sample-rate validation catches invalid engine contracts."""
    assert get_tts_sample_rate(FakeTTS(sample_rate=TTS_SAMPLE_RATE)) == TTS_SAMPLE_RATE
    with pytest.raises(ValueError, match="sample rate"):
        get_tts_sample_rate(FakeTTS(sample_rate=0))


def test_multi_voice_engine_routes_by_name_and_defaults_to_first() -> None:
    """Test the multi-voice adapter routes names and preserves list order as default."""
    second_voice = get_voice("chieu-thanh")
    default_native = FakeSherpaTTS()
    second_native = FakeSherpaTTS()
    engine = MultiVoiceTTSEngine(
        (
            NghiTTSEngine(default_native, DEFAULT_TTS_VOICE),
            NghiTTSEngine(second_native, second_voice),
        )
    )

    list(engine.infer_stream("Mặc định"))
    list(engine.infer_stream("Giọng hai", voice=second_voice.name))

    assert default_native.calls == [("Mặc định", 0, 1.0)]
    assert second_native.calls == [("Giọng hai", 0, 1.0)]
    assert list(engine._preset_voices) == [DEFAULT_TTS_VOICE.name, second_voice.name]
    with pytest.raises(ValueError, match="Unsupported"):
        list(engine.infer_stream("Lỗi", voice="missing"))
    engine.close()


def test_warm_up_tts_uses_streaming_default_voice() -> None:
    """Test warm-up exhausts the incremental API using the active default."""
    engine = FakeTTS([np.array([0.1], dtype=np.float32)])
    warm_up_tts(engine)
    assert engine.calls == [("Xin chào.", None)]
    assert engine.options == [{}]


def test_warm_up_tts_rejects_empty_audio() -> None:
    """Test warm-up rejects an engine that emits no samples."""
    with pytest.raises(RuntimeError, match="no audio"):
        warm_up_tts(FakeTTS(chunks=[]))


def _make_nghitts_model(directory: Path) -> Path:
    """Build minimal NghiTTS assets plus complete local eSpeak data markers."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TTS_MODEL_FILE).write_bytes(b"onnx")
    (directory / TTS_CONFIG_FILE).write_text("{}", encoding="utf-8")
    (directory / TTS_TOKENS_FILE).write_text("_ 0\n", encoding="utf-8")
    espeak_data = directory / "espeak-ng-data"
    for name in ("phondata", "phontab", "vi_dict", "lang/aav/vi"):
        path = espeak_data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    return espeak_data


def test_nghitts_adapter_streams_callbacks_and_validates_voice() -> None:
    """Test native sentence callbacks are exposed incrementally and bounded."""
    native = FakeSherpaTTS()
    engine = NghiTTSEngine(native, DEFAULT_TTS_VOICE)

    chunks = list(engine.infer_stream("Xin chào. Tiếp theo.", voice=DEFAULT_TTS_VOICE.name))
    assert [chunk.tolist() for chunk in chunks] == [
        pytest.approx([0.1, 0.2]),
        pytest.approx([0.3]),
    ]
    assert native.callback_results == [1, 1]

    with pytest.raises(ValueError, match="Unsupported NghiTTS voice"):
        list(engine.infer_stream("Xin chào.", voice="missing"))
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.infer_stream("Xin chào.")


@pytest.mark.parametrize(
    ("sample_rate", "speakers", "message"),
    [
        (16000, 1, "sample rate"),
        (TTS_SAMPLE_RATE, 2, "speaker count"),
    ],
)
def test_nghitts_adapter_rejects_unexpected_native_metadata(
    sample_rate: int,
    speakers: int,
    message: str,
) -> None:
    """Test model substitutions cannot silently change the audio contract."""
    with pytest.raises(RuntimeError, match=message):
        NghiTTSEngine(
            FakeSherpaTTS(sample_rate=sample_rate, num_speakers=speakers),
            DEFAULT_TTS_VOICE,
        )


def test_resolve_nghitts_espeak_data_prefers_model_and_validates_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test portable model-local eSpeak data and strict explicit overrides."""
    model_dir = tmp_path / "tts"
    espeak_data = _make_nghitts_model(model_dir)
    monkeypatch.delenv("NGHITTS_ESPEAK_DATA_DIR", raising=False)
    assert _resolve_nghitts_espeak_data(model_dir) == espeak_data.resolve()

    monkeypatch.setenv("NGHITTS_ESPEAK_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="NGHITTS_ESPEAK_DATA_DIR"):
        _resolve_nghitts_espeak_data(model_dir)


def test_initialize_tts_uses_sherpa_cpu_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test initialization passes verified local paths and effective threads."""
    model_dir = tmp_path / "tts"
    espeak_data = _make_nghitts_model(model_dir)
    native = FakeSherpaTTS()
    vits_config = Mock(return_value="vits")
    model_config = Mock(return_value="model")
    tts_config = Mock(return_value="config")
    offline_tts = Mock(return_value=native)
    fake_sherpa = types.SimpleNamespace(
        OfflineTtsVitsModelConfig=vits_config,
        OfflineTtsModelConfig=model_config,
        OfflineTtsConfig=tts_config,
        OfflineTts=offline_tts,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", cast(Any, fake_sherpa))

    engine = initialize_tts(model_dir, num_threads=4, voice=DEFAULT_TTS_VOICE)

    assert isinstance(engine, NghiTTSEngine)
    vits_config.assert_called_once_with(
        model=str((model_dir / TTS_MODEL_FILE).resolve()),
        tokens=str((model_dir / TTS_TOKENS_FILE).resolve()),
        data_dir=str(espeak_data.resolve()),
    )
    model_config.assert_called_once_with(
        vits="vits",
        num_threads=4,
        debug=False,
        provider="cpu",
    )
    tts_config.assert_called_once_with(model="model", max_num_sentences=1)
    offline_tts.assert_called_once_with("config")


def test_initialize_tts_rejects_missing_assets(tmp_path: Path) -> None:
    """Test initialization reports missing model assets before native import."""
    with pytest.raises(FileNotFoundError, match="ONNX model"):
        initialize_tts(tmp_path)


def test_initialize_tts_voices_uses_voice_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test multi-voice initialization loads each configured voice directory."""
    second_voice = get_voice("chieu-thanh")
    engines = (
        NghiTTSEngine(FakeSherpaTTS(), DEFAULT_TTS_VOICE),
        NghiTTSEngine(FakeSherpaTTS(), second_voice),
    )
    initialize = Mock(side_effect=engines)
    monkeypatch.setattr("wyoming_vietnamese.tts.initialize_tts", initialize)

    combined = initialize_tts_voices(tmp_path, (DEFAULT_TTS_VOICE, second_voice), num_threads=2)

    assert isinstance(combined, MultiVoiceTTSEngine)
    assert [item.args for item in initialize.call_args_list] == [
        (tmp_path / DEFAULT_TTS_VOICE.id, 2, DEFAULT_TTS_VOICE),
        (tmp_path / second_voice.id, 2, second_voice),
    ]


def test_initialize_tts_auto_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test initialization resolves automatic CPU thread count."""
    _make_nghitts_model(tmp_path)
    native = FakeSherpaTTS()
    fake_sherpa = types.SimpleNamespace(
        OfflineTtsVitsModelConfig=Mock(return_value="vits"),
        OfflineTtsModelConfig=Mock(return_value="model"),
        OfflineTtsConfig=Mock(return_value="config"),
        OfflineTts=Mock(return_value=native),
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", cast(Any, fake_sherpa))
    monkeypatch.setattr("wyoming_vietnamese.config.os.process_cpu_count", lambda: 3)
    initialize_tts(tmp_path)
    fake_sherpa.OfflineTtsModelConfig.assert_called_once_with(
        vits="vits",
        num_threads=3,
        debug=False,
        provider="cpu",
    )


def test_stream_clause_detector_empty_chunk_and_finish() -> None:
    """Test an empty chunk yields no clauses and finish returns an empty tail."""
    detector = StreamClauseDetector(min_clause_chars=8)
    assert not list(detector.add_chunk(""))
    assert detector.finish() == ""


def test_stream_clause_detector_streams_sentence_clauses_incrementally() -> None:
    """Test clauses are emitted as soon as enough streamed text forms a sentence."""
    detector = StreamClauseDetector(min_clause_chars=8)
    assert list(detector.add_chunk("Xin chào, ")) == ["Xin chào,"]
    assert not list(detector.add_chunk("tôi là trợ lý ảo "))
    assert list(detector.add_chunk("tiếng Việt của bạn. ")) == [
        "tôi là trợ lý ảo tiếng Việt của bạn."
    ]

    detector.add_chunk("Unfinished phrase")
    assert detector.finish() == "Unfinished phrase"


def test_stream_clause_detector_splits_sentence_within_single_chunk() -> None:
    """Test a sentence boundary inside a single chunk still produces a clause."""
    detector = StreamClauseDetector(min_clause_chars=5)
    assert "Sentence one." in list(detector.add_chunk("Sentence one. Clause two, three."))


def test_stream_clause_detector_defers_clause_without_punctuation() -> None:
    """Test text without sentence punctuation is withheld until finish."""
    detector = StreamClauseDetector(min_clause_chars=5)
    assert not list(detector.add_chunk("This is a long sentence without punctuation"))
    assert detector.finish() == "This is a long sentence without punctuation"


def test_stream_clause_detector_finish_returns_long_unsplit_word() -> None:
    """Test a single long word with no boundary is returned whole by finish."""
    detector = StreamClauseDetector(min_clause_chars=5)
    detector.add_chunk("abcdefghijklmnopqrstuvwxyz")
    assert detector.finish() == "abcdefghijklmnopqrstuvwxyz"


def test_stream_clause_detector_split_clause_text_keeps_short_text_whole() -> None:
    """Test text under the clause-length threshold is not split."""
    assert StreamClauseDetector.split_clause_text(
        "Xin chào tôi là trợ lý ảo tiếng Việt của bạn và hôm nay tôi có thể giúp gì cho bạn."
    ) == ["Xin chào tôi là trợ lý ảo tiếng Việt của bạn và hôm nay tôi có thể giúp gì cho bạn."]


def test_stream_clause_detector_emits_clause_for_long_trailing_sentence() -> None:
    """Test a short clause followed by a much longer sentence still yields output."""
    detector = StreamClauseDetector(min_clause_chars=5)
    assert list(detector.add_chunk("Short clause, and then a much longer sentence after it."))


def test_stream_clause_detector_respects_higher_min_clause_chars() -> None:
    """Test a higher min_clause_chars withholds a long unpunctuated chunk until finish."""
    detector = StreamClauseDetector(min_clause_chars=15)
    assert not list(detector.add_chunk("Hi there thisisalongwordwithoutspace"))
    assert detector.finish() == "Hi there thisisalongwordwithoutspace"


def test_stream_clause_detector_splits_natural_clause_with_tail() -> None:
    """Test a natural clause boundary splits off from its trailing tail."""
    detector = StreamClauseDetector(min_clause_chars=12)
    assert list(detector.add_chunk("Hi, this is a useful natural clause, with a tail")) == [
        "Hi, this is a useful natural clause,"
    ]


def test_stream_clause_detector_splits_default_min_clause_chars() -> None:
    """Test the default min_clause_chars splits a long padded sentence."""
    long_sentence = f"{'word ' * 30}done."
    detector = StreamClauseDetector()
    assert list(detector.add_chunk(long_sentence)) == [long_sentence.strip()]


def test_stream_clause_detector_keeps_numbers_intact() -> None:
    """Test digit separators never split a number into separate segments."""
    measurements = "Nhiệt độ 33,8 độ C và độ ẩm 72,7 phần trăm."
    assert StreamClauseDetector.split_clause_text(measurements) == [measurements]

    assert StreamClauseDetector.split_clause_text("Doanh thu 1.250.000 đồng hôm nay") == [
        "Doanh thu 1.250.000 đồng hôm nay"
    ]
    assert StreamClauseDetector.split_clause_text("Chuyến bay lúc 10:30 sáng mai") == [
        "Chuyến bay lúc 10:30 sáng mai"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Bạn hãy kết nối Wi-Fi ở phòng khách nhé",
        "Hôm nay tôi nhận được e-mail mới từ công ty",
        "Thủ tục check-in đã hoàn tất từ sáng nay",
        "Tôi sống ở TP-HCM đã rất nhiều năm rồi",
    ],
)
def test_stream_clause_detector_keeps_hyphenated_tokens_intact(text: str) -> None:
    """Test a dash joining one token never splits it into separate segments."""
    assert StreamClauseDetector.split_clause_text(text) == [text]

    detector = StreamClauseDetector()
    assert not list(detector.add_chunk(text))
    assert detector.finish() == text


def test_stream_clause_detector_still_splits_a_spaced_dash() -> None:
    """Test a dash used as punctuation remains a clause boundary."""
    detector = StreamClauseDetector()
    assert list(detector.add_chunk("Tuyến đường Sài Gòn - Hà Nội rất dài và đông")) == [
        "Tuyến đường Sài Gòn -"
    ]
    assert detector.finish() == "Hà Nội rất dài và đông"


def test_stream_clause_detector_waits_for_a_token_split_across_chunks() -> None:
    """Test a trailing dash holds until the rest of its token arrives."""
    detector = StreamClauseDetector()
    assert not list(detector.add_chunk("Bạn hãy kết nối Wi-"))
    assert list(detector.add_chunk("Fi trước đã. Rồi tiếp")) == ["Bạn hãy kết nối Wi-Fi trước đã."]
    assert detector.finish() == "Rồi tiếp"


def test_split_clause_text_merges_a_short_tail_without_rewriting_it() -> None:
    """Test merging a short trailing clause preserves the original separator."""
    spaced = "Tuyến đường Sài Gòn - Hà Nội dài"
    assert StreamClauseDetector.split_clause_text(spaced) == [spaced]

    newline = "Một câu đủ dài để tách,\nđuôi"
    assert StreamClauseDetector.split_clause_text(newline) == [newline]


def test_stream_clause_detector_still_splits_punctuation_after_digits() -> None:
    """Test a digit before a delimiter does not disable a real boundary."""
    detector = StreamClauseDetector()
    assert list(detector.add_chunk("Nhiệt độ là 33, độ ẩm là 72. Trời đẹp")) == [
        "Nhiệt độ là 33,",
        "độ ẩm là 72.",
    ]
    assert detector.finish() == "Trời đẹp"


def test_stream_clause_detector_waits_for_a_number_split_across_chunks() -> None:
    """Test a trailing digit separator holds until the rest of the number arrives."""
    detector = StreamClauseDetector()
    assert not list(detector.add_chunk("Nhiệt độ ngoài trời là 33,"))
    assert list(detector.add_chunk("8 độ C. Trời đẹp")) == ["Nhiệt độ ngoài trời là 33,8 độ C."]
    assert detector.finish() == "Trời đẹp"


async def test_tts_disconnect_log_when_stream_started() -> None:
    """Test disconnect resets an active streaming request."""
    handler, _ = make_handler()
    assert await handler.handle_event(SynthesizeStart().event()) is True
    await handler.disconnect()
    assert handler.is_streaming is False


async def test_tts_cache_hit_and_miss_too_large() -> None:
    """Test a result larger than the item limit is not retained."""
    cache = BoundedLruCache[bytes, bytes](
        max_entries=10,
        max_bytes=1024,
        max_item_bytes=10,
        max_idle_seconds=60,
    )
    handler, writer = make_handler(cache=cache)
    assert await handler.handle_event(Synthesize(text="Hello cache").event()) is False
    assert len(written_events(writer)) > 0
    assert len(cache) == 0


async def test_tts_writer_closing_skips_inference() -> None:
    """Test synthesis exits immediately if its writer is closing."""
    handler, stream = make_handler()
    stream.close()
    assert await handler._synthesize("Hello", voice_name=None) is None


async def test_tts_stream_text_too_long() -> None:
    """Test stream chunks fail when their total text exceeds the limit."""
    handler, writer = make_handler(max_text_chars=10)
    assert await handler.handle_event(SynthesizeStart().event()) is True
    assert (
        await handler.handle_event(SynthesizeChunk(text="This text is way too long").event())
        is False
    )
    assert len(written_events(writer)) > 0
