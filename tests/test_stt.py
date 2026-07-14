"""Speech-to-text handler and initialization tests."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe

from tests.helpers import make_reader, stream_writer, written_events
from wyoming_vietnamese.const import VIETNAMESE_LANGUAGE
from wyoming_vietnamese.protocol import ByteBudget
from wyoming_vietnamese.stt import (
    SherpaSTTEventHandler,
    STTRecognizer,
    get_stt_info,
    initialize_stt,
    warm_up_stt,
)


class FakeStream:
    """Provide a test double for FakeStream."""

    def __init__(self, text: str) -> None:
        """Initialize the test double."""
        self.result = SimpleNamespace(text=text)
        self.rate = 0
        self.samples = np.array([], dtype=np.float32)

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Build test support for accept waveform."""
        self.rate = sample_rate
        self.samples = samples.copy()


class FakeRecognizer:
    """Provide a test double for FakeRecognizer."""

    def __init__(self, text: str = "  Xin CHÀO  ", fail: bool = False) -> None:
        """Initialize the test double."""
        self.stream = FakeStream(text)
        self.fail = fail
        self.decoded = False

    def create_stream(self) -> FakeStream:
        """Build test support for create stream."""
        return self.stream

    def decode_stream(self, stream: FakeStream) -> None:
        """Build test support for decode stream."""
        if self.fail:
            raise RuntimeError("decode failed")
        assert stream is self.stream
        self.decoded = True


def make_handler(
    recognizer: FakeRecognizer | None = None,
    *,
    max_audio_seconds: float = 10,
    queue_timeout: float = 0.05,
    lock: asyncio.Lock | None = None,
    buffer_budget: ByteBudget | None = None,
) -> tuple[SherpaSTTEventHandler, asyncio.StreamWriter]:
    """Build test support for make handler."""
    writer = stream_writer()
    handler = SherpaSTTEventHandler(
        cast(STTRecognizer, recognizer or FakeRecognizer()),
        get_stt_info("owner/model").event(),
        lock or asyncio.Lock(),
        "owner/model",
        max_audio_seconds,
        queue_timeout,
        make_reader(),
        writer,
        buffer_budget=buffer_budget,
    )
    return handler, writer


async def test_stt_describe_and_unknown_event() -> None:
    """Test stt describe and unknown event."""
    handler, writer = make_handler()
    assert await handler.handle_event(Describe().event()) is True
    assert await handler.handle_event(Event(type="select-program")) is True
    assert written_events(writer)[0].type == "info"


async def test_stt_accepts_advertised_transcribe_language() -> None:
    """Test stt accepts its exact advertised language selector."""
    handler, _ = make_handler()
    request = Transcribe(name="owner/model", language=VIETNAMESE_LANGUAGE).event()
    assert await handler.handle_event(request) is True


@pytest.mark.parametrize(
    "event",
    [
        Event(type="transcribe", data={"name": 1}),
        Event(type="transcribe", data={"language": 1}),
        Transcribe(name="other/model").event(),
        Transcribe(language="en").event(),
        Transcribe(language="viking").event(),
        Transcribe(language="VI").event(),
        Transcribe(language=" vi ").event(),
        Transcribe(language="VI_vn").event(),
        Transcribe(language="vi-VN").event(),
        Transcribe(language="vi-Latn-VN").event(),
        Transcribe(language="vi-").event(),
        Transcribe(language="").event(),
    ],
)
async def test_stt_rejects_invalid_transcribe_request(event: Event) -> None:
    """Test stt rejects invalid transcribe request."""
    handler, writer = make_handler()
    assert await handler.handle_event(event) is False
    assert [item.type for item in written_events(writer)] == ["error", "transcript"]


async def test_stt_rejects_transcribe_during_audio() -> None:
    """Test stt rejects transcribe during audio."""
    handler, writer = make_handler()
    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(Transcribe().event()) is False
    assert written_events(writer)[0].data["code"] == "invalid-event-order"


@pytest.mark.parametrize(
    "event",
    [
        Event(type="audio-start", data={}),
        AudioStart(rate=7999, width=2, channels=1).event(),
        AudioStart(rate=16000, width=5, channels=1).event(),
        AudioStart(rate=16000, width=2, channels=3).event(),
        Event(type="audio-start", data={"rate": True, "width": 2, "channels": 1}),
    ],
)
async def test_stt_rejects_bad_audio_start(event: Event) -> None:
    """Test stt rejects bad audio start."""
    handler, writer = make_handler()
    assert await handler.handle_event(event) is False
    assert written_events(writer)[0].data["code"] == "unsupported-audio"


async def test_stt_rejects_duplicate_audio_start() -> None:
    """Test stt rejects duplicate audio start."""
    handler, writer = make_handler()
    start = AudioStart(rate=16000, width=2, channels=1).event()
    assert await handler.handle_event(start) is True
    assert await handler.handle_event(start) is False
    assert written_events(writer)[0].data["code"] == "invalid-event-order"


async def test_stt_rejects_chunk_before_start_and_malformed_chunk() -> None:
    """Test stt rejects chunk before start and malformed chunk."""
    handler, writer = make_handler()
    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event()
    assert await handler.handle_event(chunk) is False
    assert written_events(writer)[0].data["code"] == "invalid-event-order"

    handler, writer = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(Event(type="audio-chunk", data={})) is False
    assert written_events(writer)[0].data["code"] == "invalid-audio"


async def test_stt_rejects_changed_or_partial_audio_chunk() -> None:
    """Test stt rejects changed or partial audio chunk."""
    handler, writer = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    changed = AudioChunk(rate=8000, width=2, channels=1, audio=b"\0\0").event()
    assert await handler.handle_event(changed) is False
    assert written_events(writer)[0].data["code"] == "invalid-audio"

    handler, writer = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    partial = AudioChunk(rate=16000, width=2, channels=1, audio=b"x").event()
    assert await handler.handle_event(partial) is False
    assert written_events(writer)[0].data["code"] == "invalid-audio"


async def test_stt_ignores_empty_chunk_and_converts_stereo() -> None:
    """Test stt ignores empty chunk and converts stereo."""
    handler, _ = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=2).event())
    empty = AudioChunk(rate=16000, width=2, channels=2, audio=b"").event()
    assert await handler.handle_event(empty) is True
    stereo = np.array([1000, 1000, -1000, -1000], dtype="<i2").tobytes()
    chunk = AudioChunk(rate=16000, width=2, channels=2, audio=stereo).event()
    assert await handler.handle_event(chunk) is True
    assert np.frombuffer(handler.audio_bytes, dtype="<i2").tolist() == [1000, -1000]


async def test_stt_conversion_runs_outside_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stt conversion runs outside event loop thread."""
    handler, _ = make_handler()
    await handler.handle_event(AudioStart(rate=8000, width=2, channels=1).event())
    assert handler.converter is not None
    original_convert = handler.converter.convert
    conversion_threads: list[int] = []

    def record_thread(chunk: AudioChunk) -> AudioChunk:
        """Record the worker identity before delegating conversion."""
        conversion_threads.append(threading.get_ident())
        return original_convert(chunk)

    monkeypatch.setattr(handler.converter, "convert", record_thread)
    original_to_thread = asyncio.to_thread
    to_thread = Mock(wraps=original_to_thread)
    monkeypatch.setattr("wyoming_vietnamese.stt.asyncio.to_thread", to_thread)
    event_loop_thread = threading.get_ident()
    chunk = AudioChunk(
        rate=8000,
        width=2,
        channels=1,
        audio=np.zeros(6000, dtype="<i2").tobytes(),
    ).event()
    assert await handler.handle_event(chunk) is True
    assert conversion_threads
    assert all(thread_id != event_loop_thread for thread_id in conversion_threads)
    assert len(conversion_threads) == 3
    assert to_thread.call_count == 1


async def test_stt_small_realtime_conversion_avoids_thread_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stt small realtime conversion avoids thread dispatch."""
    handler, _ = make_handler()
    await handler.handle_event(AudioStart(rate=8000, width=2, channels=1).event())
    assert handler.converter is not None
    original_convert = handler.converter.convert
    conversion_threads: list[int] = []

    def record_thread(chunk: AudioChunk) -> AudioChunk:
        """Record the inline conversion thread before delegating."""
        conversion_threads.append(threading.get_ident())
        return original_convert(chunk)

    monkeypatch.setattr(handler.converter, "convert", record_thread)
    event_loop_thread = threading.get_ident()
    chunk = AudioChunk(
        rate=8000,
        width=2,
        channels=1,
        audio=np.zeros(160, dtype="<i2").tobytes(),
    ).event()
    assert await handler.handle_event(chunk) is True
    assert conversion_threads == [event_loop_thread]


async def test_stt_contains_converter_failure() -> None:
    """Test stt contains converter failure."""
    handler, writer = make_handler()
    await handler.handle_event(AudioStart(rate=8000, width=2, channels=1).event())
    assert handler.converter is not None
    handler.converter.convert = Mock(side_effect=ValueError("bad audio"))  # type: ignore[method-assign]
    chunk = AudioChunk(rate=8000, width=2, channels=1, audio=b"\0\0").event()
    assert await handler.handle_event(chunk) is False
    assert written_events(writer)[0].data["code"] == "invalid-audio"


async def test_stt_enforces_duration_limit() -> None:
    """Test stt enforces duration limit."""
    handler, writer = make_handler(max_audio_seconds=0.00001)
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event()
    assert await handler.handle_event(chunk) is False
    assert written_events(writer)[0].data["code"] == "audio-too-long"


async def test_stt_rejects_source_duration_before_conversion() -> None:
    """Test stt rejects source duration before conversion."""
    handler, writer = make_handler(max_audio_seconds=0.001)
    await handler.handle_event(AudioStart(rate=8000, width=2, channels=2).event())
    assert handler.converter is not None
    convert = Mock(wraps=handler.converter.convert)
    handler.converter.convert = convert  # type: ignore[method-assign]
    oversized = np.zeros(9 * 2, dtype="<i2").tobytes()
    event = AudioChunk(rate=8000, width=2, channels=2, audio=oversized).event()
    assert await handler.handle_event(event) is False
    convert.assert_not_called()
    assert written_events(writer)[0].data["code"] == "audio-too-long"


async def test_stt_shared_buffer_budget_bounds_concurrent_requests() -> None:
    """Test stt shared buffer budget bounds concurrent requests."""
    budget = ByteBudget(2)
    first, _ = make_handler(buffer_budget=budget)
    second, second_writer = make_handler(buffer_budget=budget)
    start = AudioStart(rate=16000, width=2, channels=1).event()
    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event()
    assert await first.handle_event(start) is True
    assert await second.handle_event(start) is True
    assert await first.handle_event(chunk) is True
    assert budget.used == 2
    assert await second.handle_event(chunk) is False
    assert written_events(second_writer)[0].data["code"] == "server-busy"
    await first.disconnect()
    assert budget.used == 0


async def test_stt_transcribes_normalized_audio() -> None:
    """Test stt transcribes normalized audio."""
    recognizer = FakeRecognizer()
    handler, writer = make_handler(recognizer)
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
    assert await handler.handle_event(AudioStop().event()) is False
    assert recognizer.decoded is True
    assert recognizer.stream.rate == 16000
    assert np.allclose(recognizer.stream.samples, [-1.0, 0.0, 32767 / 32768])
    transcript = written_events(writer)[0]
    assert transcript.data == {"text": "xin chào", "language": VIETNAMESE_LANGUAGE}
    assert handler.audio_bytes == bytearray()


async def test_stt_logs_aggregate_timings_without_chunk_noise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test stt logs aggregate timings without chunk noise."""
    handler, _ = make_handler()
    with caplog.at_level(logging.DEBUG, logger="wyoming_vietnamese.stt"):
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event()
        )
        await handler.handle_event(AudioStop().event())

    messages = caplog.messages
    assert any("STT step=audio-start" in message for message in messages)
    assert not any("STT step=audio-chunk" in message for message in messages)
    assert any(
        "STT step=queue-acquired" in message and "wait_ms=" in message for message in messages
    )
    assert any(
        "STT step=inference-complete" in message and "real_time_factor=" in message
        for message in messages
    )
    assert any(
        "Completed STT request:" in message
        and "receive_ms=" in message
        and "handle_ms=" in message
        and "real_time_factor=" in message
        for message in messages
    )
    assert not any("cache=" in message for message in messages)


async def test_stt_empty_audio_returns_empty_transcript() -> None:
    """Test stt empty audio returns empty transcript."""
    handler, writer = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    assert await handler.handle_event(AudioStop().event()) is False
    events = written_events(writer)
    assert [event.type for event in events] == ["transcript"]
    assert events[0].data["text"] == ""


async def test_stt_rejects_stop_before_start() -> None:
    """Test stt rejects stop before start."""
    handler, writer = make_handler()
    assert await handler.handle_event(AudioStop().event()) is False
    assert written_events(writer)[0].data["code"] == "invalid-event-order"


async def test_stt_reports_busy_queue() -> None:
    """Test stt reports busy queue."""
    lock = asyncio.Lock()
    await lock.acquire()
    handler, writer = make_handler(lock=lock, queue_timeout=0.001)
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event())
    assert await handler.handle_event(AudioStop().event()) is False
    lock.release()
    assert written_events(writer)[0].data["code"] == "server-busy"


async def test_stt_contains_inference_failure() -> None:
    """Test stt contains inference failure."""
    handler, writer = make_handler(FakeRecognizer(fail=True))
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event())
    assert await handler.handle_event(AudioStop().event()) is False
    assert written_events(writer)[0].data["code"] == "inference-failed"


async def test_stt_skips_inference_for_closed_client() -> None:
    """Test stt skips inference for closed client."""
    recognizer = FakeRecognizer()
    handler, writer = make_handler(recognizer)
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event())
    writer.close()
    assert await handler.handle_event(AudioStop().event()) is False
    assert recognizer.decoded is False


async def test_stt_releases_buffer_while_a_numpy_view_is_alive() -> None:
    """Test clearing the buffer never fails because a sample view still exports it."""
    handler, _ = make_handler()
    await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
    pcm = np.array([1, -1, 1, -1], dtype="<i2").tobytes()
    await handler.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())

    # Resizing an exported bytearray raises BufferError, so the buffer must be rebound.
    view = np.frombuffer(handler.audio_bytes, dtype="<i2")
    handler._clear_audio_buffer()
    assert view.tolist() == [1, -1, 1, -1]
    assert handler.audio_bytes == bytearray()


async def test_stt_disconnect_releases_buffer() -> None:
    """Test stt disconnect releases buffer."""
    handler, _ = make_handler()
    handler.audio_bytes.extend(b"audio")
    await handler.disconnect()
    assert handler.audio_bytes == bytearray()
    assert handler.converter is None


def test_stt_info_uses_configured_model() -> None:
    """Test stt info uses configured model."""
    info = get_stt_info("custom/model", "revision")
    assert info.asr[0].name == "wyoming_vietnamese"
    model = info.asr[0].models[0]
    assert model.name == "custom/model"
    assert model.version == "revision"
    assert model.attribution.url.endswith("custom/model")
    assert info.asr[0].supports_transcript_streaming is False


def test_warm_up_stt_runs_silence_through_recognizer() -> None:
    """Test warm up stt runs silence through recognizer."""
    recognizer = FakeRecognizer()
    warm_up_stt(cast(STTRecognizer, recognizer))
    assert recognizer.decoded is True
    assert recognizer.stream.rate == 16000
    assert recognizer.stream.samples.shape == (1600,)
    assert not recognizer.stream.samples.any()


def _make_stt_model(directory: Path) -> None:
    """Build test support for  make stt model."""
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (directory / name).write_bytes(b"model")


def test_initialize_stt_validates_files(tmp_path: Path) -> None:
    """Test initialize stt validates files."""
    with pytest.raises(FileNotFoundError, match=r"encoder\.onnx"):
        initialize_stt(tmp_path)


def test_initialize_stt_passes_runtime_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test initialize stt passes runtime configuration."""
    _make_stt_model(tmp_path)
    recognizer = object()
    from_transducer = Mock(return_value=recognizer)
    module = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_transducer=from_transducer))
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    assert initialize_stt(tmp_path, num_threads=3) is recognizer
    assert from_transducer.call_args.kwargs["num_threads"] == 3
    assert from_transducer.call_args.kwargs["provider"] == "cpu"


def test_initialize_stt_auto_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test initialize stt auto threads."""
    _make_stt_model(tmp_path)
    from_transducer = Mock(return_value=object())
    module = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_transducer=from_transducer))
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr("wyoming_vietnamese.config.os.process_cpu_count", lambda: 16)
    initialize_stt(tmp_path)
    assert from_transducer.call_args.kwargs["num_threads"] == 16
