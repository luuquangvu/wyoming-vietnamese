"""Combined Wyoming endpoint tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

from tests.helpers import make_reader, memory_writer, stream_writer, written_events
from wyoming_vietnamese.combined import CombinedEventHandler, combine_service_info
from wyoming_vietnamese.protocol import ConnectionLimiter, SafeAsyncEventHandler
from wyoming_vietnamese.stt import get_stt_info
from wyoming_vietnamese.tts import get_tts_info
from wyoming_vietnamese.tts_model import DEFAULT_TTS_VOICE


class RecordingHandler(SafeAsyncEventHandler):
    """Provide a test double for RecordingHandler."""

    def __init__(
        self,
        events: list[Event],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Initialize the test double."""
        super().__init__(reader, writer)
        self.events = events
        self.disconnected = False

    async def handle_event(self, event: Event) -> bool:
        """Build test support for handle event."""
        self.events.append(event)
        return True

    async def disconnect(self) -> None:
        """Build test support for disconnect."""
        self.disconnected = True


def _factory(
    events: list[Event],
    handlers: list[RecordingHandler],
) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], SafeAsyncEventHandler]:
    """Build test support for  factory."""

    def create(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> RecordingHandler:
        handler = RecordingHandler(events, reader, writer)
        handlers.append(handler)
        return handler

    return create


def _make_handler(
    peername: object = ("127.0.0.1", 12345),
) -> tuple[
    CombinedEventHandler,
    asyncio.StreamWriter,
    list[Event],
    list[Event],
    list[RecordingHandler],
]:
    """Build test support for  make handler."""
    stt_events: list[Event] = []
    tts_events: list[Event] = []
    handlers: list[RecordingHandler] = []
    writer = stream_writer()
    memory_writer(writer).peername = peername
    info = combine_service_info(
        get_stt_info("owner/stt"),
        get_tts_info(DEFAULT_TTS_VOICE),
    )
    handler = CombinedEventHandler(
        _factory(stt_events, handlers),
        _factory(tts_events, handlers),
        info.event(),
        make_reader(),
        writer,
    )
    return handler, writer, stt_events, tts_events, handlers


async def test_combined_handler_advertises_shared_service_name() -> None:
    """Test combined handler advertises shared service name."""
    handler, writer, _, _, _ = _make_handler()
    assert await handler.handle_event(Describe().event()) is True
    info = Info.from_event(written_events(writer)[0])
    assert info.asr[0].name == "wyoming_vietnamese"
    assert info.tts[0].name == "wyoming_vietnamese"
    assert info.asr[0].installed is True
    assert info.tts[0].installed is True


async def test_combined_handler_routes_stt_and_tts_events() -> None:
    """Test combined handler routes stt and tts events."""
    handler, _, stt_events, tts_events, _ = _make_handler()
    stt_requests = [
        Transcribe().event(),
        AudioStart(rate=16000, width=2, channels=1).event(),
        AudioChunk(rate=16000, width=2, channels=1, audio=b"\0\0").event(),
        AudioStop().event(),
    ]
    for event in stt_requests:
        assert await handler.handle_event(event) is True
    tts_requests = [
        SynthesizeStart().event(),
        SynthesizeChunk(text="xin chào").event(),
        Synthesize(text="xin chào").event(),
        SynthesizeStop().event(),
    ]
    for event in tts_requests:
        assert await handler.handle_event(event) is True
    assert await handler.handle_event(Event(type="unknown")) is True

    assert stt_events == stt_requests
    assert tts_events == tts_requests


async def test_combined_handler_disconnects_both_handlers() -> None:
    """Test combined handler disconnects both handlers."""
    handler, _, _, _, handlers = _make_handler()
    await handler.disconnect()
    assert len(handlers) == 2
    assert all(subhandler.disconnected for subhandler in handlers)


async def test_combined_handler_rejects_excess_connection() -> None:
    """Test combined handler rejects excess connection."""
    limiter = ConnectionLimiter(1, local_reserve=0)
    assert limiter.try_acquire()
    handler, writer, _, _, handlers = _make_handler(peername=("192.0.2.10", 12345))
    handler.connection_limiter = limiter
    await handler.run()
    assert written_events(writer)[0].data["code"] == "server-busy"
    assert all(subhandler.disconnected for subhandler in handlers)
    assert limiter.active == 1
    limiter.release()


async def test_combined_handler_keeps_headroom_for_local_health_checks() -> None:
    """Test a saturated remote limit still admits the loopback health check."""
    limiter = ConnectionLimiter(1, local_reserve=1)
    assert limiter.try_acquire()

    remote, remote_writer, _, _, _ = _make_handler(peername=("192.0.2.10", 12345))
    remote.connection_limiter = limiter
    await remote.run()
    assert written_events(remote_writer)[0].data["code"] == "server-busy"

    local, local_writer, _, _, _ = _make_handler(peername=("127.0.0.1", 12345))
    local.connection_limiter = limiter
    await local.run()
    assert written_events(local_writer) == []
    assert limiter.active == 1
    limiter.release()


@pytest.mark.parametrize(
    "peername",
    [("192.0.2.10", 12345), None, ("not-an-address", 1), ("::1", 1, 0, 0)],
)
async def test_combined_handler_releases_connection_slot(peername: object) -> None:
    """Test combined handler releases connection slot for any peer shape."""
    limiter = ConnectionLimiter(1)
    handler, _, _, _, _ = _make_handler(peername=peername)
    handler.connection_limiter = limiter
    await handler.run()
    assert limiter.active == 0
