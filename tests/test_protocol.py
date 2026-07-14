"""Bounded Wyoming framing tests."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from wyoming.event import Event

from tests.helpers import MemoryWriter, make_reader, memory_writer, stream_writer, written_events
from wyoming_vietnamese.const import MAX_EVENT_HEADER_BYTES
from wyoming_vietnamese.protocol import (
    ByteBudget,
    ConnectionLimiter,
    ProtocolError,
    ProtocolWriteError,
    SafeAsyncEventHandler,
    async_read_event_limited,
)


async def test_read_event_with_split_data_and_payload() -> None:
    """Test read event with split data and payload."""
    extra_data = b'{"rate":16000}'
    header = json.dumps(
        {
            "type": "audio-chunk",
            "data": {"width": 2, "channels": 1},
            "data_length": len(extra_data),
            "payload_length": 4,
        }
    ).encode()
    event = await async_read_event_limited(make_reader(header + b"\n" + extra_data + b"pcm!"))
    assert event == Event(
        type="audio-chunk",
        data={"width": 2, "channels": 1, "rate": 16000},
        payload=b"pcm!",
    )


async def test_read_event_eof() -> None:
    """Test read event eof."""
    assert await async_read_event_limited(make_reader()) is None


@pytest.mark.parametrize(
    "frame",
    [
        b'{"type":"x"}',
        b"not-json\n",
        b"[]\n",
        b"{}\n",
        b'{"type":"","data":{}}\n',
        b'{"type":"x","data":[]}\n',
        b'{"type":"x","payload_length":-1}\n',
        b'{"type":"x","payload_length":true}\n',
        b'{"type":"x","payload_length":5}\n12345',
    ],
)
async def test_read_event_rejects_bad_headers(frame: bytes) -> None:
    """Test read event rejects bad headers."""
    if b'"payload_length":5' in frame:
        with pytest.raises(ProtocolError, match="exceeds"):
            await async_read_event_limited(make_reader(frame), max_payload_bytes=4)
    else:
        with pytest.raises(ProtocolError):
            await async_read_event_limited(make_reader(frame))


@pytest.mark.parametrize("data", [b"bad", b"[]"])
async def test_read_event_rejects_bad_external_data(data: bytes) -> None:
    """Test read event rejects bad external data."""
    header = json.dumps({"type": "x", "data_length": len(data)}).encode() + b"\n"
    with pytest.raises(ProtocolError, match="event data"):
        await async_read_event_limited(make_reader(header + data))


async def test_read_event_rejects_oversized_header() -> None:
    """Test read event rejects oversized header."""
    reader = asyncio.StreamReader(limit=MAX_EVENT_HEADER_BYTES * 2)
    reader.feed_data(b"{" + (b" " * MAX_EVENT_HEADER_BYTES) + b"}\n")
    reader.feed_eof()
    with pytest.raises(ProtocolError, match="header is too large"):
        await async_read_event_limited(reader)


async def test_read_event_wraps_stream_header_limit() -> None:
    """Test read event wraps stream header limit."""
    reader = asyncio.StreamReader(limit=8)
    reader.feed_data(b'{"type":"too-long"}\n')
    reader.feed_eof()
    with pytest.raises(ProtocolError, match="stream limit"):
        await async_read_event_limited(reader)


class RecordingHandler(SafeAsyncEventHandler):
    """Provide a test double for RecordingHandler."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        fail: bool = False,
        event_timeout: float = 60,
        write_timeout: float = 5,
    ) -> None:
        """Initialize the test double."""
        super().__init__(
            reader,
            writer,
            event_timeout=event_timeout,
            write_timeout=write_timeout,
        )
        self.events: list[Event] = []
        self.fail = fail
        self.disconnected = False

    async def handle_event(self, event: Event) -> bool:
        """Build test support for handle event."""
        self.events.append(event)
        if self.fail:
            raise RuntimeError("handler failed")
        return False

    async def disconnect(self) -> None:
        """Build test support for disconnect."""
        self.disconnected = True


async def test_safe_handler_processes_and_closes() -> None:
    """Test safe handler processes and closes."""
    writer = stream_writer()
    handler = RecordingHandler(make_reader(b'{"type":"test"}\n'), writer)
    await handler.run()
    assert [event.type for event in handler.events] == ["test"]
    assert handler.disconnected is True
    assert memory_writer(writer).closed is True
    assert memory_writer(writer).waited is True


async def test_safe_handler_reports_protocol_error() -> None:
    """Test safe handler reports protocol error."""
    writer = stream_writer()
    handler = RecordingHandler(make_reader(b"bad\n"), writer)
    await handler.run()
    events = written_events(writer)
    assert events[0].type == "error"
    assert events[0].data["code"] == "invalid-event"


async def test_safe_handler_contains_handler_exception() -> None:
    """Test safe handler contains handler exception."""
    writer = stream_writer()
    handler = RecordingHandler(make_reader(b'{"type":"test"}\n'), writer, fail=True)
    await handler.run()
    events = written_events(writer)
    assert events[0].data["code"] == "internal-error"


class FailingReader:
    """Provide a test double for FailingReader."""

    async def readline(self) -> bytes:
        """Build test support for readline."""
        raise ConnectionError("gone")


async def test_safe_handler_contains_connection_error() -> None:
    """Test safe handler contains connection error."""
    writer = stream_writer()
    handler = RecordingHandler(cast(asyncio.StreamReader, FailingReader()), writer)
    await handler.run()
    assert written_events(writer) == []


async def test_safe_handler_times_out_incomplete_client() -> None:
    """Test safe handler times out incomplete client."""
    reader = asyncio.StreamReader()
    writer = stream_writer()
    handler = RecordingHandler(reader, writer, event_timeout=0.001)
    await handler.run()
    events = written_events(writer)
    assert events[0].data["code"] == "request-timeout"
    assert handler.disconnected is True


class BlockingWriter(MemoryWriter):
    """Provide a writer whose drain never becomes writable."""

    async def drain(self) -> None:
        """Wait indefinitely until the caller's timeout cancels the write."""
        await asyncio.Event().wait()


async def test_safe_handler_times_out_slow_write() -> None:
    """Test safe handler times out slow write."""
    handler = RecordingHandler(
        make_reader(),
        stream_writer(BlockingWriter()),
        write_timeout=0.001,
    )
    with pytest.raises(ProtocolWriteError, match="write exceeded"):
        await handler.write_event(Event(type="test"))


def test_connection_limiter_enforces_and_releases_slots() -> None:
    """Test connection limiter enforces and releases slots."""
    limiter = ConnectionLimiter(1, local_reserve=0)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    limiter.release()
    assert limiter.active == 0
    with pytest.raises(RuntimeError, match="without an active"):
        limiter.release()


def test_connection_limiter_reserves_slots_for_local_clients() -> None:
    """Test loopback clients keep headroom after the remote limit is reached."""
    limiter = ConnectionLimiter(1, local_reserve=1)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    assert limiter.try_acquire(local=True) is True
    assert limiter.try_acquire(local=True) is False
    assert limiter.active == 2
    with pytest.raises(ValueError, match="local connection reserve"):
        ConnectionLimiter(1, local_reserve=-1)


def test_connection_limiter_can_stop_new_admission_while_handlers_drain() -> None:
    """Test shutdown rejects every new client without disturbing an active slot."""
    limiter = ConnectionLimiter(1, local_reserve=1)
    assert limiter.try_acquire() is True
    limiter.stop_accepting()
    assert limiter.accepting is False
    assert limiter.try_acquire() is False
    assert limiter.try_acquire(local=True) is False
    assert limiter.active == 1
    limiter.release()
    assert limiter.active == 0


async def test_connection_limiter_waits_until_every_slot_is_released() -> None:
    """Test the shutdown drain reports whether all handlers finished in time."""
    limiter = ConnectionLimiter(2)
    assert await limiter.wait_idle(0.5) is True

    assert limiter.try_acquire() is True
    assert await limiter.wait_idle(0.01) is False

    async def release_soon() -> None:
        """Release the outstanding slot after letting the waiter start."""
        await asyncio.sleep(0.01)
        limiter.release()

    release = asyncio.create_task(release_soon())
    assert await limiter.wait_idle(1) is True
    await release


def test_byte_budget_enforces_and_releases_bytes() -> None:
    """Test byte budget enforces and releases bytes."""
    budget = ByteBudget(4)
    assert budget.try_reserve(3) is True
    assert budget.try_reserve(2) is False
    budget.release(3)
    assert budget.used == 0
    with pytest.raises(RuntimeError, match="exceeds"):
        budget.release(1)
