"""Shared in-memory Wyoming stream test doubles."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterable
from typing import Any, cast

from wyoming.event import Event, read_event


class MemoryWriter:
    """Provide a test double for MemoryWriter."""

    def __init__(self) -> None:
        """Initialize the test double."""
        self.buffer = bytearray()
        self.closed = False
        self.waited = False
        self.peername: object = ("127.0.0.1", 12345)

    def write(self, data: bytes) -> None:
        """Build test support for write."""
        self.buffer.extend(data)

    def writelines(self, lines: Iterable[bytes]) -> None:
        """Build test support for writelines."""
        for line in lines:
            self.write(line)

    async def drain(self) -> None:
        """Build test support for drain."""
        return

    def close(self) -> None:
        """Build test support for close."""
        self.closed = True

    async def wait_closed(self) -> None:
        """Build test support for wait closed."""
        self.waited = True

    def is_closing(self) -> bool:
        """Build test support for is closing."""
        return self.closed

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Build test support for get extra info."""
        return self.peername if name == "peername" else default


def stream_writer(writer: MemoryWriter | None = None) -> asyncio.StreamWriter:
    """Build test support for stream writer."""
    return cast(asyncio.StreamWriter, writer or MemoryWriter())


def memory_writer(writer: asyncio.StreamWriter) -> MemoryWriter:
    """Build test support for memory writer."""
    return cast(MemoryWriter, writer)


def make_reader(data: bytes = b"") -> asyncio.StreamReader:
    """Build test support for make reader."""
    reader = asyncio.StreamReader()
    if data:
        reader.feed_data(data)
    reader.feed_eof()
    return reader


def written_events(writer: asyncio.StreamWriter) -> list[Event]:
    """Build test support for written events."""
    source = io.BytesIO(bytes(memory_writer(writer).buffer))
    events: list[Event] = []
    while event := read_event(source):
        events.append(event)
    return events
