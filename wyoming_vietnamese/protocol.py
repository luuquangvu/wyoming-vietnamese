"""Defensive Wyoming event parsing and connection lifecycle handling."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from wyoming.error import Error as WyomingError
from wyoming.event import Event
from wyoming.server import AsyncEventHandler

from .const import (
    DEFAULT_EVENT_TIMEOUT,
    DEFAULT_LOCAL_CONNECTION_RESERVE,
    DEFAULT_WRITE_TIMEOUT,
    MAX_EVENT_DATA_BYTES,
    MAX_EVENT_HEADER_BYTES,
    MAX_EVENT_PAYLOAD_BYTES,
    VIETNAMESE_LANGUAGE,
)

_LOGGER = logging.getLogger(__name__)


class ProtocolError(ValueError):
    """Raised when an incoming Wyoming frame is malformed or exceeds a limit."""


class ProtocolWriteError(ConnectionError):
    """Raised when an outgoing Wyoming frame cannot be delivered promptly."""


def is_vietnamese_language(language: str) -> bool:
    """Return whether a Wyoming language selector matches the advertised value."""
    return language == VIETNAMESE_LANGUAGE


class ConnectionLimiter:
    """Track a process-wide bound on active client handlers.

    Loopback clients may use a small extra reserve so that shedding remote load never
    makes the container health check report an unhealthy service.
    """

    def __init__(
        self,
        maximum: int,
        local_reserve: int = DEFAULT_LOCAL_CONNECTION_RESERVE,
    ) -> None:
        """Initialize a positive maximum, its local reserve, and an empty active count."""
        if maximum < 1:
            raise ValueError("maximum connections must be positive")
        if local_reserve < 0:
            raise ValueError("local connection reserve must not be negative")
        self.maximum = maximum
        self.local_reserve = local_reserve
        self.active = 0
        self._accepting = True
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def accepting(self) -> bool:
        """Return whether new connection handlers may reserve a slot."""
        return self._accepting

    def try_acquire(self, *, local: bool = False) -> bool:
        """Reserve one connection slot without waiting."""
        if not self._accepting:
            return False
        limit = self.maximum + self.local_reserve if local else self.maximum
        if self.active >= limit:
            return False
        self.active += 1
        self._idle.clear()
        return True

    def stop_accepting(self) -> None:
        """Reject new handlers while allowing existing connections to drain."""
        self._accepting = False

    def release(self) -> None:
        """Release one previously reserved connection slot."""
        if self.active < 1:
            raise RuntimeError("connection limit released without an active reservation")
        self.active -= 1
        if not self.active:
            self._idle.set()

    async def wait_idle(self, timeout: float) -> bool:
        """Wait for every handler to release its slot, reporting whether all did."""
        try:
            async with asyncio.timeout(timeout):
                await self._idle.wait()
        except TimeoutError:
            return False
        return True


class ByteBudget:
    """Track bounded in-memory buffers shared by event-loop handlers."""

    def __init__(self, maximum: int) -> None:
        """Initialize a positive maximum and an empty byte reservation."""
        if maximum < 1:
            raise ValueError("maximum byte budget must be positive")
        self.maximum = maximum
        self.used = 0

    def try_reserve(self, size: int) -> bool:
        """Reserve bytes if doing so stays inside the process-wide budget."""
        if size < 0:
            raise ValueError("reserved byte count must not be negative")
        if self.used + size > self.maximum:
            return False
        self.used += size
        return True

    def release(self, size: int) -> None:
        """Release bytes previously reserved by a handler."""
        if size < 0 or size > self.used:
            raise RuntimeError("released byte count exceeds the active reservation")
        self.used -= size


def _read_length(event_dict: dict[str, object], name: str, maximum: int) -> int:
    """Validate a Wyoming frame length field."""
    value = event_dict.get(name, 0)
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    if value > maximum:
        raise ProtocolError(f"{name} exceeds the {maximum}-byte limit")
    return value


async def async_read_event_limited(
    reader: asyncio.StreamReader,
    *,
    max_payload_bytes: int = MAX_EVENT_PAYLOAD_BYTES,
) -> Event | None:
    """Read one Wyoming event while enforcing frame and payload limits."""
    try:
        json_line = await reader.readline()
    except ValueError as err:
        raise ProtocolError("event header exceeds the stream limit") from err

    if not json_line:
        return None
    if len(json_line) > MAX_EVENT_HEADER_BYTES:
        raise ProtocolError("event header is too large")
    if not json_line.endswith(b"\n"):
        raise ProtocolError("event header is not newline terminated")

    try:
        event_dict = json.loads(json_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError("event header is not valid JSON") from err
    if not isinstance(event_dict, dict):
        raise ProtocolError("event header must be a JSON object")

    event_type = event_dict.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ProtocolError("event type must be a non-empty string")

    data = event_dict.get("data", {})
    if not isinstance(data, dict):
        raise ProtocolError("event data must be a JSON object")

    data_length = _read_length(event_dict, "data_length", MAX_EVENT_DATA_BYTES)
    if data_length:
        data_bytes = await reader.readexactly(data_length)
        try:
            extra_data = json.loads(data_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise ProtocolError("event data is not valid JSON") from err
        if not isinstance(extra_data, dict):
            raise ProtocolError("event data payload must be a JSON object")
        data.update(extra_data)

    payload_length = _read_length(event_dict, "payload_length", max_payload_bytes)
    payload = await reader.readexactly(payload_length) if payload_length else None
    return Event(type=event_type, data=data, payload=payload)


class SafeAsyncEventHandler(AsyncEventHandler):
    """Event handler base that contains malformed input and connection failures."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        event_timeout: float = DEFAULT_EVENT_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
    ) -> None:
        """Initialize bounded event reads and writes for one connection."""
        if event_timeout <= 0 or write_timeout <= 0:
            raise ValueError("protocol timeouts must be positive")
        super().__init__(reader, writer)
        self.event_timeout = event_timeout
        self.write_timeout = write_timeout

    async def write_event(self, event: Event) -> None:
        """Write one event without letting a stalled peer block indefinitely."""
        try:
            async with asyncio.timeout(self.write_timeout):
                await super().write_event(event)
        except TimeoutError as err:
            raise ProtocolWriteError(
                f"Wyoming event write exceeded {self.write_timeout:g} seconds"
            ) from err

    async def run(self) -> None:
        """Process events with bounded parsing and deterministic connection cleanup."""
        self._is_running = True
        peer = self.writer.get_extra_info("peername")
        try:
            while self._is_running:
                async with asyncio.timeout(self.event_timeout):
                    event = await async_read_event_limited(self.reader)
                if event is None or not (await self.handle_event(event)):
                    break
        except TimeoutError:
            _LOGGER.warning(
                "Closed idle or incomplete Wyoming request from %s after %.3f seconds",
                peer,
                self.event_timeout,
            )
            await self._try_write_error("Wyoming request timed out", "request-timeout")
        except ProtocolError as err:
            _LOGGER.warning("Rejected malformed Wyoming event from %s: %s", peer, err)
            await self._try_write_error("Invalid Wyoming event", "invalid-event")
        except asyncio.IncompleteReadError:
            _LOGGER.debug("Client %s disconnected during a Wyoming frame", peer)
        except ConnectionError as err:
            _LOGGER.debug("Client %s disconnected: %s", peer, err)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Unhandled Wyoming handler failure for client %s", peer)
            await self._try_write_error("Internal server error", "internal-error")
        finally:
            with suppress(Exception):
                await self.disconnect()
            await self._close_connection()

    async def _close_connection(self) -> None:
        """Close the client stream and absorb transport teardown races."""
        self.writer.close()
        with suppress(Exception):
            await self.writer.wait_closed()

    async def _try_write_error(self, text: str, code: str) -> None:
        """Best-effort an error response without delaying connection teardown."""
        with suppress(Exception):
            async with asyncio.timeout(1):
                await self.write_event(WyomingError(text=text, code=code).event())
