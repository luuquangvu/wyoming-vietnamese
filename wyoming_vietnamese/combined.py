"""Combined Wyoming endpoint for speech-to-text and text-to-speech."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from ipaddress import ip_address

from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop

from .const import DEFAULT_EVENT_TIMEOUT, DEFAULT_WRITE_TIMEOUT
from .protocol import ConnectionLimiter, SafeAsyncEventHandler

_LOGGER = logging.getLogger(__name__)

_HandlerFactory = Callable[
    [asyncio.StreamReader, asyncio.StreamWriter],
    SafeAsyncEventHandler,
]


def combine_service_info(stt_info: Info, tts_info: Info) -> Info:
    """Combine ASR and TTS discovery data for a shared endpoint."""
    return Info(asr=list(stt_info.asr), tts=list(tts_info.tts))


class CombinedEventHandler(SafeAsyncEventHandler):
    """Route events from one Wyoming connection to its service handler."""

    def __init__(
        self,
        stt_handler_factory: _HandlerFactory,
        tts_handler_factory: _HandlerFactory,
        wyoming_info_event: Event,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        connection_limiter: ConnectionLimiter | None = None,
        event_timeout: float = DEFAULT_EVENT_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
    ) -> None:
        """Create per-connection STT and TTS handlers over shared streams."""
        super().__init__(
            reader,
            writer,
            event_timeout=event_timeout,
            write_timeout=write_timeout,
        )
        self.stt_handler = stt_handler_factory(reader, writer)
        self.tts_handler = tts_handler_factory(reader, writer)
        self.wyoming_info_event = wyoming_info_event
        self.connection_limiter = connection_limiter

    def _is_loopback_peer(self) -> bool:
        """Return whether this client connected over the loopback interface."""
        peer = self.writer.get_extra_info("peername")
        host = peer[0] if isinstance(peer, tuple) and peer else None
        if not isinstance(host, str):
            return False
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    async def run(self) -> None:
        """Reject unavailable clients before allocating any request buffers."""
        if self.connection_limiter is None:
            await super().run()
            return
        if not self.connection_limiter.try_acquire(local=self._is_loopback_peer()):
            peer = self.writer.get_extra_info("peername")
            shutting_down = not self.connection_limiter.accepting
            if shutting_down:
                reason = "server is shutting down"
                error_text = "Server is shutting down"
            else:
                reason = "active connection limit reached"
                error_text = "Server connection limit reached"
            _LOGGER.warning("Rejected connection from %s: %s", peer, reason)
            try:
                await self._try_write_error(error_text, "server-busy")
            finally:
                with suppress(Exception):
                    await self.disconnect()
                await self._close_connection()
            return
        try:
            await super().run()
        finally:
            self.connection_limiter.release()

    async def handle_event(self, event: Event) -> bool:
        """Advertise both services and route each request by event type."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent combined STT/TTS service information")
            return True

        if (
            Synthesize.is_type(event.type)
            or SynthesizeStart.is_type(event.type)
            or SynthesizeChunk.is_type(event.type)
            or SynthesizeStop.is_type(event.type)
        ):
            return await self.tts_handler.handle_event(event)

        if (
            Transcribe.is_type(event.type)
            or AudioStart.is_type(event.type)
            or AudioChunk.is_type(event.type)
            or AudioStop.is_type(event.type)
        ):
            return await self.stt_handler.handle_event(event)

        return True

    async def disconnect(self) -> None:
        """Release both subordinate handlers when their connection closes."""
        try:
            await self.stt_handler.disconnect()
        finally:
            await self.tts_handler.disconnect()
