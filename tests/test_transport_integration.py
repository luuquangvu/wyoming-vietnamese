"""Real TCP transport regression tests for the combined Wyoming endpoint."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from wyoming.event import Event, async_read_event, async_write_event
from wyoming.info import Describe, Info

from wyoming_vietnamese.combined import CombinedEventHandler, combine_service_info
from wyoming_vietnamese.protocol import SafeAsyncEventHandler
from wyoming_vietnamese.stt import get_stt_info
from wyoming_vietnamese.tts import get_tts_info
from wyoming_vietnamese.tts_model import DEFAULT_TTS_VOICE


class NoopHandler(SafeAsyncEventHandler):
    """Provide a subordinate handler for discovery-only transport tests."""

    async def handle_event(self, event: Event) -> bool:
        """Accept routed events without producing output."""
        return True


async def test_combined_endpoint_serves_discovery_over_real_tcp() -> None:
    """Test combined endpoint serves discovery over real tcp."""
    service_info = combine_service_info(
        get_stt_info("owner/stt"),
        get_tts_info(DEFAULT_TTS_VOICE),
    )
    tasks: set[asyncio.Task[None]] = set()
    factory: Callable[
        [asyncio.StreamReader, asyncio.StreamWriter],
        SafeAsyncEventHandler,
    ] = NoopHandler

    def client_connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Run one combined handler for an accepted TCP client."""
        handler = CombinedEventHandler(factory, factory, service_info.event(), reader, writer)
        task = asyncio.create_task(handler.run())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(client_connected, "127.0.0.1", 0)
    try:
        socket = server.sockets[0]
        port = int(socket.getsockname()[1])
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await async_write_event(Describe().event(), writer)
        response = await asyncio.wait_for(async_read_event(reader), timeout=1)
        assert response is not None
        info = Info.from_event(response)
        assert len(info.asr) == 1
        assert len(info.tts) == 1
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0)
    finally:
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)
