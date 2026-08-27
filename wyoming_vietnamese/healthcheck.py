"""Container healthcheck for the combined Wyoming TCP endpoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from .config import ServerConfig

_LOGGER = logging.getLogger("healthcheck")


async def check_port(
    port: int,
    timeout: float = 10.0,
) -> bool:
    """Return whether a port advertises installed STT and TTS services."""
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await async_write_event(Describe().event(), writer)
            response = await async_read_event(reader)
            if response is None or not Info.is_type(response.type):
                return False
            info = Info.from_event(response)
            has_stt = any(program.installed for program in info.asr)
            has_tts = any(program.installed for program in info.tts)
            return has_stt and has_tts
    except (TimeoutError, ConnectionError, KeyError, OSError, TypeError, ValueError) as err:
        _LOGGER.error("Healthcheck failed on port %d: %s", port, err)
        return False
    finally:
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


async def async_main() -> int:
    """Check configured services and return a process exit code."""
    try:
        config = ServerConfig.from_env()
    except ValueError:
        return 1

    healthy = await check_port(config.port)
    return 0 if healthy else 1


def main() -> None:
    """Run the asynchronous healthcheck as a command-line program."""
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
