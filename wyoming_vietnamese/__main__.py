"""Process entry point for the combined Wyoming Vietnamese server."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from functools import partial
from pathlib import Path
from typing import cast

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.modules[__name__].__package__ = "wyoming_vietnamese"

from wyoming.server import AsyncServer, HandlerFactory

from .cache import BoundedLruCache
from .combined import CombinedEventHandler, combine_service_info
from .config import ServerConfig
from .const import (
    PROGRAM_NAME,
    Timeout,
    TtsEngine,
    TtsProvider,
)
from .cpu import resolve_cpu_threads
from .download import download_models
from .inference import create_inference_executor
from .protocol import ByteBudget, ConnectionLimiter
from .stt import SherpaSTTEventHandler, get_stt_info, initialize_stt, warm_up_stt
from .stt_model import STT_MODEL
from .tts import (
    TTSEngine,
    TTSEventHandler,
    get_tts_info,
    initialize_tts_voices,
    initialize_zerotts,
    validate_tts_voices_for_engine,
    warm_up_tts,
)
from .tts_model import NghiTtsVoiceSpec, ZeroTtsVoiceSpec

_LOGGER = logging.getLogger(PROGRAM_NAME)


def _configure_logging(level: str) -> None:
    """Configure process logging once using the validated level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def _drain_inference(
    lock: asyncio.Lock,
    label: str,
    *,
    timeout: float = Timeout.SHUTDOWN_DRAIN,
) -> None:
    """Wait for one in-flight inference to finish before its model is released."""
    try:
        await asyncio.wait_for(lock.acquire(), timeout)
    except TimeoutError:
        _LOGGER.warning(
            "Timed out after %.1f seconds waiting for in-flight %s inference",
            timeout,
            label,
        )
        return
    lock.release()


def _log_startup_configuration(server_config: ServerConfig, cpu_threads: int) -> None:
    """Log configured server parameters, directory targets, and model sources."""
    _LOGGER.info(
        "Configuration: port=%d threads=%d offline=%s tts_engine=%s tts_voices=%s",
        server_config.port,
        cpu_threads,
        server_config.offline,
        server_config.tts_engine,
        ",".join(voice.id for voice in server_config.tts_voices),
    )
    _LOGGER.info(
        "Model storage: cache_dir=%s, download_dir=%s",
        server_config.cache_dir,
        server_config.download_dir,
    )
    tts_provider_name = (
        TtsProvider.ZEROTTS
        if server_config.tts_engine == TtsEngine.ZEROTTS
        else TtsProvider.NGHITTS
    )
    _LOGGER.info(
        "Model sources: stt=%s@%s %s_voices=%s",
        STT_MODEL.repo,
        STT_MODEL.revision,
        tts_provider_name.lower(),
        ", ".join(voice.name for voice in server_config.tts_voices),
    )
    _LOGGER.info(
        "Transport limits: active_connections=%d event_timeout=%.1fs write_timeout=%.1fs "
        "stt_buffer_mb=%.1f",
        server_config.max_active_connections,
        server_config.event_timeout,
        server_config.write_timeout,
        server_config.max_stt_buffer_bytes / (1024 * 1024),
    )


def _initialize_configured_tts(
    server_config: ServerConfig,
    tts_path: Path,
    cpu_threads: int,
) -> TTSEngine:
    """Initialize the configured TTS engine according to selected engine."""
    validate_tts_voices_for_engine(server_config.tts_voices, server_config.tts_engine)
    if server_config.tts_engine == TtsEngine.ZEROTTS:
        zerotts_voices = tuple(cast(ZeroTtsVoiceSpec, v) for v in server_config.tts_voices)
        return initialize_zerotts(
            tts_path,
            zerotts_voices,
            cpu_threads,
        )
    nghitts_voices = tuple(cast(NghiTtsVoiceSpec, v) for v in server_config.tts_voices)
    return initialize_tts_voices(
        tts_path,
        nghitts_voices,
        cpu_threads,
    )


async def run_server(
    config: ServerConfig | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Initialize models, bind both services, and shut down gracefully."""
    server_config = config or ServerConfig.from_env()
    cpu_threads = resolve_cpu_threads(server_config.cpu_threads)
    _configure_logging(server_config.log_level)
    _LOGGER.info("Starting Wyoming Vietnamese server initialization")

    server_config.cache_dir.mkdir(parents=True, exist_ok=True)
    server_config.download_dir.mkdir(parents=True, exist_ok=True)
    _log_startup_configuration(server_config, cpu_threads)

    validate_tts_voices_for_engine(server_config.tts_voices, server_config.tts_engine)
    paths = download_models(
        cache_dir=server_config.cache_dir,
        download_dir=server_config.download_dir,
        tts_voices=server_config.tts_voices,
        offline=server_config.offline,
        tts_engine=server_config.tts_engine,
    )

    _LOGGER.info("Initializing Speech-to-Text engine")
    stt_recognizer = initialize_stt(
        paths["stt"],
        cpu_threads,
    )
    await asyncio.to_thread(warm_up_stt, stt_recognizer)
    stt_info = get_stt_info(STT_MODEL.repo, STT_MODEL.revision)

    _LOGGER.info("Initializing Text-to-Speech engine (%s)", server_config.tts_engine)
    tts_engine = _initialize_configured_tts(server_config, paths["tts"], cpu_threads)
    try:
        await asyncio.to_thread(
            warm_up_tts,
            tts_engine,
            server_config.tts_voices,
            server_config.tts_engine,
        )
    except Exception:
        await asyncio.to_thread(tts_engine.close)
        raise
    tts_info = get_tts_info(server_config.tts_voices, engine=server_config.tts_engine)
    stt_info_event = stt_info.event()
    tts_info_event = tts_info.event()
    combined_info_event = combine_service_info(stt_info, tts_info).event()

    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    connection_limiter = ConnectionLimiter(server_config.max_active_connections)
    stt_buffer_budget = ByteBudget(server_config.max_stt_buffer_bytes)
    tts_result_cache = BoundedLruCache[bytes, bytes](
        max_entries=server_config.tts_cache_max_entries,
        max_bytes=server_config.tts_cache_max_bytes,
        max_item_bytes=server_config.tts_cache_max_item_bytes,
        max_idle_seconds=server_config.tts_cache_idle_seconds,
    )
    _LOGGER.info(
        "TTS inference cache: idle_seconds=%.0f, entries=%d, max_mb=%.1f, max_item_mb=%.1f",
        server_config.tts_cache_idle_seconds,
        server_config.tts_cache_max_entries,
        server_config.tts_cache_max_bytes / (1024 * 1024),
        server_config.tts_cache_max_item_bytes / (1024 * 1024),
    )
    inference_executor = create_inference_executor()
    stt_handler_factory: HandlerFactory = partial(
        SherpaSTTEventHandler,
        stt_recognizer,
        stt_info_event,
        stt_lock,
        STT_MODEL.repo,
        server_config.max_stt_audio_seconds,
        server_config.inference_queue_timeout,
        buffer_budget=stt_buffer_budget,
        event_timeout=server_config.event_timeout,
        write_timeout=server_config.write_timeout,
        inference_executor=inference_executor,
    )
    tts_handler_factory: HandlerFactory = partial(
        TTSEventHandler,
        tts_engine,
        tts_info_event,
        tts_lock,
        tts_result_cache,
        server_config.max_tts_text_chars,
        server_config.inference_queue_timeout,
        event_timeout=server_config.event_timeout,
        write_timeout=server_config.write_timeout,
        sentence_silence_ms=server_config.tts_sentence_silence_ms,
        clause_silence_ms=server_config.tts_clause_silence_ms,
        paragraph_silence_ms=server_config.tts_paragraph_silence_ms,
        inference_executor=inference_executor,
    )

    uri = f"tcp://0.0.0.0:{server_config.port}"
    server = AsyncServer.from_uri(uri)
    handler_factory: HandlerFactory = partial(
        CombinedEventHandler,
        stt_handler_factory,
        tts_handler_factory,
        combined_info_event,
        connection_limiter=connection_limiter,
        event_timeout=server_config.event_timeout,
        write_timeout=server_config.write_timeout,
    )

    shutdown_event = stop_event or asyncio.Event()
    registered_signals: list[signal.Signals] = []
    loop = asyncio.get_running_loop()
    if stop_event is None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, shutdown_event.set)
            except NotImplementedError:
                break
            registered_signals.append(signum)

    server_started = False
    try:
        await server.start(handler_factory)
        server_started = True
        _LOGGER.info("Wyoming STT/TTS service is ready at %s", uri)
        await shutdown_event.wait()
    finally:
        _LOGGER.info("Stopping Wyoming services")
        for signum in registered_signals:
            loop.remove_signal_handler(signum)
        if server_started:
            # Reject new handlers before observing the locks so a newly admitted
            # connection cannot start inference after a service has been drained.
            # Existing clients retain their writers until both independent model calls
            # have completed or timed out.
            connection_limiter.stop_accepting()
            await asyncio.gather(
                _drain_inference(stt_lock, "STT"),
                _drain_inference(tts_lock, "TTS"),
            )
            try:
                await server.stop()
            except Exception as err:
                _LOGGER.exception("Server shutdown failed: %s", err)
            if not await connection_limiter.wait_idle(Timeout.SHUTDOWN_DRAIN):
                _LOGGER.warning(
                    "Releasing models while %d client handler(s) are still active",
                    connection_limiter.active,
                )
        tts_entries, tts_bytes = tts_result_cache.clear()
        _LOGGER.debug(
            "Cleared TTS inference cache: entries=%d bytes=%d",
            tts_entries,
            tts_bytes,
        )
        await asyncio.to_thread(inference_executor.shutdown)
        await asyncio.to_thread(tts_engine.close)
        _LOGGER.info("Wyoming services stopped")


def main() -> None:
    """Run the service process and translate startup failures into an exit code."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        _LOGGER.info("Shutdown requested")
    except Exception:
        if not logging.getLogger().handlers:
            _configure_logging("INFO")
        _LOGGER.critical("Wyoming server startup or execution failed", exc_info=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
