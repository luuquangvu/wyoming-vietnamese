"""Process entry point for the combined Wyoming Vietnamese server."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import signal
import sys
from functools import partial
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.modules[__name__].__package__ = "wyoming_vietnamese"

from wyoming.server import AsyncServer, HandlerFactory

from .cache import BoundedLruCache, PersistentAudioCache, ResultCache
from .combined import CombinedEventHandler, combine_service_info
from .config import ServerConfig
from .const import (
    PROGRAM_NAME,
    STT_DIR,
    TTS_CACHE_DIR,
    TTS_DIR,
    Timeout,
    TtsCacheFile,
    TtsEngine,
    TtsProvider,
)
from .cpu import resolve_cpu_threads
from .download import download_models
from .inference import create_inference_executor
from .protocol import ByteBudget, ConnectionLimiter
from .resources import DeviceResources, detect_device_resources, warn_if_low_startup_memory
from .stt import SherpaSTTEventHandler, get_stt_info, initialize_stt, warm_up_stt
from .stt_model import SttModelSpec, get_stt_model
from .tts import (
    TTSEngine,
    TTSEventHandler,
    get_tts_info,
    initialize_lazy_tts_voices,
    initialize_tts_voices,
    initialize_zerotts,
    validate_tts_voices_for_engine,
    warm_up_tts,
)
from .tts_model import ZEROTTS_MODEL, NghiTtsVoiceSpec, ZeroTtsVoiceSpec
from .voice_usage import RecentVoiceStore

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


def _log_startup_configuration(
    server_config: ServerConfig,
    cpu_threads: int,
    stt_model: SttModelSpec | None = None,
) -> None:
    """Log configured server parameters, directory targets, and model sources."""
    model = stt_model or get_stt_model(server_config.stt_engine)
    _LOGGER.info(
        "Configuration: port=%d threads=%d offline=%s stt_engine=%s tts_engine=%s "
        "tts_voices=%s tts_cache_enabled=%s",
        server_config.port,
        cpu_threads,
        server_config.offline,
        server_config.stt_engine,
        server_config.tts_engine,
        ",".join(voice.id for voice in server_config.tts_voices),
        server_config.tts_cache_enabled,
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
        model.repo,
        model.revision,
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
    voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...] | None = None,
    *,
    all_voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...] | None = None,
    max_loaded_voices: int | None = None,
) -> TTSEngine:
    """Initialize the configured TTS engine according to selected engine."""
    selected_voices = server_config.tts_voices if voices is None else voices
    validate_tts_voices_for_engine(selected_voices, server_config.tts_engine)
    if server_config.tts_engine == TtsEngine.ZEROTTS:
        zerotts_voices = tuple(v for v in selected_voices if isinstance(v, ZeroTtsVoiceSpec))
        return initialize_zerotts(
            tts_path,
            zerotts_voices,
            cpu_threads,
        )
    nghitts_voices = tuple(v for v in selected_voices if isinstance(v, NghiTtsVoiceSpec))
    configured_nghitts = (
        nghitts_voices
        if all_voices is None
        else tuple(v for v in all_voices if isinstance(v, NghiTtsVoiceSpec))
    )
    if max_loaded_voices is not None and len(configured_nghitts) > len(nghitts_voices):
        return initialize_lazy_tts_voices(
            tts_path,
            nghitts_voices,
            configured_nghitts,
            cpu_threads,
            max_loaded_voices,
        )
    return initialize_tts_voices(
        tts_path,
        configured_nghitts,
        cpu_threads,
    )


def _select_startup_voices(
    configured: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
    recent_names: tuple[str, ...],
    max_voices: int | None,
) -> tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...]:
    """Select recent voices first, then configured defaults, within the device limit."""
    if max_voices == 0:
        return ()
    by_name = {voice.name: voice for voice in configured} | {
        voice.id: voice for voice in configured
    }
    selected: list[NghiTtsVoiceSpec | ZeroTtsVoiceSpec] = []
    for name in (*recent_names, *(voice.name for voice in configured)):
        voice = by_name.get(name)
        if voice is not None and voice not in selected:
            selected.append(voice)
    return tuple(selected if max_voices is None else selected[:max_voices])


def _select_loaded_voices(
    configured: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
    recent_names: tuple[str, ...],
    tts_engine: str,
    max_voices: int | None,
) -> tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...]:
    """Limit resident NghiTTS models by RAM while retaining all shared ZeroTTS voices."""
    if tts_engine == TtsEngine.ZEROTTS:
        return configured
    return _select_startup_voices(configured, recent_names, max_voices)


def _tts_cache_namespace(
    server_config: ServerConfig,
    voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
) -> bytes:
    """Fingerprint engine, sentence silence, and voice artifacts for cache safety."""
    digest = hashlib.sha256(server_config.tts_engine.encode("utf-8"))
    if server_config.tts_engine == TtsEngine.ZEROTTS:
        digest.update(ZEROTTS_MODEL.revision.encode("ascii"))
    digest.update(server_config.tts_sentence_silence_ms.to_bytes(4, "big"))
    for voice in sorted(voices, key=lambda v: v.id):
        digest.update(voice.id.encode("utf-8"))
        for artifact in sorted(voice.artifacts, key=lambda a: a.local_name):
            digest.update(artifact.sha256.encode("ascii"))
    return digest.digest()


def _remove_obsolete_cache_namespaces(base_dir: Path, current_namespace: str) -> None:
    """Remove persistent cache entries that belong to a previous engine or voice revision."""
    try:
        if not base_dir.is_dir():
            return
        children = list(base_dir.iterdir())
    except OSError as err:
        _LOGGER.debug("Could not list TTS cache directory %s: %s", base_dir, err)
        return
    for child in children:
        if child.name == current_namespace:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            elif child.suffix == TtsCacheFile.PCM_SUFFIX or child.name.startswith(
                TtsCacheFile.TEMP_PREFIX
            ):
                child.unlink(missing_ok=True)
        except OSError:
            _LOGGER.debug("Could not remove obsolete cache entry: %s", child)


def _initialize_tts_cache(
    server_config: ServerConfig,
    resources: DeviceResources,
    memory_cache_limits: tuple[int, int, int],
    tts_namespace: bytes,
) -> ResultCache:
    """Initialize persistent or RAM-only cache according to configuration and resources."""
    if not server_config.tts_cache_enabled:
        _LOGGER.info("TTS inference cache is disabled")
        return BoundedLruCache[bytes, bytes](
            max_entries=0,
            max_bytes=0,
            max_item_bytes=0,
            max_idle_seconds=0.0,
        )

    memory_entries, memory_bytes, memory_item_bytes = memory_cache_limits
    disk_entries, disk_bytes, disk_item_bytes, cache_idle_seconds = resources.disk_cache_limits()
    memory_cache = BoundedLruCache[bytes, bytes](
        max_entries=memory_entries,
        max_bytes=memory_bytes,
        max_item_bytes=memory_item_bytes,
        max_idle_seconds=cache_idle_seconds,
    )
    tts_namespace_label = tts_namespace.hex()[:16]
    tts_audio_base = server_config.cache_dir / TTS_CACHE_DIR
    _remove_obsolete_cache_namespaces(tts_audio_base, tts_namespace_label)
    try:
        tts_result_cache: ResultCache = PersistentAudioCache(
            tts_audio_base / tts_namespace_label,
            memory_cache,
            max_entries=disk_entries,
            max_bytes=disk_bytes,
            max_item_bytes=disk_item_bytes,
            max_idle_seconds=cache_idle_seconds,
        )
    except OSError as err:
        _LOGGER.warning("Could not initialize persistent TTS cache; using RAM-only cache: %s", err)
        tts_result_cache = memory_cache
    _LOGGER.info(
        "TTS inference cache: memory_tier=%s idle_seconds=%.0f, disk_entries=%d, "
        "disk_max_mb=%.1f, disk_max_item_mb=%.1f, ram_max_mb=%.1f",
        resources.tier,
        cache_idle_seconds,
        disk_entries,
        disk_bytes / (1024 * 1024),
        disk_item_bytes / (1024 * 1024),
        memory_bytes / (1024 * 1024),
    )
    return tts_result_cache


def _log_selected_voices(
    active_voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
    configured_voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
    tier: str,
) -> None:
    """Log startup voice selection and on-demand loading status."""
    _LOGGER.info(
        "Startup TTS voices selected by recent use: %s",
        ", ".join(voice.name for voice in active_voices),
    )
    if len(active_voices) < len(configured_voices):
        _LOGGER.info(
            "Memory profile %s: loading %d of %d configured NghiTTS voices at startup; "
            "remaining voices load on demand with LRU eviction",
            tier,
            len(active_voices),
            len(configured_voices),
        )


async def _warm_up_tts_safely(
    tts_engine: TTSEngine,
    active_voices: tuple[NghiTtsVoiceSpec | ZeroTtsVoiceSpec, ...],
    tts_engine_name: str,
) -> None:
    """Warm up TTS engine and ensure engine is closed if warm-up raises an exception."""
    try:
        await asyncio.to_thread(warm_up_tts, tts_engine, active_voices, tts_engine_name)
    except Exception:
        await asyncio.to_thread(tts_engine.close)
        raise


async def run_server(
    config: ServerConfig | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Initialize models, bind both services, and shut down gracefully."""
    server_config = config or ServerConfig.from_env()
    stt_model = get_stt_model(server_config.stt_engine)
    cpu_threads = resolve_cpu_threads(server_config.cpu_threads)
    _configure_logging(server_config.log_level)
    resources = detect_device_resources()
    memory_cache_limits = resources.ram_cache_limits()
    voice_usage_store = RecentVoiceStore(server_config.cache_dir / "recent_tts_voices.json")
    recent_voice_names = await asyncio.to_thread(voice_usage_store.get_recent)
    active_voices = _select_loaded_voices(
        server_config.tts_voices,
        recent_voice_names,
        server_config.tts_engine,
        resources.max_tts_voices,
    )
    warn_if_low_startup_memory(
        server_config.tts_engine,
        len(active_voices),
        resources.memory_bytes,
        resources.memory_capacity_bytes,
        memory_cache_limits[1] if server_config.tts_cache_enabled else 0,
    )
    _log_selected_voices(active_voices, server_config.tts_voices, str(resources.tier))
    _LOGGER.info("Starting Wyoming Vietnamese server initialization")

    server_config.cache_dir.mkdir(parents=True, exist_ok=True)
    server_config.download_dir.mkdir(parents=True, exist_ok=True)
    _log_startup_configuration(server_config, cpu_threads, stt_model)

    validate_tts_voices_for_engine(server_config.tts_voices, server_config.tts_engine)
    paths = download_models(
        cache_dir=server_config.cache_dir,
        download_dir=server_config.download_dir,
        tts_voices=server_config.tts_voices,
        offline=server_config.offline,
        tts_engine=server_config.tts_engine,
        stt_engine=server_config.stt_engine,
    )

    _LOGGER.info("Initializing Speech-to-Text engine (%s)", server_config.stt_engine)
    stt_recognizer = initialize_stt(
        paths[STT_DIR],
        cpu_threads,
    )
    await asyncio.to_thread(warm_up_stt, stt_recognizer)
    stt_info = get_stt_info(stt_model.repo, stt_model.revision, description=stt_model.description)

    _LOGGER.info("Initializing Text-to-Speech engine (%s)", server_config.tts_engine)
    tts_engine = _initialize_configured_tts(
        server_config,
        paths[TTS_DIR],
        cpu_threads,
        active_voices,
        all_voices=server_config.tts_voices,
        max_loaded_voices=resources.max_tts_voices,
    )
    await _warm_up_tts_safely(tts_engine, active_voices, server_config.tts_engine)
    tts_info = get_tts_info(server_config.tts_voices, engine=server_config.tts_engine)
    stt_info_event = stt_info.event()
    tts_info_event = tts_info.event()
    combined_info_event = combine_service_info(stt_info, tts_info).event()

    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    connection_limiter = ConnectionLimiter(server_config.max_active_connections)
    stt_buffer_budget = ByteBudget(server_config.max_stt_buffer_bytes)
    tts_namespace = _tts_cache_namespace(server_config, server_config.tts_voices)
    tts_result_cache = _initialize_tts_cache(
        server_config,
        resources,
        memory_cache_limits,
        tts_namespace,
    )
    inference_executor = create_inference_executor()
    stt_handler_factory: HandlerFactory = partial(
        SherpaSTTEventHandler,
        stt_recognizer,
        stt_info_event,
        stt_lock,
        stt_model.repo,
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
        voice_usage_store=voice_usage_store,
        cache_namespace=tts_namespace,
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
            "Cleared TTS RAM cache: entries=%d bytes=%d; persistent disk entries retained",
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
