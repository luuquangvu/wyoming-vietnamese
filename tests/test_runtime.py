"""Server lifecycle and healthcheck tests."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from wyoming.event import Event, write_event
from wyoming.info import Info
from wyoming.server import AsyncServer, HandlerFactory

from tests.helpers import make_reader, memory_writer, stream_writer
from wyoming_vietnamese import __main__ as main_module
from wyoming_vietnamese import healthcheck
from wyoming_vietnamese.__main__ import (
    _create_server_handler_factory,
    _download_configured_models,
    _drain_inference,
    _initialize_configured_tts,
    _register_shutdown_signals,
    _remove_obsolete_cache_namespaces,
    _resolve_zerotts_identity,
    _run_cleanup_bounded,
    _select_loaded_voices,
    _select_startup_voices,
    _shutdown_server,
    _tts_cache_namespace,
    configure_process_environment,
    main,
    run_server,
)
from wyoming_vietnamese.cache import BoundedLruCache as RealBoundedLruCache
from wyoming_vietnamese.cache import PersistentAudioCache as RealPersistentAudioCache
from wyoming_vietnamese.combined import CombinedEventHandler, combine_service_info
from wyoming_vietnamese.config import ServerConfig
from wyoming_vietnamese.const import (
    PROGRAM_NAME,
    STT_DIR,
    TTS_CACHE_DIR,
    TTS_DIR,
    SttEngine,
    TtsEngine,
    ZeroTtsEnv,
    ZeroTtsFile,
    ZeroTtsVariant,
)
from wyoming_vietnamese.cpu import resolve_cpu_threads
from wyoming_vietnamese.protocol import ByteBudget, ConnectionLimiter
from wyoming_vietnamese.resources import DeviceResources, DeviceTier
from wyoming_vietnamese.stt import get_stt_info
from wyoming_vietnamese.stt_model import GIPFORMER_MODEL
from wyoming_vietnamese.tts import get_tts_info
from wyoming_vietnamese.tts_model import (
    DEFAULT_NGHITTS_VOICE,
    DEFAULT_ZEROTTS_VOICE,
    NGHITTS_VOICES_BY_ID,
    ZEROTTS_VOICES_BY_ID,
    ZeroTtsModelResolution,
)


def _event_bytes(event: Event) -> bytes:
    """Build test support for  event bytes."""
    output = io.BytesIO()
    write_event(event, output)
    return output.getvalue()


def _config(tmp_path: Path) -> ServerConfig:
    """Build test support for  config."""
    return ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
        }
    )


def test_startup_voice_selection_prefers_recent_with_default_fallback() -> None:
    """Select most recently used voices first and configured defaults as fallback."""
    first = DEFAULT_NGHITTS_VOICE
    second = NGHITTS_VOICES_BY_ID["ban-mai"]
    third = NGHITTS_VOICES_BY_ID["duy-onyx-moi"]
    configured = (first, second, third)
    assert _select_startup_voices(configured, (third.name, second.name), 1) == (third,)
    assert _select_startup_voices(configured, (), 1) == (first,)
    assert _select_startup_voices(configured, (third.name,), 2) == (third, first)
    assert _select_startup_voices(configured, (), 2) == (first, second)
    assert _select_startup_voices(configured, (third.id,), 1) == (third,)
    assert _select_startup_voices(configured, ("unknown",), 1) == (first,)
    assert _select_startup_voices(configured, (third.id,), 0) == ()
    assert _select_startup_voices(configured, (), None) == configured


def test_tts_cache_namespace_tracks_engine_and_voice_artifact_revision(
    tmp_path: Path,
) -> None:
    """Invalidate persistent results when the engine or a pinned voice artifact changes."""
    nghitts_config = _config(tmp_path)
    voice = DEFAULT_NGHITTS_VOICE
    initial = _tts_cache_namespace(nghitts_config, (voice,))
    changed_artifact = replace(
        voice,
        model=replace(voice.model, sha256="0" * 64),
    )
    assert _tts_cache_namespace(nghitts_config, (changed_artifact,)) != initial

    changed_sentence_silence = replace(
        nghitts_config,
        tts_sentence_silence_ms=nghitts_config.tts_sentence_silence_ms + 100,
    )
    assert _tts_cache_namespace(changed_sentence_silence, (voice,)) != initial

    changed_clause_silence = replace(
        nghitts_config,
        tts_clause_silence_ms=nghitts_config.tts_clause_silence_ms + 100,
    )
    assert _tts_cache_namespace(changed_clause_silence, (voice,)) == initial

    changed_paragraph_silence = replace(
        nghitts_config,
        tts_paragraph_silence_ms=nghitts_config.tts_paragraph_silence_ms + 100,
    )
    assert _tts_cache_namespace(changed_paragraph_silence, (voice,)) == initial

    zerotts_config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_ENGINE": "zerotts",
        }
    )
    zerotts_default_ns = _tts_cache_namespace(zerotts_config, (DEFAULT_ZEROTTS_VOICE,))
    assert zerotts_default_ns != initial

    # Codec precision and bundled codec/runtime revisions affect generated audio.
    int8_engine = Mock(codec_int8=True)
    fp32_engine = Mock(codec_int8=False)
    assert _tts_cache_namespace(
        zerotts_config, (DEFAULT_ZEROTTS_VOICE,), tts_engine=int8_engine
    ) != _tts_cache_namespace(zerotts_config, (DEFAULT_ZEROTTS_VOICE,), tts_engine=fp32_engine)

    # ZeroTTS namespace changes when GGUF model identity changes
    zerotts_q4_ns = _tts_cache_namespace(
        zerotts_config, (DEFAULT_ZEROTTS_VOICE,), zerotts_model_identity="q4_sha256"
    )
    zerotts_q8_ns = _tts_cache_namespace(
        zerotts_config, (DEFAULT_ZEROTTS_VOICE,), zerotts_model_identity="q8_sha256"
    )
    assert zerotts_q4_ns != zerotts_q8_ns
    assert zerotts_q4_ns != zerotts_default_ns

    # ZeroTTS namespace changes when zerotts_max_codebooks changes
    zerotts_med_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=8),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    zerotts_low_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=4),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    assert zerotts_med_ns != zerotts_low_ns

    # Initialized engine's max_codebooks takes precedence over config
    engine_mock = Mock()
    engine_mock.max_codebooks = 12
    engine_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=16),
        (DEFAULT_ZEROTTS_VOICE,),
        tts_engine=engine_mock,
    )
    expected_12_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=12),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    assert engine_ns == expected_12_ns

    # Out of bounds zerotts_max_codebooks are clamped
    clamped_low_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=1),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    assert clamped_low_ns == zerotts_low_ns

    clamped_high_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=32),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    assert clamped_high_ns == zerotts_default_ns

    # Unset max_codebooks falls back to quality resolution
    quality_low_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=None, zerotts_quality="low"),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    explicit_low_ns = _tts_cache_namespace(
        replace(zerotts_config, zerotts_max_codebooks=10),
        (DEFAULT_ZEROTTS_VOICE,),
    )
    assert quality_low_ns == explicit_low_ns

    # Non-ZeroTTS engines ignore zerotts_model_identity
    nghitts_with_id = _tts_cache_namespace(
        nghitts_config, (voice,), zerotts_model_identity="some_id"
    )
    assert nghitts_with_id == initial

    second_voice = NGHITTS_VOICES_BY_ID["chieu-thanh"]
    assert _tts_cache_namespace(nghitts_config, (voice, second_voice)) != _tts_cache_namespace(
        nghitts_config, (second_voice, voice)
    )


def test_remove_obsolete_cache_namespaces_cleans_stale_entries(tmp_path: Path) -> None:
    """Remove old namespace directories and legacy flat entries, keeping the current one."""
    base = tmp_path / "tts-audio"
    current = "abcdef0123456789"
    old_ns = "9876543210fedcba"
    fixture_files = {
        base / current / "entry.pcm": b"keep",
        base / old_ns / "stale.pcm": b"stale",
        base / "legacy.pcm": b"flat",
        base / ".tts-partial": b"tmp",
    }
    for file_path, content in fixture_files.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    _remove_obsolete_cache_namespaces(base, current)

    assert (base / current / "entry.pcm").read_bytes() == b"keep"
    assert not (base / old_ns).exists()
    assert not (base / "legacy.pcm").exists()
    assert not (base / ".tts-partial").exists()


def test_remove_obsolete_cache_namespaces_tolerates_missing_base(tmp_path: Path) -> None:
    """Do nothing when the base directory does not yet exist."""
    _remove_obsolete_cache_namespaces(tmp_path / "nonexistent", "any")


def test_initialize_configured_tts_preserves_configured_order_in_eager_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager initialization must retain configured voice order rather than recent-first order."""
    init_eager = Mock()
    init_lazy = Mock()
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_tts_voices", init_eager)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_lazy_tts_voices", init_lazy)

    voice1 = DEFAULT_NGHITTS_VOICE
    voice2 = NGHITTS_VOICES_BY_ID["ban-mai"]
    configured_order = (voice1, voice2)
    recent_order = (voice2, voice1)

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_VOICE": f"{voice1.id},{voice2.id}",
        }
    )

    _initialize_configured_tts(
        config,
        tmp_path / TTS_DIR,
        cpu_threads=1,
        voices=recent_order,
        all_voices=configured_order,
        max_loaded_voices=2,
    )
    init_eager.assert_called_once_with(tmp_path / TTS_DIR, configured_order, 1)
    init_lazy.assert_not_called()

    init_eager.reset_mock()
    _initialize_configured_tts(
        config,
        tmp_path / TTS_DIR,
        cpu_threads=1,
        voices=(voice2,),
        all_voices=configured_order,
        max_loaded_voices=1,
    )
    init_lazy.assert_called_once_with(
        tmp_path / TTS_DIR,
        (voice2,),
        configured_order,
        1,
        1,
    )
    init_eager.assert_not_called()


def test_loaded_voice_limits_are_specific_to_nghitts_memory_use() -> None:
    """Limit NghiTTS resident models by RAM while retaining every ZeroTTS voice."""
    nghitts_voices = (
        DEFAULT_NGHITTS_VOICE,
        NGHITTS_VOICES_BY_ID["ban-mai"],
        NGHITTS_VOICES_BY_ID["duy-onyx-moi"],
    )
    zerotts_voices = (
        DEFAULT_ZEROTTS_VOICE,
        ZEROTTS_VOICES_BY_ID["baotrang"],
        ZEROTTS_VOICES_BY_ID["giahuy"],
    )

    assert _select_loaded_voices(nghitts_voices, (), TtsEngine.NGHITTS, 1) == (
        DEFAULT_NGHITTS_VOICE,
    )
    assert _select_loaded_voices(zerotts_voices, (), TtsEngine.ZEROTTS, 1) == zerotts_voices


class FakeServer(AsyncServer):
    """Provide a test double for FakeServer."""

    def __init__(
        self, *, start_error: Exception | None = None, stop_error: Exception | None = None
    ) -> None:
        """Initialize the test double."""
        super().__init__()
        self.start_error = start_error
        self.stop_error = stop_error
        self.factory: object | None = None
        self.stopped = False

    async def run(self, handler_factory: HandlerFactory) -> None:
        """Build test support for run."""
        await self.start(handler_factory)

    async def start(self, handler_factory: HandlerFactory) -> None:
        """Build test support for start."""
        self.factory = handler_factory
        if self.start_error is not None:
            raise self.start_error

    async def stop(self) -> None:
        """Build test support for stop."""
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error


class ClosableTTS:
    """Provide a test double for ClosableTTS."""

    def __init__(self) -> None:
        """Initialize the test double."""
        self._preset_voices = {"voice": {"description": "Voice"}}
        self.sample_rate = 22050
        self.chunks_are_sentences = False
        self.closed = False

    def infer_stream(self, text: str, *, voice: str | None = None) -> Iterator[np.ndarray]:
        """Synthesize dummy waveform."""
        return iter(())

    def close(self) -> None:
        """Build test support for close."""
        self.closed = True


async def test_healthcheck_accepts_combined_info_and_closes_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test healthcheck accepts combined info and closes writer."""
    info = combine_service_info(
        get_stt_info("owner/stt"),
        get_tts_info(DEFAULT_NGHITTS_VOICE),
    )
    writer = stream_writer()
    reader = make_reader(_event_bytes(info.event()))
    monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
    assert await healthcheck.check_port(1234) is True
    assert memory_writer(writer).closed is True
    assert memory_writer(writer).waited is True


async def test_healthcheck_rejects_incomplete_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test healthcheck rejects incomplete discovery."""
    writer = stream_writer()
    reader = make_reader(_event_bytes(get_stt_info("owner/stt").event()))
    monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
    assert await healthcheck.check_port(1234) is False


async def test_healthcheck_rejects_wrong_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test healthcheck rejects wrong response."""
    writer = stream_writer()
    reader = make_reader(_event_bytes(Event(type="pong")))
    monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
    assert await healthcheck.check_port(1234) is False


async def test_healthcheck_contains_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test healthcheck contains connection failure."""
    monkeypatch.setattr(
        asyncio,
        "open_connection",
        AsyncMock(side_effect=ConnectionError("refused")),
    )
    assert await healthcheck.check_port(1234) is False


async def test_healthcheck_async_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test healthcheck async main."""
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(healthcheck, "check_port", check)
    monkeypatch.setenv("WYOMING_PORT", "12000")
    assert await healthcheck.async_main() == 0
    check.assert_awaited_once_with(12000)

    check.reset_mock()
    check.return_value = False
    assert await healthcheck.async_main() == 1
    check.assert_awaited_once_with(12000)


async def test_healthcheck_async_main_rejects_bad_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test healthcheck async main rejects bad port."""
    monkeypatch.setenv("WYOMING_PORT", "bad")
    assert await healthcheck.async_main() == 1


def test_healthcheck_main_uses_async_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test healthcheck main uses async exit code."""
    async_main = AsyncMock(return_value=1)
    monkeypatch.setattr(healthcheck, "async_main", async_main)
    with pytest.raises(SystemExit) as err:
        healthcheck.main()
    assert err.value.code == 1
    async_main.assert_awaited_once_with()


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    servers: list[FakeServer],
) -> tuple[ClosableTTS, Mock, Mock]:
    """Build test support for  patch runtime."""
    model_paths = {
        STT_DIR: tmp_path / STT_DIR,
        TTS_DIR: tmp_path / TTS_DIR,
    }
    tts = ClosableTTS()
    warm_up = Mock()
    init_tts = Mock(return_value=tts)
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.download_models", Mock(return_value=model_paths)
    )
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_stt", Mock(return_value=object()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_stt", Mock())
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_tts_voices", init_tts)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_zerotts", Mock(return_value=tts))
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_tts", warm_up)
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_stt_info", Mock(return_value=Info()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_tts_info", Mock(return_value=Info()))
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.AsyncServer.from_uri",
        Mock(side_effect=servers),
    )
    return tts, warm_up, init_tts


async def test_drain_inference_waits_for_an_active_request() -> None:
    """Test the shutdown drain releases the lock it borrows and bounds its wait."""
    lock = asyncio.Lock()
    await _drain_inference(lock, "TTS")
    assert lock.locked() is False

    await lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(_drain_inference(lock, "TTS"), timeout=0.05)
    finally:
        lock.release()


async def test_drain_inference_reports_a_stuck_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a request that outlives the drain window is reported instead of ignored."""
    lock = asyncio.Lock()
    await lock.acquire()
    try:
        with caplog.at_level(logging.WARNING, logger=PROGRAM_NAME):
            await _drain_inference(lock, "STT", timeout=0.01)
    finally:
        lock.release()
    assert any("in-flight STT inference" in message for message in caplog.messages)


async def test_run_cleanup_bounded_reports_timeout_and_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test blocking cleanup is bounded and failures are contained."""
    assert await _run_cleanup_bounded("quick", lambda: None, 1.0) is True

    with caplog.at_level(logging.WARNING, logger=PROGRAM_NAME):
        assert await _run_cleanup_bounded("slow", lambda: time.sleep(0.2), 0.01) is False
    assert any("slow cleanup did not finish" in message for message in caplog.messages)

    def broken_cleanup() -> None:
        """Raise to exercise the cleanup error path."""
        raise RuntimeError("cleanup failed")

    with caplog.at_level(logging.ERROR, logger=PROGRAM_NAME):
        assert await _run_cleanup_bounded("broken", broken_cleanup, 1.0) is True
    assert any("broken cleanup failed" in message for message in caplog.messages)


async def test_run_server_starts_one_combined_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server starts one combined service."""
    server = FakeServer()
    tts, warm_up, init_tts = _patch_runtime(monkeypatch, tmp_path, [server])
    executor = Mock()
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=executor),
    )
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(_config(tmp_path), stop_event)
    executor.shutdown.assert_called_once_with()
    assert server.factory is not None
    assert getattr(server.factory, "func", None) is CombinedEventHandler
    factory_keywords = getattr(server.factory, "keywords", None)
    assert isinstance(factory_keywords, dict)
    limiter = factory_keywords["connection_limiter"]
    assert isinstance(limiter, ConnectionLimiter)
    assert limiter.accepting is False
    assert server.stopped is True
    assert tts.closed is True
    warm_up.assert_called_once_with(tts, (DEFAULT_NGHITTS_VOICE,), TtsEngine.NGHITTS)
    init_tts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_NGHITTS_VOICE,),
        resolve_cpu_threads(0),
    )


def test_register_shutdown_signals() -> None:
    """Test _register_shutdown_signals attaches handlers to the event loop."""
    loop = Mock()
    shutdown_event = asyncio.Event()
    signals = _register_shutdown_signals(loop, shutdown_event)
    assert signals == [signal.SIGINT, signal.SIGTERM]
    assert loop.add_signal_handler.call_count == 2


def test_register_shutdown_signals_not_implemented() -> None:
    """Test _register_shutdown_signals gracefully handles NotImplementedError."""
    loop = Mock()
    loop.add_signal_handler.side_effect = NotImplementedError
    shutdown_event = asyncio.Event()
    signals = _register_shutdown_signals(loop, shutdown_event)
    assert signals == []


async def test_shutdown_server_when_not_started() -> None:
    """Test _shutdown_server skips stopping server if server_started is False."""
    server = FakeServer()
    limiter = ConnectionLimiter(5)
    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    cache = Mock()
    cache.clear.return_value = (2, 100)
    executor = Mock()
    tts_engine = ClosableTTS()
    loop = Mock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "get_running_loop", Mock(return_value=loop))
        await _shutdown_server(
            server,
            limiter,
            stt_lock,
            tts_lock,
            cache,
            executor,
            tts_engine,
            [signal.SIGINT],
            server_started=False,
        )

    loop.remove_signal_handler.assert_called_once_with(signal.SIGINT)
    assert server.stopped is False
    assert limiter.accepting is True
    cache.clear.assert_called_once_with()
    executor.shutdown.assert_called_once_with()
    assert tts_engine.closed is True


async def test_shutdown_server_when_started_clean() -> None:
    """Test _shutdown_server drains inference, stops server, and releases resources."""
    server = FakeServer()
    limiter = ConnectionLimiter(5)
    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    cache = Mock()
    cache.clear.return_value = (0, 0)
    executor = Mock()
    tts_engine = ClosableTTS()
    loop = Mock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "get_running_loop", Mock(return_value=loop))
        await _shutdown_server(
            server,
            limiter,
            stt_lock,
            tts_lock,
            cache,
            executor,
            tts_engine,
            [signal.SIGINT, signal.SIGTERM],
            server_started=True,
        )

    assert loop.remove_signal_handler.call_count == 2
    assert server.stopped is True
    assert limiter.accepting is False
    cache.clear.assert_called_once_with()
    executor.shutdown.assert_called_once_with()
    assert tts_engine.closed is True


async def test_shutdown_server_handles_stop_failure_and_idle_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _shutdown_server handles server.stop exception and wait_idle timeout gracefully."""
    server = FakeServer(stop_error=RuntimeError("stop crashed"))
    limiter = ConnectionLimiter(5)
    limiter.try_acquire()
    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    cache = Mock()
    cache.clear.return_value = (1, 50)
    executor = Mock()
    tts_engine = ClosableTTS()
    loop = Mock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "get_running_loop", Mock(return_value=loop))
        with caplog.at_level(logging.WARNING, logger=PROGRAM_NAME):
            await _shutdown_server(
                server,
                limiter,
                stt_lock,
                tts_lock,
                cache,
                executor,
                tts_engine,
                [],
                server_started=True,
            )

    assert server.stopped is True
    assert any("Server shutdown failed" in m for m in caplog.messages)
    assert any("Releasing models while" in m for m in caplog.messages)
    cache.clear.assert_called_once_with()
    executor.shutdown.assert_called_once_with()
    assert tts_engine.closed is True


async def test_shutdown_server_skips_tts_cleanup_when_executor_times_out(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test native TTS resources stay open while an executor task may still be running."""
    server = FakeServer()
    limiter = ConnectionLimiter(5)
    stt_lock = asyncio.Lock()
    tts_lock = asyncio.Lock()
    cache = Mock()
    cache.clear.return_value = (0, 0)
    executor = Mock()
    tts_engine = ClosableTTS()
    loop = Mock()
    bounded_cleanup = _run_cleanup_bounded

    async def cleanup(label: str, callback: Callable[[], None], timeout: float) -> bool:
        """Simulate an inference executor that does not stop in time."""
        if label == "inference executor":
            return False
        return await bounded_cleanup(label, callback, timeout)

    monkeypatch.setattr(main_module, "_run_cleanup_bounded", cleanup)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "get_running_loop", Mock(return_value=loop))
        with caplog.at_level(logging.WARNING, logger=PROGRAM_NAME):
            await _shutdown_server(
                server,
                limiter,
                stt_lock,
                tts_lock,
                cache,
                executor,
                tts_engine,
                [],
                server_started=False,
            )

    assert tts_engine.closed is False
    executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert any("Skipping TTS engine cleanup" in message for message in caplog.messages)


def test_resolve_zerotts_identity_cases(tmp_path: Path) -> None:
    """Test _resolve_zerotts_identity returns pinned or custom content digests."""
    assert _resolve_zerotts_identity(TtsEngine.NGHITTS, None) is None
    assert _resolve_zerotts_identity(TtsEngine.ZEROTTS, None) is None

    spec = Mock()
    spec.gguf_model = Mock(sha256="test-sha256")
    resolution = ZeroTtsModelResolution(spec=spec, gguf_path=tmp_path / "model.gguf")
    assert _resolve_zerotts_identity(TtsEngine.ZEROTTS, resolution) == "test-sha256"

    spec_no_gguf = Mock()
    spec_no_gguf.gguf_model = None
    first_model = tmp_path / "a" / "custom.gguf"
    first_model.parent.mkdir(parents=True)
    first_model.write_bytes(b"first model")
    second_model = tmp_path / "b" / "custom.gguf"
    second_model.parent.mkdir(parents=True)
    second_model.write_bytes(b"second model")
    first_resolution = ZeroTtsModelResolution(spec=spec_no_gguf, gguf_path=first_model)
    second_resolution = ZeroTtsModelResolution(spec=spec_no_gguf, gguf_path=second_model)
    first_identity = _resolve_zerotts_identity(TtsEngine.ZEROTTS, first_resolution)
    second_identity = _resolve_zerotts_identity(TtsEngine.ZEROTTS, second_resolution)
    assert first_identity == f"custom:{sha256(b'first model').hexdigest()}"
    assert second_identity == f"custom:{sha256(b'second model').hexdigest()}"
    assert first_identity != second_identity


def test_create_server_handler_factory(tmp_path: Path) -> None:
    """Test _create_server_handler_factory returns partial CombinedEventHandler."""
    config = _config(tmp_path)
    stt_rec = Mock()
    stt_event = Event(type="info")
    stt_lock = asyncio.Lock()
    stt_budget = ByteBudget(1024)
    tts_engine = ClosableTTS()
    tts_event = Event(type="info")
    tts_lock = asyncio.Lock()
    cache = Mock()
    store = Mock()
    executor = Mock()
    limiter = ConnectionLimiter(5)
    combined_event = Event(type="info")

    factory = _create_server_handler_factory(
        config,
        stt_rec,
        stt_event,
        stt_lock,
        "repo/stt",
        stt_budget,
        tts_engine,
        tts_event,
        tts_lock,
        cache,
        b"namespace",
        store,
        executor,
        limiter,
        combined_event,
    )
    assert getattr(factory, "func", None) is CombinedEventHandler
    args = getattr(factory, "args", ())
    assert len(args) == 3
    assert args[2] is combined_event
    keywords = getattr(factory, "keywords", {})
    assert keywords["connection_limiter"] is limiter


def test_download_configured_models_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _download_configured_models forwards zerotts_quant only when set."""
    download_mock = Mock(return_value={})
    monkeypatch.setattr("wyoming_vietnamese.__main__.download_models", download_mock)
    monkeypatch.setattr("wyoming_vietnamese.__main__.validate_tts_voices_for_engine", Mock())

    cfg_default = _config(tmp_path)
    _download_configured_models(cfg_default)
    assert "zerotts_quant" not in download_mock.call_args.kwargs

    download_mock.reset_mock()
    cfg_quant = replace(_config(tmp_path), tts_engine=TtsEngine.ZEROTTS, zerotts_model="q4_hybrid")
    _download_configured_models(cfg_quant)
    assert download_mock.call_args.kwargs["zerotts_quant"] == "q4_hybrid"


def test_configure_process_environment_sets_passive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test configure_process_environment sets OMP_WAIT_POLICY and GOMP_SPINCOUNT if unset."""
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("GOMP_SPINCOUNT", raising=False)
    configure_process_environment()
    assert os.environ.get("OMP_WAIT_POLICY") == "PASSIVE"
    assert os.environ.get("GOMP_SPINCOUNT") == "1000"


@pytest.mark.parametrize("existing_val", ["ACTIVE", "active", "PASSIVE", "passive", ""])
def test_configure_process_environment_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
    existing_val: str,
) -> None:
    """Test configure_process_environment preserves existing pre-configured OMP_WAIT_POLICY."""
    monkeypatch.setenv("OMP_WAIT_POLICY", existing_val)
    configure_process_environment()
    assert os.environ.get("OMP_WAIT_POLICY") == existing_val


async def test_run_server_applies_resource_limits_to_selected_recent_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use detected limits for cache construction and the recent voice for startup."""
    server = FakeServer()
    tts, warm_up, init_tts = _patch_runtime(monkeypatch, tmp_path, [server])
    recent_voice = NGHITTS_VOICES_BY_ID["duy-onyx-moi"]
    download = Mock(return_value={STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR})
    monkeypatch.setattr(main_module, "download_models", download)
    voice_store = Mock()
    voice_store.get_recent.return_value = (recent_voice.id, "unknown-voice")
    monkeypatch.setattr(main_module, "RecentVoiceStore", Mock(return_value=voice_store))
    resources = DeviceResources(
        cpu_count=2,
        memory_bytes=1024**3,
        memory_capacity_bytes=1024**3,
        tier=DeviceTier.WEAK,
    )
    monkeypatch.setattr(main_module, "detect_device_resources", Mock(return_value=resources))
    memory_caches: list[RealBoundedLruCache[bytes, bytes]] = []

    class RecordingBoundedLruCache(RealBoundedLruCache):
        """Capture generic cache construction while remaining subscriptable at runtime."""

        def __class_getitem__(cls, _params: object) -> type:
            """Allow ``cls[bytes, bytes]`` without consuming generic type parameters."""
            return cls

        def __init__(
            self,
            *,
            max_entries: int,
            max_bytes: int,
            max_item_bytes: int,
            max_idle_seconds: float,
        ) -> None:
            """Record the cache construction parameters for later inspection."""
            memory_caches.append(self)
            super().__init__(
                max_entries=max_entries,
                max_bytes=max_bytes,
                max_item_bytes=max_item_bytes,
                max_idle_seconds=max_idle_seconds,
            )

    disk_cache = Mock(side_effect=RealPersistentAudioCache)
    monkeypatch.setattr(main_module, "BoundedLruCache", RecordingBoundedLruCache)
    monkeypatch.setattr(main_module, "PersistentAudioCache", disk_cache)
    monkeypatch.setattr(
        main_module,
        "create_inference_executor",
        Mock(return_value=Mock()),
    )
    init_lazy = Mock(return_value=tts)
    monkeypatch.setattr(main_module, "initialize_lazy_tts_voices", init_lazy)

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_VOICE": "ngoc-huyen-moi,ban-mai,duy-onyx-moi",
        }
    )
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(config, stop_event)

    selected = (recent_voice,)
    download.assert_called_once_with(
        cache_dir=config.cache_dir,
        download_dir=config.download_dir,
        tts_voices=config.tts_voices,
        offline=False,
        tts_engine=TtsEngine.NGHITTS,
        stt_engine=SttEngine.ZIPFORMER,
    )
    init_tts.assert_not_called()
    init_lazy.assert_called_once_with(
        tmp_path / TTS_DIR,
        selected,
        config.tts_voices,
        resolve_cpu_threads(0),
        1,
    )
    warm_up.assert_called_once_with(tts, selected, TtsEngine.NGHITTS)
    assert len(memory_caches) == 1
    memory_cache = memory_caches[0]
    assert memory_cache.max_entries == 64
    assert memory_cache.max_bytes == 16 * 1024 * 1024
    assert memory_cache.max_item_bytes == 2 * 1024 * 1024
    assert memory_cache.max_idle_seconds == 2_592_000.0
    tts_namespace = _tts_cache_namespace(config, config.tts_voices)
    disk_cache.assert_called_once_with(
        config.cache_dir / TTS_CACHE_DIR / tts_namespace.hex()[:16],
        memory_cache,
        max_entries=256,
        max_bytes=128 * 1024 * 1024,
        max_item_bytes=8 * 1024 * 1024,
        max_idle_seconds=2_592_000.0,
    )
    handler_factory = server.factory
    factory_args = getattr(handler_factory, "args", None)
    assert isinstance(factory_args, tuple)
    tts_handler_factory = factory_args[1]
    factory_keywords = getattr(tts_handler_factory, "keywords", None)
    assert isinstance(factory_keywords, dict)
    assert factory_keywords["cache_namespace"] == (_tts_cache_namespace(config, config.tts_voices))


async def test_run_server_starts_zerotts_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server starts combined service with ZeroTTS engine."""
    server = FakeServer()
    tts = ClosableTTS()
    warm_up = Mock()
    init_zerotts = Mock(return_value=tts)
    init_nghitts = Mock(return_value=tts)
    download = Mock(return_value={STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR})
    get_tts_info_mock = Mock(return_value=Info())

    monkeypatch.setattr("wyoming_vietnamese.__main__.download_models", download)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_stt", Mock(return_value=object()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_stt", Mock())
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_tts_voices", init_nghitts)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_zerotts", init_zerotts)
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_tts", warm_up)
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_stt_info", Mock(return_value=Info()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_tts_info", get_tts_info_mock)
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.AsyncServer.from_uri",
        Mock(side_effect=[server]),
    )

    executor = Mock()
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=executor),
    )
    stop_event = asyncio.Event()
    stop_event.set()

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_ENGINE": "zerotts",
        }
    )
    await run_server(config, stop_event)

    executor.shutdown.assert_called_once_with()
    assert server.stopped is True
    assert tts.closed is True
    download.assert_called_once_with(
        cache_dir=config.cache_dir,
        download_dir=config.download_dir,
        tts_voices=(DEFAULT_ZEROTTS_VOICE,),
        offline=False,
        tts_engine=TtsEngine.ZEROTTS,
        stt_engine=SttEngine.ZIPFORMER,
        zerotts_quant="auto",
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        gguf_path=tmp_path / TTS_DIR / "gguf" / ZeroTtsFile.DEFAULT_GGUF_MODEL,
        max_codebooks=config.zerotts_max_codebooks,
    )
    init_nghitts.assert_not_called()
    warm_up.assert_called_once_with(tts, (DEFAULT_ZEROTTS_VOICE,), TtsEngine.ZEROTTS)
    get_tts_info_mock.assert_called_once_with(
        (DEFAULT_ZEROTTS_VOICE,),
        engine=TtsEngine.ZEROTTS,
        zerotts_model_name="auto",
        zerotts_quality="auto",
    )


async def _run_configured_zerotts_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download_result: object,
    env_overrides: dict[str, str] | None = None,
) -> tuple[FakeServer, ClosableTTS, Mock, Mock, ServerConfig]:
    """Mock ZeroTTS server dependencies and run the server until stopped."""
    server = FakeServer()
    tts = ClosableTTS()
    init_zerotts = Mock(return_value=tts)
    download = Mock(return_value=download_result)

    monkeypatch.setattr("wyoming_vietnamese.__main__.download_models", download)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_stt", Mock(return_value=object()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_stt", Mock())
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_zerotts", init_zerotts)
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_tts", Mock())
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_stt_info", Mock(return_value=Info()))
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_tts_info", Mock(return_value=Info()))
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.AsyncServer.from_uri",
        Mock(side_effect=[server]),
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=Mock()),
    )
    stop_event = asyncio.Event()
    stop_event.set()

    base_env = {
        "WYOMING_PORT": "10300",
        "CACHE_DIR": str(tmp_path / "cache"),
        "DOWNLOAD_DIR": str(tmp_path / "models"),
        "TTS_ENGINE": "zerotts",
    }
    if env_overrides:
        base_env |= env_overrides
    config = ServerConfig.from_env(base_env)
    await run_server(config, stop_event)
    return server, tts, init_zerotts, download, config


async def test_run_server_forwards_custom_zerotts_model_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server forwards zerotts_model to download_models and initialize_zerotts."""
    custom_gguf = tmp_path / "custom-zerotts.gguf"
    download_ret = {STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR}
    _server, _tts, init_zerotts, download, config = await _run_configured_zerotts_server(
        tmp_path,
        monkeypatch,
        download_ret,
        env_overrides={"ZEROTTS_MODEL": str(custom_gguf)},
    )

    download.assert_called_once_with(
        cache_dir=config.cache_dir,
        download_dir=config.download_dir,
        tts_voices=(DEFAULT_ZEROTTS_VOICE,),
        offline=False,
        tts_engine=TtsEngine.ZEROTTS,
        stt_engine=SttEngine.ZIPFORMER,
        zerotts_quant=str(custom_gguf),
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        gguf_path=custom_gguf,
        max_codebooks=config.zerotts_max_codebooks,
    )


async def test_run_server_auto_selects_zerotts_q4_on_edge_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server auto-selects Q4-Hybrid model on edge CPU architectures when unset."""
    download_ret = {STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR}
    monkeypatch.setattr(
        "wyoming_vietnamese.tts_model.detect_cpu_variant", lambda: ZeroTtsVariant.SSE4
    )
    _server, _tts, init_zerotts, _download, config = await _run_configured_zerotts_server(
        tmp_path,
        monkeypatch,
        download_ret,
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        gguf_path=tmp_path / TTS_DIR / "gguf" / ZeroTtsFile.Q4_HYBRID_GGUF_MODEL,
        max_codebooks=config.zerotts_max_codebooks,
    )


async def test_run_server_incorporates_zerotts_resolution_in_cache_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server passes resolved ZeroTTS GGUF artifact hash to cache namespace."""
    from wyoming_vietnamese.download import ModelDownloadResult
    from wyoming_vietnamese.tts_model import ZEROTTS_Q4_MODEL, ZeroTtsModelResolution

    res = ZeroTtsModelResolution(
        spec=ZEROTTS_Q4_MODEL,
        gguf_path=tmp_path / "zerotts-q4-hybrid.gguf",
    )
    download_result = ModelDownloadResult(
        tmp_path / STT_DIR,
        tmp_path / TTS_DIR,
        zerotts_resolution=res,
    )
    server, _tts, init_zerotts, _download, config = await _run_configured_zerotts_server(
        tmp_path,
        monkeypatch,
        download_result,
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        resolution=res,
        max_codebooks=config.zerotts_max_codebooks,
    )

    handler_factory = server.factory
    factory_args = getattr(handler_factory, "args", None)
    assert isinstance(factory_args, tuple)
    tts_handler_factory = factory_args[1]
    factory_keywords = getattr(tts_handler_factory, "keywords", None)
    assert isinstance(factory_keywords, dict)
    expected_ns = _tts_cache_namespace(
        config,
        config.tts_voices,
        zerotts_model_identity=(
            ZEROTTS_Q4_MODEL.gguf_model.sha256 if ZEROTTS_Q4_MODEL.gguf_model else None
        ),
    )
    assert factory_keywords["cache_namespace"] == expected_ns


async def test_run_server_forwards_zerotts_opportunistic_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server forwards zerotts_opportunistic=False to initialize_zerotts."""
    download_ret = {STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR}
    _server, _tts, init_zerotts, _download, config = await _run_configured_zerotts_server(
        tmp_path,
        monkeypatch,
        download_ret,
        env_overrides={ZeroTtsEnv.OPPORTUNISTIC: "false"},
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        gguf_path=tmp_path / TTS_DIR / "gguf" / ZeroTtsFile.DEFAULT_GGUF_MODEL,
        opportunistic=False,
        max_codebooks=config.zerotts_max_codebooks,
    )


async def test_run_server_forwards_zerotts_max_codebooks_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server forwards zerotts_max_codebooks setting to initialize_zerotts."""
    download_ret = {STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR}
    _server, _tts, init_zerotts, _download, _config = await _run_configured_zerotts_server(
        tmp_path,
        monkeypatch,
        download_ret,
        env_overrides={ZeroTtsEnv.MAX_CODEBOOKS: "10"},
    )
    init_zerotts.assert_called_once_with(
        tmp_path / TTS_DIR,
        (DEFAULT_ZEROTTS_VOICE,),
        resolve_cpu_threads(0),
        gguf_path=tmp_path / TTS_DIR / "gguf" / ZeroTtsFile.DEFAULT_GGUF_MODEL,
        max_codebooks=10,
    )


async def test_run_server_drains_stt_and_tts_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test shutdown starts both independent model drains before awaiting either one."""
    server = FakeServer()
    _patch_runtime(monkeypatch, tmp_path, [server])
    executor = Mock()
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=executor),
    )
    started: set[str] = set()
    both_started = asyncio.Event()

    async def drain_together(_lock: asyncio.Lock, label: str) -> None:
        """Wait for the other model drain so sequential execution cannot pass."""
        started.add(label)
        if started == {"STT", "TTS"}:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)

    monkeypatch.setattr("wyoming_vietnamese.__main__._drain_inference", drain_together)
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(_config(tmp_path), stop_event)

    assert started == {"STT", "TTS"}
    assert server.stopped is True
    executor.shutdown.assert_called_once_with()


async def test_run_server_cleans_up_after_bind_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server cleans up after bind failure."""
    server = FakeServer(start_error=OSError("address in use"))
    tts, _, _ = _patch_runtime(monkeypatch, tmp_path, [server])
    with pytest.raises(OSError, match="address in use"):
        await run_server(_config(tmp_path), asyncio.Event())
    assert server.stopped is False
    assert tts.closed is True


async def test_run_server_contains_shutdown_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server contains shutdown errors."""
    server = FakeServer(stop_error=RuntimeError("stop failed"))
    tts, _, _ = _patch_runtime(monkeypatch, tmp_path, [server])
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(_config(tmp_path), stop_event)
    assert tts.closed is True


async def test_run_server_closes_tts_after_warm_up_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server closes tts after warm up failure."""
    server = FakeServer()
    tts, warm_up, _ = _patch_runtime(monkeypatch, tmp_path, [server])
    warm_up.side_effect = RuntimeError("warm-up failed")

    with pytest.raises(RuntimeError, match="warm-up failed"):
        await run_server(_config(tmp_path), asyncio.Event())

    assert server.factory is None
    assert tts.closed is True


async def test_run_server_with_tts_cache_disabled_creates_no_disk_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling TTS caching avoids persistent cache construction and disk storage."""
    server = FakeServer()
    _patch_runtime(monkeypatch, tmp_path, [server])
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=Mock()),
    )
    persistent_cache_mock = Mock()
    monkeypatch.setattr("wyoming_vietnamese.__main__.PersistentAudioCache", persistent_cache_mock)

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_CACHE_ENABLED": "false",
        }
    )
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(config, stop_event)

    persistent_cache_mock.assert_not_called()
    assert (tmp_path / "cache" / TTS_CACHE_DIR).exists() is False

    handler_factory = server.factory
    factory_args = getattr(handler_factory, "args", None)
    assert isinstance(factory_args, tuple)
    tts_handler_factory = factory_args[1]
    tts_args = getattr(tts_handler_factory, "args", None)
    assert isinstance(tts_args, tuple)
    tts_cache = tts_args[3]
    assert tts_cache.enabled is False


async def test_run_server_with_strong_tier_loads_all_voices_eagerly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Device tier with no voice cap forces all voices to load eagerly at startup."""
    server = FakeServer()
    tts, _, init_tts = _patch_runtime(monkeypatch, tmp_path, [server])
    init_lazy = Mock()
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_lazy_tts_voices", init_lazy)
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=Mock()),
    )
    resources = DeviceResources(
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        memory_capacity_bytes=16 * 1024**3,
        tier=DeviceTier.STRONG,
    )
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.detect_device_resources", Mock(return_value=resources)
    )

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "TTS_VOICE": "ban-mai,chieu-thanh",
        }
    )
    stop_event = asyncio.Event()
    stop_event.set()
    await run_server(config, stop_event)

    init_tts.assert_called_once()
    init_lazy.assert_not_called()
    assert tts.closed is True


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("boom")])
def test_main_handles_terminal_failures(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test main handles terminal failures."""
    monkeypatch.setattr("wyoming_vietnamese.__main__.asyncio.run", Mock(side_effect=error))
    if isinstance(error, KeyboardInterrupt):
        assert main() is None
    else:
        with pytest.raises(SystemExit) as exit_error:
            main()
        assert exit_error.value.code == 1


async def test_run_server_starts_gipformer_stt_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test run server starts combined service with Gipformer STT engine."""
    server = FakeServer()
    tts = ClosableTTS()
    warm_up_stt_mock = Mock()
    init_stt_mock = Mock(return_value=object())
    download = Mock(return_value={STT_DIR: tmp_path / STT_DIR, TTS_DIR: tmp_path / TTS_DIR})
    get_stt_info_mock = Mock(return_value=Info())

    monkeypatch.setattr("wyoming_vietnamese.__main__.download_models", download)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_stt", init_stt_mock)
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_stt", warm_up_stt_mock)
    monkeypatch.setattr("wyoming_vietnamese.__main__.initialize_tts_voices", Mock(return_value=tts))
    monkeypatch.setattr("wyoming_vietnamese.__main__.warm_up_tts", Mock())
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_stt_info", get_stt_info_mock)
    monkeypatch.setattr("wyoming_vietnamese.__main__.get_tts_info", Mock(return_value=Info()))
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.AsyncServer.from_uri",
        Mock(side_effect=[server]),
    )

    executor = Mock()
    monkeypatch.setattr(
        "wyoming_vietnamese.__main__.create_inference_executor",
        Mock(return_value=executor),
    )
    stop_event = asyncio.Event()
    stop_event.set()

    config = ServerConfig.from_env(
        {
            "WYOMING_PORT": "10300",
            "CACHE_DIR": str(tmp_path / "cache"),
            "DOWNLOAD_DIR": str(tmp_path / "models"),
            "STT_ENGINE": "gipformer",
        }
    )
    await run_server(config, stop_event)

    download.assert_called_once_with(
        cache_dir=config.cache_dir,
        download_dir=config.download_dir,
        tts_voices=config.tts_voices,
        offline=False,
        tts_engine=TtsEngine.NGHITTS,
        stt_engine=SttEngine.GIPFORMER,
    )
    init_stt_mock.assert_called_once_with(
        tmp_path / STT_DIR,
        resolve_cpu_threads(0),
    )
    warm_up_stt_mock.assert_called_once()
    get_stt_info_mock.assert_called_once_with(
        GIPFORMER_MODEL.repo,
        GIPFORMER_MODEL.revision,
        description=GIPFORMER_MODEL.description,
    )
