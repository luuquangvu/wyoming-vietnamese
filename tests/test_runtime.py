"""Server lifecycle and healthcheck tests."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from wyoming.event import Event, write_event
from wyoming.info import Info

from tests.helpers import make_reader, memory_writer, stream_writer
from wyoming_vietnamese import healthcheck
from wyoming_vietnamese.__main__ import _drain_inference, main, run_server
from wyoming_vietnamese.combined import CombinedEventHandler, combine_service_info
from wyoming_vietnamese.config import ServerConfig, resolve_cpu_threads
from wyoming_vietnamese.protocol import ConnectionLimiter
from wyoming_vietnamese.stt import get_stt_info
from wyoming_vietnamese.tts import get_tts_info
from wyoming_vietnamese.tts_model import DEFAULT_TTS_VOICE


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


class FakeServer:
    """Provide a test double for FakeServer."""

    def __init__(
        self, *, start_error: Exception | None = None, stop_error: Exception | None = None
    ):
        """Initialize the test double."""
        self.start_error = start_error
        self.stop_error = stop_error
        self.factory: object | None = None
        self.stopped = False

    async def start(self, factory: object) -> None:
        """Build test support for start."""
        self.factory = factory
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
        self.closed = False

    def close(self) -> None:
        """Build test support for close."""
        self.closed = True


async def test_healthcheck_accepts_combined_info_and_closes_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test healthcheck accepts combined info and closes writer."""
    info = combine_service_info(
        get_stt_info("owner/stt"),
        get_tts_info(DEFAULT_TTS_VOICE),
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
        "stt": tmp_path / "stt",
        "tts": tmp_path / "tts",
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
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test a request that outlives the drain window is reported instead of ignored."""
    monkeypatch.setattr("wyoming_vietnamese.__main__.SHUTDOWN_DRAIN_TIMEOUT", 0.01)
    lock = asyncio.Lock()
    await lock.acquire()
    try:
        with caplog.at_level(logging.WARNING, logger="wyoming_vietnamese"):
            await _drain_inference(lock, "STT")
    finally:
        lock.release()
    assert any("in-flight STT inference" in message for message in caplog.messages)


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
    warm_up.assert_called_once_with(tts)
    init_tts.assert_called_once_with(
        tmp_path / "tts",
        (DEFAULT_TTS_VOICE,),
        resolve_cpu_threads(0),
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
