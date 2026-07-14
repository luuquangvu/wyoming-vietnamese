"""Dedicated inference executor tests."""

from __future__ import annotations

import threading

import pytest

from wyoming_vietnamese.inference import create_inference_executor, run_inference


def _worker_identity(marker: str) -> tuple[str, int]:
    """Return the calling thread identity for one recorded inference call."""
    return marker, threading.get_ident()


async def test_run_inference_uses_the_default_pool_without_an_executor() -> None:
    """Test handlers stay usable when no dedicated pool is configured."""
    marker, thread_id = await run_inference(None, _worker_identity, "default")
    assert marker == "default"
    assert thread_id != threading.get_ident()


async def test_run_inference_isolates_calls_on_the_configured_pool() -> None:
    """Test inference runs on the dedicated pool and forwards keyword arguments."""
    executor = create_inference_executor(max_workers=1)
    try:
        _, first = await run_inference(executor, _worker_identity, marker="one")
        _, second = await run_inference(executor, _worker_identity, "two")
        assert first == second
        assert first != threading.get_ident()
    finally:
        executor.shutdown()


def test_create_inference_executor_rejects_an_empty_pool() -> None:
    """Test an unusable pool size is refused instead of silently deadlocking."""
    with pytest.raises(ValueError, match="worker count must be positive"):
        create_inference_executor(max_workers=0)
