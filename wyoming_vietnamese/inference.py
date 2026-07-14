"""Dedicated worker pool for blocking model inference calls.

Audio conversion and model inference both leave the event loop, but they have very
different durations. Sharing one pool lets a burst of short conversions queue ahead of
a synthesis step that is holding the shared TTS inference lock, so inference is given
its own small pool and conversion keeps using the default executor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial

from .const import INFERENCE_EXECUTOR_WORKERS


def create_inference_executor(
    max_workers: int = INFERENCE_EXECUTOR_WORKERS,
) -> ThreadPoolExecutor:
    """Create the pool that isolates model inference from short audio conversions."""
    if max_workers < 1:
        raise ValueError("inference worker count must be positive")
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="inference")


async def run_inference[T](
    executor: Executor | None,
    func: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """Run one blocking inference call off the event loop.

    A ``None`` executor falls back to the default thread pool so handlers stay usable
    without a configured server process.
    """
    if executor is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))
