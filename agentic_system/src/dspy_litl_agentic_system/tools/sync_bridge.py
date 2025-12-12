# sync_bridge.py

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")

_EXECUTOR = ThreadPoolExecutor(max_workers=8)

def run_async_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from sync code.

    - If no event loop is running in this thread: uses asyncio.run().
    - If an event loop IS running (e.g., Jupyter): runs coro in a new thread
      with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread → safe to use asyncio.run
        return asyncio.run(coro)

    # Running loop exists in this thread → run in another thread
    fut = _EXECUTOR.submit(asyncio.run, coro)
    return fut.result()
