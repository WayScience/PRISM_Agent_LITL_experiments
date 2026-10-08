"""
rate_limiter.py

File-based rate limiting.
Tool methods that involve API calls should invoke `.acquire()` before
    making the call to avoid hitting the API end rate limits. The file-based
    nature of this rate limiter allows it to work across multiple
    processes/threads, which is important for agentic experiments that
    are parallelized.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo
Added some guardrails against bad init parameters and corrupted state files, 
potential cross-platform compatibility issues.
Uses `fcntl.flock` on POSIX systems and `msvcrt.locking` on Windows at best
effort. Please note that this package is developed and tested exclusively on
POSIX systems; Windows support is not guaranteed.

Classes:
- FileBasedRateLimiter: enables cross-process/thread rate limiting 
    using file system locks
"""

import os
import json
import time
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import IO, Union, Callable, TypeVar, Protocol, Optional
from functools import wraps

logger = logging.getLogger(__name__)

FilePath = Union[str, Path]
F = TypeVar("F", bound=Callable[..., object])
_WINDOWS_LOCK_LENGTH = 1

# Persist wall time by default; legacy monotonic state is migrated under lock.
_TIME_FUNC: Callable[[], float] = time.time
_STATE_DIR: Optional[Path] = None


# --- Platform-specific imports ------------------------------------------------

if os.name == "nt":
    import msvcrt  # type: ignore[import]
    fcntl = None
elif os.name == "posix":
    import fcntl  # type: ignore[import]
    msvcrt = None
else:
    msvcrt = None
    fcntl = None
    raise RuntimeError(
        "FileBasedRateLimiter only supports POSIX or Windows "
        f"(os.name 'posix'/'nt'), got os.name='{os.name}'"
    )

# --- Locking helpers ----------------------------------------------------------


def _lock_file(f: IO[bytes]) -> None:
    """Cross-platform *exclusive* blocking file lock."""
    if fcntl is not None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return

    if msvcrt is not None:
        # Lock 1 byte from offset 0 to represent a lock on the whole file.
        pos = f.tell()
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, _WINDOWS_LOCK_LENGTH)
        finally:
            f.seek(pos)
        return

    # Should be unreachable with the RuntimeError above.
    logger.debug("File locking unavailable; proceeding without explicit lock")


def _unlock_file(f: IO[bytes]) -> None:
    """Release cross-platform file lock."""
    if fcntl is not None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return

    if msvcrt is not None:
        pos = f.tell()
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, _WINDOWS_LOCK_LENGTH)
        finally:
            f.seek(pos)
        return

    logger.debug("File locking unavailable; nothing to unlock")


# --- Config setter/getter -----------------------------------------------------


def set_default_time_func(
    time_func: Callable[[], float]
) -> None:
    
    if not callable(time_func):
        raise TypeError("time_func must be callable")

    global _TIME_FUNC
    _TIME_FUNC = time_func


def resolve_default_time_func() -> Callable[[], float]:
    return _TIME_FUNC


def set_default_state_dir(
    state_dir: FilePath
) -> None:
    
    if not isinstance(state_dir, (str, Path)):
        raise TypeError("state_dir must be a str or Path")
    
    if not Path(state_dir).is_dir():
        raise ValueError(f"state_dir '{state_dir}' is not a valid directory")

    global _STATE_DIR
    _STATE_DIR = Path(state_dir)


def resolve_default_state_dir() -> Optional[Path]:
    return _STATE_DIR


# --- Rate limiter -------------------------------------------------------------


class SupportsAcquireSync(Protocol):
    def acquire_sync(self) -> None: ...
    

def make_rate_limited_decorator(limiter: SupportsAcquireSync) -> Callable[[F], F]:
    """
    Given any limiter object with acquire_sync(), return a decorator that
    enforces the limiter before each call to the wrapped function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire_sync()
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorator


class FileBasedRateLimiter:
    """
    This rate limiter uses file system locking to coordinate rate limiting
    across multiple processes and threads.

    How it works:
    1. Request timestamps are stored as user specified time function 
        return values in a JSON file
        with file locking for thread safety
    2. Clock metadata allows legacy monotonic timestamps to be migrated
    3. Before each request, old timestamps outside the time window are removed
    4. If the request count exceeds the limit, the caller sleeps until the
        oldest request falls outside the time window
    5. New request timestamps are appended and the state is persisted
    6. If the state file is corrupted, it's automatically cleaned up and
        the rate limiter assumes full capacity
    7. Backward clock changes rebase pending requests rather than clearing them;
        elapsed time while waiting is always measured with a monotonic clock
    """

    def __init__(
        self, 
        max_requests: int = 3, 
        time_window: float = 1.0, 
        name: str = "default",
    ):
        """
        Initialize the rate limiter.
        
        :param max_requests: Maximum requests allowed in the time window
        :param time_window: Time window in seconds
        :param name: Name for the rate limiter (used in state file name)
        """
        if not isinstance(max_requests, int) or max_requests <= 0:
            raise ValueError(
                f"max_requests must be a positive integer, got {max_requests}"
            )
        if not isinstance(time_window, (int, float)) or time_window <= 0:
            raise ValueError(
                f"time_window must be a positive number, got {time_window}"
            )
        
        self.max_requests = max_requests
        self.time_window = time_window
        self.time_func = resolve_default_time_func()
        self.state_dir = resolve_default_state_dir()
        if self.state_dir is None:
            temp_dir = Path(tempfile.gettempdir())
            self.state_file = temp_dir / f"{name}_rate_limiter.json"
        else:
            self.state_file = Path(self.state_dir) / f"{name}_rate_limiter.json"

    async def acquire(self):
        """
        Acquire the rate limiter asynchronously.
        This method runs the synchronous acquire method in a thread pool
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._acquire_sync)

    def acquire_sync(self):
        self._acquire_sync()

    def _normalize_clock(self, data, now_ts, now_wall, now_monotonic):
        """Migrate built-in clocks and rebase backward jumps, under the lock.

        Older files have no clock tag. For the built-in clocks, infer their
        domain from proximity to the current wall/monotonic readings. Legacy
        monotonic files must originate on this host; they cannot be accurately
        translated using another host's uptime. Custom clocks retain their
        existing timestamp convention and must use a separate limiter name
        when changing that convention.
        """
        clock = (
            "wall" if self.time_func is time.time else
            "monotonic" if self.time_func is time.monotonic else "custom"
        )
        requests = data["requests"]
        source_clock = data.get("clock")
        if requests and clock != "custom":
            if source_clock is None:
                newest = max(requests)
                source_clock = (
                    "monotonic" if abs(now_monotonic - newest) < abs(now_wall - newest)
                    else "wall"
                )
            if source_clock in ("wall", "monotonic") and source_clock != clock:
                source_now = now_wall if source_clock == "wall" else now_monotonic
                offset = now_ts - source_now
                requests = [t + offset for t in requests]
                if "last_ts" in data:
                    data["last_ts"] += offset

        # A clock rollback is not proof of a reboot. Preserve request ages at
        # the last observation, bounding the remaining wait to one window.
        last_ts = max(data.get("last_ts", now_ts), max(requests, default=now_ts))
        if now_ts < last_ts:
            requests = [t + (now_ts - last_ts) for t in requests]
        data["requests"] = sorted(requests)
        data["clock"] = clock

    def _acquire_sync(self):
        """
        Acquire the rate limiter synchronously.
        This method uses file locking to ensure that only one process/thread
        can modify the state file at a time. Persist the configured clock,
        but use monotonic elapsed time while holding the lock and waiting.
        """
        try:
            if not self.state_file.exists():
                self.state_file.write_text(
                    json.dumps({"requests": [], "boot_wall_time": time.time()})
                )
        except (OSError, IOError) as e:
            logger.warning(
                f"Failed to create state file {self.state_file}: {e}. "
                "Proceeding without rate limiting for this request."
            )
            return
        
        try:
            with open(self.state_file, "r+") as f:
                _lock_file(f)
                try:
                    data = self._read_and_validate_state(f)
                    now_ts = self.time_func()
                    now_wall = time.time()
                    start_monotonic = time.monotonic()
                    self._normalize_clock(data, now_ts, now_wall, start_monotonic)

                    elapsed = 0.0
                    while True:
                        data["requests"] = [
                            t for t in data["requests"]
                            if (now_ts - t) + elapsed < self.time_window
                        ]
                        if len(data["requests"]) < self.max_requests:
                            break
                        # Also handle queues written with a higher request limit.
                        oldest = data["requests"][-self.max_requests]
                        wait = self.time_window - ((now_ts - oldest) + elapsed)
                        time.sleep(wait)
                        elapsed = time.monotonic() - start_monotonic

                    # Translate surviving ages back to the configured clock,
                    # even if it moved backward while we slept.
                    end_ts = self.time_func()
                    data["requests"] = [
                        end_ts - ((now_ts - t) + elapsed) for t in data["requests"]
                    ]
                    data["requests"].append(end_ts)
                    data["last_ts"] = end_ts
                    data["boot_wall_time"] = time.time()
                    self._write_state(f, data)
                finally:
                    _unlock_file(f)
        except (OSError, IOError) as e:
            logger.warning(
                f"Failed to access state file {self.state_file}: {e}. "
                "Proceeding without rate limiting for this request."
            )
            return

    def _read_and_validate_state(self, f):
        """
        Read and validate the state from the file.
        If corrupted, reset to empty state with full capacity.
        
        :param f: Open file handle
        :return: Validated state dictionary
        """
        try:
            f.seek(0)
            content = f.read().strip()
            
            # Remove any trailing null bytes or extra data
            if '\x00' in content:
                content = content[:content.index('\x00')]
            content = content.strip()
            
            if not content:
                logger.debug("Empty state file, initializing fresh state")
                return {"requests": [], "boot_wall_time": time.time()}
            
            data = json.loads(content)
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("State is not a dictionary")
            if "requests" not in data:
                raise ValueError("State missing 'requests' key")
            if not isinstance(data["requests"], list):
                raise ValueError("'requests' is not a list")
            
            # Validate timestamps are numbers
            for ts in data["requests"]:
                if not isinstance(ts, (int, float)):
                    raise ValueError(f"Invalid timestamp: {ts}")
            if "last_ts" in data and not isinstance(data["last_ts"], (int, float)):
                raise ValueError("Invalid last_ts timestamp")
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Corrupted state file detected: {e}. "
                "Resetting to fresh state with full capacity."
            )
            # Return fresh state, allowing full capacity
            return {"requests": [], "boot_wall_time": time.time()}

    def _write_state(self, f, data):
        """
        Write state to file with proper error handling.
        
        :param f: Open file handle
        :param data: State dictionary to write
        """
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        except (OSError, IOError) as e:
            logger.error(f"Failed to write state file: {e}")
            # Continue without updating state - conservative approach
            raise
