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
from typing import IO, Union

logger = logging.getLogger(__name__)

FilePath = Union[str, Path]
_WINDOWS_LOCK_LENGTH = 1


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


class FileBasedRateLimiter:
    """
    This rate limiter uses file system locking to coordinate rate limiting
    across multiple processes and threads.

    How it works:
    1. Request timestamps are stored as monotonic time values in a JSON file
        with file locking for thread safety
    2. Wall-clock time is recorded to detect reboots
    3. Before each request, old timestamps outside the time window are removed
    4. If the request count exceeds the limit, the caller sleeps until the
        oldest request falls outside the time window
    5. New request timestamps are appended and the state is persisted
    6. If the state file is corrupted, it's automatically cleaned up and
        the rate limiter assumes full capacity
    7. Reboots are detected by comparing wall-clock time; if a reboot occurred,
        the request queue is cleared automatically
    """

    def __init__(
        self, 
        max_requests: int = 3, 
        time_window: float = 1.0, 
        name: str = "default"
    ):
        """
        Initialize the rate limiter.
        
        :param max_requests: Maximum requests allowed in the time window
        :param time_window: Time window in seconds
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
        temp_dir = Path(tempfile.gettempdir())
        self.state_file = temp_dir / f"{name}_rate_limiter.json"
        # Store current wall-clock time to detect reboots
        self._init_wall_time = time.time()

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

    def _acquire_sync(self):
        """
        Acquire the rate limiter synchronously.
        This method uses file locking to ensure that only one process/thread
        can modify the state file at a time. Uses monotonic time for rate
        limiting and wall-clock time for reboot detection.
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
                    now_monotonic = time.monotonic()
                    now_wall = time.time()
                    
                    # Detect reboots: if wall-clock time went backward significantly,
                    # a reboot occurred. Clear the request queue.
                    boot_wall = data.get("boot_wall_time", now_wall)
                    if now_wall < boot_wall - 60:  # 60s threshold for clock adjustments
                        logger.info(
                            f"Reboot detected (wall time went back from {boot_wall} "
                            f"to {now_wall}). Clearing request queue."
                        )
                        data["requests"] = []
                        data["boot_wall_time"] = now_wall
                    elif abs(now_wall - boot_wall) > 3600 and data["requests"]:
                        # If more than 1 hour has passed, update boot time
                        # This helps with long-running processes
                        data["boot_wall_time"] = now_wall
                    
                    # Filter out old requests using monotonic time
                    data["requests"] = [
                        t for t in data["requests"] if now_monotonic - t < self.time_window
                    ]
                    
                    if len(data["requests"]) >= self.max_requests:
                        oldest = data["requests"][0]
                        wait = self.time_window - (now_monotonic - oldest)
                        if wait > 0:
                            time.sleep(wait)
                            now_monotonic = time.monotonic()
                            data["requests"] = [
                                t for t in data["requests"]\
                                    if now_monotonic - t < self.time_window
                            ]
                    
                    data["requests"].append(now_monotonic)
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
                return {"requests": []}
            
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
            
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Corrupted state file detected: {e}. "
                "Resetting to fresh state with full capacity."
            )
            # Return fresh state, allowing full capacity
            return {"requests": []}

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
        except (OSError, IOError) as e:
            logger.error(f"Failed to write state file: {e}")
            # Continue without updating state - conservative approach
            raise
