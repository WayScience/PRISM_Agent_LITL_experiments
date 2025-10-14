"""
conftest.py

Shared pytest fixtures and utilities for tools with rate limit/cache testing.
"""

import pytest
import time
from dspy_litl_agentic_system.tools.rate_limiter import FileBasedRateLimiter


# Test timeout constant - can be imported in tests
TEST_TIMEOUT = 30


@pytest.fixture
def temp_rate_limiter():
    """
    Fixture that creates a rate limiter with a unique name and cleans up after.
    
    Yields:
        FileBasedRateLimiter: A rate limiter instance with 3 max requests 
            and 1.0 second time window.
    """
    name = f"test_{int(time.monotonic() * 1000000)}"
    limiter = FileBasedRateLimiter(max_requests=3, time_window=1.0, name=name)
    yield limiter
    # Cleanup
    if limiter.state_file.exists():
        limiter.state_file.unlink()


def make_request_process(args):
    """
    Helper function for multiprocess testing.
    Must be at module level for pickling by multiprocessing.
    
    Args:
        args: Tuple of (index, rate_limiter_name)
        
    Returns:
        Tuple of (index, duration) where duration is the time taken 
            to acquire the rate limiter.
    """
    i, name = args
    limiter = FileBasedRateLimiter(
        max_requests=3, 
        time_window=1.0, 
        name=name
    )
    start = time.monotonic()
    limiter.acquire_sync()
    duration = time.monotonic() - start
    return (i, duration)
