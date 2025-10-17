"""
temp.py

Temporary fixture file for ChEMBL tools as the actual
    implementations are now under PR review.
Only for the current PR review, will be replaced with actual
    implementations once all the relevant PRs are merged.
"""
import time
from functools import wraps


class DummyRateLimiter:
    """Dummy rate limiter needed for the tools to work for now"""

    def acquire_sync(self) -> None:
        """Fixed small sleep to prevent triggering rate limits."""
        time.sleep(0.3)  # 0.3s sleep ~ 3 req/sec


def dummy_cache_decorator(name: str, **kwargs):
    """
    Dummy no-op cache decorator needed for the tools to work for now.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                kwargs.pop("_force_refresh")
            except Exception:
                pass
            return func(*args, **kwargs)
        return wrapper
    
    return decorator


def dummy_fetch_limit() -> int:
    """
    Dummy fetch limit function needed for the tools to work for now.
    """
    return 10
