"""
cache_config.py

Cache configuration and global state management. 
See tool_cache.py for the main caching decorator and detailed motivation of 
    design. 
We intend to use persistent caching on all API/database tools
    related to compound queries, and will cache based on the query method
    arguments. To avoid caching redundant data that are just subsets of larger
    queries, we will use a fixed fetch limit for all API calls that return
    multiple results (e.g. 50). This will ensure that the same query with a
    different fetch limit will be cached separately.
This module provides programmatic and environment variable based configuration
    of global cache settings that apply to all tools decorated with @tool_cache.
Note that this only takes effect when configured before actually using tools 
    decorated with @tool_cache.

Exposes to the user:
- `set_default_cache_root` to programmatically set the cache root
- `set_cache_defaults` to programmatically set cache limit and expire
- `set_fetch_limit` to programmatically set the fixed API fetch limit
This enables the same set of cache to be re-usable across multiple runs

Provides to the decorator:
- `resolve_cache_root` to determine the effective cache root directory
- `resolve_global_size_limit` to determine the effective size limit
- `resolve_global_expire` to determine the effective expire (TTL, seconds)
so the decorator can resolve the global cache settings at runtime.
Generally, these global settings have lower precedence over decorator-end
    overrides.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# ---- Global state
_AGENTIC_CACHE_ROOT: Optional[Path] = None
_GLOBAL_CACHE_DEFAULTS = {
    "root": None,
    "size_limit_bytes": None,
    "expire": None,
}
_FETCH_LIMIT: Optional[int] = None


def set_default_cache_root(path: str | Path):
    """
    Programmatically set the default cache root (overrides env).
    Intended to be called by the user
    """
    global _AGENTIC_CACHE_ROOT
    _AGENTIC_CACHE_ROOT = Path(path)


def resolve_cache_root() -> Path:
    """
    Decide the effective cache root directory:
      1) programmatic global default if set
      2) environment variable AGENTIC_CACHE_DIR
      3) fallback to ~/.cache/agentic_tools
    Intended to be used internally by the decorator at runtime.
    """
    if _AGENTIC_CACHE_ROOT is not None:
        return _AGENTIC_CACHE_ROOT
    env = os.environ.get("AGENTIC_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "agentic_tools"


def set_cache_defaults(
        *, 
        size_limit_bytes: Optional[int] = None, 
        expire: Optional[float] = None
    ):
    """
    Programmatically set global defaults for cache size limit and expire 
        (TTL, seconds).
        - size_limit_bytes: int or None (None -> uses env or sys.maxsize)
        - expire: float seconds or None (None -> never expire)
    Intended to be called by the user.
    """
    if size_limit_bytes is not None and not isinstance(size_limit_bytes, int):
        raise TypeError("size_limit_bytes must be an int or None")
    if expire is not None and not isinstance(expire, (int, float)):
        raise TypeError("expire must be a number (seconds) or None")

    _GLOBAL_CACHE_DEFAULTS["size_limit_bytes"] = size_limit_bytes
    _GLOBAL_CACHE_DEFAULTS["expire"] = expire


def resolve_global_size_limit(default_from_decorator: Optional[int]) -> int:
    """
    Decide the effective size limit for diskcache:
      1) decorator argument if provided (not None)
      2) programmatic global default if set
      3) environment variable AGENTIC_CACHE_SIZE_LIMIT_BYTES
      4) fallback to sys.maxsize
    Intended to be used internally by the decorator at runtime.
    """
    if default_from_decorator is not None:
        return int(default_from_decorator)
    if _GLOBAL_CACHE_DEFAULTS["size_limit_bytes"] is not None:
        return int(_GLOBAL_CACHE_DEFAULTS["size_limit_bytes"])
    env_val = os.environ.get("AGENTIC_CACHE_SIZE_LIMIT_BYTES")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return int(sys.maxsize)


def resolve_global_expire(
        default_from_decorator: Optional[float]) -> Optional[float]:
    """
    Decide the effective expire (TTL, seconds):
      1) decorator argument if provided (not None)
      2) programmatic global default if set
      3) environment variable AGENTIC_CACHE_EXPIRE_SECS
      4) fallback to None (never expire)
    Intended to be used internally by the decorator at runtime.
    """
    if default_from_decorator is not None:
        return float(default_from_decorator)
    if _GLOBAL_CACHE_DEFAULTS["expire"] is not None:
        return float(_GLOBAL_CACHE_DEFAULTS["expire"])
    env_val = os.environ.get("AGENTIC_CACHE_EXPIRE_SECS")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    return None


def set_fetch_limit(n: int) -> None:
    """
    Programmatically set the fixed API fetch limit used for canonical caching.
    Intended to be called by the user.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("fetch limit must be a positive integer")
    global _FETCH_LIMIT
    _FETCH_LIMIT = n


def get_fetch_limit() -> int:
    """
    Resolve the canonical fetch limit from programmatic set, env, or fallback.
    Intended to be used internally by the tool methods at runtime.
    """
    global _FETCH_LIMIT
    if _FETCH_LIMIT is not None:
        return _FETCH_LIMIT
    env_val = os.environ.get("AGENTIC_TOOL_FETCH_LIMIT")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                _FETCH_LIMIT = n
                return n
        except ValueError:
            pass
    _FETCH_LIMIT = 50
    return _FETCH_LIMIT
