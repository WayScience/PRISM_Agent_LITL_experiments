"""
tool_utils.py

General utility functions and classes for tools.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo
Modified to make disk cache persistent and location configurable so the query 
speeds up over experiments that runs on the same PRISM dataset and tools.

This module sets up a file-based caching system for API calls made to
external databases/services such as chembl and pubchem, as well as
a file-based rate limiter to prevent exceeding API rate limits when
multiple processes/threads are making requests concurrently. 
The caching system benefits the experiment as we 

CACHING SYSTEM:
- Implements a configurable disk-based cache using diskcache that persists across
    experiment runs, dramatically speeding up repeated API calls on the same datasets
- Features late-binding directory resolution, allowing cache locations to be set
    globally or per-call without breaking existing cached data
- Supports cache versioning through function fingerprinting and manual versioning
- Provides cache management utilities (stats, clear, import/export) for debugging
    and cache transfer between environments
- Handles serialization edge cases gracefully with fallback strategies

RATE LIMITING:
- File-based rate limiter coordinates API request throttling across multiple
    processes/threads, preventing rate limit violations in concurrent experiments
- Uses file locking to ensure thread-safe coordination when multiple agents
    are making simultaneous API calls to external databases/services

FOUNDATION FOR AGENTIC TOOLS:
This infrastructure enables building sophisticated agentic tools by:
1. Eliminating redundant API calls through intelligent caching, making iterative
     agent development and experimentation much faster
2. Preventing API rate limit violations in multi-agent or parallel experiment scenarios
3. Providing configurable, environment-aware defaults that work across different
     deployment contexts (local dev, containers, clusters)
4. Supporting offline-only modes for testing and development without external dependencies

Tools built on this foundation can focus on their core logic rather than handling
caching, rate limiting, and cross-process coordination concerns.
"""

import os
import sys
import json
import time
import asyncio
import inspect
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict
import diskcache
from functools import wraps
import fcntl

# ---- Global default & registry for cache instances
_AGENTIC_CACHE_ROOT: Optional[Path] = None
_CACHE_REGISTRY: Dict[str, diskcache.Cache] = {}
_GLOBAL_CACHE_DEFAULTS = {
    "root": None,
    "size_limit_bytes": None,
    "expire": None,
}
_FETCH_LIMIT: int | None = None

def set_default_cache_root(path: str | Path):
    """Programmatically set the default cache root (overrides env)."""
    global _AGENTIC_CACHE_ROOT
    _AGENTIC_CACHE_ROOT = Path(path)

def _resolve_cache_root() -> Path:
    """
    Decide the effective cache root directory:
      1) programmatic global default if set
      2) environment variable AGENTIC_CACHE_DIR
      3) fallback to ~/.cache/agentic_tools
    """
    if _AGENTIC_CACHE_ROOT is not None:
        return _AGENTIC_CACHE_ROOT
    env = os.environ.get("AGENTIC_CACHE_DIR")
    if env:
        return Path(env)
    # Fallback to home cache if nothing else provided
    return Path.home() / ".cache" / "agentic_tools"

def _resolve_global_size_limit(default_from_decorator: int | None) -> int:
    """
    Decide the effective size limit for diskcache:
      1) decorator argument if provided (not None)
      2) programmatic global default if set
      3) environment variable AGENTIC_CACHE_SIZE_LIMIT_BYTES
      4) fallback to sys.maxsize
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
    return int(sys.maxsize)  # safe, effectively “unlimited”

def _resolve_global_expire(default_from_decorator: float | None) -> float | None:
    """
    Decide the effective expire (TTL, seconds):
      1) decorator argument if provided (not None)
      2) programmatic global default if set
      3) environment variable AGENTIC_CACHE_EXPIRE_SECS
      4) fallback to None (never expire)
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
    return None  # never expire

def set_cache_defaults(*, size_limit_bytes: int | None = None, expire: float | None = None):
    """
    Programmatically set global defaults for cache size limit and expire 
        (TTL, seconds).
    - size_limit_bytes: int or None (None -> uses env or sys.maxsize)
    - expire: float seconds or None (None -> never expire)
    """
    if size_limit_bytes is not None and not isinstance(size_limit_bytes, int):
        raise TypeError("size_limit_bytes must be an int or None")
    if expire is not None and not isinstance(expire, (int, float)):
        raise TypeError("expire must be a number (seconds) or None")

    _GLOBAL_CACHE_DEFAULTS["size_limit_bytes"] = size_limit_bytes
    _GLOBAL_CACHE_DEFAULTS["expire"] = expire

def _get_cache(directory: Path, size_limit: Optional[int]) -> diskcache.Cache:
    """
    Get or create a diskcache.Cache instance for the given directory, with the
    given size limit. Caches are singletons per directory path.
    Creates the directory if it does not exist.
    Uses _CACHE_REGISTRY to avoid multiple instances for the same directory.
    """
    key = str(directory.resolve())
    if key not in _CACHE_REGISTRY:
        directory.mkdir(parents=True, exist_ok=True)
        eff_limit = _resolve_global_size_limit(size_limit)  # ensures int, not None
        _CACHE_REGISTRY[key] = diskcache.Cache(
            directory=str(directory), size_limit=eff_limit)
    return _CACHE_REGISTRY[key]

def _fingerprint_func(func: Callable) -> str:
    """
    Create a short fingerprint of the function source code for cache versioning.
    """
    try:
        src = inspect.getsource(func)
    except Exception:
        src = func.__name__
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]

def _default_key_fn(
    func: Callable, 
    args: Tuple[Any, ...], 
    kwargs: dict, *, 
    version: str, 
    tag: Optional[str]
) -> str:
    """
    Default key function: SHA256 of JSON-serialized payload of:
      - version
      - function module + qualname
      - args
      - kwargs (sorted by key)
      - tag (optional)
    """
    try:
        payload = {
            "v": version,
            "func": func.__module__ + "." + func.__qualname__,
            "args": args,
            "kwargs": kwargs,
            "tag": tag,
        }
        text = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        text = json.dumps(
            {
                "v": version,
                "func": func.__module__ + "." + func.__qualname__,
                "args": [repr(a) for a in args],
                "kwargs": {k: repr(v) for k, v in sorted(kwargs.items())},
                "tag": tag,
            },
            sort_keys=True,
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def set_fetch_limit(n: int) -> None:
    """
    Programmatically set the fixed API fetch limit used for canonical caching.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("fetch limit must be a positive integer")
    global _FETCH_LIMIT
    _FETCH_LIMIT = n

def get_fetch_limit() -> int:
    """
    Resolve the canonical fetch limit from programmatic set, env, or fallback.
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
    _FETCH_LIMIT = 50  # sensible default
    return _FETCH_LIMIT

def tool_cache(
    name: str,
    *,
    base_dir: Optional[Path | str] = None, 
    size_limit_bytes: Optional[int] = None,
    expire: Optional[float] = None,
    offline_only: bool = False,
    cache_version: str = "1",
    include_func_fingerprint: bool = True,
    tag: Optional[str] = None,
    key_fn: Optional[Callable[[Callable, tuple, dict], str]] = None,
):
    """
    Persistent, portable disk cache decorator.

    Late-binding rules:
    - If _cache_dir is passed at call time, use it.
    - Else if decorator base_dir was provided, use it.
    - Else resolve global root at call time via 
        set_default_cache_root()/AGENTIC_CACHE_DIR, then append {name}.

    Other per-call kwargs:
    - _cache_expire_override: override TTL for this write
    - _offline_only: force offline behavior for this call (bool)
    """
    # No default_root / base_dir computation at decoration time.

    def _resolve_effective_dir(call_override: Optional[str | Path]) -> Path:
        if call_override:
            return Path(call_override)
        if base_dir is not None:
            return Path(base_dir)
        return _resolve_cache_root() / name  # late-binding to current global root

    def decorator(func):
        func_fp = _fingerprint_func(func) if include_func_fingerprint else "na"
        version_str = f"{cache_version}+{func_fp}" if include_func_fingerprint else cache_version

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Per-call overrides
            call_cache_dir = kwargs.pop("_cache_dir", None)
            call_offline_only = kwargs.pop("_offline_only", None)
            call_expire = kwargs.pop("_cache_expire_override", None)

            # ---- LATE-BINDING directory resolution (do not override afterward)
            cache_dir = _resolve_effective_dir(call_cache_dir)

            cache = _get_cache(cache_dir, size_limit_bytes)

            # Build key
            kf = key_fn or (lambda f, a, kw: _default_key_fn(
                f, a, kw, version=version_str, tag=tag
            ))
            key = kf(func, args, kwargs)

            # Read
            try:
                if key in cache:
                    return cache[key]
            except Exception:
                pass

            # Miss behavior
            oo = (call_offline_only if call_offline_only is not None else offline_only)
            if oo:
                raise KeyError(
                    f"Cache miss in offline_only mode for key={key[:10]}… (cache={cache_dir})."
                )

            # Compute + write
            result = func(*args, **kwargs)
            effective_expire = _resolve_global_expire(expire)
            ttl = effective_expire if call_expire is None else call_expire

            try:
                cache.set(key, result, expire=ttl)
            except Exception:
                import json
                try:
                    cache.set(key, json.loads(json.dumps(result, default=str)), expire=ttl)
                except Exception:
                    cache.set(key, str(result), expire=ttl)
            return result

        # ---- Helper methods should also late-bind the directory
        def _dir_from_optional(path: Optional[str | Path]) -> Path:
            if path:
                return Path(path)
            if base_dir is not None:
                return Path(base_dir)
            return _resolve_cache_root() / name

        def cache_stats(path: Optional[str | Path] = None):
            d = _dir_from_optional(path)
            c = _get_cache(d, size_limit_bytes)
            return {
                "name": name,
                "directory": str(d),
                "size_limit_bytes": c.size_limit,
                "bytes": c.volume(),
                "count": len(c),
                "version": version_str,
                "tag": tag,
            }

        def cache_clear(confirm: bool = False, path: Optional[str | Path] = None):
            if not confirm:
                raise RuntimeError("Pass confirm=True to clear the cache.")
            d = _dir_from_optional(path)
            c = _get_cache(d, size_limit_bytes)
            c.clear()

        def cache_export(path: str | Path, source_dir: Optional[str | Path] = None):
            d = _dir_from_optional(source_dir)
            c = _get_cache(d, size_limit_bytes)
            p = Path(path)
            with p.open("w", encoding="utf-8") as f:
                for k in c:
                    try:
                        v = c[k]
                        f.write(json.dumps({"k": k, "v": v}, default=str) + "\n")
                    except Exception:
                        f.write(json.dumps({"k": k, "v": str(c[k])}) + "\n")
            return str(p)

        def cache_import(path: str | Path, dest_dir: Optional[str | Path] = None):
            d = _dir_from_optional(dest_dir)
            c = _get_cache(d, size_limit_bytes)
            p = Path(path)
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        c.set(row["k"], row["v"], expire=None)
                    except Exception:
                        continue

        wrapper.cache_stats = cache_stats
        wrapper.cache_clear = cache_clear
        wrapper.cache_export = cache_export
        wrapper.cache_import = cache_import
        wrapper.set_default_cache_root = set_default_cache_root  # expose setter

        return wrapper

    return decorator


class FileBasedRateLimiter:
    """
    A simple file-based rate limiter to limit the number of requests in a 
    time window. Prevents exceeding rate limits when multiple
    processes/threads are making requests concurrently.

    This rate limiter uses file system locking to coordinate rate limiting 
    across multiple processes and threads, (i.e. running multiple experiments
    with agents calling API calls to databases simultaneously.

    How it works:
    1. Request timestamps are stored in a JSON file with file locking for 
        thread safety
    2. Before each request, old timestamps outside the time window are removed
    3. If the request count exceeds the limit, the caller sleeps until the 
        oldest request falls outside the time window
    4. New request timestamps are appended and the state is persisted
    """
    def __init__(
            self, 
            max_requests: int = 3, 
            time_window: float = 1.0, 
            name: str = "default"
        ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.state_file = Path(f"/tmp/{name}_rate_limiter.json")

    async def acquire(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._acquire_sync)

    def acquire_sync(self):
        self._acquire_sync()

    def _acquire_sync(self):
        if not self.state_file.exists():
            self.state_file.write_text(json.dumps({"requests": []}))
        with open(self.state_file, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = json.load(f)
                now = time.time()
                data["requests"] = [
                    t for t in data["requests"] if now - t < self.time_window]
                if len(data["requests"]) >= self.max_requests:
                    oldest = data["requests"][0]
                    wait = self.time_window - (now - oldest)
                    if wait > 0:
                        time.sleep(wait)
                        now = time.time()
                        data["requests"] = [
                            t for t in data["requests"] \
                                if now - t < self.time_window]
                data["requests"].append(now)
                f.seek(0)
                json.dump(data, f)
                f.truncate()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
