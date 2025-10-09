"""
cache_decorator.py

Cache decorator and key generation utilities.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo
This module is motivated by experimental design of this repository, where
    we will operate only on the PRISM secondary dataset, which covers a fixed,
    very managable set of ~1, 400 compounds. Due to this, we anticipate the
    same set of compound based queries to be shared by multiple instances of
    agentic systems across the course of experiments (with replication). 
    Caching these queries to disk will speed up experiments and avoid rate
    limiting issues with APIs, especially when parallelized. 

Implements:
- The @tool_cache decorator for persistent function caching
- Function fingerprinting for automatic cache versioning
- Configurable key generation strategies
"""

import json
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from functools import wraps

from .cache_config import (
    resolve_cache_root,
    resolve_global_expire,
    set_default_cache_root,
)
from .cache_manager import (
    get_cache,
    get_cache_stats,
)


def fingerprint_func(func: Callable) -> str:
    """
    Create a short fingerprint of the function source code for cache versioning.
    """
    try:
        src = inspect.getsource(func)
    except Exception:
        src = func.__name__
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def default_key_fn(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: dict,
    *,
    version: str,
    tag: Optional[str],
) -> str:
    """
    Default key function: SHA256 of JSON-serialized payload.
    Used to generate unique cache keys based on function identity and args.
    Intended to be used with @tool_cache decorator and operate on tool methods.
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
    Persistent, portable disk cache decorator to be attached to all
        tool methods that returns (deterministic) results from calling 
        external APIs, databases, or expensive computations that can be 
        shared across multiple agentic system instances.

    Late-binding rules:
    - If _cache_dir is passed at call time, use it.
    - Else if decorator base_dir was provided, use it.
    - Else resolve global root at call time, then append {name}.

    Per-call kwargs:
    - _cache_dir: override cache directory for this call
    - _cache_expire_override: override TTL for this write
    - _offline_only: force offline behavior for this call (bool)
    """

    def _resolve_effective_dir(call_override: Optional[str | Path]) -> Path:
        if call_override:
            return Path(call_override)
        if base_dir is not None:
            return Path(base_dir)
        return resolve_cache_root() / name

    def decorator(func):
        func_fp = fingerprint_func(func) if include_func_fingerprint else "na"
        version_str = f"{cache_version}+{func_fp}" \
            if include_func_fingerprint else cache_version

        @wraps(func)
        def wrapper(*args, **kwargs):

            # Per-call overrides
            call_cache_dir = kwargs.pop("_cache_dir", None)
            call_offline_only = kwargs.pop("_offline_only", None)
            call_expire = kwargs.pop("_cache_expire_override", None)

            cache_dir = _resolve_effective_dir(call_cache_dir)
            cache = get_cache(cache_dir, size_limit_bytes)

            # Get key for tool method call
            kf = key_fn or (
                lambda f, a, kw: default_key_fn(
                    f, a, kw, version=version_str, tag=tag)
            )
            key = kf(func, args, kwargs)

            # Cache hit
            try:
                if key in cache:
                    return cache[key]
            except Exception:
                pass

            # Miss behavior
            oo = call_offline_only \
                if call_offline_only is not None else offline_only
            if oo:
                raise KeyError(
                    f"Cache miss in offline_only mode for key={key[:10]}… "
                    f"(cache={cache_dir})."
                )

            # Compute + write
            result = func(*args, **kwargs)
            effective_expire = resolve_global_expire(expire)
            ttl = effective_expire if call_expire is None else call_expire

            try:
                cache.set(key, result, expire=ttl)
            except Exception:
                try:
                    cache.set(
                        key, 
                        json.loads(json.dumps(result, default=str)), 
                        expire=ttl
                    )
                except Exception:
                    cache.set(key, str(result), expire=ttl)
            return result

        # Helper to resolve directory for utility methods
        def _dir_from_optional(path: Optional[str | Path]) -> Path:
            if path:
                return Path(path)
            if base_dir is not None:
                return Path(base_dir)
            return resolve_cache_root() / name

        # Attach utility methods
        def cache_stats_wrapper(
                path: Optional[str | Path] = None):
            d = _dir_from_optional(path)
            return get_cache_stats(d, size_limit_bytes, name, version_str, tag)

        wrapper.cache_stats = cache_stats_wrapper
        wrapper.set_default_cache_root = set_default_cache_root

        return wrapper

    return decorator
