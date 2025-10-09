"""
cache_manager.py

Cache instance management and utility operations.
See tool_cache.py for the main caching decorator and detailed motivation of 
    design. 
This module provides utilities to interact with diskcache.Cache instances
    and perform operations like stats.

Implements:
- Creating and managing diskcache.Cache instances
- Cache statistics
- Singleton pattern for cache instances per directory
"""

from pathlib import Path
from typing import Dict, Optional
import diskcache

from .cache_config import resolve_global_size_limit

# ---- Cache registry (singleton pattern)
_CACHE_REGISTRY: Dict[str, diskcache.Cache] = {}


def get_cache(directory: Path, size_limit: Optional[int]) -> diskcache.Cache:
    """
    Get or create a diskcache.Cache instance for the given directory, with the
    given size limit. Caches are singletons per directory path.
    Creates the directory if it does not exist.
    """
    key = str(directory.resolve())
    if key not in _CACHE_REGISTRY:
        directory.mkdir(parents=True, exist_ok=True)
        eff_limit = resolve_global_size_limit(size_limit)
        _CACHE_REGISTRY[key] = diskcache.Cache(
            directory=str(directory), size_limit=eff_limit
        )
    return _CACHE_REGISTRY[key]


def get_cache_stats(
    cache_dir: Path,
    size_limit: Optional[int],
    name: str,
    version_str: str,
    tag: Optional[str],
) -> dict:
    """Get statistics for a cache instance."""
    c = get_cache(cache_dir, size_limit)
    return {
        "name": name,
        "directory": str(cache_dir),
        "size_limit_bytes": c.size_limit,
        "bytes": c.volume(),
        "count": len(c),
        "version": version_str,
        "tag": tag,
    }
