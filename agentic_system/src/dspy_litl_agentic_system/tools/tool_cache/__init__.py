from .cache_config import (
    set_default_cache_root,
    resolve_cache_root,
    set_cache_defaults,
    resolve_global_size_limit,
    resolve_global_expire,
    set_fetch_limit,
    get_fetch_limit,
    _GLOBAL_CACHE_DEFAULTS,
)

# Cache decorator
from .cache_decorator import tool_cache

__all__ = [
    # Configuration
    "set_default_cache_root",
    "resolve_cache_root",
    "set_cache_defaults",
    "resolve_global_size_limit",
    "resolve_global_expire",
    "set_fetch_limit",
    "get_fetch_limit",
    "_GLOBAL_CACHE_DEFAULTS",    
]
