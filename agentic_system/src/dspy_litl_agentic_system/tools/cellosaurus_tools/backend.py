"""
backend.py

Cellosaurus API backend functions with caching.

These functions are intended for internal use, agents will call wrapped
    versions of these implemented in for_agents.py.
Tools are limited to accession search and summary retrieval only due to the
    complexity of the full Cellosaurus data structure. 
"""

from typing import Any, Dict, List

import requests
from cellosaurus_mcp.tools import (
    search_cell_lines,
    get_cell_line_info
)
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from ..tool_cache.cache_decorator import tool_cache
from ..tool_cache.cache_config import get_fetch_limit
from ..sync_bridge import run_async_sync
from ..cellosaurus_tools.extractor import extract_bio_summary, parse_cell_line_list
from ..rate_limiter import FileBasedRateLimiter, make_rate_limited_decorator

# cache config
cache_name = "cellosaurus"

# rate limiter config
cellosaurus_limiter  = FileBasedRateLimiter(
    max_requests=2,
    time_window=1.0,
    name="cellosaurus"
)
rate_limited_cellosaurus = make_rate_limited_decorator(cellosaurus_limiter)

# retry config
TENACITY_CONFIG = {
    "retry": retry_if_exception_type(requests.exceptions.RequestException),
    "stop": stop_after_attempt(4),
    "wait": wait_exponential(multiplier=1.0, min=1, max=5),
    "reraise": True
}


@tool_cache(cache_name)
@rate_limited_cellosaurus
@retry(**TENACITY_CONFIG)
def _search_ac_cached(query: str) -> List[str]:
    """
    Search Cellosaurus for a cell line name/synonym to get its accession (AC).
    Returns the top-ranked AC if found, else None.
    """
    result = run_async_sync(
        search_cell_lines.fn(
            query=query,
            fields=["ac"],
            rows=get_fetch_limit()
        )
    )

    if not result:
        return []
    
    return parse_cell_line_list(
        result.get("Cellosaurus", {}).get("cell-line-list", [])
    )


@tool_cache(cache_name)
@rate_limited_cellosaurus
@retry(**TENACITY_CONFIG)
def _get_ac_info_cached(accession: str) -> Dict[str, Any]:
    """
    Retrieve full Cellosaurus record for the given accession code (AC).
    """
    result = run_async_sync(
        get_cell_line_info.fn(
            accession=accession
        )
    )

    if not result:
        return {}
    
    cell_line_list = result.get("Cellosaurus", {}).get("cell-line-list", [])    
    return extract_bio_summary(
        cell_line_list[0] if cell_line_list else []
    )
