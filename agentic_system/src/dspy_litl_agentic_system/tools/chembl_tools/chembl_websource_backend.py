"""
chembl_websource_backend.py

Cached backend for ChEMBL querying using the chembl_webresource_client library.
Actual agent-facing tools are wrappers around these functions.
Uses custom disk caching decorator and rate limiting to facilitate good querying
    behavior when used in agentic systems.
Also uses tenacity for retrying failed requests with exponential backoff.
"""

from typing import Any, Dict, Optional
from functools import lru_cache

import requests
from chembl_webresource_client.http_errors import (
    HttpTooManyRequests,
    HttpApplicationError,
    HttpBadGateway,
    HttpServiceUnavailable,
    HttpGatewayTimeout,
)
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from ..tool_cache.cache_config import get_fetch_limit
from ..tool_cache.cache_decorator import tool_cache
from ..rate_limiter import FileBasedRateLimiter, make_rate_limited_decorator

# cache config
cache_name = "chembl"

# rate limiter config
MAX_REQUESTS = 4
WINDOW=1.0
_chembl_limiter  = FileBasedRateLimiter(
    max_requests=MAX_REQUESTS,
    time_window=1.0,
    name="chembl"
)
rate_limited_chembl = make_rate_limited_decorator(_chembl_limiter)

# retry config
TENACITY_CONFIG = {
    "retry": retry_if_exception_type((
        requests.exceptions.RequestException,
        HttpTooManyRequests,
        HttpApplicationError,
        HttpBadGateway,
        HttpServiceUnavailable,
        HttpGatewayTimeout,
        )
    ),
    "stop": stop_after_attempt(4),
    "wait": wait_exponential(multiplier=1.0, min=1, max=5),
    "reraise": True
}

# ChEMBL webresource config
TIMEOUT = 30.0
@lru_cache(maxsize=1)
def get_chembl_client():
    """
    Lazy load and cache the ChEMBL client to allow for
        tenacity retry upon chembl client initialization failures.
    """
    from chembl_webresource_client.settings import Settings
    Settings.Instance().TIMEOUT = TIMEOUT
    from chembl_webresource_client.new_client import new_client
    available_resources = [
        resource for resource in dir(new_client) if not resource.startswith('_')
    ]
    if not available_resources:
        raise ValueError("No resources available in ChEMBL client.")
    return new_client


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _search_chembl_molecule_cached(query: str) -> Dict[str, Any]:
    """
    Search ChEMBL molecules by query with global fetch limit.
    Query can be any string (canonical name, synonym, smiles, etc).
    """

    # force evalaute so cache serialization works properly
    results = list(get_chembl_client().molecule.search(query)[:get_fetch_limit()])
    
    return {"results": results, "error": None}


def _search_chembl_id(query: str, _force_refresh: bool = False) -> Dict[str, Any]:
    """
    Wrapper around the molecule search to return ChEMBL IDs and names.
    Not cached separately, relies on cached _search_chembl_molecule_cached.
    """

    results = _search_chembl_molecule_cached(query, _force_refresh=_force_refresh)
    try:
        
        compounds = [
                (
                    f"{mol.get('molecule_chembl_id', 'Not found')} "
                    f"({mol.get('pref_name', 'Not found')})"
                ) for mol in results.get("results", []) 
            ]
        error = results.get("error", None)
    except Exception as e:
        compounds = []
        error = str(e)
        pass
    
    return {"compounds": compounds, "error": error}


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_compound_properties_cached(chembl_id: str) -> Dict[str, Any]:
    """
    Get compound properties from ChEMBL by ChEMBL ID.
    """
    
    results = list(
        get_chembl_client().molecule.filter(
            chembl_id=chembl_id
        ).only(
            ['molecule_chembl_id', 'pref_name', 'molecule_properties']
        )[:get_fetch_limit()]
    )

    if results:
        molecule = results[0]
        properties = molecule.get('molecule_properties', {})
        error = None if properties else f"No properties found for {chembl_id}"
    else:
        properties = {}
        molecule = {}
        error = f"No data found for {chembl_id}"
    
    return {
        "properties": properties, 
        "molecule": molecule,
        "error": error
    }
    

@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_compound_activities_cached(
    chembl_id: str,
    activity_type: Optional[str] = None
) -> Dict[str, Any]:    
    """
    Get compound activities from ChEMBL by ChEMBL ID.
    """
    
    results = list(
        get_chembl_client().activity.filter(
            molecule_chembl_id=chembl_id,
            **{"activity_type": activity_type} if activity_type else {}
        )[:get_fetch_limit()]
    )

    return {
        "activities": results or [],
        "error": None if results else f"No activities found for {chembl_id}"
    }


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_drug_info_cached(chembl_id: str) -> Dict[str, Any]:
    """
    Get drug information from ChEMBL by ChEMBL ID.
    """
    
    results = list(get_chembl_client().drug.filter(chembl_id=chembl_id)[:get_fetch_limit()])

    return {
        "info": results or [],
        "error": None if results else f"No drug info found for {chembl_id}"
    }


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_drug_moa_cached(chembl_id: str) -> Dict[str, Any]:
    """
    Get drug mechanism of action from ChEMBL by ChEMBL ID.
    """

    results = list(get_chembl_client().mechanism.filter(chembl_id=chembl_id)[:get_fetch_limit()])

    return {
        "moa": results or [],
        "error": None if results else f"No mechanism of action found for {chembl_id}"
    }


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_drug_indications_cached(chembl_id: str) -> Dict[str, Any]:
    """
    Get drug indications from ChEMBL by ChEMBL ID.
    """

    results = list(get_chembl_client().drug_indication.filter(chembl_id=chembl_id)[:get_fetch_limit()])

    return {
        "indications": results or [],
        "error": None if results else f"No indications found for {chembl_id}"
    }


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _search_target_id_cached(query: str) -> Dict[str, Any]:
    """
    Search ChEMBL targets by query with global fetch limit.
    Query can be any string (target name, synonym, etc).
    """

    results = list(get_chembl_client().target.search(query)[:get_fetch_limit()])

    return {
        "targets": results or [],
        "error": None if results else f"No targets found for query '{query}'"
    }


@tool_cache(cache_name)
@rate_limited_chembl
@retry(**TENACITY_CONFIG)
def _get_target_activities_summary_cached(
    target_chembl_id: str,
    activity_type: Optional[str] = "IC50"
) -> Dict[str, Any]:
    """
    Get target activities summary from ChEMBL by target ChEMBL ID.
    """

    results = list(
        get_chembl_client().activity.filter(
            target_chembl_id=target_chembl_id,
            **{"activity_type": activity_type} if activity_type else {}
        )[:get_fetch_limit()]
    )

    return {
        "activities_summary": results or [],
        "error": None if results else f"No activities found for target {target_chembl_id}"
    }
