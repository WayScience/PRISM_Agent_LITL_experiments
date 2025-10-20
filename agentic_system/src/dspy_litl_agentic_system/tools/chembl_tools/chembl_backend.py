"""
chembl_backend.py

Backend, cached ChEMBL API access methods to cache queries with results
    up to a global fetch limit. 
A `_shared_client_get_process` method is implemented because the get request
    process and error handling is largely the same across all methods.
The tools that will be made accessible to the agentic systems are thin, 
    un-cached wrappers around these cached backend methods to have a
    a separate limit parameter to allow the agents to request any amount of 
    results as it sees fit while every query still has a single source of truth
    cache based on the global fetch limit.
All ChEMBL backend methods make a get request to the 2.x ChEMBL API 
    (see chembl_client.py for details) with specific endpoints as documented
    in https://www.ebi.ac.uk/chembl/api/data/docs. E.g. for searching compounds,
    the request is made as /chembl/api/data/molecule/search?q={query}. Given
    that the ChEMBL API isn't strictly versioned but is quite stable, these
    methods should hopefully work for the foreseeable future.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo.
"""

from typing import Any, Dict, Optional, Tuple, List

from .temp import dummy_cache_decorator as tool_cache  # temporary, will be replaced
from .temp import dummy_fetch_limit as get_fetch_limit  # temporary, will be replaced
from .chembl_client import ChEMBLClient

# Initialize the ChEMBL client
chembl_client = ChEMBLClient()
cache_name = "chembl"


def _shared_client_get_process(
    endpoint: str,
    fields: str | List[str],
    params: Dict[str, Any] = {}
) -> Tuple[Any, str | None]:    
    """
    Shared process for GET requests to ChEMBL API.
    Centralizes some error handling. 
    """
    result = chembl_client.get(endpoint, params=params)
    try:
        params.pop("limit")
    except Exception:
        pass
    param_str = ",".join(f"{k}={v}" for k, v in params.items())

    # Produce natural language error messages for failed searches
    error = result.get("error", None)
    if error:
        error = (
            f"Error searching chembl for {fields} with "
            f"{param_str}: "
            f"{error}"
        )
    
    # When no error but also empty results, produce a no-results error
    payloads = []
    for field in (fields if isinstance(fields, list) else [fields]):
        payloads.append(result.get(field, None))
    if not error and not any([payload is not None] for payload in payloads):
        error = (
            f"No results searching chembl for {fields} with "
            f"{param_str}"
        )
    if len(payloads) == 1:
        payloads = payloads[0]

    return payloads, error

@tool_cache(cache_name)
def _search_chembl_id_global(query: str) -> Dict[str, Any]:
    """
    Search ChEMBL for a compound by query with global fetch limit.
    Query can be any string (canonical name, synonym, smiles, etc).
    Makes a request equivalent to /molecule/search?q={query}
    """
    payload, error = _shared_client_get_process(
        endpoint="/molecule/search.json",
        fields="molecules",
        params={"q": query, "limit": get_fetch_limit()}
    )    
    return {
        "compounds": [
            (
                f"{mol.get('molecule_chembl_id', 'Not found')} "
                f"({mol.get('pref_name', 'Not found')})"
            ) for mol in (payload if payload else []) 
        ],
        "error": error
    }


@tool_cache(cache_name)
def _get_compound_properties_global(chembl_id: str) -> Dict[str, Any]:
    """
    Get compound properties from ChEMBL by ChEMBL ID with global fetch limit.
    Does not use the shared client get process due to needing to interact with
        a different response structure.
    GET from /chembl/api/data/molecule/{chembl_id}
    
    :param chembl_id: ChEMBL ID of the compound to retrieve properties for.
    """
    result = chembl_client.get("/molecule.json", params={
        "molecule_chembl_id": chembl_id,
        "limit": get_fetch_limit()
    })
    if "error" in result:
        return {
            "properties": {}, 
            "error": f"Error retrieving compound properties: {result['error']}"
        }
    
    molecules = result.get("molecules", [])
    if not molecules:
        return {
            "properties": {}, 
            "error": f"No data found for {chembl_id}"
        }

    props = molecules[0].get("molecule_properties", {})
    if not props:
        return {
            "properties": {}, 
            "error": f"{chembl_id} has no calculated properties available"
        }    

    return {"properties": props, "error": None, "molecule": molecules[0]}
    

@tool_cache(cache_name)
def _get_compound_activities_global(
    chembl_id: str,
    activity_type: Optional[str] = None
) -> Dict[str, Any]:    
    """
    Get compound activities from ChEMBL by ChEMBL ID with global fetch limit.
    GET from /chembl/api/data/activity.json

    :param chembl_id: ChEMBL ID of the compound to retrieve activities for.
    :param activity_type: Optional activity type filter (e.g., "IC50").
    """
    payload, error = _shared_client_get_process(
        endpoint="/activity.json",
        fields="activities",
        params={
            "molecule_chembl_id": chembl_id, 
            "limit": get_fetch_limit(),
            **({"activity_type": activity_type} if activity_type else {})
        }
    )    
    return {
        "activities": payload if payload else [],
        "error": error
    }

    
@tool_cache(cache_name)
def _get_drug_info_global(chembl_id: str) -> Dict[str, Any]:
    """
    Get drug information from ChEMBL by ChEMBL ID with global fetch limit.
    GET from /chembl/api/data/drug.json

    :param chembl_id: ChEMBL ID of the drug to retrieve information for.
    """
    payload, error = _shared_client_get_process(
        endpoint="/drug.json",
        fields="drugs",
        params={"molecule_chembl_id": chembl_id, "limit": get_fetch_limit()}
    )
    return {
        "info": payload if payload else [],
        "error": error
    }


@tool_cache(cache_name)
def _get_drug_moa_global(chembl_id: str) -> Dict[str, Any]:
    """Get drug mechanism of action from ChEMBL by ChEMBL ID
        with global fetch limit.
    GET from /chembl/api/data/mechanism.json

    :param chembl_id: ChEMBL ID of the drug to retrieve mechanism of action for.
    """
    
    payload, error = _shared_client_get_process(
        endpoint="/mechanism.json",
        fields="mechanisms",
        params={"molecule_chembl_id": chembl_id, "limit": get_fetch_limit()}
    )
    return {
        "moa": payload if payload else [],
        "error": error
    }


@tool_cache(cache_name)
def _get_drug_indications_global(chembl_id: str) -> Dict[str, Any]:
    """
    Get drug indications from ChEMBL by ChEMBL ID with global fetch limit.
    GET from /chembl/api/data/drug_indication.json

    :param chembl_id: ChEMBL ID of the drug to retrieve indications for.
    """
    payload, error = _shared_client_get_process(
        endpoint="/drug_indication.json",
        fields="drug_indications",
        params={"molecule_chembl_id": chembl_id, "limit": get_fetch_limit()}
    )
    return {
        "drug_indications": payload if payload else [],
        "error": error
    }


@tool_cache(cache_name)
def _search_target_id_global(query: str) -> Dict[str, Any]:
    """
    Search ChEMBL for a target by query with global fetch limit.
    Query can be any string (name, synonym, etc).
    Requests made to /target/search?q={query}
    """
    payload, error = _shared_client_get_process(
        endpoint="/target/search.json",
        fields="targets",
        params={"q": query, "limit": get_fetch_limit()}
    )
    return {
        "targets": payload if payload else [],
        "error": error
    }


@tool_cache(cache_name)
def _get_target_activities_summary_global(
    chembl_id: str, activity_type: Optional[str] = "IC50") -> Dict[str, Any]:
    """Get target activities summary from ChEMBL by target ID
        with global fetch limit.
    GET from /chembl/api/data/activity.json

    :param chembl_id: ChEMBL ID of the target to retrieve activities for.
    :param activity_type: Optional activity type filter (e.g., "IC50").
    """
    payload, error = _shared_client_get_process(
        endpoint="/activity.json",
        fields="activities",
        params={
            "target_chembl_id": chembl_id, 
            "limit": get_fetch_limit(),
            **({"activity_type": activity_type} if activity_type else {})
        }
    )
    return {
        "target_activities": payload if payload else [],
        "error": error
    }
