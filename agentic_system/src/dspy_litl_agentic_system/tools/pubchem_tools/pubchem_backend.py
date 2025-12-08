"""
pubchem_backend.py

Backend, cached PubChem API access methods to cache queries with results
    up to a global fetch limit.
A `_shared_get` method is implemented because the get request
    process and error handling is largely the same across all methods.
The tools that will be made accessible to the agentic systems are thin, 
    un-cached wrappers around these cached backend methods to have a
    a separate limit parameter to allow the agents to request any amount of 
    results as it sees fit while every query still has a single source of truth
    cache based on the global fetch limit.
All PubChem backend methods make a get request to the PubChem PUG REST API
    (see pubchem_client.py for details) with specific endpoints as documented
    in https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest. 
    
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo.
"""

from typing import Any, Dict, Optional, Tuple, Union
import urllib

from ..tool_cache.cache_decorator import tool_cache
from ..tool_cache.cache_config import get_fetch_limit
from .pubchem_client import PubChemClient

# Initialize the PubChem client
pubchem_client = PubChemClient()
cache_name = "pubchem"

def _shared_get(
    endpoint: str,
    params: Dict[str, Any] = {},
) -> Tuple[Optional[Any], Optional[str]]:
    result = pubchem_client.get(endpoint, params=params)
    if "error" in result:
        return None, result["error"]
    return result, None


@tool_cache(cache_name)
def _search_pubchem_cids_global(query: str) -> Dict[str, Any]:
    limit = get_fetch_limit()
    endpoint = f"/compound/name/{urllib.parse.quote(query)}/cids/JSON"
    result, error = _shared_get(endpoint, params={"MaxRecords": limit})

    cids = result.get("IdentifierList", {}).get("CID", [])\
        if (not error) and result else []
    return {"cids": cids[:limit], "error": error}


@tool_cache(cache_name)
def _get_ipuac_name_global(cid: Union[int, str]) -> Dict[str, Any]:
    endpoint = f"/compound/cid/{cid}/property/IUPACName/JSON"
    result, error = _shared_get(endpoint)

    props = result.get("PropertyTable", {}).get("Properties", [])\
        if (not error) and result else []
    name = props[0].get("IUPACName") if props else None
    return {"name": name, "error": error}

@tool_cache(cache_name)
def _get_molecular_formula_global(cid: Union[int, str]) -> Dict[str, Any]:
    endpoint = f"/compound/cid/{cid}/property/MolecularFormula/JSON"
    result, error = _shared_get(endpoint)

    props = result.get("PropertyTable", {}).get("Properties", [])\
        if (not error) and result else []
    formula = props[0].get("MolecularFormula") if props else None
    return {"molecular_formula": formula, "error": error}

@tool_cache(cache_name)
def _get_cid_properties_global(cid: Union[int, str]) -> Dict[str, Any]:
    props = [
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Complexity",
        "HeavyAtomCount",
        "Charge",
        "CanonicalSMILES",
    ]
    endpoint = f"/compound/cid/{cid}/property/{','.join(props)}/JSON"    
    result, error = _shared_get(endpoint)

    table = result.get("PropertyTable", {}).get("Properties", [])\
        if (not error) and result else []
    return {"properties": table[0] or {}, "error": error}


@tool_cache(cache_name)
def _get_assay_summary_global(cid: Union[int, str]) -> Dict[str, Any]:
    endpoint = f"/compound/cid/{cid}/assaysummary/JSON"
    result, error = _shared_get(endpoint)

    table = result.get("Table", {}) if (not error) and result else {}
    return {"table": table, "error": error}


@tool_cache(cache_name)
def _get_ghs_classification_global(cid: Union[int, str]) -> Dict[str, Any]:
    result = pubchem_client.get_view("GHS Classification", cid)

    if "error" in result:
        return {"record": {}, "error": result["error"]}
    record = result.get("Record", {}) if result else {}
    return {"record": record, "error": None}


@tool_cache(cache_name)
def _get_drug_med_info_global(cid: Union[int, str]) -> Dict[str, Any]:
    result = pubchem_client.get_view("Drug and Medication Information", cid)
    
    if "error" in result:
        return {"info": {}, "error": result["error"]}
    record = result.get("Record", {}) if result else {}
    return {"info": record, "error": None}


@tool_cache(cache_name)
def _get_similar_cids_global(
    cid: Union[int, str],
    threshold: int
) -> Dict[str, Any]:
    # 1) SMILES
    # canonical SMILES is preferred
    candidate_fields = ["CanonicalSMILES", "ConnectivitySMILES"]
    smi = _get_cid_properties_global(cid)
    if smi["error"]:
        return {"similar_cids": [], "error": f"SMILES fetch error: {smi['error']}"}
    smiles = None
    for field in candidate_fields:
        smiles = smi["properties"].get(field)
        if smiles:
            break
    if not smiles:
        return {"similar_cids": [], "error": "No SMILES available"}
    
    # 2) Use fastsimilarity_2d here to avoid the time-consuming async
    # structure search that pubchem does not recommend. 
    endpoint = f"/compound/fastsimilarity_2d/smiles/{smiles}/cids/JSON"
    params = {
        "Threshold": int(threshold), 
        "MaxRecords": get_fetch_limit()
    }
    result, error = _shared_get(endpoint, params=params)
    similar_cids = result.get("IdentifierList", {}).get("CID", [])\
        if (not error) and result else []
    return {"similar_cids": similar_cids, "error": error}
