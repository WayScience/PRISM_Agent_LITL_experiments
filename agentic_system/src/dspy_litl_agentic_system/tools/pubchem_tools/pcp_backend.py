"""
pcp_backend.py

Cached backend for PubChem querying using the PubChemPy library.
"""

from typing import Any, Dict, Union

import pubchempy as pcp
from rdkit import DataStructs

from ..tool_cache.cache_decorator import tool_cache
from ..tool_cache.cache_config import get_fetch_limit
from ..request_utils import _json_get

cache_name = "pubchem"
PUBCHEM_VIEW_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
TIMEOUT = 30.0

@tool_cache(cache_name)
def _search_pubchem_cid_cached(query: str):
    """
    Search for PubChem CIDs based on a compound name.
    """

    try:
        cids = [
            str(compound.cid) for compound in
            pcp.get_compounds(
                identifier=query,
                namespace="name",
                MaxRecords=get_fetch_limit(),
            )
        ]
        error = None
    except Exception as e:
        cids = []
        error = str(e)
        pass
    
    return {"cids": cids, "error": error}


@tool_cache(cache_name)
def _get_cid_properties_cached(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Get various properties for a given PubChem CID.
    """

    try:
        compound = pcp.Compound.from_cid(int(cid))
        properties = {
            "IUPACName": compound.iupac_name,
            "MolecularFormula": compound.molecular_formula,
            "MolecularWeight": compound.molecular_weight,
            "XLogP": compound.xlogp,
            "HBondDonorCount": compound.h_bond_donor_count,
            "HBondAcceptorCount": compound.h_bond_acceptor_count,
            "RotatableBondCount": compound.rotatable_bond_count,
            "Complexity": compound.complexity,
            "HeavyAtomCount": compound.heavy_atom_count,
            "Charge": compound.charge,
            # canonical smiles is deprecated in pubchempy
            "ConnectivitySMILES": compound.connectivity_smiles,
        }
        error = None
    except Exception as e:
        properties = {}
        error = str(e)
        pass
    
    return {"properties": properties, "error": error}


@tool_cache(cache_name)
def _get_assay_summary_cached(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Get assay summary for a given PubChem CID.
    """
    
    try:
        assay_summary = pcp.get_json(
            identifier=str(cid),
            namespace='cid',
            operation='assaysummary',
        )
        # expected return being a json object
        # defining a column oriented table.
        # the get request is expected to return something like:
        # {
        #   "Table": {
        #       "Columns": {"Column": [...]},
        #       "Row": [{"Cell": [...] }],
        #   }
        # }
        table = assay_summary.get('Table', {})
        error = None
    except Exception as e:
        table = {}
        error = str(e)
        
    return {"table": table, "error": error}


@tool_cache(cache_name)
def _get_ghs_classification_cached(cid):
    """
    Get GHS classification for a given PubChem CID.
    """
    url = PUBCHEM_VIEW_BASE_URL.format(cid=cid)

    def extract_record(json_data):
        if "Fault" in json_data:
            msg = json_data["Fault"].get("Message", "Unknown fault")
            raise RuntimeError(msg)
        return json_data.get("Record", {})

    result = _json_get(
        url,
        params={"heading": "GHS Classification"},
        response_handler=extract_record,
    )
    return {"record": result["data"] or {}, "error": result["error"]}


@tool_cache(cache_name)
def _get_drug_med_info_cached(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Get drug medication information for a given PubChem CID.
    """
    
    url = PUBCHEM_VIEW_BASE_URL.format(cid=cid)

    def extract_record(json_data):
        if "Fault" in json_data:
            msg = json_data["Fault"].get("Message", "Unknown fault")
            raise RuntimeError(msg)
        return json_data.get("Record", {})

    result = _json_get(
        url,
        params={"heading": "Drug and Medication Information"},
        response_handler=extract_record,
    )
    return {"info": result["data"] or {}, "error": result["error"]}


@tool_cache(cache_name)
def _get_similar_cids_cached(
    cid: Union[int, str],
    threshold: int = 90
) -> Dict[str, Any]:
    """
    Get similar CIDs for a given PubChem CID based on Tanimoto similarity.
    """
    
    try:
        similar_cids = [
            compound.cid for compound in
            pcp.get_compounds(
                identifier=cid,
                namespace="cid",
                searchtype="similarity",
                threshold=threshold,
                MaxRecords=get_fetch_limit(),
            )
        ]
        error = None
    except Exception as e:
        similar_cids = []
        error = str(e)
        pass
    
    return {"similar_cids": similar_cids, "error": error}


@tool_cache(cache_name)
def _get_fingerprint_cached(cid: Union[int, str]) -> Dict[str, Any]:
    """
    Get the fingerprint for a given PubChem CID.
    """
    
    try:
        compound = pcp.Compound.from_cid(int(cid))
        fingerprint = compound.fingerprint  # This is a bit string
        error = None
    except Exception as e:
        fingerprint = None
        error = str(e)
        pass
    
    return {"fingerprint": fingerprint, "error": error}


@tool_cache(cache_name)
def _compute_tanimoto_cached(
    cid1: Union[int, str],
    cid2: Union[int, str]
) -> Dict[str, Any]:
    """
    Compute Tanimoto similarity between two PubChem CIDs.
    """
    # using cached tool call here so that
    # each compound only needs to be fetched once
    fp1 = _get_fingerprint_cached(cid1).get("fingerprint")
    fp2 = _get_fingerprint_cached(cid2).get("fingerprint")
    if not fp1 or not fp2:
        return {"tanimoto": None, "error": "Could not retrieve fingerprints for one or both CIDs."}
    
    try:
        tanimoto = DataStructs.TanimotoSimilarity(
            DataStructs.CreateFromBitString(fp1),
            DataStructs.CreateFromBitString(fp2)
        )
    except Exception as e:
        return {"tanimoto": None, "error": f"Error computing Tanimoto similarity: {str(e)}"}
    
    return {"tanimoto": tanimoto, "error": None}
