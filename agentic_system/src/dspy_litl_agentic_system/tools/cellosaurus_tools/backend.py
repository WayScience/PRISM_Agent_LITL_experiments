"""
backend.py

Cellosaurus API backend functions with caching.

These functions are intended for internal use, agents will call wrapped
    versions of these implemented in for_agents.py.
Tools are limited to accession search and summary retrieval only due to the
    complexity of the full Cellosaurus data structure. 
"""

from typing import Any, Dict, List, Optional

from ..tool_cache.cache_decorator import tool_cache
from ..tool_cache.cache_config import get_fetch_limit
from .client import CellosaurusClient

client = CellosaurusClient()
cache_name = "cellosaurus"


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    return [str(x)]


def _shared_search(
    query: str,
    fetch_limit: int,
    fields: Optional[List[str] | str] = None
) -> Dict[str, Any]:
    """
    Wrapper around Cellosaurus search API.
    Accepts fields as a list or comma-separated string.
    """
    params: Dict[str, Any] = {
        "q": f'idsy:"{query}"',    # recommended name + synonyms
        "rows": fetch_limit,
        "format": "json",
    }
    if fields:
        params["fields"] = ",".join(fields) if isinstance(fields, list) else fields
    return client.get("/search/cell-line", params=params)


@tool_cache(cache_name)
def _search_ac_global(query: str) -> List[str]:
    """
    Search Cellosaurus for a cell line name/synonym to get its accession (AC).
    Returns the top-ranked AC if found, else None.

    :param query: cell line name or synonym to search for
    :return: list of matching Cellosaurus ACs (may be empty)
    """
    fetch_limit = get_fetch_limit()

    # Fast path: ask only for 'ac' so we get compact results
    result = _shared_search(query, fetch_limit, fields=["ac"])

    entries = (result.get("Cellosaurus") or {}).get("cell-line-list") or []
    ac_list = []
    if entries:
        for entry in entries:
            accession_list = entry.get("accession-list", [])
            for a in accession_list:
                if a.get("type") == "primary" and a.get("value"):
                    ac_list.append(a["value"])

    return ac_list


@tool_cache(cache_name)
def _get_ac_summary_global(ac: str) -> Dict[str, Any]:
    """
    Return a minimal dict for the given AC with keys:
      recommended_name, tissues, diseases, species, sex, age, cell_type
    Uses the compact fields API and only falls back to full JSON if needed.

    :param ac: Cellosaurus accession code
    :return: dict with summary information, or empty dict if not found
    """
    fields = "id,site,di,ox,sx,age,cell"
    flat = client.get(f"/cell-line/{ac}", params={"fields": fields, "format": "json"})

    # Preferred compact path
    if isinstance(flat, dict) and any(k in flat for k in ("id", "site", "di", "ox", "sx", "age", "cell")):
        return {
            "recommended_name": flat.get("id"),
            "tissues": _as_list(flat.get("site")),
            "diseases": _as_list(flat.get("di")),
            "species": _as_list(flat.get("ox")),
            "sex": flat.get("sx"),
            "age": flat.get("age"),
            "cell_type": flat.get("cell"),
        }

    # Fallback to full JSON (rare, but robust)
    full = client.get(f"/cell-line/{ac}", params={"format": "json"})
    entry = ((full.get("Cellosaurus") or {}).get("cell-line-list") or [None])[0]
    if not entry:
        return {}

    # recommended name
    rec = None
    for n in entry.get("name-list", []):
        if n.get("type") == "identifier":
            rec = n.get("value")
            break

    # tissues (labels or values)
    tissues: List[str] = []
    for s in entry.get("derived-from-site-list", []):
        site = s.get("site", {})
        label = site.get("label") or site.get("value")
        if label:
            tissues.append(label)

    # diseases (labels)
    diseases = [
        d.get("label") or d.get("value")
        for d in entry.get("disease-list", [])
        if (d.get("label") or d.get("value"))
    ]

    # species (labels)
    species = [
        sp.get("label") or sp.get("value")
        for sp in entry.get("species-list", [])
        if (sp.get("label") or sp.get("value"))
    ]

    # age, sex, cell_type (via BTO/CL/CLO label if present)
    sex = entry.get("sex")
    age = entry.get("age")
    cell_type = None
    for xr in entry.get("xref-list", []):
        if xr.get("database") in {"BTO", "CL", "CLO"} and xr.get("label"):
            cell_type = xr["label"]
            break

    return {
        "recommended_name": rec,
        "tissues": tissues,
        "diseases": diseases,
        "species": species,
        "sex": sex,
        "age": age,
        "cell_type": cell_type,
    }
