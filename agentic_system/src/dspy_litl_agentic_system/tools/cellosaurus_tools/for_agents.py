"""
for_agents.py

Convenience entry points for Cellosaurus tools for use by agents.
"""

from .backend import (
    _search_ac_global,
    _get_ac_summary_global
)


def search_cellosaurus_ac(query: str) -> str:
    """
    Convenience entry point for tools:
      search Cellosaurus for a cell line name/synonym to get its accession (AC).
    
    Args:
        query (str): Cell line name or synonym to search for
    Returns:
        str: Search results with Cellosaurus ACs or error/no match message
    """
    try:
        ac_list = _search_ac_global(query)
    except Exception:
        return f"Error searching Cellosaurus for '{query}'"
    
    if not ac_list:
        return f"No Cellosaurus match for '{query}'"
    
    return f"Cellosaurus ACs found for '{query}': " + ", ".join(ac_list)


def get_cellosaurus_summary(ac: str) -> str:
    """
    Retrieve summary information regarding cell line with cellosaurus accession.
    Specifically, recommended name, species, tissues, diseases, cell type,
    if available.

    Args:
        ac (str): Cellosaurus accession code
    Returns:
        str: Summary information about the cell line        
    """
    
    try:
        record = _get_ac_summary_global(ac)
    except Exception:
        return f"Error fetching Cellosaurus summary for AC '{ac}'"
    
    if not record:
        return f"No Cellosaurus record found for AC '{ac}'"
    
    parts = [f"Cellosaurus summary for AC '{ac}':"]
    if record.get("recommended_name"):
        parts.append(f"- Recommended Name: {record['recommended_name']}")
    if record.get("species"):
        parts.append(f"- Species: {', '.join(record['species'][:3])}")
    if record.get("tissues"):
        parts.append(f"- Tissues: {', '.join(record['tissues'][:3])}")
    if record.get("diseases"):
        parts.append(f"- Diseases: {', '.join(record['diseases'][:3])}")
    if record.get("cell_type"):
        parts.append(f"- Cell Type: {', '.join(record['cell_type'][:3])}")

    return "\n".join(parts)
