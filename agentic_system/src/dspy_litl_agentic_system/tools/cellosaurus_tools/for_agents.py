"""
for_agents.py

Convenience entry points for Cellosaurus tools for use by agents.
"""

from .backend import (
    _search_ac_cached,
    _get_ac_info_cached
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
        ac_list = _search_ac_cached(query)
    except Exception as e:
        return f"Error searching Cellosaurus for '{query}': {e}"
    
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
        info = _get_ac_info_cached(ac)
    except Exception as e:
        return f"Error fetching Cellosaurus summary for AC '{ac}': {e}"
    
    if not info:
        return f"No Cellosaurus record found for AC '{ac}'"
    
    summary_lines = [f"Cellosaurus Summary for AC '{ac}':"]
    for key, value in info.items():
        line = f"- {key.replace('_', ' ').title()}: "
        if isinstance(value, list):
            line += ", ".join(value) if value else "N/A"
        else:
            line += str(value) if value else "N/A"
        summary_lines.append(line[:197] + ("..." if len(line) > 200 else ""))

    return "\n".join(summary_lines)
