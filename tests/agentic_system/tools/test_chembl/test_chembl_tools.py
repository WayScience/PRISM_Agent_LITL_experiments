"""
Basic pytest tests for ChEMBL tools.
Tests each function with known working queries from the notebook.
"""

import pytest

try:
    # This import triggers the schema fetch at import-time inside the library.
    from chembl_webresource_client.new_client import new_client  # noqa: F401
except Exception as e:
    pytest.skip(
        f"ChEMBL client unavailable (chembl_webresource_client failed to init): {e}",
        allow_module_level=True,
    )

from dspy_litl_agentic_system.tools.chembl_tools.for_agents import (
    search_chembl_id,
    get_compound_properties,
    get_compound_activities,
    get_drug_approval_status,
    get_drug_moa,
    get_drug_indications,
    search_target_id,
    get_target_activities_summary,
)


def test_search_chembl_id():
    """Test searching for compound IDs by name."""
    result = search_chembl_id("IMATINIB")
    
    assert isinstance(result, str)
    assert "CHEMBL941" in result
    assert "IMATINIB" in result
    assert "Found" in result
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "not found" not in result.lower()


def test_get_compound_properties():
    """Test retrieving compound properties."""
    result = get_compound_properties("CHEMBL941")
    
    assert isinstance(result, str)
    assert "CHEMBL941" in result
    assert "molecular weight" in result.lower()
    assert "Da" in result
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "not found" not in result.lower()


def test_get_compound_activities():
    """Test retrieving compound bioactivity data."""
    result = get_compound_activities("CHEMBL941")
    
    assert isinstance(result, str)
    assert "CHEMBL941" in result
    assert "Bioactivity" in result or "activity" in result.lower()
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "No bioactivity data found" not in result


def test_get_drug_approval_status():
    """Test checking drug approval status."""
    result = get_drug_approval_status("CHEMBL941")
    
    assert isinstance(result, str)
    assert "CHEMBL941" in result
    assert "approved" in result.lower()
    assert "2001" in result
    # Ensure no error messages
    assert "error" not in result.lower()


def test_get_drug_moa():
    """Test retrieving drug mechanism of action."""
    result = get_drug_moa("CHEMBL25")
    
    assert isinstance(result, str)
    assert "Mechanisms of action" in result or "mechanism" in result.lower()
    assert "Cyclooxygenase" in result or "INHIBITOR" in result
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "No mechanism of action data found" not in result


def test_get_drug_indications():
    """Test retrieving drug indications."""
    result = get_drug_indications("CHEMBL1370561")
    
    assert isinstance(result, str)
    assert "Drug indications" in result or "indication" in result.lower()
    assert "Phase" in result
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "No indication data found" not in result


def test_search_target_id():
    """Test searching for target IDs by name."""
    result = search_target_id("ABL1")
    
    assert isinstance(result, str)
    assert "CHEMBL1862" in result
    assert "Found" in result
    assert "target" in result.lower()
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "not found" not in result.lower()


def test_get_target_activities_summary():
    """Test retrieving target activity summary."""
    result = get_target_activities_summary("CHEMBL1862")
    
    assert isinstance(result, str)
    assert "CHEMBL1862" in result
    assert "compounds" in result.lower()
    assert "Tyrosine-protein kinase ABL1" in result or "ABL" in result
    # Ensure no error messages
    assert "error" not in result.lower()
    assert "No valid" not in result


def test_search_chembl_id_with_limit():
    """Test search with custom limit."""
    result = search_chembl_id("IMATINIB", limit=3)
    
    assert isinstance(result, str)
    assert "Found 3 compound(s)" in result
    assert "CHEMBL941" in result


def test_get_compound_activities_with_limit():
    """Test compound activities with custom limit."""
    result = get_compound_activities("CHEMBL941", limit=3)
    
    assert isinstance(result, str)
    assert "CHEMBL941" in result
    # Should have fewer targets shown
    assert "activity" in result.lower() or "Bioactivity" in result


def test_get_target_activities_summary_with_limit():
    """Test target activities summary with custom limit."""
    result = get_target_activities_summary("CHEMBL1862", limit=3)
    
    assert isinstance(result, str)
    assert "Top 3 compounds" in result
    assert "CHEMBL1862" in result
