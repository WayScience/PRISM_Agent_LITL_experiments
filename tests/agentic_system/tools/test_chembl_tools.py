"""
Basic pytest tests for ChEMBL tools.
Tests each function with known working queries from the notebook.
"""

import pytest
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


@pytest.mark.parametrize("func,query,expected_substrings,forbidden_substrings", [
    (
        search_chembl_id,
        "IMATINIB",
        ["CHEMBL941", "IMATINIB", "Found"],
        ["error", "not found"]
    ),
    (
        get_compound_properties,
        "CHEMBL941",
        ["CHEMBL941", "molecular weight", "Da"],
        ["error", "not found"]
    ),
    (
        get_compound_activities,
        "CHEMBL941",
        ["CHEMBL941", "Bioactivity"],
        ["error", "No bioactivity data found"]
    ),
    (
        get_drug_approval_status,
        "CHEMBL941",
        ["CHEMBL941", "approved", "2001"],
        ["error"]
    ),
    (
        get_drug_moa,
        "CHEMBL25",
        ["Mechanisms of action", "Cyclooxygenase"],
        ["error", "No mechanism of action data found"]
    ),
    (
        get_drug_indications,
        "CHEMBL1370561",
        ["Drug indications", "Phase"],
        ["error", "No indication data found"]
    ),
    (
        search_target_id,
        "ABL1",
        ["CHEMBL1862", "Found", "target"],
        ["error", "not found"]
    ),
    (
        get_target_activities_summary,
        "CHEMBL1862",
        ["CHEMBL1862", "compounds", "Tyrosine-protein kinase ABL1"],
        ["error", "No valid"]
    ),
])
def test_chembl_functions(func, query, expected_substrings, forbidden_substrings):
    """Test ChEMBL functions with known working queries."""
    result = func(query)
    
    assert isinstance(result, str)
    
    # Check for expected content
    for substring in expected_substrings:
        assert substring in result, f"Expected '{substring}' in result"
    
    # Check for forbidden content (case-insensitive for error messages)
    for substring in forbidden_substrings:
        assert substring.lower() not in result.lower(), f"Unexpected '{substring}' found in result"


@pytest.mark.parametrize("func,query,limit,expected_in_result", [
    (search_chembl_id, "IMATINIB", 3, "Found 3 compound(s)"),
    (get_compound_activities, "CHEMBL941", 3, "CHEMBL941"),
    (get_target_activities_summary, "CHEMBL1862", 3, "Top 3 compounds"),
])
def test_chembl_functions_with_limit(func, query, limit, expected_in_result):
    """Test ChEMBL functions with custom limit parameter."""
    result = func(query, limit=limit)
    
    assert isinstance(result, str)
    assert expected_in_result in result
