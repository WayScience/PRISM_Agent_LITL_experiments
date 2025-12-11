"""
Tests for agent-facing PubChem tools.

These tests verify that the agent-facing tools return properly formatted string outputs
and handle errors gracefully.
"""

from dspy_litl_agentic_system.tools.pubchem_tools.for_agents import (
    search_pubchem_cid,
    get_properties,
    get_assay_summary,
    get_safety_summary,
    get_drug_summary,
    find_similar_compounds,
    compute_tanimoto,
)


class TestSearchPubChemCID:
    """Tests for search_pubchem_cid function."""
    
    def test_search_aspirin_single_result(self):
        """Test searching for aspirin returns string with CID."""
        result = search_pubchem_cid("aspirin", limit=1)
        assert isinstance(result, str)
        assert "2244" in result
        assert "CID" in result
    
    def test_search_caffeine_multiple_results(self):
        """Test searching for caffeine with multiple results."""
        result = search_pubchem_cid("caffeine", limit=5)
        assert isinstance(result, str)
        assert "CID" in result or "compound" in result.lower()
    
    def test_search_limit_parameter(self):
        """Test that limit parameter is respected."""
        result = search_pubchem_cid("aspirin", limit=3)
        assert isinstance(result, str)
        # Should not have too many CIDs listed
        assert result.count("CID") <= 10  # Loose check
    
    def test_search_nonexistent_compound(self):
        """Test searching for nonexistent compound."""
        result = search_pubchem_cid("xyznonexistentcompound12345")
        assert isinstance(result, str)
        assert "no" in result.lower() or "not found" in result.lower() or "error" in result.lower()
    
    def test_search_returns_enriched_info_for_single_hit(self):
        """Test that single hit returns enriched information."""
        result = search_pubchem_cid("water", limit=1)
        assert isinstance(result, str)
        # Should contain CID 962 for water
        assert "962" in result


class TestGetProperties:
    """Tests for get_properties function."""
    
    def test_get_aspirin_properties(self):
        """Test getting properties for aspirin."""
        result = get_properties("2244")
        assert isinstance(result, str)
        assert "CID 2244" in result or "2244" in result
        assert "molecular formula" in result.lower() or "formula" in result.lower()
        assert "C9H8O4" in result
    
    def test_get_water_properties(self):
        """Test getting properties for water."""
        result = get_properties("962")
        assert isinstance(result, str)
        assert "H2O" in result
        assert "962" in result
    
    def test_properties_contain_expected_fields(self):
        """Test that properties contain expected molecular descriptors."""
        result = get_properties("2244")
        assert isinstance(result, str)
        # Should contain at least some of these terms
        contains_descriptor = any(term in result.lower() for term in [
            "molecular", "weight", "xlogp", "bond", "formula"
        ])
        assert contains_descriptor
    
    def test_invalid_cid_properties(self):
        """Test getting properties for invalid CID."""
        result = get_properties("999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "no" in result.lower()


class TestGetAssaySummary:
    """Tests for get_assay_summary function."""
    
    def test_get_aspirin_assay_summary(self):
        """Test getting assay summary for aspirin."""
        result = get_assay_summary("2244", limit=5)
        assert isinstance(result, str)
        assert "2244" in result
        # Should mention assay or activity
        assert "assay" in result.lower() or "active" in result.lower() or "no" in result.lower()
    
    def test_get_caffeine_assay_summary(self):
        """Test getting assay summary for caffeine."""
        result = get_assay_summary("2519")
        assert isinstance(result, str)
        assert "2519" in result
    
    def test_assay_limit_parameter(self):
        """Test that limit parameter works."""
        result = get_assay_summary("2244", limit=3)
        assert isinstance(result, str)
    
    def test_invalid_cid_assay(self):
        """Test assay summary for invalid CID."""
        result = get_assay_summary("999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "no" in result.lower()


class TestGetSafetySummary:
    """Tests for get_safety_summary function."""
    
    def test_get_aspirin_safety(self):
        """Test getting safety summary for aspirin."""
        result = get_safety_summary("2244")
        assert isinstance(result, str)
        assert "2244" in result
        # Should mention GHS or safety
        assert "ghs" in result.lower() or "safety" in result.lower() or "limited" in result.lower()
    
    def test_get_ethanol_safety(self):
        """Test getting safety summary for ethanol."""
        result = get_safety_summary("702")
        assert isinstance(result, str)
        assert "702" in result
    
    def test_invalid_cid_safety(self):
        """Test safety summary for invalid CID."""
        result = get_safety_summary("999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "no" in result.lower() or "limited" in result.lower()


class TestGetDrugSummary:
    """Tests for get_drug_summary function."""
    
    def test_get_aspirin_drug_info(self):
        """Test getting drug info for aspirin."""
        result = get_drug_summary("2244")
        assert isinstance(result, str)
        assert "2244" in result
        # Aspirin is a drug, should have some info
        assert "drug" in result.lower() or "medication" in result.lower() or "therapeutic" in result.lower()
    
    def test_get_caffeine_drug_info(self):
        """Test getting drug info for caffeine."""
        result = get_drug_summary("2519")
        assert isinstance(result, str)
        assert "2519" in result
    
    def test_get_water_drug_info(self):
        """Test getting drug info for water (not a drug)."""
        result = get_drug_summary("962")
        assert isinstance(result, str)
        # Should indicate no drug info available
        assert "no" in result.lower() or "not" in result.lower() or "962" in result
    
    def test_invalid_cid_drug_info(self):
        """Test drug info for invalid CID."""
        result = get_drug_summary("999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "no" in result.lower()


class TestFindSimilarCompounds:
    """Tests for find_similar_compounds function."""
    
    def test_find_similar_to_aspirin(self):
        """Test finding similar compounds to aspirin."""
        result = find_similar_compounds("2244", threshold=90, limit=5)
        assert isinstance(result, str)
        assert "2244" in result
        assert "similar" in result.lower() or "tanimoto" in result.lower()
        # Should contain table header
        assert "cid" in result.lower() or "CID" in result
    
    def test_find_similar_with_high_threshold(self):
        """Test finding similar compounds with high threshold."""
        result = find_similar_compounds("2244", threshold=95, limit=3)
        assert isinstance(result, str)
        assert "2244" in result
    
    def test_find_similar_limit_parameter(self):
        """Test that limit parameter is respected."""
        result = find_similar_compounds("2244", threshold=85, limit=2)
        assert isinstance(result, str)
        # Should have limited results
        lines = result.split("\n")
        # Header lines + limited data lines
        assert len(lines) <= 15  # Loose check
    
    def test_invalid_cid_similar(self):
        """Test finding similar compounds for invalid CID."""
        result = find_similar_compounds("999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "no" in result.lower()
    
    def test_similar_compounds_include_properties(self):
        """Test that similar compounds include molecular properties."""
        result = find_similar_compounds("2244", threshold=90, limit=3)
        assert isinstance(result, str)
        # Should contain property information
        assert "iupac" in result.lower() or "formula" in result.lower() or "molecular" in result.lower()


class TestComputeTanimoto:
    """Tests for compute_tanimoto function."""
    
    def test_tanimoto_same_compound(self):
        """Test Tanimoto similarity of a compound with itself."""
        result = compute_tanimoto("2244", "2244")
        assert isinstance(result, str)
        assert "1.0000" in result or "1.00" in result
        assert "tanimoto" in result.lower()
        assert "2244" in result
    
    def test_tanimoto_different_compounds(self):
        """Test Tanimoto similarity between aspirin and caffeine."""
        result = compute_tanimoto("2244", "2519")
        assert isinstance(result, str)
        assert "tanimoto" in result.lower()
        assert "2244" in result
        assert "2519" in result
        # Should contain a decimal number
        assert any(char.isdigit() for char in result)
    
    def test_tanimoto_similar_compounds(self):
        """Test Tanimoto between aspirin and salicylic acid."""
        result = compute_tanimoto("2244", "338")
        assert isinstance(result, str)
        assert "tanimoto" in result.lower()
        # Should be a reasonable similarity value
        assert any(char.isdigit() for char in result)
    
    def test_invalid_cid_tanimoto(self):
        """Test Tanimoto with invalid CID."""
        result = compute_tanimoto("2244", "999999999999")
        assert isinstance(result, str)
        assert "error" in result.lower() or "could not" in result.lower()
    
    def test_tanimoto_both_invalid(self):
        """Test Tanimoto with both invalid CIDs."""
        result = compute_tanimoto("999999999999", "888888888888")
        assert isinstance(result, str)
        assert "error" in result.lower() or "could not" in result.lower()


class TestErrorHandling:
    """Tests for consistent error handling across all functions."""
    
    def test_all_functions_return_strings(self):
        """Test that all functions return strings even on errors."""
        invalid_cid = "999999999999"
        
        # Test all functions with invalid input
        functions = [
            lambda: search_pubchem_cid("xyznonexistent12345"),
            lambda: get_properties(invalid_cid),
            lambda: get_assay_summary(invalid_cid),
            lambda: get_safety_summary(invalid_cid),
            lambda: get_drug_summary(invalid_cid),
            lambda: find_similar_compounds(invalid_cid),
            lambda: compute_tanimoto(invalid_cid, "2244"),
        ]
        
        for func in functions:
            result = func()
            assert isinstance(result, str), f"Function {func} did not return string"
    
    def test_error_messages_are_informative(self):
        """Test that error messages contain helpful information."""
        result = get_properties("999999999999")
        assert isinstance(result, str)
        # Should mention the CID and that there's an issue
        assert "999999999999" in result or "error" in result.lower() or "no" in result.lower()
