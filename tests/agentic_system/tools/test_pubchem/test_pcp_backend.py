"""
Tests for PubChem backend functions.

These are loose tests to confirm basic functionality of the PubChem API interactions.
"""

import string

from dspy_litl_agentic_system.tools.pubchem_tools.pcp_backend import (
    _search_pubchem_cid_cached,
    _get_cid_properties_cached,
    _get_assay_summary_cached,
    _get_ghs_classification_cached,
    _get_drug_med_info_cached,
    _get_similar_cids_cached,
    _get_fingerprint_cached,
    _compute_tanimoto_cached,
)


class TestSearchPubChemCID:
    """Tests for CID search functionality."""
    
    def test_search_aspirin(self):
        """Test searching for a common drug (aspirin)."""
        result = _search_pubchem_cid_cached("aspirin", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["cids"], list)
        assert len(result["cids"]) > 0
        # Aspirin's primary CID is 2244
        assert "2244" in result["cids"]
    
    def test_search_caffeine(self):
        """Test searching for caffeine."""
        result = _search_pubchem_cid_cached("caffeine", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["cids"], list)
        assert len(result["cids"]) > 0
    
    def test_search_nonexistent(self):
        """Test searching for a nonexistent compound."""
        result = _search_pubchem_cid_cached("xyznonexistentcompound12345", _force_refresh=True)
        # Should either return empty list or have an error
        assert isinstance(result["cids"], list)


class TestGetCIDProperties:
    """Tests for retrieving compound properties."""
    
    def test_get_aspirin_properties(self):
        """Test getting properties for aspirin (CID 2244)."""
        result = _get_cid_properties_cached("2244", _force_refresh=True)
        assert result["error"] is None
        props = result["properties"]
        assert isinstance(props, dict)
        assert "MolecularFormula" in props
        assert props["MolecularFormula"] == "C9H8O4"
        assert "MolecularWeight" in props
        assert "IUPACName" in props
    
    def test_get_water_properties(self):
        """Test getting properties for water (CID 962)."""
        result = _get_cid_properties_cached("962", _force_refresh=True)
        assert result["error"] is None
        props = result["properties"]
        assert props["MolecularFormula"] == "H2O"
        assert props["MolecularWeight"] is not None
    
    def test_invalid_cid(self):
        """Test with an invalid CID."""
        result = _get_cid_properties_cached("999999999999", _force_refresh=True)
        # Should handle error gracefully
        assert isinstance(result["properties"], dict)


class TestGetAssaySummary:
    """Tests for assay summary retrieval."""
    
    def test_get_aspirin_assay(self):
        """Test getting assay summary for aspirin."""
        result = _get_assay_summary_cached("2244", _force_refresh=True)
        assert isinstance(result["table"], dict)
        # May or may not have assay data, but should return valid structure
    
    def test_get_caffeine_assay(self):
        """Test getting assay summary for caffeine (CID 2519)."""
        result = _get_assay_summary_cached("2519", _force_refresh=True)
        assert isinstance(result["table"], dict)


class TestGetGHSClassification:
    """Tests for GHS classification retrieval."""
    
    def test_get_aspirin_ghs(self):
        """Test getting GHS classification for aspirin."""
        result = _get_ghs_classification_cached("2244", _force_refresh=True)
        assert isinstance(result["record"], dict)
        # May or may not have GHS data
    
    def test_get_ethanol_ghs(self):
        """Test getting GHS classification for ethanol (CID 702)."""
        result = _get_ghs_classification_cached("702", _force_refresh=True)
        assert isinstance(result["record"], dict)


class TestGetDrugMedInfo:
    """Tests for drug medication information retrieval."""
    
    def test_get_aspirin_drug_info(self):
        """Test getting drug info for aspirin."""
        result = _get_drug_med_info_cached("2244", _force_refresh=True)
        assert isinstance(result["info"], dict)
        # Aspirin is a drug, so should have some info
    
    def test_get_water_drug_info(self):
        """Test getting drug info for water (not a drug)."""
        result = _get_drug_med_info_cached("962", _force_refresh=True)
        # Water is not a drug, but should handle gracefully
        assert isinstance(result["info"], dict)


class TestGetSimilarCIDs:
    """Tests for similar compound search."""
    
    def test_get_similar_to_aspirin(self):
        """Test finding similar compounds to aspirin."""
        result = _get_similar_cids_cached("2244", threshold=90, _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["similar_cids"], list)
        assert len(result["similar_cids"]) > 0
        # Aspirin itself should be in the results
        assert 2244 in result["similar_cids"]
    
    def test_get_similar_with_high_threshold(self):
        """Test with high similarity threshold."""
        result = _get_similar_cids_cached("2244", threshold=95, _force_refresh=True)
        assert isinstance(result["similar_cids"], list)


class TestFingerprint:
    """Tests for fingerprint retrieval."""
    
    def test_get_aspirin_fingerprint(self):
        """Test getting fingerprint for aspirin."""
        result = _get_fingerprint_cached("2244", _force_refresh=True)
        assert result["error"] is None
        assert result["fingerprint"] is not None
        assert isinstance(result["fingerprint"], str)
        # Fingerprint should be a binary string
        assert all(c in string.hexdigits for c in result["fingerprint"])
    
    def test_get_caffeine_fingerprint(self):
        """Test getting fingerprint for caffeine."""
        result = _get_fingerprint_cached("2519", _force_refresh=True)
        assert result["error"] is None
        assert result["fingerprint"] is not None


class TestComputeTanimoto:
    """Tests for Tanimoto similarity computation."""
    
    def test_tanimoto_same_compound(self):
        """Test Tanimoto similarity of a compound with itself."""
        result = _compute_tanimoto_cached("2244", "2244", _force_refresh=True)
        assert result["error"] is None
        assert result["tanimoto"] is not None
        # Should be 1.0 for identical compounds
        assert result["tanimoto"] == 1.0
    
    def test_tanimoto_different_compounds(self):
        """Test Tanimoto similarity between aspirin and caffeine."""
        result = _compute_tanimoto_cached("2244", "2519", _force_refresh=True)
        assert result["error"] is None
        assert result["tanimoto"] is not None
        assert 0.0 <= result["tanimoto"] <= 1.0
        # They're different, so should be less than 1
        assert result["tanimoto"] < 1.0
    
    def test_tanimoto_similar_compounds(self):
        """Test Tanimoto between aspirin and salicylic acid (338)."""
        result = _compute_tanimoto_cached("2244", "338", _force_refresh=True)
        assert result["error"] is None
        assert result["tanimoto"] is not None
        # These are structurally related
        assert result["tanimoto"] > 0.5
