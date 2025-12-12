"""
Tests for Cellosaurus backend functions.

These are loose tests to confirm basic functionality of the Cellosaurus API interactions.
"""

from dspy_litl_agentic_system.tools.cellosaurus_tools.backend import (
    _search_ac_cached,
    _get_ac_info_cached,
)


class TestSearchAccession:
    """Tests for accession search functionality."""
    
    def test_search_hela(self, hela_query, hela_accession):
        """Test searching for HeLa cell line."""
        result = _search_ac_cached(hela_query, _force_refresh=True)
        assert isinstance(result, list)
        assert len(result) > 0
        # HeLa's primary accession is CVCL_0030
        assert hela_accession in result
    
    def test_search_mcf7(self, mcf7_query, mcf7_accession):
        """Test searching for MCF-7 cell line."""
        result = _search_ac_cached(mcf7_query, _force_refresh=True)
        assert isinstance(result, list)
        assert len(result) > 0
        # MCF-7's primary accession is CVCL_0031
        assert mcf7_accession in result
    
    def test_search_nonexistent(self):
        """Test searching for a nonexistent cell line."""
        result = _search_ac_cached("xyznonexistentcellline12345", _force_refresh=True)
        # Should return empty list for nonexistent cell line
        assert isinstance(result, list)


class TestGetAccessionInfo:
    """Tests for retrieving cell line information by accession."""
    
    def test_get_hela_info(self, hela_accession):
        """Test getting info for HeLa (CVCL_0030)."""
        result = _get_ac_info_cached(hela_accession, _force_refresh=True)
        assert isinstance(result, dict)
        # Should have some content for a valid accession
        assert len(result) > 0
    
    def test_get_mcf7_info(self, mcf7_accession):
        """Test getting info for MCF-7 (CVCL_0031)."""
        result = _get_ac_info_cached(mcf7_accession, _force_refresh=True)
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_invalid_accession(self):
        """Test with an invalid accession code."""
        result = _get_ac_info_cached("CVCL_INVALID999", _force_refresh=True)
        # Should handle error gracefully and return empty dict
        assert isinstance(result, dict)
