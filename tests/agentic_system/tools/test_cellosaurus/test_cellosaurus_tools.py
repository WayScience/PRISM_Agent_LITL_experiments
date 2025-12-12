"""
Tests for Cellosaurus agent-facing tools.

These are loose tests to confirm basic functionality of the agent-facing wrappers.
"""

from dspy_litl_agentic_system.tools.cellosaurus_tools.for_agents import (
    search_cellosaurus_ac,
    get_cellosaurus_summary,
)


class TestSearchCellosaurusAC:
    """Tests for agent-facing accession search."""
    
    def test_search_hela(self, hela_query, hela_accession):
        """Test searching for HeLa cell line."""
        result = search_cellosaurus_ac(hela_query)
        assert isinstance(result, str)
        assert hela_accession in result
        assert "Cellosaurus ACs found" in result
    
    def test_search_mcf7(self, mcf7_query, mcf7_accession):
        """Test searching for MCF-7 cell line."""
        result = search_cellosaurus_ac(mcf7_query)
        assert isinstance(result, str)
        assert mcf7_accession in result
    
    def test_search_nonexistent(self):
        """Test searching for a nonexistent cell line."""
        result = search_cellosaurus_ac("xyznonexistentcellline12345")
        assert isinstance(result, str)
        assert "No Cellosaurus match" in result


class TestGetCellosaurusSummary:
    """Tests for agent-facing summary retrieval."""
    
    def test_get_hela_summary(self, hela_accession):
        """Test getting summary for HeLa."""
        result = get_cellosaurus_summary(hela_accession)
        assert isinstance(result, str)
        assert "Cellosaurus Summary" in result
        assert hela_accession in result
    
    def test_get_mcf7_summary(self, mcf7_accession):
        """Test getting summary for MCF-7."""
        result = get_cellosaurus_summary(mcf7_accession)
        assert isinstance(result, str)
        assert "Cellosaurus Summary" in result
    
    def test_invalid_accession(self):
        """Test with an invalid accession code."""
        result = get_cellosaurus_summary("CVCL_INVALID999")
        assert isinstance(result, str)
        # Should return no record found message
        assert "No Cellosaurus record" in result or "Error" in result
