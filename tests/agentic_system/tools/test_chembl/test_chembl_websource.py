"""
Tests for ChEMBL backend functions.

These are loose tests to confirm basic functionality of the ChEMBL API interactions.
"""

from dspy_litl_agentic_system.tools.chembl_tools.chembl_websource_backend import (
    _search_chembl_molecule_cached,
    _search_chembl_id,
    _get_compound_properties_cached,
    _get_compound_activities_cached,
    _get_drug_info_cached,
    _get_drug_moa_cached,
    _get_drug_indications_cached,
    _search_target_id_cached,
    _get_target_activities_summary_cached,
)


class TestSearchChEMBLMolecule:
    """Tests for ChEMBL molecule search functionality."""
    
    def test_search_aspirin(self):
        """Test searching for aspirin."""
        result = _search_chembl_molecule_cached("aspirin", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        # Check that results contain molecule data
        assert "molecule_chembl_id" in result["results"][0]
    
    def test_search_imatinib(self):
        """Test searching for imatinib."""
        result = _search_chembl_molecule_cached("imatinib", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
    
    def test_search_nonexistent(self):
        """Test searching for nonexistent compound."""
        result = _search_chembl_molecule_cached("xyznonexistentcompound12345", _force_refresh=True)
        assert isinstance(result["results"], list)
        # Should return empty list for nonexistent compounds


class TestSearchChEMBLID:
    """Tests for ChEMBL ID search functionality."""
    
    def test_search_aspirin_id(self):
        """Test searching for aspirin and getting ChEMBL IDs."""
        result = _search_chembl_id("aspirin", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["compounds"], list)
        assert len(result["compounds"]) > 0
        # Should contain CHEMBL25 for aspirin
        assert any("CHEMBL25" in compound for compound in result["compounds"])
    
    def test_search_imatinib_id(self):
        """Test searching for imatinib IDs."""
        result = _search_chembl_id("imatinib", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["compounds"], list)
        assert len(result["compounds"]) > 0
    
    def test_search_nonexistent_id(self):
        """Test ID search for nonexistent compound."""
        result = _search_chembl_id("xyznonexistentcompound12345", _force_refresh=True)
        assert isinstance(result["compounds"], list)


class TestGetCompoundProperties:
    """Tests for retrieving compound properties."""
    
    def test_get_aspirin_properties(self):
        """Test getting properties for aspirin (CHEMBL25)."""
        result = _get_compound_properties_cached("CHEMBL25", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["properties"], dict)
        assert isinstance(result["molecule"], dict)
        # Check for common property fields
        assert len(result["properties"]) > 0
        # Should have molecular weight or formula
        assert any(key in result["properties"] for key in [
            "mw_freebase", "molecular_weight", "full_mwt"
        ])
    
    def test_get_imatinib_properties(self):
        """Test getting properties for imatinib (CHEMBL941)."""
        result = _get_compound_properties_cached("CHEMBL941", _force_refresh=True)
        assert result["error"] is None
        assert isinstance(result["properties"], dict)
        assert len(result["properties"]) > 0
    
    def test_invalid_chembl_id_properties(self):
        """Test getting properties for invalid ChEMBL ID."""
        result = _get_compound_properties_cached("CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["properties"], dict)
        assert isinstance(result["molecule"], dict)
        # Should have error message
        assert result["error"] is not None


class TestGetCompoundActivities:
    """Tests for compound activities retrieval."""
    
    def test_get_aspirin_activities(self):
        """Test getting activities for aspirin."""
        result = _get_compound_activities_cached("CHEMBL25", _force_refresh=True)
        assert isinstance(result["activities"], list)
        # Aspirin should have some activity data
    
    def test_get_imatinib_activities(self):
        """Test getting activities for imatinib."""
        result = _get_compound_activities_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["activities"], list)
        # Imatinib is well-studied, should have activities
        assert len(result["activities"]) > 0
    
    def test_get_activities_with_type_filter(self):
        """Test getting activities filtered by type."""
        result = _get_compound_activities_cached(
            "CHEMBL941", activity_type="IC50", _force_refresh=True)
        assert isinstance(result["activities"], list)
        # If there are results, they should be IC50 type
        if result["activities"]:
            assert any(
                act.get("standard_type") == "IC50" 
                for act in result["activities"]
            )
    
    def test_invalid_chembl_id_activities(self):
        """Test getting activities for invalid ChEMBL ID."""
        result = _get_compound_activities_cached(
            "CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["activities"], list)


class TestGetDrugInfo:
    """Tests for drug information retrieval."""
    
    def test_get_aspirin_drug_info(self):
        """Test getting drug info for aspirin."""
        result = _get_drug_info_cached("CHEMBL25", _force_refresh=True)
        assert isinstance(result["info"], list)
        # Aspirin is a drug, should have info
        if result["info"]:
            assert isinstance(result["info"][0], dict)
    
    def test_get_imatinib_drug_info(self):
        """Test getting drug info for imatinib."""
        result = _get_drug_info_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["info"], list)
        # Imatinib (Gleevec) is an approved drug
        assert len(result["info"]) > 0
    
    def test_invalid_chembl_id_drug_info(self):
        """Test getting drug info for invalid ChEMBL ID."""
        result = _get_drug_info_cached("CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["info"], list)


class TestGetDrugMOA:
    """Tests for drug mechanism of action retrieval."""
    
    def test_get_aspirin_moa(self):
        """Test getting MOA for aspirin."""
        result = _get_drug_moa_cached("CHEMBL25", _force_refresh=True)
        assert isinstance(result["moa"], list)
        # Aspirin has known mechanism (COX inhibition)
        if result["moa"]:
            assert isinstance(result["moa"][0], dict)
    
    def test_get_imatinib_moa(self):
        """Test getting MOA for imatinib."""
        result = _get_drug_moa_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["moa"], list)
        # Imatinib has well-defined MOA
        assert len(result["moa"]) > 0
    
    def test_invalid_chembl_id_moa(self):
        """Test getting MOA for invalid ChEMBL ID."""
        result = _get_drug_moa_cached("CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["moa"], list)


class TestGetDrugIndications:
    """Tests for drug indications retrieval."""
    
    def test_get_aspirin_indications(self):
        """Test getting indications for aspirin."""
        result = _get_drug_indications_cached("CHEMBL25", _force_refresh=True)
        assert isinstance(result["indications"], list)
        # Aspirin has known indications
    
    def test_get_imatinib_indications(self):
        """Test getting indications for imatinib."""
        result = _get_drug_indications_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["indications"], list)
        # Imatinib is used for CML, should have indications
        assert len(result["indications"]) > 0
    
    def test_invalid_chembl_id_indications(self):
        """Test getting indications for invalid ChEMBL ID."""
        result = _get_drug_indications_cached("CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["indications"], list)


class TestSearchTargetID:
    """Tests for target search functionality."""
    
    def test_search_egfr_target(self):
        """Test searching for EGFR target."""
        result = _search_target_id_cached("EGFR", _force_refresh=True)
        assert isinstance(result["targets"], list)
        assert len(result["targets"]) > 0
        # Should contain target_chembl_id
        assert "target_chembl_id" in result["targets"][0]
    
    def test_search_kinase_target(self):
        """Test searching for kinase targets."""
        result = _search_target_id_cached("kinase", _force_refresh=True)
        assert isinstance(result["targets"], list)
        # Kinases are well-represented in ChEMBL
        assert len(result["targets"]) > 0
    
    def test_search_nonexistent_target(self):
        """Test searching for nonexistent target."""
        result = _search_target_id_cached("xyznonexistenttarget12345", _force_refresh=True)
        assert isinstance(result["targets"], list)


class TestGetTargetActivitiesSummary:
    """Tests for target activities summary retrieval."""
    
    def test_get_egfr_activities(self):
        """Test getting activities for EGFR target (CHEMBL203)."""
        result = _get_target_activities_summary_cached(
            "CHEMBL203", _force_refresh=True)
        assert isinstance(result["activities_summary"], list)
        # EGFR is well-studied, should have many activities
        assert len(result["activities_summary"]) > 0
    
    def test_get_activities_with_type_filter(self):
        """Test getting target activities filtered by type."""
        result = _get_target_activities_summary_cached(
            "CHEMBL203", activity_type="IC50", _force_refresh=True)
        assert isinstance(result["activities_summary"], list)
        # If there are results, they should be IC50 type
        if result["activities_summary"]:
            assert any(
                act.get("standard_type") == "IC50" 
                for act in result["activities_summary"]
            )
    
    def test_get_activities_with_ki_filter(self):
        """Test getting target activities filtered by Ki."""
        result = _get_target_activities_summary_cached(
            "CHEMBL203", activity_type="Ki", _force_refresh=True)
        assert isinstance(result["activities_summary"], list)
    
    def test_invalid_target_id_activities(self):
        """Test getting activities for invalid target ID."""
        result = _get_target_activities_summary_cached(
            "CHEMBL999999999", _force_refresh=True)
        assert isinstance(result["activities_summary"], list)


class TestDataStructures:
    """Tests for verifying data structure consistency."""
    
    def test_molecule_search_returns_expected_fields(self):
        """Test that molecule search returns expected fields."""
        result = _search_chembl_molecule_cached("aspirin")
        if result["results"]:
            mol = result["results"][0]
            assert isinstance(mol, dict)
            # Should have at least molecule_chembl_id
            assert "molecule_chembl_id" in mol
    
    def test_properties_returns_dict(self):
        """Test that properties returns dictionary."""
        result = _get_compound_properties_cached("CHEMBL25", _force_refresh=True)
        assert isinstance(result["properties"], dict)
        assert isinstance(result["molecule"], dict)
    
    def test_activities_returns_list_of_dicts(self):
        """Test that activities returns list of dictionaries."""
        result = _get_compound_activities_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["activities"], list)
        if result["activities"]:
            assert isinstance(result["activities"][0], dict)
    
    def test_drug_info_returns_list(self):
        """Test that drug info returns list."""
        result = _get_drug_info_cached("CHEMBL941", _force_refresh=True)
        assert isinstance(result["info"], list)
        if result["info"]:
            assert isinstance(result["info"][0], dict)


class TestErrorHandling:
    """Tests for error handling across all functions."""
    
    def test_all_functions_handle_invalid_input_gracefully(self):
        """Test that all functions handle invalid input without crashing."""
        invalid_id = "CHEMBL999999999"
        invalid_query = "xyznonexistent12345"
        
        # Test all functions with invalid input
        functions = [
            lambda: _search_chembl_molecule_cached(invalid_query, _force_refresh=True),
            lambda: _search_chembl_id(invalid_query, _force_refresh=True),
            lambda: _get_compound_properties_cached(invalid_id, _force_refresh=True),
            lambda: _get_compound_activities_cached(invalid_id, _force_refresh=True),
            lambda: _get_drug_info_cached(invalid_id, _force_refresh=True),
            lambda: _get_drug_moa_cached(invalid_id, _force_refresh=True),
            lambda: _get_drug_indications_cached(invalid_id, _force_refresh=True),
            lambda: _search_target_id_cached(invalid_query, _force_refresh=True),
            lambda: _get_target_activities_summary_cached(invalid_id, _force_refresh=True),
        ]
        
        for func in functions:
            result = func()
            assert isinstance(result, dict), f"Function {func} did not return dict"
            # Should have expected keys
            assert "error" in result or any(
                key in result for key in ["results", "compounds", "properties", "activities"]
            )
