import pytest
import pandas as pd
import numpy as np

from dspy_litl_agentic_system.tasks.prism_lookup import PrismKey, PrismLookup


class TestPrismKey:
    """Test cases for PrismKey class."""
    
    @pytest.mark.parametrize("drug,cell,expected_drug,expected_cell", [
        ("DrugA", "CellX", "DrugA", "CellX"),
        ("DrugB", "CellY", "DrugB", "CellY"),
        ("", "", "", ""),
        ("Drug-1_test", "Cell@2#test", "Drug-1_test", "Cell@2#test"),
    ])
    def test_prism_key_creation(self, drug, cell, expected_drug, expected_cell):
        key = PrismKey(drug, cell)
        assert key.drug == expected_drug
        assert key.cell == expected_cell
    
    @pytest.mark.parametrize("drug,cell", [
        ("DrugA", "CellX"),
        ("DrugB", "CellY"),
        ("", ""),
    ])
    def test_prism_key_immutable(self, drug, cell):
        key = PrismKey(drug, cell)
        with pytest.raises(AttributeError):
            key.drug = "NewDrug"
        with pytest.raises(AttributeError):
            key.cell = "NewCell"
    
    @pytest.mark.parametrize("key1_data,key2_data,key3_data,expected_unique_count", [
        (("DrugA", "CellX"), ("DrugA", "CellX"), ("DrugB", "CellX"), 2),
        (("DrugA", "CellX"), ("DrugA", "CellY"), ("DrugB", "CellX"), 3),
        (("", ""), ("", ""), ("DrugA", "CellX"), 2),
    ])
    def test_prism_key_hashable(self, key1_data, key2_data, key3_data, expected_unique_count):
        key1 = PrismKey(*key1_data)
        key2 = PrismKey(*key2_data)
        key3 = PrismKey(*key3_data)
        
        if key1_data == key2_data:
            assert hash(key1) == hash(key2)
        else:
            assert hash(key1) != hash(key2)
        
        # Can be used in sets/dicts
        key_set = {key1, key2, key3}
        assert len(key_set) == expected_unique_count
    
    @pytest.mark.parametrize("input_drug,input_cell,casefold,expected_drug,expected_cell", [
        (" DrugA ", " CellX ", True, "druga", "cellx"),
        (" DrugA ", " CellX ", False, "DrugA", "CellX"),
        ("DrugA", "CellX", True, "druga", "cellx"),
        ("DrugA", "CellX", False, "DrugA", "CellX"),
        (" DRUG_A ", " CELL_X ", True, "drug_a", "cell_x"),
        (" DRUG_A ", " CELL_X ", False, "DRUG_A", "CELL_X"),
    ])
    def test_prism_key_norm(self, input_drug, input_cell, casefold, expected_drug, expected_cell):
        key = PrismKey(input_drug, input_cell)
        normalized = key.norm(casefold=casefold)
        
        assert normalized.drug == expected_drug
        assert normalized.cell == expected_cell


class TestPrismLookup:
    """Test cases for PrismLookup class."""
    
    def test_initialization_default(self, sample_data):
        lookup = PrismLookup(sample_data)
        assert lookup.drug_col == "drug"
        assert lookup.cell_col == "cell_line"
        assert lookup.ic50_col == "ic50"
        assert lookup.casefold is False
        assert len(lookup) == 5
    
    def test_initialization_custom_columns(self, sample_data):
        lookup = PrismLookup(
            sample_data, 
            drug_col="drug", 
            cell_col="cell_line", 
            ic50_col="auc"
        )
        assert lookup.ic50_col == "auc"
    
    def test_initialization_casefold(self, sample_data):
        lookup = PrismLookup(sample_data, casefold=True)
        assert lookup.casefold is True
        # Should strip and casefold the index
        keys = lookup.keys()
        assert ("drugd", "cellx") in keys
    
    def test_duplicate_validation(self, sample_data):
        # Add duplicate row
        duplicate_data = pd.concat([
            sample_data, 
            pd.DataFrame(
                {'drug': ['DrugA'], 'cell_line': ['CellX'], 'ic50': [999]})
        ])
        
        # Pandas raises this error during set_index() with verify_integrity=True
        with pytest.raises(ValueError, match="Index has duplicate keys"):
            PrismLookup(duplicate_data, validate_unique=True)
    
    @pytest.mark.parametrize("drug,cell,expected_ic50", [
        ("DrugA", "CellX", 1.5),
        ("DrugB", "CellY", 2.3),
        ("DrugC", "CellZ", 0.8),
        ("DrugD", "CellX", 3.1),  # Tests whitespace stripping
        ("DrugA", "CellY", 1.9),
    ])
    def test_ic50_lookup(self, sample_lookup, drug, cell, expected_ic50):
        assert sample_lookup.ic50(drug, cell) == expected_ic50
    
    @pytest.mark.parametrize("drug,cell", [
        ("NonExistentDrug", "CellX"),
        ("DrugA", "NonExistentCell"),
        ("", ""),
        ("DrugA", ""),
        ("", "CellX"),
    ])
    def test_ic50_lookup_keyerror(self, sample_lookup, drug, cell):
        with pytest.raises(KeyError):
            sample_lookup.ic50(drug, cell)
    
    @pytest.mark.parametrize("drug,cell,expected_values", [
        ("DrugA", "CellX", {"ic50": 1.5, "auc": 0.75, "viability": 0.25}),
        ("DrugB", "CellY", {"ic50": 2.3, "auc": 0.82, "viability": 0.18}),
        ("DrugC", "CellZ", {"ic50": 0.8, "auc": 0.65, "viability": 0.35}),
    ])
    def test_row_lookup(self, sample_lookup, drug, cell, expected_values):
        row = sample_lookup.row(drug, cell)
        for col, expected_val in expected_values.items():
            assert row[col] == expected_val
    
    @pytest.mark.parametrize("drug,cell,default,expected", [
        ("DrugA", "CellX", None, 1.5),  # Existing key
        ("DrugA", "CellX", 999, 1.5),   # Existing key with default
        ("NonExistent", "CellX", 999, 999),  # Non-existing with default
        ("NonExistent", "CellX", None, None),  # Non-existing without default
        ("NonExistent", "CellX", "N/A", "N/A"),  # Non-existing with string default
    ])
    def test_get_with_default(self, sample_lookup, drug, cell, default, expected):
        if default is None:
            result = sample_lookup.get(drug, cell)
        else:
            result = sample_lookup.get(drug, cell, default=default)
        assert result == expected
    
    @pytest.mark.parametrize("drug,cell,should_exist", [
        ("DrugA", "CellX", True),
        ("DrugB", "CellY", True),
        ("DrugD", "CellX", True),  # Tests whitespace handling
        ("NonExistent", "CellX", False),
        ("DrugA", "NonExistent", False),
        ("", "", False),
    ])
    def test_contains(self, sample_lookup, drug, cell, should_exist):
        assert ((drug, cell) in sample_lookup) == should_exist
    
    @pytest.mark.parametrize("drug_variants,cell_variants,expected_ic50", [
        (["druga", "DRUGA", "DrUgA"], ["cellx", "CELLX", "CeLlX"], 1.5),
        (["drugb", "DRUGB"], ["celly", "CELLY"], 2.3),
        (["drugd", "DRUGD"], ["cellx", "CELLX"], 3.1),
    ])
    def test_casefold_functionality(self, casefold_lookup, drug_variants, cell_variants, expected_ic50):
        for drug in drug_variants:
            for cell in cell_variants:
                assert casefold_lookup.ic50(drug, cell) == expected_ic50
                assert (drug, cell) in casefold_lookup


class TestPrismLookupEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['drug', 'cell_line', 'ic50'])
        lookup = PrismLookup(empty_df)
        assert len(lookup) == 0
        assert lookup.keys() == []
        assert list(lookup) == []
    
    def test_single_row(self):
        single_row_df = pd.DataFrame({
            'drug': ['DrugA'],
            'cell_line': ['CellX'],
            'ic50': [1.5]
        })
        lookup = PrismLookup(single_row_df)
        assert len(lookup) == 1
        assert lookup.ic50("DrugA", "CellX") == 1.5
    
    def test_special_characters_in_names(self):
        special_df = pd.DataFrame({
            'drug': ['Drug-A_1', 'Drug.B/2'],
            'cell_line': ['Cell@X', 'Cell#Y'],
            'ic50': [1.5, 2.3]
        })
        lookup = PrismLookup(special_df)
        assert lookup.ic50('Drug-A_1', 'Cell@X') == 1.5
        assert lookup.ic50('Drug.B/2', 'Cell#Y') == 2.3
    
    def test_nan_values(self):
        nan_df = pd.DataFrame({
            'drug': ['DrugA', 'DrugB'],
            'cell_line': ['CellX', 'CellY'],
            'ic50': [1.5, np.nan]
        })
        lookup = PrismLookup(nan_df)
        assert lookup.ic50("DrugA", "CellX") == 1.5
        assert pd.isna(lookup.ic50("DrugB", "CellY"))
