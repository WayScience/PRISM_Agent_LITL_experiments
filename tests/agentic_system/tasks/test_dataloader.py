import pytest
import pandas as pd
import numpy as np

from dspy_litl_agentic_system.tasks.prism_lookup import PrismKey, PrismLookup


@pytest.fixture
def sample_data():
    """Create sample PRISM data for testing."""
    return pd.DataFrame({
        'drug': ['DrugA', 'DrugB', 'DrugC', ' DrugD ', 'DrugA'],
        'cell_line': ['CellX', 'CellY', 'CellZ', 'CellX', 'CellY'],
        'ic50': [1.5, 2.3, 0.8, 3.1, 1.9],
        'auc': [0.75, 0.82, 0.65, 0.91, 0.78],
        'viability': [0.25, 0.18, 0.35, 0.09, 0.22]
    })


@pytest.fixture
def sample_lookup(sample_data):
    """Create a PrismLookup instance with sample data."""
    return PrismLookup(sample_data)


@pytest.fixture
def casefold_lookup(sample_data):
    """Create a PrismLookup instance with casefold=True."""
    return PrismLookup(sample_data, casefold=True)


class TestPrismKey:
    """Test cases for PrismKey class."""
    
    def test_prism_key_creation(self):
        key = PrismKey("DrugA", "CellX")
        assert key.drug == "DrugA"
        assert key.cell == "CellX"
    
    def test_prism_key_immutable(self):
        key = PrismKey("DrugA", "CellX")
        with pytest.raises(AttributeError):
            key.drug = "DrugB"
    
    def test_prism_key_hashable(self):
        key1 = PrismKey("DrugA", "CellX")
        key2 = PrismKey("DrugA", "CellX")
        key3 = PrismKey("DrugB", "CellX")
        
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)
        
        # Can be used in sets/dicts
        key_set = {key1, key2, key3}
        assert len(key_set) == 2
    
    def test_prism_key_norm_casefold(self):
        key = PrismKey(" DrugA ", " CellX ")
        normalized = key.norm(casefold=True)
        
        assert normalized.drug == "druga"
        assert normalized.cell == "cellx"
    
    def test_prism_key_norm_no_casefold(self):
        key = PrismKey(" DrugA ", " CellX ")
        normalized = key.norm(casefold=False)
        
        assert normalized.drug == "DrugA"
        assert normalized.cell == "CellX"


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
    
    def test_ic50_lookup(self, sample_lookup):
        assert sample_lookup.ic50("DrugA", "CellX") == 1.5
        assert sample_lookup.ic50("DrugB", "CellY") == 2.3
        assert sample_lookup.ic50("DrugD", "CellX") == 3.1
    
    def test_ic50_lookup_keyerror(self, sample_lookup):
        with pytest.raises(KeyError):
            sample_lookup.ic50("NonExistentDrug", "CellX")
    
    def test_row_lookup(self, sample_lookup):
        row = sample_lookup.row("DrugA", "CellX")
        assert row['ic50'] == 1.5
        assert row['auc'] == 0.75
        assert row['viability'] == 0.25
    
    def test_get_with_default(self, sample_lookup):
        # Existing key
        assert sample_lookup.get("DrugA", "CellX") == 1.5
        
        # Non-existing key with default
        assert sample_lookup.get("NonExistent", "CellX", default=999) == 999
        
        # Non-existing key without default
        assert sample_lookup.get("NonExistent", "CellX") is None
    
    def test_get_row(self, sample_lookup):
        # Existing key
        row = sample_lookup.get_row("DrugA", "CellX")
        assert row is not None
        assert row['ic50'] == 1.5
        
        # Non-existing key
        row = sample_lookup.get_row("NonExistent", "CellX")
        assert row is None
    
    def test_contains(self, sample_lookup):
        assert sample_lookup.contains("DrugA", "CellX") is True
        assert sample_lookup.contains("DrugD", "CellX") is True 
        assert sample_lookup.contains("NonExistent", "CellX") is False
    
    def test_keys(self, sample_lookup):
        keys = sample_lookup.keys()
        assert len(keys) == 5
        assert ("DrugA", "CellX") in keys
        assert ("DrugD", "CellX") in keys  # Should be stripped
    
    def test_to_frame(self, sample_lookup):
        df = sample_lookup.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert df.index.names == ["drug", "cell_line"]
    
    def test_len(self, sample_lookup):
        assert len(sample_lookup) == 5
    
    def test_casefold_functionality(self, casefold_lookup):
        # Should work with different cases
        assert casefold_lookup.ic50("druga", "cellx") == 1.5
        assert casefold_lookup.ic50("DRUGA", "CELLX") == 1.5
        assert casefold_lookup.contains("DrUgA", "CeLlX") is True
    
    def test_subset_with_query_string(self, sample_lookup):
        subset = sample_lookup.subset("ic50 > 2.0")
        assert len(subset) == 2  # DrugB and DrugD
        assert subset.contains("DrugB", "CellY")
        assert subset.contains("DrugD", "CellX")
    
    def test_subset_with_boolean_mask(self, sample_lookup):
        df = sample_lookup.to_frame().reset_index()
        mask = df['drug'].str.contains('Drug[AB]')
        subset = sample_lookup.subset(mask)
        
        assert len(subset) == 3  # DrugA (2 entries) + DrugB (1 entry)
    
    def test_subset_with_dataframe(self, sample_lookup):
        df = sample_lookup.to_frame().reset_index()
        filtered_df = df[df['ic50'] < 2.0]
        subset = sample_lookup.subset(filtered_df)
        
        assert len(subset) == 3  # DrugA (2 entries) + DrugC (1 entry)
    
    def test_subset_invalid_input(self, sample_lookup):
        with pytest.raises(TypeError):
            sample_lookup.subset(123)  # Invalid type
    
    def test_iteration(self, sample_lookup):
        entries = list(sample_lookup)
        assert len(entries) == 5
        
        # Check first entry structure
        drug, cell, ic50_val, row = entries[0]
        assert isinstance(drug, str)
        assert isinstance(cell, str)
        assert isinstance(ic50_val, (int, float))
        assert isinstance(row, pd.Series)
        
        # Verify specific values (depends on sort order)
        drugs = [entry[0] for entry in entries]
        cells = [entry[1] for entry in entries]
        ic50_vals = [entry[2] for entry in entries]
        
        assert "DrugA" in drugs
        assert "CellX" in cells
        assert 1.5 in ic50_vals
    
    def test_norm_helper(self, sample_lookup, casefold_lookup):
        # Without casefold
        assert sample_lookup._norm(" DrugA ") == "DrugA"
        
        # With casefold
        assert casefold_lookup._norm(" DrugA ") == "druga"


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
