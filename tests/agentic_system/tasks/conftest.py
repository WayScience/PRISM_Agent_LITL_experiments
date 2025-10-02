import pytest
import pandas as pd
from typing import List, Tuple

from dspy_litl_agentic_system.tasks.prism_lookup import PrismLookup


class FakePrismLookup:
    """Mock PrismLookup for testing dispatcher and other components."""
    
    def __init__(self, data: List[Tuple[str, str, float]], ic50_col: str = "ic50"):
        self.ic50_col = ic50_col
        self._data = {}
        self._keys = []
        
        for drug, cell, ic50 in data:
            key = (drug, cell)
            self._keys.append(key)
            # Create a mock Series with ic50 and some additional columns
            row_data = {
                ic50_col: ic50,
                "drug_name": drug,
                "cell_line": cell,
                "other_col": f"data_{drug}_{cell}"
            }
            self._data[key] = pd.Series(row_data)
    
    def keys(self) -> List[Tuple[str, str]]:
        return list(self._keys)
    
    def contains(self, drug: str, cell: str) -> bool:
        return (drug, cell) in self._data
    
    def row(self, drug: str, cell: str) -> pd.Series:
        if (drug, cell) not in self._data:
            raise KeyError(f"Key ({drug}, {cell}) not found")
        return self._data[(drug, cell)]
    
    def __contains__(self, key) -> bool:
        """Support 'in' operator for compatibility with PrismLookup.__contains__"""
        if isinstance(key, tuple) and len(key) == 2:
            return key in self._data
        return False


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


@pytest.fixture
def dispatcher_sample_data():
    """Sample data specifically formatted for dispatcher testing."""
    return [
        ("drug1", "cell1", 1.5),
        ("drug2", "cell2", 2.0),
        ("drug3", "cell3", 0.5),
        ("drug1", "cell2", 1.8),
    ]


@pytest.fixture
def fake_lookup(dispatcher_sample_data):
    """Create a FakePrismLookup instance for dispatcher testing."""
    return FakePrismLookup(dispatcher_sample_data)


@pytest.fixture
def empty_fake_lookup():
    """Create an empty FakePrismLookup instance for testing edge cases."""
    return FakePrismLookup([])
