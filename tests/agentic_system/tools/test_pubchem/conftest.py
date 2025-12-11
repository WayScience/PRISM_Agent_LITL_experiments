"""
Shared fixtures for PubChem tool tests.
"""

import pytest


@pytest.fixture
def common_cids():
    """Common compound CIDs for testing."""
    return {
        "aspirin": "2244",
        "caffeine": "2519",
        "water": "962",
        "ethanol": "702",
        "salicylic_acid": "338",
    }


@pytest.fixture
def sample_compound_names():
    """Sample compound names for search testing."""
    return [
        "aspirin",
        "caffeine",
        "ibuprofen",
        "acetaminophen",
        "glucose",
    ]
