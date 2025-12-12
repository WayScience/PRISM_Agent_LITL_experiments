"""
Shared fixtures for ChEMBL backend tests.
"""

import pytest


@pytest.fixture
def common_chembl_ids():
    """Common ChEMBL IDs for testing."""
    return {
        "aspirin": "CHEMBL25",
        "caffeine": "CHEMBL113",
        "imatinib": "CHEMBL941",  # Well-known drug
        "paracetamol": "CHEMBL112",
        "viagra": "CHEMBL192",  # Sildenafil
    }


@pytest.fixture
def common_target_ids():
    """Common target ChEMBL IDs for testing."""
    return {
        "egfr": "CHEMBL203",  # EGFR receptor
        "bcr_abl": "CHEMBL1862",  # BCR-ABL fusion
        "cox2": "CHEMBL230",  # COX-2
    }


@pytest.fixture
def sample_compound_names():
    """Sample compound names for search testing."""
    return [
        "aspirin",
        "ibuprofen",
        "paracetamol",
        "metformin",
        "atorvastatin",
    ]
