"""
Pytest fixtures for Cellosaurus backend tests.
"""

import pytest


@pytest.fixture
def hela_query():
    """Common HeLa cell line query."""
    return "HeLa"


@pytest.fixture
def hela_accession():
    """HeLa cell line accession code."""
    return "CVCL_0030"


@pytest.fixture
def mcf7_query():
    """MCF-7 breast cancer cell line query."""
    return "MCF-7"


@pytest.fixture
def mcf7_accession():
    """MCF-7 cell line accession code."""
    return "CVCL_0031"
