import pytest


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provides a temporary cache directory."""
    return tmp_path / "test_cache"
