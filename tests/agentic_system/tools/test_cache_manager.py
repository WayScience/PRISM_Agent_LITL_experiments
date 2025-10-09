from dspy_litl_agentic_system.tools.tool_cache.cache_manager import (
    get_cache,
    get_cache_stats,
    _CACHE_REGISTRY,
)


class TestCacheManager:
    def test_get_cache_creates_directory(self, temp_cache_dir):
        cache = get_cache(temp_cache_dir, size_limit=1000)
        assert temp_cache_dir.exists()
        assert cache.directory == str(temp_cache_dir)

    def test_get_cache_singleton(self, temp_cache_dir):
        cache1 = get_cache(temp_cache_dir, size_limit=1000)
        cache2 = get_cache(temp_cache_dir, size_limit=1000)
        assert cache1 is cache2

    def test_get_cache_stats(self, temp_cache_dir):
        cache = get_cache(
            temp_cache_dir, 
            # a larger limit is needed here to avoid eviction
            # of the existing cache entry when calling .set()
            # which causes the stats["count"] to behave unexpectedly
            size_limit=10_000_000
        )
        cache.set("key1", "value1")
        
        stats = get_cache_stats(
            temp_cache_dir, 1000, "test_cache", "v1", "test_tag"
        )
        
        assert stats["name"] == "test_cache"
        assert stats["count"] == 1
        assert stats["version"] == "v1"
        assert stats["tag"] == "test_tag"
