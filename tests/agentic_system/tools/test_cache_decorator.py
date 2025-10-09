import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

from dspy_litl_agentic_system.tools.tool_cache.cache_decorator import (
    fingerprint_func,
    default_key_fn,
    tool_cache,
)


class TestCacheDecorator:
    def test_fingerprint_func(self):
        def sample_func():
            return 42
        
        fp = fingerprint_func(sample_func)
        assert isinstance(fp, str)
        assert len(fp) == 12

    def test_default_key_fn(self):
        def sample_func(x, y=10):
            return x + y
        
        key = default_key_fn(
            sample_func, (5,), {"y": 10}, version="v1", tag="test"
        )
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex

    def test_basic_caching(self, temp_cache_dir):
        call_count = 0
        
        @tool_cache("test_tool", base_dir=temp_cache_dir)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - cache miss
        result1 = expensive_func(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - cache hit
        result2 = expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Not called again

    def test_different_args_different_cache(self, temp_cache_dir):
        call_count = 0
        
        @tool_cache("test_tool", base_dir=temp_cache_dir)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        func(5)
        func(10)
        assert call_count == 2  # Both calls execute

    def test_offline_only_mode(self, temp_cache_dir):
        @tool_cache("test_tool", base_dir=temp_cache_dir, offline_only=True)
        def func(x):
            return x * 2
        
        # Cache miss in offline mode raises error
        with pytest.raises(KeyError, match="Cache miss in offline_only mode"):
            func(5)

    def test_per_call_overrides(self, temp_cache_dir):
        @tool_cache("test_tool", base_dir=temp_cache_dir)
        def func(x):
            return x * 2
        
        # Populate cache
        func(5)
        
        # Use different cache dir for this call
        alt_dir = temp_cache_dir / "alt"
        with pytest.raises(KeyError):  # Cache miss in offline mode
            func(5, _cache_dir=alt_dir, _offline_only=True)

    def test_cache_stats_helper(self, temp_cache_dir):
        @tool_cache("test_tool", base_dir=temp_cache_dir, tag="my_tag")
        def func(x):
            return x * 2
        
        func(5)
        stats = func.cache_stats()
        
        assert stats["name"] == "test_tool"
        assert stats["count"] == 1
        assert stats["tag"] == "my_tag"

    def test_function_fingerprint_versioning(self, temp_cache_dir):
        @tool_cache("test_tool", base_dir=temp_cache_dir, 
                   include_func_fingerprint=True)
        def func_v1(x):
            return x * 2
        
        result1 = func_v1(5)
        
        # Redefine function (simulates code change)
        @tool_cache("test_tool", base_dir=temp_cache_dir,
                   include_func_fingerprint=True)
        def func_v2(x):
            return x * 3  # Different logic
        
        # Should be cache miss due to different fingerprint
        result2 = func_v2(5)
        assert result2 == 15  # New logic executed

    def test_custom_key_function(self, temp_cache_dir):
        def custom_key(func, args, kwargs):
            # Ignore function name, only use first arg
            return f"custom_{args[0]}"
        
        @tool_cache("test_tool", base_dir=temp_cache_dir, key_fn=custom_key)
        def func(x, y=10):
            return x + y
        
        result1 = func(5, y=10)
        result2 = func(5, y=20)  # Different y but same custom key
        
        assert result1 == result2 == 15  # Second call returns cached result

    def test_expire_ttl(self, temp_cache_dir):
        call_count = 0
        
        @tool_cache("test_tool", base_dir=temp_cache_dir, expire=0.01)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = func(5)
        assert result1 == 10
        assert call_count == 1
        
        # Wait for expiration
        import time
        time.sleep(0.02)
        
        # Cache should be expired, function should be called again
        result2 = func(5)
        assert result2 == 10
        assert call_count == 2  # Called again after expiration
