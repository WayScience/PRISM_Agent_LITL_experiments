from pathlib import Path
import importlib

import dspy_litl_agentic_system.tools.tool_cache.cache_config as cfg


class TestCacheConfig:
    def test_set_and_resolve_cache_root(self, temp_cache_dir):
        importlib.reload(cfg)
        cfg.set_default_cache_root(temp_cache_dir)
        assert cfg.resolve_cache_root() == temp_cache_dir

    def test_resolve_cache_root_env_var(self, temp_cache_dir, monkeypatch):
        importlib.reload(cfg)
        monkeypatch.setenv("AGENTIC_CACHE_DIR", str(temp_cache_dir))
        assert cfg.resolve_cache_root() == temp_cache_dir

    def test_resolve_cache_root_fallback(self):
        importlib.reload(cfg)
        root = cfg.resolve_cache_root()
        assert root == Path.home() / ".cache" / "agentic_tools"

    def test_set_cache_defaults(self):
        importlib.reload(cfg)
        cfg.set_cache_defaults(size_limit_bytes=1000, expire=60.0)
        assert cfg._GLOBAL_CACHE_DEFAULTS["size_limit_bytes"] == 1000
        assert cfg._GLOBAL_CACHE_DEFAULTS["expire"] == 60.0

    def test_resolve_global_size_limit_precedence(self, monkeypatch):
        importlib.reload(cfg)
        # Decorator arg wins
        assert cfg.resolve_global_size_limit(500) == 500
        
        # Programmatic default
        cfg.set_cache_defaults(size_limit_bytes=1000)
        assert cfg.resolve_global_size_limit(None) == 1000
        
        # Env var
        monkeypatch.setenv("AGENTIC_CACHE_SIZE_LIMIT_BYTES", "2000")
        cfg.set_cache_defaults(size_limit_bytes=None)
        assert cfg.resolve_global_size_limit(None) == 2000

    def test_resolve_global_expire_precedence(self, monkeypatch):
        importlib.reload(cfg)
        # Decorator arg wins
        assert cfg.resolve_global_expire(30.0) == 30.0
        
        # Programmatic default
        cfg.set_cache_defaults(expire=60.0)
        assert cfg.resolve_global_expire(None) == 60.0
        
        # Env var
        monkeypatch.setenv("AGENTIC_CACHE_EXPIRE_SECS", "120")
        cfg.set_cache_defaults(expire=None)
        assert cfg.resolve_global_expire(None) == 120.0

    def test_fetch_limit(self, monkeypatch):
        importlib.reload(cfg)
        cfg.set_fetch_limit(100)
        assert cfg.get_fetch_limit() == 100
        
        # Env var after reset
        importlib.reload(cfg)
        cfg._FETCH_LIMIT = None
        monkeypatch.setenv("AGENTIC_TOOL_FETCH_LIMIT", "200")
        assert cfg.get_fetch_limit() == 200
