"""
client.py

Cellosaurus API client. 
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import httpx

from ..rate_limiter import FileBasedRateLimiter as RateLimiter
from .temp import Client # will be removed later

BASE_URL = "https://api.cellosaurus.org"


class CellosaurusClient(Client):
    """
    Minimal REST client for Cellosaurus:
      - name/synonym -> accession (AC)
      - AC -> slim record dict
    """

    def __init__(
            self,
            *,
            max_requests: int = 5,
            time_window: float = 1.0,
            rl_name: str = "cellosaurus", 
            timeout: float = 30.0, 
            max_retries: int = 3
        ):
        super().__init__()
        self._timeout = timeout
        self._max_retries = max_retries
        self.rate_limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            name=rl_name
        )
        self._client = httpx.Client(timeout=self._timeout)

    # ---------------- Core GET with Retry-After handling ----------------

    def get(self, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Perform a GET with simple retry on 429/5xx while honoring Retry-After.
        endpoint: path beginning with '/' (e.g., '/search/cell-line')
        """

        try:
            self.rate_limiter.acquire_sync()
        except Exception:
            pass

        url = f"{BASE_URL}{endpoint}"
        params = params or {}
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code in (429, 500, 502, 503, 504) and \
                    attempt < self._max_retries:
                    self._respect_retry_after(resp)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                # no more retries?
                if attempt >= self._max_retries:
                    raise
                # backoff: rely on Retry-After if present
                if hasattr(e, "response") and getattr(e, "response") is not None:
                    self._respect_retry_after(e.response)
                else:
                    # tiny pause if no response object (network hiccup)
                    import time
                    time.sleep(1.0)
        # Should not reach here
        raise last_exc or RuntimeError("Unknown HTTP error")
