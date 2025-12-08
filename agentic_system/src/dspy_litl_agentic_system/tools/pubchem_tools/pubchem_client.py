"""
pubchem_client.py

HTTP client for PubChem PUG REST API incorporating a rate limiter and gentle retry.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo
"""

from __future__ import annotations
from typing import Dict, Any, Union
import httpx

from ..rate_limiter import FileBasedRateLimiter as RateLimiter
from ..client import Client

# PubChem API client configuration
# Notes: TODO
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
TIMEOUT = 30.0


class PubChemClient(Client):
    """HTTP client for PubChem API interactions (rate-limited, gentle retry)."""

    def __init__(
        self,
        *,
        max_requests: int = 2,
        time_window: float = 1.0,
        rl_name: str = "pubchem"
    ):
        """
        Initialize PubChemClient with rate limiter.

        :max_requests: Maximum requests allowed in the time window.
            Default 2 req/sec to be gentle to PubChem servers.
        :time_window: Time window in seconds for rate limiting.
        :rl_name: Unique name for the rate limiter state file.
        """

        self.client = httpx.Client(
            base_url=PUBCHEM_BASE_URL,
            timeout=TIMEOUT,
            headers={
                "User-Agent": "PubChem-Tools/1.0.0",
                "Accept": "application/json",
            },
        )
        self.rate_limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            name=rl_name
        )

    def _do(
        self, 
        method: str, 
        url_or_endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        # Acquire RL token (best-effort)
        try:
            self.rate_limiter.acquire_sync()
        except Exception:
            pass

        # First attempt
        resp = self.client.request(method, url_or_endpoint, **kwargs)
        if resp.status_code in (429, 503):
            self._respect_retry_after(resp)
            try:
                self.rate_limiter.acquire_sync()
            except Exception:
                pass
            resp = self.client.request(method, url_or_endpoint, **kwargs)

        try:
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"API error: {e.response.status_code} - {e.response.text[:200]}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def get(self, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._do("GET", endpoint, params=params)

    def post(
        self,
        endpoint: str,
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._do("POST", endpoint, params=params, data=data)

    # For PUG-View (note: different base path)
    def get_view(self, heading: str, cid: Union[int, str]) -> Dict[str, Any]:
        # Use absolute URL; bypass base_url
        url = f"{PUBCHEM_VIEW_BASE_URL}/data/compound/{cid}/JSON"
        # We still want rate limiting + retry on this path:
        return self._do("GET", url, params={"heading": heading})
