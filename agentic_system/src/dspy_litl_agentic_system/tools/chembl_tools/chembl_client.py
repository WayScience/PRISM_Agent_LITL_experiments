"""
chembl_client.py

HTTP client for ChEMBL API incorporating a rate limiter and gentle retry.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo.
"""

from typing import Any, Dict, Optional
import time
import httpx

from .temp import DummyRateLimiter as RateLimiter # will be replaced with real RL

# ChEMBL API client configuration
# Note that this is the 2.x ChEMBL API base URL
# https://github.com/chembl/chembl_webservices_2
# which is still active as of 2025. 
# In the future this may cease to be supported.
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT = 30.0


class ChEMBLClient:
    """HTTP client for ChEMBL API."""

    def __init__(
        self,
        *,
        max_requests: int = 4,       # up to 4 req/sec is typically safe
        time_window: float = 1.0,    # second(s)
        rl_name: str = "chembl"      # unique name for the RL state file
    ):
        self.client = httpx.Client(
            base_url=CHEMBL_BASE_URL,
            timeout=TIMEOUT,
            headers={
                "User-Agent": "ChEMBL-Tools/1.0.0",
                "Accept": "application/json",
            },
        )
        
        self.rate_limiter = RateLimiter()

    def _respect_retry_after(self, response: httpx.Response) -> None:
        """Sleep according to Retry-After header (seconds), if present."""
        ra: Optional[str] = response.headers.get("Retry-After")
        if ra:
            try:
                delay = float(ra)
                if delay > 0:
                    time.sleep(delay)
            except ValueError:
                # If non-numeric (e.g., HTTP-date), just do a small pause
                time.sleep(1.0)

    def get(
        self,
        endpoint: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make GET request to ChEMBL API (rate-limited, with gentle retry)."""

        # ---- RATE LIMIT ACQUIRE (synchronous)
        try:
            self.rate_limiter.acquire_sync()
        except Exception:
            # If RL has a transient issue, fail open but keep going.
            pass

        try:
            response = self.client.get(endpoint, params=params)
            # Soft retry on 429/503 with Retry-After support
            if response.status_code in (429, 503):
                self._respect_retry_after(response)
                # Acquire again before retry to keep RL honest
                try:
                    self.rate_limiter.acquire_sync()
                except Exception:
                    pass
                response = self.client.get(endpoint, params=params)

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": (
                    f"API error: {e.response.status_code} - "
                    f"{e.response.text[:200]}"
                )
            }
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
