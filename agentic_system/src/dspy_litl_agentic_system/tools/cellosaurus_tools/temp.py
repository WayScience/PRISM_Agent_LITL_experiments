"""
temp.py

Copied over from another in review branch. 
Feel free to disregard in PR review. 

Module defining an abstract base Client class with retry-after handling,
    will be shared by various API clients.
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time
import httpx


class Client(ABC):
    """Abstract base class for clients with retry-after handling."""

    def __init__(self):
        super().__init__()

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

    @abstractmethod
    def get(
        self, 
        endpoint: str, 
        params: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        pass
