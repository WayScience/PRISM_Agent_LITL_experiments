import time
from typing import Any, Callable, Dict, Optional

import requests

def _json_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: float = 30.0,
    retry_delay: float = 1.0,
    response_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generic robust GET + JSON + retry helper for any REST API.

    Returns a standardized structure:
        {
            "data": dict | list | None,
            "error": str | None
        }

    :param url: The URL to send the GET request to.
    :param params: Optional query parameters to include in the request.
    :param max_retries: Maximum number of retries on failure.
    :param timeout: Timeout for the request in seconds.
    :param retry_delay: Delay between retries in seconds.
    :param response_handler: Optional function to process the JSON response.
    """
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()

            try:
                json_data = resp.json()
            except ValueError as e:
                last_error = f"JSON decoding failed on attempt {attempt}: {e}"
                if attempt < max_retries:
                    time.sleep(retry_delay)
                continue

            # Optional post-processing hook
            if response_handler:
                try:
                    json_data = response_handler(json_data)
                except Exception as e:
                    last_error = f"response_handler failed on attempt {attempt}: {e}"
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue
                    return {"data": None, "error": last_error}

            return {"data": json_data, "error": None}

        except requests.RequestException as e:
            last_error = f"Request error on attempt {attempt}: {e}"
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return {"data": None, "error": last_error}

    return {"data": None, "error": last_error or "Unknown error"}
