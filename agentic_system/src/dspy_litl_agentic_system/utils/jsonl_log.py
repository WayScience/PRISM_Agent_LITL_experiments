"""
jsonl_log.py

Simple utility for logging agentic trace to JSONL files.
"""

from __future__ import annotations
from typing import Dict, Any, Union
from pathlib import Path
import json

def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> bool:
    """
    Append a single JSON object per line to a JSONL file.

    :param path: Path to the JSONL file.
    :param record: Dictionary representing the JSON object to append.
    :return: True if the operation was successful, False otherwise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing to JSONL file: {e}")
        return False
    
    return True
