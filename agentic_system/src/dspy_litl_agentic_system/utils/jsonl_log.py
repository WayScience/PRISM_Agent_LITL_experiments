"""
jsonl_log.py

Simple utility for logging agentic trace to JSONL files.
"""

from __future__ import annotations
from typing import Dict, Any, Union
from pathlib import Path
import json

def append_jsonl(path: Union[str, Path], record: Dict[str, Any]) -> None:
    """
    Append a single JSON object per line to a JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
