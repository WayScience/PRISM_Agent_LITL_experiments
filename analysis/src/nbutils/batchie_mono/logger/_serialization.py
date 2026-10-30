"""Encode and decode metadata and DataFrames for storage"""

import json
from io import StringIO

import numpy as np
import pandas as pd


def _json(value):
    """Return sorted-key strict JSON, converting NumPy scalars and arrays."""
    def convert(item):
        """Convert one NumPy value, or reject an unsupported object."""
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")

    return json.dumps(value, default=convert, allow_nan=False, sort_keys=True)


def _frame(frame):
    """Encode a DataFrame as table JSON without retaining its index."""
    if not frame.columns.is_unique:
        raise ValueError("Frame columns must be unique")
    return frame.reset_index(drop=True).to_json(
        orient="table", index=False, double_precision=15, date_format="iso"
    )


def _read_frame(value):
    """Decode table JSON into a DataFrame using its stored pandas schema."""
    return pd.read_json(StringIO(value), orient="table")


def _decode_run_metadata(record):
    """Decode the four run JSON fields in place and return the same dictionary."""
    for key in ("model_config", "inference_config", "trace_metadata", "metadata"):
        record[key] = json.loads(record[key])
    return record
