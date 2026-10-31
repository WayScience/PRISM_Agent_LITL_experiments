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


def _frame(frame, *, lossless=False):
    """Encode table JSON, optionally retaining floating columns as exact hex."""
    if not frame.columns.is_unique:
        raise ValueError("Frame columns must be unique")
    value = frame.reset_index(drop=True).to_json(
        orient="table", index=False, double_precision=15, date_format="iso"
    )
    if lossless:
        table = json.loads(value)
        table["float_hex"] = {
            column: [None if pd.isna(item) else float(item).hex() for item in frame[column]]
            for column in frame.select_dtypes(include="floating")
        }
        return json.dumps(table)
    return value


def _read_frame(value):
    """Decode table JSON and restore exact floats when present; accept old rows."""
    frame = pd.read_json(StringIO(value), orient="table")
    for column, values in json.loads(value).get("float_hex", {}).items():
        frame[column] = pd.Series(
            [None if item is None else float.fromhex(item) for item in values],
            dtype=frame[column].dtype,
        )
    return frame


def _decode_run_metadata(record):
    """Decode the four run JSON fields in place and return the same dictionary."""
    for key in ("model_config", "inference_config", "trace_metadata", "metadata"):
        record[key] = json.loads(record[key])
    return record
