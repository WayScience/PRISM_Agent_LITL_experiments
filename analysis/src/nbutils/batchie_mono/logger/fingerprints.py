"""Verify ordered response inputs without importing Batchie model code.
"""

import hashlib
import json
import math


def fingerprint_responses(frame):
    """
    Fingerprints a drug-model response dataset accompanying the experiment log.
   
    :params frame: DataFrame containing ``ModelID``, ``broad_id``, ``dose``, and
        ``logfold_change``. IDs must be strings; numeric values must be
        convertible to finite Python floats (float64 precision).
    :returns: Fingerprint string of the form ``responses-v1:sha256:<digest>``.
    """
    columns = ["ModelID", "broad_id", "dose", "logfold_change"]
    digest = hashlib.sha256()
    digest.update(json.dumps(["responses-v1", columns]).encode("utf-8"))
    for model, compound, dose, response in frame[columns].itertuples(index=False, name=None):
        if not isinstance(model, str) or not isinstance(compound, str):
            raise ValueError("Response fingerprints require string identifiers")
        numbers = [float(dose), float(response)]
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Response fingerprints require finite dose and response")
        row = [model, compound, *(value.hex() for value in numbers)]
        digest.update(b"\n" + json.dumps(row, ensure_ascii=True).encode("utf-8"))
    return "responses-v1:sha256:" + digest.hexdigest()
