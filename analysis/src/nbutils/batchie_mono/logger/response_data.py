"""Shared response-frame conversion and dataset lookups for logger adapters.

Frames follow the ModelID/broad_id/dose/logfold_change column convention. These
helpers do not import a model integration or require pandas at import time.
"""

from math import isfinite

KEYS = ["ModelID", "broad_id", "dose"]
RESPONSE = "logfold_change"


def response_records(frame):
    """Convert measured response rows to core records, preserving row order."""
    return [
        dict(cell_id=r[0], drug_id=r[1], dose=r[2], value=r[3])
        for r in frame[KEYS + [RESPONSE]].itertuples(index=False, name=None)
    ]


def frame_ids(logger, dataset_id, frame):
    """Resolve response identities in frame order, without requiring values."""
    return logger.response_ids(
        dataset_id, frame[KEYS].itertuples(index=False, name=None)
    )


def dataset_for_action(logger, action_id):
    """Look up an action's experiment and dataset through its registered split."""
    return logger.one(
        """SELECT e.*,s.dataset_id FROM actions a JOIN experiments e USING(experiment_id)
                        JOIN splits s USING(split_id) WHERE a.action_id=?""",
        (action_id,),
    )


def verify_truth(logger, dataset_id, frame):
    """Reject nonfinite or altered values against registered dataset truth."""
    ids = frame_ids(logger, dataset_id, frame)
    truth = {
        r["response_id"]: r["value"]
        for r in logger.rows(
            "SELECT response_id,value FROM responses WHERE dataset_id=?", (dataset_id,)
        )
    }
    for rid, value in zip(ids, frame[RESPONSE]):
        if not isfinite(value) or float(value) != truth[rid]:
            raise ValueError("Supplied response differs from registered dataset truth")
