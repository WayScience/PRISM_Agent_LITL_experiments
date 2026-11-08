"""Response-data fingerprints, independent of model integration and logging."""

from nbutils.batchie_mono.logger._storage import digest, encode
from nbutils.batchie_mono.logger.response_data import response_records


def fingerprint_responses(frame):
    """Order-independent fingerprint of response identities and measured values.

    Only ModelID, broad_id, dose and logfold_change participate; the DataFrame
    index and unrelated columns are ignored. Duplicate rows are retained.
    """
    records = response_records(frame)
    return digest(
        sorted(records, key=lambda r: (r["cell_id"], r["drug_id"], r["dose"], encode(r)))
    )
