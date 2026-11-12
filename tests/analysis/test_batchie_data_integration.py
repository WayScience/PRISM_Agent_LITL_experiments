"""Optional local PRISM-to-encoding integration; no downloads or synthetic fallback."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbutils.batchie_mono.config import CORE_COLUMNS, RESPONSE_KEY_COLUMNS
from nbutils.batchie_mono.data import EncodedObservations


@pytest.mark.integration
def test_prism_parquet_encoding_and_candidate_alignment():
    path = (
        Path(__file__).resolve().parents[2]
        / "data/secondary-screen-observed-response-long.parquet"
    )
    if not path.is_file():
        pytest.skip("Local wrangled PRISM response Parquet is not available")
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    try:
        if not set(CORE_COLUMNS).issubset(pq.ParquetFile(path).schema_arrow.names):
            pytest.skip("Local PRISM Parquet is missing required response columns")
        frame = pd.read_parquet(
            path, columns=CORE_COLUMNS, filters=[("dose", "==", 2.5)], engine="pyarrow"
        )
    except (OSError, pa.ArrowException) as error:
        pytest.skip(f"Local PRISM Parquet cannot be read as expected: {error}")

    frame = frame.dropna(subset=CORE_COLUMNS)
    if (
        frame.empty
        or not pd.api.types.is_numeric_dtype(frame.logfold_change)
        or not np.isfinite(frame.logfold_change).all()
        or not frame.ModelID.map(lambda value: isinstance(value, str)).all()
        or not frame.broad_id.map(lambda value: isinstance(value, str)).all()
    ):
        pytest.skip("Expected finite responses and string identifiers at PRISM dose 2.5")

    # Match the demos: one dose, drop missing core values, mean repeated screens.
    frame = frame.groupby(RESPONSE_KEY_COLUMNS, as_index=False).logfold_change.mean()
    cells = sorted(frame.ModelID.unique())[:2]
    frame = frame[frame.ModelID.isin(cells)].groupby("ModelID", sort=True).head(8)
    if len(cells) < 2 or frame.broad_id.nunique() < 2:
        pytest.skip("PRISM subset needs at least two cells and two compounds")
    frame = frame.iloc[::-1].copy()  # Exercise encoding independently of sorted row order.

    # Do not catch encoding/assertion failures: valid input must satisfy the contract.
    encoded = EncodedObservations.from_frame(frame)
    assert encoded.fixed_dose == 2.5
    np.testing.assert_array_equal(encoded.responses, frame.logfold_change.to_numpy())
    np.testing.assert_array_equal(encoded.models[encoded.model_ids], frame.ModelID)
    np.testing.assert_array_equal(encoded.compounds[encoded.compound_ids], frame.broad_id)
    for array in (
        encoded.responses, encoded.model_ids, encoded.compound_ids,
        encoded.models, encoded.compounds,
    ):
        assert not array.flags.writeable

    candidates = frame.iloc[[-1, 0, -1]][RESPONSE_KEY_COLUMNS]
    cells, compounds = encoded.candidate_indices(candidates)
    np.testing.assert_array_equal(encoded.models[cells], candidates.ModelID)
    np.testing.assert_array_equal(encoded.compounds[compounds], candidates.broad_id)
    with pytest.raises(ValueError, match="duplicate response keys"):
        EncodedObservations.from_frame(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="fixed dose"):
        encoded.candidate_indices(candidates.assign(dose=1.0))
