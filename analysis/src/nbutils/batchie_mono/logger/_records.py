"""
Helper to handle multiple types of records the experiment will involve.
"""

import json

import pandas as pd

from ._serialization import _frame, _json, _read_frame


class RecordsMixin:
    """
    Add run-linked record helpers to a host implementing SQLite primitives.
    """

    def log_prediction(self, run_id, frame, metadata=None):
        """
        Commit one prediction call's candidate IDs and outputs.

        :param run_id: Existing parent fit UUID.
        :param frame: DataFrame with unique column names and explicit identity
            columns. Row order and duplicates survive, but the index is
            discarded and JSON floating-point storage is not bitwise exact.
        :param metadata: Optional JSON-compatible call context, such as a seed.
        """
        return self._insert(
            "INSERT INTO predictions(run_id,metadata,frame_json) VALUES (?,?,?)",
            (run_id, _json(metadata or {}), _frame(frame)),
        )


    def load_prediction(self, prediction_id):
        """
        Return a prediction record with decoded metadata and a DataFrame.

        :param prediction_id: Integer returned by ``log_prediction``.
        :returns: Dictionary with prediction details and decoded metadata.
        """
        row = self._row(
            "SELECT * FROM predictions WHERE prediction_id=?", prediction_id
        )
        row["metadata"] = json.loads(row["metadata"])
        row["frame"] = _read_frame(row.pop("frame_json"))
        return row


    def log_batch(self, run_id, selected, revealed=None, metadata=None):
        """
        Commit a controller-supplied selection and optional revealed data.

        :param run_id: Existing parent fit UUID.
        :param selected: DataFrame of selected identities in caller-defined order.
        :param revealed: Optional DataFrame of actually revealed responses. None
            stores SQL NULL and is distinct from an empty DataFrame.
        :param metadata: Optional JSON-compatible acquisition context.
        """
        return self._insert(
            "INSERT INTO batches(run_id,metadata,selected_json,revealed_json) "
            "VALUES (?,?,?,?)",
            (
                run_id,
                _json(metadata or {}),
                _frame(selected),
                None if revealed is None else _frame(revealed),
            ),
        )


    def load_batch(self, batch_id):
        """
        Return one acquisition record, decoding frames and metadata.

        :param batch_id: Integer returned by ``log_batch``.
        :returns: Dictionary with batch details and decoded metadata.
        """
        row = self._row("SELECT * FROM batches WHERE batch_id=?", batch_id)
        row["metadata"] = json.loads(row["metadata"])
        row["selected"] = _read_frame(row.pop("selected_json"))
        value = row.pop("revealed_json")
        row["revealed"] = None if value is None else _read_frame(value)
        return row


    def list_predictions(self, run_id):
        """
        List a run's prediction IDs in ascending ID order.

        :param run_id: Parent fit UUID to match exactly.
        :returns: pandas.DataFrame with one ``prediction_id`` column; empty if the run
            has no predictions or does not exist.
        """
        return pd.read_sql_query(
            "SELECT prediction_id FROM predictions WHERE run_id=? ORDER BY prediction_id",
            self.connection,
            params=(run_id,),
        )


    def list_batches(self, run_id):
        """
        List a run's batch IDs in ascending ID order.

        :param run_id: Parent fit UUID to match exactly.
        :returns: pandas.DataFrame with one ``batch_id`` column; empty if the run has
            no batches or does not exist.
        """
        return pd.read_sql_query(
            "SELECT batch_id FROM batches WHERE run_id=? ORDER BY batch_id",
            self.connection,
            params=(run_id,),
        )
