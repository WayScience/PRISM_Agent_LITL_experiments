"""
Model facing API for logging experiment runs and associated metadata.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import arviz as az
import numpy as np
import pandas as pd

from ._records import RecordsMixin
from ._serialization import _decode_run_metadata, _json
from ._storage import SQLiteStore


class ExperimentLogger(RecordsMixin, SQLiteStore):
    """Store fit, predict, acquisition artifacts in one SQLite database."""


    def log_run(
        self,
        experiment_name: str,
        trace: any,
        *,
        trajectory_id: str | None = None,
        batch_index: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Log an experiment with its associated trace, metadata, and optional labels.

        :param experiment_name: Nonempty string grouping related fits.
        :param trace: ArviZ InferenceData with nonempty posterior ``chain`` and
            ``draw`` axes. Every group, coordinate, and attribute supported
            by the h5netcdf backend is serialized, not just the posterior.
        :param trajectory_id: Optional label grouping successive adaptive fits.
        :param batch_index: Optional nonnegative Python integer; booleans are
            rejected. This is a label, not a uniqueness constraint.
        :param metadata: Optional JSON-compatible caller metadata. NumPy scalars
            and arrays are supported; nonfinite numbers are rejected.
        :returns: str: Newly generated UUID identifying the committed fit row.
        :raises ValueError: Invalid name, batch index, posterior axes, or JSON
            values (including malformed configuration JSON strings).
        :raises TypeError: Metadata contains an unsupported JSON value.
        """
        if not experiment_name or not isinstance(experiment_name, str):
            raise ValueError("experiment_name must be a nonempty string")
        if batch_index is not None and (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or batch_index < 0
        ):
            raise ValueError("batch_index must be a nonnegative integer")
        if not hasattr(trace, "posterior") or any(
            trace.posterior.sizes.get(axis, 0) < 1 for axis in ("chain", "draw")
        ):
            raise ValueError("Trace requires nonempty posterior chain and draw axes")
        attrs = dict(trace.attrs)
        scalar_keys = (
            "backend",
            "sampling_parameterization",
            "seed",
            "raw_postwarmup_divergences",
            "raw_postwarmup_tree_depth_hits",
        )
        scalars = [attrs.pop(key, None) for key in scalar_keys]
        scalars = [x.item() if isinstance(x, np.generic) else x for x in scalars]
        configs = []
        for key in ("model_config", "inference_config"):
            value = attrs.pop(key, {})
            configs.append(
                _json(json.loads(value) if isinstance(value, str) else value)
            )
        remaining, caller = _json(attrs), _json(metadata or {})
        run_id = str(uuid4())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.nc"
            trace.to_netcdf(str(path), engine="h5netcdf")
            blob = path.read_bytes()
        self._insert(
            "INSERT INTO experiment_runs(run_id,experiment_name,trajectory_id,"
            "batch_index,backend,parameterization,seed,raw_postwarmup_divergences,"
            "raw_postwarmup_tree_depth_hits,model_config,inference_config,"
            "trace_metadata,metadata,trace_netcdf) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                experiment_name,
                trajectory_id,
                batch_index,
                *scalars,
                *configs,
                remaining,
                caller,
                blob,
            ),
        )
        return run_id


    def get_run_metadata(self, run_id: str):
        """
        Read one fit's metadata without fetching its NetCDF BLOB.
        
        :param run_id: Fit UUID returned by ``log_run``.
        :return: Decoded metadata dictionary.
        :raises KeyError: No fit has the requested ID.
        """
        return _decode_run_metadata(
            self._row(
                "SELECT run_id,experiment_name,trajectory_id,batch_index,timestamp,"
                "backend,parameterization,seed,raw_postwarmup_divergences,"
                "raw_postwarmup_tree_depth_hits,model_config,inference_config,"
                "trace_metadata,metadata FROM experiment_runs WHERE run_id=?",
                run_id,
            )
        )


    def load_run(self, run_id: str):
        """
        Reload all stored trace groups into memory without refitting.

        :param run_id: Fit UUID returned by ``log_run``.
        :return: Tuple ``(metadata, trace)`` with the same decoded metadata as
            ``get_run_metadata`` and a fully loaded ArviZ InferenceData.
        """
        record = self._row("SELECT * FROM experiment_runs WHERE run_id=?", run_id)
        blob = record.pop("trace_netcdf")
        _decode_run_metadata(record)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.nc"
            path.write_bytes(blob)
            trace = az.from_netcdf(str(path), engine="h5netcdf")
            try:
                for group in trace.groups():
                    trace[group].load()
            finally:
                trace.close()
        return record, trace


    def list_runs(self, experiment_name: str = None, *, trajectory_id: str = None):
        """
        List lightweight fit summaries in insertion order, without traces.

        :param experiment_name: Optional exact experiment-name filter.
        :param trajectory_id: Optional exact trajectory filter. When both filters
            are supplied, a row must match both. None omits a filter;
            an empty string is a literal value, not a wildcard.
        :return: pandas.DataFrame with columns ``run_id``, ``experiment_name``,
            ``trajectory_id``, ``batch_index``, ``timestamp``, ``backend``,
            ``parameterization``, and ``seed``. No matches yield an empty frame
            with these columns. Configuration and JSON metadata are omitted;
            use ``get_run_metadata`` for one run's full metadata.
        """
        predicates, params = [], []
        if experiment_name is not None:
            predicates.append("experiment_name=?")
            params.append(experiment_name)
        if trajectory_id is not None:
            predicates.append("trajectory_id=?")
            params.append(trajectory_id)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        return pd.read_sql_query(
            "SELECT run_id,experiment_name,trajectory_id,batch_index,timestamp,"
            "backend,parameterization,seed FROM experiment_runs" + where + " ORDER BY rowid",
            self.connection,
            params=params,
        )
