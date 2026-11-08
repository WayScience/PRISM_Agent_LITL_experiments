"""
Model facing API for logging experiment runs and associated metadata.
"""

import json


from nbutils.batchie_mono.logger._storage import SQLiteStore, encode, uid
from nbutils.batchie_mono.logger.artifacts import Artifact, ArtifactStore, load_inference_data
from nbutils.batchie_mono.logger.datasets import DatasetRecords
from nbutils.batchie_mono.logger.predictions import PredictionRecords
from nbutils.batchie_mono.logger.states import StateRecords


class ExperimentLogger(DatasetRecords, StateRecords, PredictionRecords, SQLiteStore):
    """Store fit, predict, acquisition artifacts in one SQLite database."""

    def __init__(self, db_path="experiments.sqlite", *, timeout=30):
        super().__init__(db_path, timeout=timeout)
        try:
            self.artifacts = ArtifactStore(self.path)
        except BaseException:
            # Artifact directory setup can fail after SQLite has opened.
            self.close()
            raise

    def complete_fit(self, action_id, response_ids, *, manifest, artifacts=()):
        """Publish a fit snapshot with exact input membership and artifact links."""
        return self.complete_model_update(
            action_id,
            response_ids,
            manifest=manifest,
            artifacts=artifacts,
            expected_kind="fit",
        )

    def complete_model_update(
        self, action_id, response_ids, *, manifest, artifacts=(), expected_kind=None
    ):
        """Publish a snapshot for fit/reflection/etc., without changing revealed data.

        The caller serializes artifacts first. A manifest must identify its
        schema version, model family and capabilities (e.g. predict, resume).
        response_ids records exactly which revealed responses this update used.
        Prior model lineage is the input experiment state; it does not imply
        warm-starting. Record fresh/warm-start semantics in the action config.
        """
        if (
            not isinstance(manifest, dict)
            or not {"schema_version", "model_family", "capabilities"} <= manifest.keys()
        ):
            raise ValueError(
                "Snapshot requires schema_version, model_family, capabilities"
            )
        if manifest["schema_version"] != 1 or not isinstance(
            manifest["capabilities"], list
        ):
            raise ValueError("Invalid snapshot manifest")
        payload = encode(manifest)
        ids = list(response_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate update inputs")
        artifacts = list(artifacts)
        # File checks are deliberately outside the database transaction.
        for artifact in artifacts:
            self.artifacts.verify(artifact)
        with self.transaction():
            action, state = self._active(action_id, expected_kind)
            if action["kind"] == "acquire":
                raise ValueError("Use complete_acquisition for acquisition actions")
            exp = self.one(
                "SELECT * FROM experiments WHERE experiment_id=?",
                (action["experiment_id"],),
            )
            if manifest["model_family"] != exp["model_family"]:
                raise ValueError("Snapshot family differs from experiment family")
            if not set(ids) <= self.revealed_ids(state["data_state_id"]):
                raise ValueError(
                    "Model update attempted to consume unrevealed responses"
                )
            mid = uid()
            self.insert(
                "model_states",
                model_state_id=mid,
                experiment_id=action["experiment_id"],
                producing_action_id=action_id,
                manifest=payload,
            )
            self.connection.executemany(
                "INSERT INTO fit_inputs VALUES (?,?,?)",
                [(action_id, rid, i) for i, rid in enumerate(ids)],
            )
            for artifact in artifacts:
                self.insert(
                    "artifacts",
                    artifact_id=uid(),
                    action_id=action_id,
                    role=artifact.role,
                    relative_path=artifact.relative_path,
                    format=artifact.format,
                    sha256=artifact.sha256,
                    size_bytes=artifact.size_bytes,
                    metadata=encode(artifact.metadata),
                )
            return self._advance(
                action, state, model_state_id=mid, data_state_id=state["data_state_id"]
            )

    def artifact_records(self, action_id):
        """Return validated descriptor shapes; bytes are checked when loaded."""
        return [
            Artifact(
                r["relative_path"],
                r["format"],
                r["sha256"],
                r["size_bytes"],
                r["role"],
                json.loads(r["metadata"]),
            )
            for r in self.rows(
                "SELECT * FROM artifacts WHERE action_id=?", (action_id,)
            )
        ]

    def load_run(self, action_id):
        """Compatibility convenience: return fit metadata and a fully loaded trace."""
        metadata = self.get_run_metadata(action_id)
        posterior = [
            a for a in self.artifact_records(action_id) if a.role == "posterior"
        ]
        if len(posterior) != 1:
            raise ValueError("Fit does not have exactly one posterior artifact")
        return metadata, load_inference_data(self.artifacts, posterior[0])

    def get_run_metadata(self, action_id):
        """Read a fit/action and snapshot without opening any artifact files."""
        action = self.one("SELECT * FROM actions WHERE action_id=?", (action_id,))
        action["config"] = json.loads(action["config"])
        states = self.rows(
            "SELECT manifest FROM model_states WHERE producing_action_id=?",
            (action_id,),
        )
        action["manifest"] = json.loads(states[0]["manifest"]) if states else None
        return action

    def log_batch(self, run_id, selected, revealed=None, metadata=None):
        """Existing simulation socket; DataFrame handling is isolated in the adapter."""
        from nbutils.batchie_mono.logger.integration import log_batch

        return log_batch(self, run_id, selected, revealed, metadata)

    def record_evaluations(self, action_id, records):
        """Attach optional diagnostics to a completed action.

        Records require metric, scope, value and n. Use distinct scopes for
        chain-level summaries. Existing values cannot be overwritten silently.
        Definitions use version 1; include changed definitions in metric names
        until an explicit schema migration introduces additional versions.
        """
        with self.transaction():
            self.one(
                "SELECT action_id FROM actions WHERE action_id=? AND status='completed'",
                (action_id,),
            )
            for row in records:
                if (
                    isinstance(row["n"], bool)
                    or not isinstance(row["n"], int)
                    or row["n"] < 0
                ):
                    raise ValueError("Evaluation n must be a nonnegative integer")
                self._metric(
                    action_id, row["metric"], row["scope"], row["value"], row["n"]
                )
