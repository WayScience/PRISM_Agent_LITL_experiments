"""
Records for model predictions / next batch suggestion
"""

import math


from nbutils.batchie_mono.logger._storage import encode, uid


class PredictionRecords:
    """
    Handles recording of model predictions and next batch suggestions for retrospective analysis.
    """
    def record_predictions(self, fit_action_id, records, *, metadata=None):
        """Save predictions made now, before reveal, from the current model.

        Each record has response_id and point_estimate; optional standard
        deviations and intervals must be finite when present. Other values go
        into an explicitly named `extra` JSON object. No model calls occur here.
        """
        records = list(records)
        ids = [r["response_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate prediction response IDs")
        fields = (
            "point_estimate",
            "posterior_std",
            "predictive_std",
            "interval_lower",
            "interval_upper",
        )
        values = []
        for r in records:
            row = []
            for field in fields:
                value = r.get(field)
                if value is None and field == "point_estimate":
                    raise ValueError("Point estimate is required")
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(
                            "Predictions must be finite; absent uncertainty must be None"
                        )
                row.append(value)
            values.append((*row, encode(r.get("extra", {}))))
        pid = uid()
        with self.transaction():
            fit = self.one(
                "SELECT * FROM actions WHERE action_id=? AND status='completed'",
                (fit_action_id,),
            )
            state = self.current_state(fit["experiment_id"])
            model = self.one(
                "SELECT * FROM model_states WHERE model_state_id=?",
                (state["model_state_id"],),
            )
            if model["producing_action_id"] != fit_action_id:
                raise ValueError("Predictions must use the current model state")
            exp = self.one(
                "SELECT * FROM experiments WHERE experiment_id=?",
                (fit["experiment_id"],),
            )
            if exp["status"] != "running":
                raise ValueError("Experiment is not running")
            available = {
                r["response_id"]
                for r in self.rows(
                    "SELECT response_id FROM split_memberships WHERE split_id=?",
                    (exp["split_id"],),
                )
            }
            if not set(ids) <= available:
                raise ValueError("Prediction references a response outside the split")
            self.insert(
                "prediction_sets",
                prediction_id=pid,
                fit_action_id=fit_action_id,
                state_id=state["state_id"],
                metadata=encode(metadata or {}),
            )
            self.connection.executemany(
                "INSERT INTO prediction_values VALUES (?,?,?,?,?,?,?,?,?)",
                [(pid, rid, i, *v) for i, (rid, v) in enumerate(zip(ids, values))],
            )
        return pid

    def complete_acquisition(
        self,
        action_id,
        selected,
        *,
        prediction_id=None,
        policy,
        requested_size,
        objective="none",
        metrics=None
    ):
        """
        Select and immediately reveal a batch from a retrospective dataset.        
        """
        selected = list(selected)
        if (
            isinstance(requested_size, bool)
            or not isinstance(requested_size, int)
            or requested_size < 0
        ):
            raise ValueError("requested_size must be a nonnegative integer")
        if len(selected) != len(set(selected)) or len(selected) > requested_size:
            raise ValueError("Duplicate selections or batch exceeds requested_size")
        with self.transaction():
            action, state = self._active(action_id, "acquire")
            if not set(selected) <= self.candidate_ids(action["experiment_id"]):
                raise ValueError(
                    "Selections must be unrevealed, eligible measured responses"
                )
            if prediction_id is not None:
                pred = self.one(
                    "SELECT * FROM prediction_sets WHERE prediction_id=?",
                    (prediction_id,),
                )
                if pred["state_id"] != state["state_id"]:
                    raise ValueError(
                        "Acquisition prediction set is from a different state"
                    )
                predicted = {
                    r["response_id"]
                    for r in self.rows(
                        "SELECT response_id FROM prediction_values WHERE prediction_id=?",
                        (prediction_id,),
                    )
                }
                if not set(selected) <= predicted:
                    raise ValueError("Predictions do not cover all selected responses")
            self.insert(
                "acquisitions",
                action_id=action_id,
                prediction_id=prediction_id,
                policy=policy,
                requested_size=requested_size,
                selected_size=len(selected),
                objective=objective,
            )
            self.connection.executemany(
                "INSERT INTO acquisition_items VALUES (?,?,?)",
                [(action_id, rid, i) for i, rid in enumerate(selected)],
            )
            did = uid()
            self.insert(
                "data_states",
                data_state_id=did,
                experiment_id=action["experiment_id"],
                parent_id=state["data_state_id"],
            )
            self.connection.executemany(
                "INSERT INTO reveals VALUES (?,?)", [(did, rid) for rid in selected]
            )
            errors = self.rows(
                "SELECT * FROM acquisition_errors WHERE action_id=?", (action_id,)
            )
            measured = len(errors)
            if measured:
                self._metric(
                    action_id,
                    "min_observed_lfc",
                    "acquired_batch",
                    min(r["actual"] for r in errors),
                    measured,
                )
                self._metric(
                    action_id,
                    "max_observed_lfc",
                    "acquired_batch",
                    max(r["actual"] for r in errors),
                    measured,
                )
            if prediction_id is not None and measured:
                self._metric(
                    action_id,
                    "rmse",
                    "acquired_batch",
                    math.sqrt(sum(r["residual"] ** 2 for r in errors) / measured),
                    measured,
                )
                self._metric(
                    action_id,
                    "mae",
                    "acquired_batch",
                    sum(abs(r["residual"]) for r in errors) / measured,
                    measured,
                )
            for metric, value in (metrics or {}).items():
                self._metric(action_id, metric, "simulation", value, measured)
            return self._advance(
                action, state, model_state_id=state["model_state_id"], data_state_id=did
            )

    def _metric(self, action_id, metric, scope, value, n):
        """
        Helps with inserting evaluation metrics for a given action within a specific scope.
        """
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Metric values must be finite")
        self.insert(
            "evaluations",
            action_id=action_id,
            metric=metric,
            scope=scope,
            value=value,
            n=n,
        )
