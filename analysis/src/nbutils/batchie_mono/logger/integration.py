"""
Bayesian Factorization integration adapter for the experiment logger.
"""

import json
import numpy as np
import pandas as pd

from nbutils.batchie_mono.logger.artifacts import write_inference_data
from nbutils.batchie_mono.logger.fingerprints import fingerprint_responses
from nbutils.batchie_mono.logger.response_data import (
    KEYS,
    dataset_for_action,
    frame_ids,
    response_records,
    verify_truth,
)


def default_snapshot(model, store):
    """Bayesian snapshot, or an explicit future model-provided serializer.

    Non-Bayesian implementations implement logging_snapshot(store) returning
    (manifest, artifacts), or pass a snapshot callable to LoggedModel. We never
    silently pickle models or claim that a trace is a resumable sampler state.
    """
    trace = getattr(model, "inference_data_", None)
    if trace is None:
        method = getattr(model, "logging_snapshot", None)
        if method is None:
            raise TypeError("Model needs inference_data_ or logging_snapshot(store)")
        return method(store)
    attrs = dict(trace.attrs)
    for field in ("model_config", "inference_config"):
        if isinstance(attrs.get(field), str):
            attrs[field] = json.loads(attrs[field])
    attrs = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in attrs.items()
    }
    artifact = write_inference_data(store, trace)
    return {
        "schema_version": 1,
        "model_family": "bayesian_factorization",
        "capabilities": ["posterior_analysis", "prediction_with_adapter"],
        "implementation": type(model).__module__ + "." + type(model).__qualname__,
        "prediction_seed": getattr(model, "prediction_seed", None),
        "trace_metadata": attrs,
    }, [artifact]


class LoggedModel:
    """Delegate fit/predict exactly once while recording their actual outcomes.

    posterior_draws and all other attributes delegate unchanged. Only calls to
    predict create prediction records. A future selector that uses draws alone
    should explicitly supply its pre-reveal prediction summary to the logger.
    """

    def __init__(self, model, logger, experiment_name, *, snapshot=default_snapshot):
        if not experiment_name:
            raise ValueError("experiment_name is required when logging")
        self.model, self.logger, self.name = model, logger, experiment_name
        self.snapshot = snapshot
        self.last_run_id = self.last_prediction_id = None
        self.trajectory_id = self.dataset_id = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def configure(
        self,
        *,
        background,
        target,
        initial,
        selector,
        batch_size,
        iterations,
        metadata=None,
        initial_metrics=None,
    ):
        """Register immutable source data and a split, then create one trajectory.

        Ground truth lives in the evaluator/logger; only the observed frame is
        passed to model.fit. Caller provenance should supply source versions,
        code commit, preparation recipe, and external split identifiers.
        """
        all_data = pd.concat([background, target], ignore_index=True)
        provenance = dict(metadata or {})
        self.dataset_id = self.logger.register_dataset(
            response_records(all_data),
            description={
                "response": "viability_lfc",
                "adapter_schema": 1,
                "provenance": provenance.get("dataset", {}),
            },
        )
        initial_frame = pd.concat([background, initial], ignore_index=True)
        initial_ids = frame_ids(self.logger, self.dataset_id, initial_frame)
        target_ids = frame_ids(self.logger, self.dataset_id, target)
        split = self.logger.create_split(
            self.dataset_id,
            initial=initial_ids,
            pool=sorted(set(target_ids) - set(initial_ids)),
            definition={"protocol": "background_plus_adaptive_target", "version": 1},
        )
        self.family = (
            "bayesian_factorization"
            if hasattr(self.model, "inference_data_")
            else type(self.model).__module__ + "." + type(self.model).__qualname__
        )
        config = {
            "provenance": provenance,
            "dataset": {
                "background": fingerprint_responses(background),
                "target": fingerprint_responses(target),
                "background_rows": len(background),
                "target_rows": len(target),
            },
            "selector": selector.name,
            "batch_size": batch_size,
            "iterations": iterations,
            "initial_compounds": initial["broad_id"].tolist(),
            "initial_metrics": dict(initial_metrics or {}),
            "prediction_seed": getattr(self.model, "prediction_seed", None),
        }
        # Capture settings, not arbitrary model internals or mutable live references.
        for label, obj in [
            ("model", getattr(self.model, "spec", None)),
            ("sampling", getattr(getattr(self.model, "sampler", None), "config", None)),
        ]:
            if hasattr(obj, "model_dump"):
                config[label] = obj.model_dump(mode="json")
        rng = getattr(selector, "_rng", None)
        if rng is not None:
            config["selector_rng_state"] = rng.bit_generator.state
        self.trajectory_id = self.logger.start_experiment(
            self.name, split, model_family=self.family, config=config
        )

    def fit(self, observed_data):
        if self.trajectory_id is None:
            raise RuntimeError("Configure the experiment before fitting")
        ids = frame_ids(self.logger, self.dataset_id, observed_data)
        revealed = self.logger.revealed_ids(
            self.logger.current_state(self.trajectory_id)["data_state_id"]
        )
        if not set(ids) <= revealed:
            raise ValueError("Fit input contains unrevealed responses")
        # Verify the actual values supplied to fit, not only the identifiers.
        verify_truth(self.logger, self.dataset_id, observed_data)
        self.last_run_id = self.last_prediction_id = None
        aid = self.logger.start_action(
            self.trajectory_id, "fit", config={"initialization": "fresh"}
        )
        try:
            self.model.fit(observed_data)
            manifest, artifacts = self.snapshot(self.model, self.logger.artifacts)
            self.logger.complete_fit(aid, ids, manifest=manifest, artifacts=artifacts)
        except BaseException as error:
            self.logger.fail_action(
                aid,
                error,
                interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)),
            )
            raise
        self.last_run_id = aid
        return self

    def begin_acquisition(self, iteration):
        """Reserve an action before selection so prediction/selector failures are logged."""
        return self.logger.start_action(
            self.trajectory_id, "acquire", config={"iteration": iteration}
        )

    def predict(self, candidate_pairs):
        result = self.model.predict(candidate_pairs)
        ids = frame_ids(self.logger, self.dataset_id, result)
        expected = frame_ids(self.logger, self.dataset_id, candidate_pairs)
        if ids != expected:
            raise ValueError("Prediction output must preserve requested response order")
        records = []
        known = set(
            KEYS
            + [
                "predicted_mean",
                "posterior_std",
                "predictive_std",
                "interval_lower",
                "interval_upper",
            ]
        )
        for rid, row in zip(ids, result.to_dict("records")):
            records.append(
                {
                    "response_id": rid,
                    "point_estimate": row["predicted_mean"],
                    **{
                        k: row.get(k)
                        for k in (
                            "posterior_std",
                            "predictive_std",
                            "interval_lower",
                            "interval_upper",
                        )
                    },
                    "extra": {
                        k: None
                        if isinstance(v, (float, np.floating)) and np.isnan(v)
                        else v
                        for k, v in row.items()
                        if k not in known
                    },
                }
            )
        self.last_prediction_id = self.logger.record_predictions(
            self.last_run_id,
            records,
            metadata={
                "estimator": (
                    "posterior_mean"
                    if self.family == "bayesian_factorization"
                    else "model_point_estimate"
                ),
                "prediction_seed": getattr(self.model, "prediction_seed", None),
                "interval_level": (
                    0.9 if self.family == "bayesian_factorization" else None
                ),
                "posterior_std_semantics": (
                    "latent_mean" if self.family == "bayesian_factorization" else None
                ),
            },
        )
        return result


def log_batch(logger, run_id, selected, revealed, metadata=None):
    """Compatibility socket for the existing simulation's log_batch call.

    This adapter requires complete synchronous reveal, matching simulation.py.
    Core APIs take IDs directly and do not depend on this DataFrame convention.
    """
    info = dataset_for_action(logger, run_id)
    state = logger.current_state(info["experiment_id"])
    model = logger.one(
        "SELECT * FROM model_states WHERE model_state_id=?", (state["model_state_id"],)
    )
    if model["producing_action_id"] != run_id:
        raise ValueError("Batch references a stale model")
    selected_ids = frame_ids(logger, info["dataset_id"], selected)
    if (
        revealed is None
        or set(frame_ids(logger, info["dataset_id"], revealed)) != set(selected_ids)
        or len(revealed) != len(selected_ids)
    ):
        raise ValueError(
            "Synchronous acquisition requires one revealed response per selection"
        )
    verify_truth(logger, info["dataset_id"], revealed)
    metadata = dict(metadata or {})
    config = json.loads(info["config"])
    aid = metadata.get("acquisition_action_id")
    if aid is None:
        aid = logger.start_action(info["experiment_id"], "acquire", config=metadata)
    elif (
        logger.one("SELECT experiment_id FROM actions WHERE action_id=?", (aid,))[
            "experiment_id"
        ]
        != info["experiment_id"]
    ):
        raise ValueError("Acquisition action belongs to a different experiment")
    try:
        metrics = {
            k: v
            for k, v in metadata.get("metrics_after", {}).items()
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
        }
        logger.complete_acquisition(
            aid,
            selected_ids,
            prediction_id=metadata.get("last_prediction_id"),
            policy=config["selector"],
            requested_size=config["batch_size"],
            objective=(
                "minimize" if config["selector"] == "predicted_lowest" else "none"
            ),
            metrics=metrics,
        )
    except BaseException as error:
        logger.fail_action(aid, error)
        raise
    return aid
