"""
BATCHIE simulation orchestration module. 
"""

import numpy as np
import pandas as pd

from nbutils.batchie_mono.config import (
    COMPOUND_COLUMN,
    RESPONSE_COLUMN,
    RESPONSE_KEY_COLUMNS,
)
from nbutils.batchie_mono.models import ResponseModel
from nbutils.batchie_mono.model_acquisition import BatchSelector


def run_simulation(
    *,
    background: pd.DataFrame,
    target_responses: pd.DataFrame,
    initial_compounds: list[str],
    model: ResponseModel,
    selector: BatchSelector,
    batch_size: int = 8,
    iterations: int = 10,
    logger: object | None = None,
    experiment_name: str | None = None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """
    Fit, choose a batch, reveal its target responses, and repeat.

    Inputs are prefiltered to one dose and unique response keys. Background
    excludes the target cell line and covers every candidate compound. Target
    truth is used only for revealing selected rows and retrospective scoring;
    selectors receive identifier columns only. ``evaluation_pairs`` is kept in
    the selector interface for compatibility but unused by these two policies.

    Return retrieval metrics, including the shared initial observations at
    iteration zero. Hits are the lowest ceil(10% * pool size) target LFCs,
    breaking ties by compound ID. The denominator is fixed across all rounds.
    Each fit starts afresh; there is no extra fit after the final reveal.

    Optional logging creates a fresh trajectory per invocation and adds its ID
    to returned metrics. Pass a bare model exposing inference_data_ and
    prediction_seed; orchestration wraps it to record actual fit/predict calls.
    Input fingerprints are computed once; caller metadata holds the external
    source checksums, preparation recipe and seeds. Only revealed responses
    enter the model. Each write commits independently and errors propagate.
    Zero fits means zero database records (including zero requested rounds).

    :param background: DataFrame containing background response observations.
    :param target_responses: DataFrame containing target observations (prediction targets).
    :param initial_compounds: List of initial compound IDs revealed to the model outright.
    :param model: ResponseModel instance used for fitting.
    :param selector: BatchSelector instance used for selecting batches.
    :param batch_size: Number of compounds to select in each batch.
    :param iterations: Number of iterations to run the simulation.
    :param logger: Optional socket for logging all outcomes and traces as db.
    :param experiment_name: Required when logging; groups related trajectories.
    :param metadata: Caller-supplied provenance, stored separately from simulation context.
    :return: DataFrame containing retrieval metrics for each iteration.
    """
    observed_target = target_responses[
        target_responses[COMPOUND_COLUMN].isin(initial_compounds)
    ].copy()
    remaining = target_responses[
        ~target_responses[COMPOUND_COLUMN].isin(initial_compounds)
    ].copy()
    evaluation_pairs = target_responses[RESPONSE_KEY_COLUMNS].copy()
    n_hits = max(1, int(np.ceil(0.1 * len(target_responses))))
    hits = set(
        target_responses.sort_values([RESPONSE_COLUMN, COMPOUND_COLUMN])
        .head(n_hits)[COMPOUND_COLUMN]
    )
    def metrics(iteration):
        seen = set(observed_target[COMPOUND_COLUMN])
        return {
            "selector": selector.name,
            "iteration": iteration,
            "compounds_seen": len(seen),
            "top_hits_found": len(seen & hits),
            "top_hits_total": len(hits),
            "top_10pct_retrieval": len(seen & hits) / len(hits),
        }

    logged_model = None
    if logger is not None:
        from nbutils.batchie_mono.logger import LoggedModel, fingerprint_responses

        if isinstance(model, LoggedModel):
            raise ValueError("Pass a bare model when supplying logger to run_simulation")
        logged_model = LoggedModel(model, logger, experiment_name)
        logged_model.metadata = {
            "provenance": dict(metadata or {}),
            "dataset": {
                "background": fingerprint_responses(background),
                "target": fingerprint_responses(target_responses),
                "background_rows": len(background),
                "target_rows": len(target_responses),
            },
            "simulation": {
                "selector": selector.name,
                "batch_size": batch_size,
                "iterations": iterations,
                "initial_compounds": observed_target[COMPOUND_COLUMN].tolist(),
                "initial_metrics": metrics(0),
            },
        }
        model = logged_model

    records = []
    for iteration in range(iterations + 1):
        records.append(metrics(iteration))
        if iteration == iterations or remaining.empty:
            break

        model.fit(pd.concat([background, observed_target], ignore_index=True))
        selected = selector.select(
            model=model,
            candidate_pairs=remaining[RESPONSE_KEY_COLUMNS].copy(),
            evaluation_pairs=evaluation_pairs,
            batch_size=min(batch_size, len(remaining)),
        )
        newly_revealed = remaining[remaining[COMPOUND_COLUMN].isin(selected)]
        observed_target = pd.concat([observed_target, newly_revealed], ignore_index=True)
        if logged_model is not None:
            # Preserve selector order, not the pool's order, in the selection artifact.
            selected_pairs = remaining.set_index(COMPOUND_COLUMN, drop=False).loc[
                selected, RESPONSE_KEY_COLUMNS
            ]
            logger.log_batch(
                logged_model.last_run_id,
                selected_pairs,
                newly_revealed,
                metadata={
                    "iteration": iteration + 1,
                    "last_prediction_id": logged_model.last_prediction_id,
                    "metrics_before": records[-1],
                    "metrics_after": metrics(iteration + 1),
                },
            )
        remaining = remaining[~remaining[COMPOUND_COLUMN].isin(selected)].copy()

    result = pd.DataFrame.from_records(records)
    if logged_model is not None:
        result["trajectory_id"] = logged_model.trajectory_id
    return result
