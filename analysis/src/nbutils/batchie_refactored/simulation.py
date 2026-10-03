"""
BATCHIE simulation orchestration module. 
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .batchie_config import COMPOUND_COLUMN, RESPONSE_COLUMN, RESPONSE_KEY_COLUMNS
from .models import ResponseModel
from .batchie_selector import BatchSelector


def run_simulation(
    *,
    background: pd.DataFrame,
    target_responses: pd.DataFrame,
    initial_compounds: list[str],
    model: ResponseModel,
    selector: BatchSelector,
    batch_size: int = 8,
    iterations: int = 10,
) -> pd.DataFrame:
    """Fit, choose a batch, reveal its target responses, and repeat.

    Inputs are prefiltered to one dose and unique response keys. Background
    excludes the target cell line and covers every candidate compound. Target
    truth is used only for revealing selected rows and retrospective scoring;
    selectors receive identifier columns only. ``evaluation_pairs`` is kept in
    the selector interface for compatibility but unused by these two policies.

    Return retrieval metrics, including the shared initial observations at
    iteration zero. Hits are the lowest ceil(10% * pool size) target LFCs,
    breaking ties by compound ID. The denominator is fixed across all rounds.
    Each fit starts afresh; there is no extra fit after the final reveal.
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
    records = []

    for iteration in range(iterations + 1):
        seen = set(observed_target[COMPOUND_COLUMN])
        records.append({
            "selector": selector.name,
            "iteration": iteration,
            "compounds_seen": len(seen),
            "top_hits_found": len(seen & hits),
            "top_hits_total": len(hits),
            "top_10pct_retrieval": len(seen & hits) / len(hits),
        })
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
        remaining = remaining[~remaining[COMPOUND_COLUMN].isin(selected)].copy()

    return pd.DataFrame.from_records(records)
