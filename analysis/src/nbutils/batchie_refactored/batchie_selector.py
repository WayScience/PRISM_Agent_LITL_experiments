"""
BATCHIE selector implementations. 
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .batchie_config import COMPOUND_COLUMN, RESPONSE_COLUMN, RESPONSE_KEY_COLUMNS
from .models import ResponseModel


class BatchSelector(ABC):
    """Choose compound IDs from response-free, single-target candidate rows."""

    name: str

    @abstractmethod
    def select(
        self,
        *,
        model: ResponseModel,
        candidate_pairs: pd.DataFrame,
        evaluation_pairs: pd.DataFrame,
        batch_size: int,
    ) -> list[str]:
        raise NotImplementedError


class RandomSelector(BatchSelector):
    """Uniform sampling without replacement; RNG advances between rounds."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def select(self, *, model, candidate_pairs, evaluation_pairs, batch_size) -> list[str]:
        compounds = np.sort(candidate_pairs[COMPOUND_COLUMN].unique())
        selected = self._rng.choice(
            compounds, size=min(batch_size, len(compounds)), replace=False
        )
        return sorted(str(compound) for compound in selected)


class PredictedLowestSelector(BatchSelector):
    """Greedily select lowest predicted LFC (strongest predicted killing)."""

    name = "predicted_lowest"

    def select(self, *, model, candidate_pairs, evaluation_pairs, batch_size) -> list[str]:
        return (
            model.predict(candidate_pairs)
            .sort_values(["predicted_mean", COMPOUND_COLUMN])
            .head(batch_size)[COMPOUND_COLUMN]
            .astype(str)
            .tolist()
        )
