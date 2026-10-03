"""Response models for single-agent selective screening."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ResponseModel(ABC):
    """Interface shared by batchie and other models."""

    @abstractmethod
    def fit(self, observed_data: pd.DataFrame) -> "ResponseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BayesianResponseModel(ResponseModel):
    """Response model that retains aligned joint posterior parameter draws."""

    @abstractmethod
    def posterior_draws(self, candidate_pairs: pd.DataFrame) -> np.ndarray:
        """Return conditional-mean draws with shape (draw, candidate)."""
        raise NotImplementedError

    @abstractmethod
    def observation_variance_draws(self) -> np.ndarray:
        raise NotImplementedError
