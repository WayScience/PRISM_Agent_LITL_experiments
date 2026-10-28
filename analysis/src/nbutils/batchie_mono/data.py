"""
Data class for validated and encoded drug screens to unify interface
    across sampler backends.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .batchie_config import (
    COMPOUND_COLUMN,
    CORE_COLUMNS,
    DOSE_COLUMN,
    MODEL_COLUMN,
    RESPONSE_COLUMN,
    RESPONSE_KEY_COLUMNS,
)


@dataclass(frozen=True)
class EncodedObservations:
    responses: np.ndarray
    model_ids: np.ndarray
    compound_ids: np.ndarray
    models: np.ndarray
    compounds: np.ndarray
    fixed_dose: float

    @classmethod
    def from_frame(cls, frame: pd.DataFrame):
        """
        Create an EncodedObservations instance from a pandas DataFrame.        

        :param frame: A pandas DataFrame containing the observed data.
            Expects just one dose and all unique model_id/compound_id combinations.
        """
        missing = sorted(set(CORE_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Observed data are missing columns: {missing}")
        if frame.empty or frame[CORE_COLUMNS].isna().any().any():
            raise ValueError(
                "Observed data must be nonempty and contain no missing values"
            )
        if frame[DOSE_COLUMN].nunique() != 1:
            raise ValueError("BayesianFixedDoseModel requires exactly one fixed dose")
        if frame.duplicated(RESPONSE_KEY_COLUMNS).any():
            raise ValueError("Observed data contain duplicate response keys")
        responses = frame[RESPONSE_COLUMN].to_numpy(dtype=float, copy=True)
        dose = float(frame[DOSE_COLUMN].iloc[0])
        if not np.isfinite(responses).all() or not np.isfinite(dose):
            raise ValueError("Responses and dose must be finite")
        models = np.sort(frame[MODEL_COLUMN].unique())
        compounds = np.sort(frame[COMPOUND_COLUMN].unique())
        model_ids = (
            frame[MODEL_COLUMN]
            .map(dict(zip(models, range(len(models)))))
            .to_numpy(dtype=int)
        )
        compound_ids = (
            frame[COMPOUND_COLUMN]
            .map(dict(zip(compounds, range(len(compounds)))))
            .to_numpy(dtype=int)
        )
        arrays = (responses, model_ids, compound_ids, models, compounds)
        for array in arrays:
            array.flags.writeable = False
        return cls(*arrays, dose)

    def candidate_indices(self, frame: pd.DataFrame):
        """
        Get the indices of candidate model/compound pairs in the encoded observations.

        :param frame: A pandas DataFrame containing candidate pairs.
            Must have the same fixed dose as the encoded observations.
        """
        missing = sorted(set(RESPONSE_KEY_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Candidate pairs are missing columns: {missing}")
        if not np.allclose(frame[DOSE_COLUMN], self.fixed_dose):
            raise ValueError("Candidate pairs must use the model's fixed dose")
        try:
            cells = dict(zip(self.models, range(len(self.models))))
            drugs = dict(zip(self.compounds, range(len(self.compounds))))
            return (
                np.array([cells[value] for value in frame[MODEL_COLUMN]], dtype=int),
                np.array([drugs[value] for value in frame[COMPOUND_COLUMN]], dtype=int),
            )
        except KeyError as error:
            raise KeyError(
                f"Candidate contains an unseen identifier: {error.args[0]}"
            ) from error
