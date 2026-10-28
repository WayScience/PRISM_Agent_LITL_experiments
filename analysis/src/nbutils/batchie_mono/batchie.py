"""
Dose-response model with BATCHIE architecture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nbutils.batchie_mono.batchie_config import RESPONSE_KEY_COLUMNS
from nbutils.batchie_mono.models import BayesianResponseModel
from nbutils.batchie_mono.posterior import Posterior, posterior_from_trace
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.data import EncodedObservations
from nbutils.batchie_mono.sampling import PosteriorSampler
from nbutils.batchie_mono.trace import validate_trace


__all__ = ["BayesianFixedDoseModel", "Posterior"]


class BayesianFixedDoseModel(BayesianResponseModel):
    """
    Fixed-dose Gaussian factorization with PyMC NUTS or blocked Gibbs sampling.
    Adapted from BATCHIE (Tosh et al., 2025, 10.1038/s41467-024-55287-7) for monotherapy screens.

    Fits observed log2-fold-changes in a viability screen as:
    alpha + a[k] + b[i] + U[k] @ V[i].
    - alpha is the global intercept
    - a[k] is the model-specific effect (learnable constant per model)
    - b[i] is the compound-specific effect (learnable constant per compound)
    - U[k] @ V[i] is the low-rank interaction between model k and compound i,
        where U[k] and V[i] are both learnable embeddings of dimension `rank`,
        added on top of the scalar a[k] and b[i] effects to model similarly
        "natured" interactions between the same compound and different models and vice versa.

    This class owns just the factorization vocabulary and API facing the selector,
        to propose next screen experiments based on current state of model. 
    """

    def __init__(
        self,
        *,
        sampler: PosteriorSampler,
        spec: ModelSpec = ModelSpec(),
        prediction_seed: int = 0,
    ) -> None:
        """
        Initialize the BayesianFixedDoseModel

        :param sampler: Posterior sampler instance for generating posterior draws.
        :param spec: Model specification containing hyperparameters and settings.
        :param prediction_seed: Random seed for prediction reproducibility.
        """
        
        self.sampler = sampler
        self.spec = spec
        self.prediction_seed = prediction_seed
        self.clear_fit()

    def clear_fit(self):
        """Clear the current fit, resetting all model attributes."""
        self._is_fitted = False
        self.inference_data_ = None
        self.posterior_ = None
        self.training_data_ = None
        self.models_ = self.compounds_ = self.fixed_dose_ = None
        self.model_index_ = self.compound_index_ = None

    def fit(self, observed_data: pd.DataFrame) -> BayesianFixedDoseModel:
        """
        Fresh fit at one dose; a failed refit cannot expose an old posterior.
        """
        self.clear_fit()
        
        data = EncodedObservations.from_frame(observed_data)
        result = self.sampler.sample(data, self.spec)
        validate_trace(result.trace, data, self.spec)
        posterior = posterior_from_trace(result.trace)

        self.training_data_ = data
        self.inference_data_ = result.trace
        self.posterior_ = posterior
        self.models_, self.compounds_ = data.models, data.compounds
        self.fixed_dose_ = data.fixed_dose
        self.model_index_ = dict(zip(data.models, range(len(data.models))))
        self.compound_index_ = dict(zip(data.compounds, range(len(data.compounds))))
        self._is_fitted = True
        return self

    def posterior_draws(self, candidate_pairs: pd.DataFrame) -> np.ndarray:
        """
        Return aligned noise-free mean draws, shape (draw, candidate).

        Preserve candidate row order and duplicates. Both IDs must have been
        observed in training; the pairing need not have been. Dose must match
        under np.allclose. Raises RuntimeError before fit and KeyError for new IDs.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit before requesting posterior draws")
        
        model_ids, compound_ids = self.training_data_.candidate_indices(candidate_pairs)

        posterior = self.posterior_

        return (
            posterior.global_offset[:, None] # intercept (ambient)
            + posterior.cell_offsets[:, model_ids] # intercept for each model (cell-line)
            + posterior.compound_offsets[:, compound_ids] # intercept for each compound (broad_id)

            # the interaction term from multiplicative embeddings between models and compounds
            + np.einsum( 
                # einsum pattern for summing over the embedding dimension (r) 
                # to get interaction term for each draw (s) and candidate (n)
                "snr,snr->sn", 
                posterior.cell_embeddings[:, model_ids, :],
                posterior.compound_embeddings[:, compound_ids, :],
            )
        )

    def observation_variance_draws(self) -> np.ndarray:
        """Return shared observation variances, shape (draw,), aligned to mean draws."""
        if not self._is_fitted:
            raise RuntimeError("Call fit before requesting observation variance")
        return 1.0 / self.posterior_.observation_precision

    def predict(self, candidate_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Return identifiers, posterior mean/SD, predictive SD, and 90% predictive limits.

        posterior_std uses ddof=1; predictive_std adds mean observation variance.
        Limits are 5th/95th percentiles of one noisy value per posterior draw.
        Noise uses seed + 1 on each call. Candidate index/order are preserved.
        At least two draws are needed for finite SDs (not enforced).

        :param candidate_pairs: DataFrame containing candidate model-compound pairs.
        :return: DataFrame with predicted mean, posterior SD, predictive SD, and 90% predictive intervals.        
        """
        draws = self.posterior_draws(candidate_pairs)
        variance = self.observation_variance_draws()
        result = candidate_pairs[RESPONSE_KEY_COLUMNS].copy()
        result["predicted_mean"] = draws.mean(axis=0)
        result["posterior_std"] = draws.std(axis=0, ddof=1)
        result["predictive_std"] = np.sqrt(draws.var(axis=0, ddof=1) + variance.mean())
        noise = np.random.default_rng(self.prediction_seed + 1).normal(size=draws.shape)
        predictive_draws = draws + np.sqrt(variance)[:, None] * noise
        result["interval_lower"] = np.quantile(predictive_draws, 0.05, axis=0)
        result["interval_upper"] = np.quantile(predictive_draws, 0.95, axis=0)
        return result
