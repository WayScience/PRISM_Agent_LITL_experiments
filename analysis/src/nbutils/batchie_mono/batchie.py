"""
Dose-response model with BATCHIE architecture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .batchie_config import CONDITION_COLUMNS, DOSE_COLUMN, MODEL_COLUMN, RESPONSE_COLUMN
from .models import BayesianResponseModel
from .posterior import Posterior

__all__ = ["BayesianFixedDoseModel", "Posterior"]


class BayesianFixedDoseModel(BayesianResponseModel):
    """
    Fixed-dose Gaussian factorization with PyMC NUTS sampling.
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

    Class parameters are primarily hyperparameters that control rank (model complexity), 
        the MCMC sampling schedule, and priors on what the
        distributions for model parameters should be. 
    """

    def __init__(
        self,
        *,
        rank: int = 4,
        burnin: int = 40,
        posterior_samples: int = 30,
        thin: int = 2,
        prior_shape: float = 1.1,
        prior_rate: float = 1.1,
        global_precision: float = 0.01,
        seed: int = 0,
        backend: str = "pymc",
        chains: int = 1,
        target_accept: float = 0.9,
    ) -> None:
        """
        Initialize the BayesianFixedDoseModel with the specified hyperparameters.

        :param rank: Dimensionality of the low-rank interaction embeddings.
        :param burnin: Gibbs burn-in sweeps or PyMC NUTS tuning steps.
        :param posterior_samples: Retained draws per chain after burn-in and thinning.
        :param thin: Thinning interval for posterior samples.
        :param prior_shape: Shape parameter for the Gamma prior on precision terms.
        :param prior_rate: Rate parameter for the Gamma prior on precision terms.
        :param global_precision: Precision of the global intercept prior.
        :param seed: Random seed for reproducibility.
        :param backend: "pymc" (default, NUTS) or "gibbs" (not implemented yet).
        :param chains: Number of PyMC chains, run sequentially; Gibbs requires one.
        :param target_accept: PyMC NUTS target acceptance probability.
        """
        values = [rank, burnin, posterior_samples, thin, prior_shape, prior_rate, global_precision]
        if any(value <= 0 for value in values):
            raise ValueError("All model hyperparameters must be positive")
        if backend not in {"gibbs", "pymc"}:
            raise ValueError("backend must be 'gibbs' or 'pymc'")
        if not isinstance(chains, int) or chains < 1 or (backend == "gibbs" and chains != 1):
            raise ValueError("chains must be positive; the Gibbs backend requires chains=1")
        if backend == "gibbs":
            raise NotImplementedError("Gibbs backend is not implemented yet")
        if not 0 < target_accept < 1:
            raise ValueError("target_accept must be between zero and one")
        
        self.rank = rank
        self.burnin = burnin
        self.posterior_samples = posterior_samples
        self.thin = thin
        self.prior_shape = prior_shape
        self.prior_rate = prior_rate
        self.global_precision = global_precision
        self.seed = seed
        self.backend = backend
        self.chains = chains
        self.target_accept = target_accept
        self.inference_data_ = None
        self._is_fitted = False

    def fit(self, observed_data: pd.DataFrame) -> "BayesianFixedDoseModel":
        """Fit unique, nonmissing response keys at exactly one dose.

        Extra metadata are ignored; responses are not normalized. Sorted IDs
        define posterior axes. Repeated screens must be aggregated upstream.
        Returns self with a fresh posterior_ and identifier mappings."""
        required = [MODEL_COLUMN, *CONDITION_COLUMNS, RESPONSE_COLUMN]
        missing_columns = sorted(set(required).difference(observed_data.columns))
        if missing_columns:
            raise ValueError(f"Observed data are missing columns: {missing_columns}")
        if observed_data[required].isna().any().any():
            raise ValueError("Observed data contain missing model, compound, dose, or response")
        if observed_data[DOSE_COLUMN].nunique() != 1:
            raise ValueError("BayesianFixedDoseModel requires exactly one fixed dose")
        if observed_data.duplicated([MODEL_COLUMN, *CONDITION_COLUMNS]).any():
            raise ValueError("Observed data contain duplicate response keys")

        # models refers to the disease-models (cell-lines)
        models = np.sort(observed_data[MODEL_COLUMN].unique())
        compounds = np.sort(observed_data["broad_id"].unique())
        model_index = {model: index for index, model in enumerate(models)}
        compound_index = {compound: index for index, compound in enumerate(compounds)}
        model_ids = observed_data[MODEL_COLUMN].map(model_index).to_numpy(dtype=int)
        compound_ids = observed_data["broad_id"].map(compound_index).to_numpy(dtype=int)
        responses = observed_data[RESPONSE_COLUMN].to_numpy(dtype=float)

        self.models_ = models
        self.compounds_ = compounds
        self.fixed_dose_ = float(observed_data[DOSE_COLUMN].iloc[0])
        self.model_index_ = model_index
        self.compound_index_ = compound_index
        self._is_fitted = False
        self.inference_data_ = None
        if self.backend == "pymc":
            from .pymc_backend import sample_posterior as sample_pymc

            self.posterior_, self.inference_data_ = sample_pymc(
                responses, model_ids, compound_ids, self
            )
        else:
            from .gibbs import sample_posterior
            from .model_utils import _group_row_indices

            self.posterior_ = sample_posterior(
                responses, model_ids, compound_ids,
                _group_row_indices(model_ids, len(models)),
                _group_row_indices(compound_ids, len(compounds)), self,
            )
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
        required = [MODEL_COLUMN, "broad_id", DOSE_COLUMN]
        missing_columns = sorted(set(required).difference(candidate_pairs.columns))
        if missing_columns:
            raise ValueError(f"Candidate pairs are missing columns: {missing_columns}")
        if not np.allclose(candidate_pairs[DOSE_COLUMN], self.fixed_dose_):
            raise ValueError("Candidate pairs must use the model's fixed dose")
        try:
            model_ids = np.array([self.model_index_[value] for value in candidate_pairs[MODEL_COLUMN]])
            compound_ids = np.array(
                [self.compound_index_[value] for value in candidate_pairs["broad_id"]]
            )
        except KeyError as error:
            raise KeyError(f"Candidate contains an unseen identifier: {error.args[0]}") from error

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
        result = candidate_pairs[[MODEL_COLUMN, "broad_id", DOSE_COLUMN]].copy()
        result["predicted_mean"] = draws.mean(axis=0)
        result["posterior_std"] = draws.std(axis=0, ddof=1)
        result["predictive_std"] = np.sqrt(
            draws.var(axis=0, ddof=1) + self.observation_variance_draws().mean()
        )
        predictive_draws = draws + np.sqrt(self.observation_variance_draws())[:, None] * (
            np.random.default_rng(self.seed + 1).normal(size=draws.shape)
        )
        result["interval_lower"] = np.quantile(predictive_draws, 0.05, axis=0)
        result["interval_upper"] = np.quantile(predictive_draws, 0.95, axis=0)
        return result
