"""
Defines the posterior distribution data structure (descriptive of a model given data observations)
Contracts the data structure of posterior sampler returns.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Posterior:
    """
    Posterior distribution for a Bayesian, monotherapy, fixed-dose model.

    Because we fit S models through sampling S posterior draws to account for modelling uncertainty,
        all parameters have a S dimension.
    """

    # screen native scalar offset, learned per posterior draw (model fit)
    global_offset: np.ndarray 

    # cell-specific scalar offsets, learned per cell per posterior draw
    cell_offsets: np.ndarray  # shape (S, K)

    # compound-specific scalar offsets, learned per compound per posterior draw
    compound_offsets: np.ndarray  # shape (S, I)

    # These are the non-scalar parameters intended for modelling the interaction between cells and compounds
    # cell-specific low-rank embeddings, learned per cell per posterior draw
    cell_embeddings: np.ndarray  # shape (S, K, R)
    # compound-specific low-rank embeddings, learned per compound per posterior draw
    compound_embeddings: np.ndarray  # shape (S, I, R)

    # observation precision, learned per posterior draw (model fit)
    observation_precision: np.ndarray  # shape (S,)
