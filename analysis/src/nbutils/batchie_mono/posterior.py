"""
Defines the posterior distribution data structure (descriptive of a model given data observations)
Contracts the data structure of posterior sampler returns.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict


class Posterior(BaseModel):
    """
    Posterior distribution for a Bayesian, monotherapy, fixed-dose model.

    Because we fit S models through sampling S posterior draws to account for modelling uncertainty,
        all parameters have a S dimension.
    """

    # Array shapes, finiteness and precision positivity are checked by validate_trace.
    # Frozen prevents field reassignment, not in-place mutation of array contents.
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

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


def posterior_from_trace(trace):
    """
    Flatten an already-retained trace once, using chain-major ordering.
    """

    arrays = {}

    for name in Posterior.model_fields:
        values = trace.posterior[name].transpose("chain", "draw", ...).values
        arrays[name] = values.reshape((-1, *values.shape[2:]))

    return Posterior(**arrays)
