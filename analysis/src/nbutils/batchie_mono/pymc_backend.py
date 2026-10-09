"""
Defines the prior distributions and sampling procedure for the Bayesian fixed-dose model.
Same prior distributions as the original BATCHIE model.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np
import pymc as pm

from .posterior import Posterior

if TYPE_CHECKING:
    from .batchie import BayesianFixedDoseModel


def sample_posterior(responses: np.ndarray, model_ids: np.ndarray,
                     compound_ids: np.ndarray, config: BayesianFixedDoseModel):
    """
    Same joint density sampling of model parameters as the original BATCHIE model with gibbs sampler.
    The original work hand-crafts sampling using a closed-form solved posterior
        function (leveraging the relationships between gamma and normal) 
        distributions. This implementation leaves that posterior solving and
        sampling to NUTS and PyMC and only specified to them the prior distribution.

    :param responses: Observed response values.
    :param model_ids: Array of model indices corresponding to each response.
    :param compound_ids: Array of compound indices corresponding to each response.
    :param config: Configuration object containing model hyperparameters and sampling settings.
    :return: A tuple containing the Posterior object with thinned samples and the full PyMC trace.
    """
    # Define the coordinate system for the screen
    # which is rather simple with monotherapy, fixed dose screens,
    # as it is simply a matrix of cell id by compound id.
    coords = {"cell": config.models_, "compound": config.compounds_,
              "factor": np.arange(config.rank)}
    
    with pm.Model(coords=coords):
        # as defined in the original BATCHIE model,
        # the precisions for the cell, dimension, and compound effects are gamma
        cell_precision = pm.Gamma("cell_offset_precision", config.prior_shape,
                                  config.prior_rate)
        dimension_precision = pm.Gamma("dimension_precisions", config.prior_shape,
                                       config.prior_rate, dims="factor")
        compound_precision = pm.Gamma("compound_precisions", config.prior_shape,
                                      config.prior_rate, dims="compound")
        tau = pm.Gamma("observation_precision", config.prior_shape, config.prior_rate)

        # all offsets and multi-dimensional embeddings are modeled as normal distributions.
        alpha = pm.Normal("global_offset", mu=0, tau=config.global_precision)
        a = pm.Normal("cell_offsets", mu=0, tau=cell_precision, dims="cell")
        b = pm.Normal("compound_offsets", mu=0, tau=compound_precision, dims="compound")
        u = pm.Normal("cell_embeddings", mu=0, tau=dimension_precision,
                      dims=("cell", "factor"))
        v = pm.Normal("compound_embeddings", mu=0, tau=compound_precision[:, None],
                      dims=("compound", "factor"))

        # defines the relationships between model parameters (of which we
        # are computing the posterior distributions for) and the observed responses.
        mu = alpha + a[model_ids] + b[compound_ids] + (
            u[model_ids] * v[compound_ids]
        ).sum(axis=1)

        # we assume that the observed responses are normally distributed 
        # around the mean `mu` with precision `tau`.
        pm.Normal("response", mu=mu, tau=tau, observed=responses)

        # from here PyMC and NUTS takes over the sampling process.
        # This will be slower than hand-crafted sampling with solved conjugate
        # but it offers a more general and flexible approach.
        trace = pm.sample(
            tune=config.burnin, draws=config.posterior_samples * config.thin,
            chains=config.chains, cores=1, random_seed=config.seed,
            target_accept=config.target_accept, nuts_sampler="pymc",
            progressbar=False, return_inferencedata=True,
        )

    # Slice before flattening so thinning never crosses a chain boundary.
    retained = trace.posterior.isel(draw=slice(None, None, config.thin))
    posterior = {}
    for field in fields(Posterior):
        values = retained[field.name].transpose("chain", "draw", ...).values
        posterior[field.name] = values.reshape((-1, *values.shape[2:]))
    return Posterior(**posterior), trace
