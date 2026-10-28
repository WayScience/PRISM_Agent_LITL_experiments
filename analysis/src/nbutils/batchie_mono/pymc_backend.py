"""
Defines the prior distributions and sampling procedure for the Bayesian fixed-dose model.
Same prior distributions as the original BATCHIE model.
"""

from __future__ import annotations

from typing import Literal
from time import perf_counter

import numpy as np
import pymc as pm
from pydantic import BaseModel, ConfigDict, Field

from nbutils.batchie_mono.data import EncodedObservations
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.sampling import NUTSConfig, SamplingResult
from nbutils.batchie_mono.trace import retain_trace


def build_model(    
    data: EncodedObservations,
    spec: ModelSpec,
    *, 
    parameterization: Literal["centered", "noncentered"]="centered"
):
    """
    Same joint density sampling of model parameters as the original BATCHIE model with gibbs sampler.
    The original work hand-crafts sampling using a closed-form solved posterior
        function (leveraging the relationships between gamma and normal) 
        distributions. This implementation leaves that posterior solving and
        sampling to NUTS and PyMC and only specified to them the prior distribution.

    :param data: EncodedObservations containing the observed responses and indices.
    :param spec: ModelSpec containing the model specification.
    :param parameterization: Either "centered" or "noncentered" parameterization for hierarchical priors.
    :return: The model graph, usable for both prior checks and posterior fitting.
    """

    if parameterization not in {"centered", "noncentered"}:
            raise ValueError("Unknown parameterization")

    responses, model_ids, compound_ids = (
        data.responses,
        data.model_ids,
        data.compound_ids,
    )
    
    # Define the coordinate system for the screen
    # which is rather simple with monotherapy, fixed dose screens,
    # as it is simply a matrix of cell id by compound id.
    coords = {
        "cell": data.models,
        "compound": data.compounds,
        "factor": np.arange(spec.rank),
        "observation": np.arange(len(responses)),
    }
    
    with pm.Model(coords=coords) as graph:
        # as defined in the original BATCHIE model,
        # the precisions for the cell, dimension, and compound effects are gamma
        cell_precision = pm.Gamma(
            "cell_offset_precision", spec.prior_shape, spec.prior_rate
        )
        dimension_precision = pm.Gamma(
            "dimension_precisions", spec.prior_shape, spec.prior_rate, dims="factor"
        )
        compound_precision = pm.Gamma(
            "compound_precisions", spec.prior_shape, spec.prior_rate, dims="compound"
        )
        tau = pm.Gamma("observation_precision", spec.prior_shape, spec.prior_rate)

        # all offsets and multi-dimensional embeddings are modeled as normal distributions.
        alpha = pm.Normal("global_offset", mu=0, tau=spec.global_precision)

        def hierarchical_normal(name, precision, dims):
            if parameterization == "centered":
                return pm.Normal(name, mu=0, tau=precision, dims=dims)
            z = pm.Normal("z_" + name, mu=0, sigma=1, dims=dims)
            return pm.Deterministic(name, z / pm.math.sqrt(precision), dims=dims)

        a = hierarchical_normal("cell_offsets", cell_precision, "cell")
        b = hierarchical_normal("compound_offsets", compound_precision, "compound")
        u = hierarchical_normal(
            "cell_embeddings", dimension_precision, ("cell", "factor")
        )
        v = hierarchical_normal(
            "compound_embeddings", compound_precision[:, None], ("compound", "factor")
        )

        # defines the relationships between model parameters (of which we
        # are computing the posterior distributions for) and the observed responses.
        mu = (
            alpha
            + a[model_ids]
            + b[compound_ids]
            + (u[model_ids] * v[compound_ids]).sum(axis=1)
        )

        # we assume that the observed responses are normally distributed 
        # around the mean `mu` with precision `tau`.
        pm.Normal("response", mu=mu, tau=tau, observed=responses, dims="observation")

    return graph


class PyMCSampler(BaseModel):
    """
    Thin wrapper of a configuration and re-usable sample handle.
    For unifying bayesian sampler interface with BayesianFixedDoseModel and other models. 
    """
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", validate_default=True)

    config: NUTSConfig = Field(default_factory=NUTSConfig)

    def sample(self, data: EncodedObservations, spec: ModelSpec) -> SamplingResult:
        """
        Main entry point for sampling from the posterior distribution using PyMC.

        :param data: Encoded observations containing model and compound IDs and responses.
        :param spec: Model specification defining priors and dimensions.
        :return: SamplingResult containing the posterior samples and metadata.
        """
        import pymc as pm

        config: NUTSConfig = self.config
        start = perf_counter()
        with build_model(data, spec, parameterization=config.parameterization):
            trace = pm.sample(
                tune=config.burnin,
                draws=config.raw_draws,
                chains=config.chains,
                cores=config.cores,
                blas_cores=config.blas_cores,
                random_seed=config.seed,
                target_accept=config.target_accept,
                nuts_sampler="pymc",
                nuts={"max_treedepth": config.max_treedepth},
                progressbar=False,
                return_inferencedata=True,
            )
        return SamplingResult(
            trace=retain_trace(
                trace,
                data,
                spec,
                config,
                backend="pymc",
                parameterization=config.parameterization,
                wall_seconds=perf_counter() - start,
            )
        )
