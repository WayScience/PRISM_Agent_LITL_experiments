"""
BATCHIE/factorization model trace helpers
"""

import json

import numpy as np
import xarray as xr

from .data import EncodedObservations
from .spec import ModelSpec
from .sampling import SamplingConfig


PARAMETER_DIMS = {
    "global_offset": (),
    "cell_offset_precision": (),
    "observation_precision": (),
    "cell_offsets": ("cell",),
    "compound_offsets": ("compound",),
    "cell_embeddings": ("cell", "factor"),
    "compound_embeddings": ("compound", "factor"),
    "dimension_precisions": ("factor",),
    "compound_precisions": ("compound",),
}


def _numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def validate_trace(trace, data: EncodedObservations, spec: ModelSpec):
    """
    Reject mislabeled, incomplete or nonfinite traces before prediction.

    :param trace: Inference trace object containing posterior and optionally sample statistics.
    :param data: EncodedObservations instance containing training data.
    :param spec: ModelSpec instance describing the factorization model.
    :raises ValueError: If the trace is invalid or inconsistent with the data/spec.
    """
    if not hasattr(trace, "posterior"):
        raise ValueError("Sampler result has no posterior group")
    posterior = trace.posterior
    for axis in ("chain", "draw"):
        if posterior.sizes.get(axis, 0) < 1:
            raise ValueError(f"Posterior must have a nonempty {axis} axis")
        labels = posterior.coords[axis].values
        if len(np.unique(labels)) != len(labels):
            raise ValueError(f"Duplicate {axis} labels")
    for name, dims in PARAMETER_DIMS.items():
        if name not in posterior or posterior[name].dims != ("chain", "draw", *dims):
            raise ValueError(f"Invalid or missing posterior variable: {name}")
        values = posterior[name].values
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite posterior variable: {name}")
        if "precision" in name and (values <= 0).any():
            raise ValueError(f"Nonpositive precision: {name}")
    for name, expected in {
        "cell": data.models,
        "compound": data.compounds,
        "factor": np.arange(spec.rank),
    }.items():
        if not np.array_equal(posterior.coords[name].values, expected):
            raise ValueError(f"Trace {name} coordinates do not match training data")
    for group, expected in (
        ("observed_data", {"response": data.responses}),
        (
            "constant_data",
            {"model_ids": data.model_ids, "compound_ids": data.compound_ids},
        ),
    ):
        dataset = getattr(trace, group, None)
        for name, values in expected.items():
            if (
                dataset is None
                or name not in dataset
                or dataset[name].dims != ("observation",)
            ):
                raise ValueError(f"Missing or invalid {group}.{name}")
            if not np.array_equal(dataset[name].values, values):
                raise ValueError(f"Trace {group}.{name} differs from training data")
    if hasattr(trace, "sample_stats"):
        for name in ("chain", "draw"):
            if not np.array_equal(
                trace.sample_stats.coords[name], posterior.coords[name]
            ):
                raise ValueError(
                    f"Sample statistics have misaligned {name} coordinates"
                )


def retain_trace(
    trace, 
    data: EncodedObservations, 
    spec: ModelSpec, 
    config: SamplingConfig, 
    *, 
    backend: str, 
    parameterization: str, 
    **metadata
):
    """
    Unifying helper for sampling output (trace) from multiple engines/backends. 
    Consistently keeps draws 0, thin, 2*thin, ... in every group,
        which is a convention in bayesian posterior analysis to retain some 
        records while reducing storage and computational burden. This is based
        on the heuristic that neighboring draws are often highly correlated, 
        and can be safely "thinned" without losing much information.

    Warmup, if supplied by another backend, is kept separate and is not thinned.

    :param trace: The trace object containing posterior samples and sample statistics.
    :param data: The encoded observations used for the model.
    :param spec: The model specification.
    :param config: The sampling configuration.
    :param backend: The backend used for sampling. 
        Truly a literal in configuring the sampling but allowed as freeform string in trace logging.
    :param parameterization: The parameterization used for sampling.
        Truly a literal in configuring the sampling but allowed as freeform string in trace logging.
    :param metadata: Additional metadata to attach to the trace.
    :return: The validated and thinned trace object.
    """

    # Validate the number of draws and chains against the sampling configuration.
    if trace.posterior.sizes["draw"] != config.raw_draws:
        raise ValueError("Sampler returned an unexpected number of raw draws")
    if trace.posterior.sizes["chain"] != config.chains:
        raise ValueError("Sampler returned an unexpected number of chains")
    # Capture event counts over ALL post-warmup transitions before thinning.
    stats = getattr(trace, "sample_stats", None)
    if stats is not None and "diverging" in stats:
        metadata["raw_postwarmup_divergences"] = int(stats.diverging.sum())
    if stats is not None and "reached_max_treedepth" in stats:
        metadata["raw_postwarmup_tree_depth_hits"] = int(
            stats.reached_max_treedepth.sum()
        )
    groups = [
        group
        for group in trace.groups()
        if not group.startswith("warmup_") and "draw" in trace[group].dims
    ]
    trace = trace.isel(groups=groups, draw=slice(None, None, config.thin))
    constants = xr.Dataset(
        {
            "model_ids": ("observation", data.model_ids),
            "compound_ids": ("observation", data.compound_ids),
        },
        coords={"observation": np.arange(len(data.responses))},
    )
    if hasattr(trace, "constant_data"):
        trace.constant_data = constants
    else:
        trace.add_groups({"constant_data": constants})
    trace.attrs.update(
        trace_schema_version=1,
        backend=backend,
        sampling_parameterization=parameterization,
        model_config=json.dumps(spec.model_dump(mode="json"), sort_keys=True, default=_numpy_scalar),
        inference_config=json.dumps(
            config.model_dump(mode="json"), sort_keys=True, default=_numpy_scalar
        ),
        fixed_dose=data.fixed_dose,
        seed=int(config.seed),
        warmup_per_chain=int(config.burnin),
        raw_draws_per_chain=int(config.raw_draws),
        retained_draws_per_chain=int(config.posterior_samples),
        thin=int(config.thin),
        thinning_rule="post-warmup draw positions [::thin]; chain-major flattening",
        **metadata,
    )
    validate_trace(trace, data, spec)
    return trace
