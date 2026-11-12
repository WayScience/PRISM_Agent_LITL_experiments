"""Regression coverage for Gibbs integration with the retained-trace contract."""

from contextlib import closing
import json

import arviz as az
import numpy as np
import pandas as pd
import pytest

from nbutils.batchie_mono.config import RESPONSE_KEY_COLUMNS
from nbutils.batchie_mono.data import EncodedObservations
from nbutils.batchie_mono.factorization_model import BayesianFixedDoseModel
from nbutils.batchie_mono.gibbs import GibbsSampler
from nbutils.batchie_mono.model_acquisition import PredictedLowestSelector
from nbutils.batchie_mono.sampling import NUTSConfig, SamplingConfig, SamplingResult
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.trace import PARAMETER_DIMS, validate_trace


@pytest.fixture
def observed():
    return pd.DataFrame({
        "ModelID": ["b", "a", "a", "b"],
        "broad_id": ["y", "x", "y", "x"],
        "dose": [1.0] * 4,
        "logfold_change": [-1.0, -2.0, -0.5, -1.5],
    })


@pytest.mark.parametrize("workers", [1, 2], ids=["serial", "spawned"])
def test_gibbs_result_and_model_fit(observed, workers):
    config = SamplingConfig(
        burnin=2, posterior_samples=4, thin=2, chains=2,
        cores=workers, blas_cores=workers, seed=31,
    )
    spec = ModelSpec(rank=2)
    sampler = GibbsSampler(config=config)
    data = EncodedObservations.from_frame(observed)
    result = sampler.sample(data, spec)

    assert isinstance(result, SamplingResult)
    trace = result.trace
    validate_trace(trace, data, spec)
    assert trace.posterior.sizes["chain"] == config.chains
    assert trace.posterior.sizes["draw"] == config.posterior_samples
    np.testing.assert_array_equal(trace.posterior.draw, [0, 2, 4, 6])
    np.testing.assert_array_equal(trace.sample_stats.draw, trace.posterior.draw)
    assert trace.attrs["backend"] == "gibbs"
    assert trace.attrs["sampling_parameterization"] == "centered"
    assert trace.attrs["chain_workers"] == workers
    assert json.loads(trace.attrs["inference_config"]) == config.model_dump(mode="json")

    model = BayesianFixedDoseModel(sampler=sampler, spec=spec)
    assert model.fit(observed) is model
    for name in PARAMETER_DIMS:
        np.testing.assert_array_equal(
            model.inference_data_.posterior[name], trace.posterior[name],
        )
    candidates = observed.iloc[[3, 0, 3]][RESPONSE_KEY_COLUMNS]
    draws = model.posterior_draws(candidates)
    assert draws.shape == (config.chains * config.posterior_samples, 3)
    assert np.isfinite(draws).all()
    np.testing.assert_array_equal(draws[:, 0], draws[:, 2])
    prediction = model.predict(candidates)
    assert prediction.index.equals(candidates.index)
    np.testing.assert_allclose(prediction.predicted_mean, draws.mean(axis=0))
    assert np.isfinite(prediction.select_dtypes(include="number")).all().all()


def test_gibbs_simulation_acquires_remaining_compound(observed):
    model = BayesianFixedDoseModel(
        sampler=GibbsSampler(config=SamplingConfig(burnin=2, posterior_samples=4)),
        spec=ModelSpec(rank=2),
    )
    metrics = run_simulation(
        background=observed[observed.ModelID == "b"],
        target_responses=observed[observed.ModelID == "a"],
        initial_compounds=["x"],
        model=model,
        selector=PredictedLowestSelector(),
        batch_size=1,
        iterations=1,
    )
    assert model._is_fitted
    assert metrics.compounds_seen.tolist() == [1, 2]
    assert metrics.top_10pct_retrieval.tolist() == [1.0, 1.0]


def test_gibbs_serial_parallel_and_roundtrip(observed, tmp_path):
    config = SamplingConfig(
        burnin=4, posterior_samples=6, thin=2, chains=2, seed=17, blas_cores=2,
    )
    serial = BayesianFixedDoseModel(
        sampler=GibbsSampler(config=config), spec=ModelSpec(rank=2),
    ).fit(observed)
    parallel_config = SamplingConfig(**{**config.model_dump(), "cores": 2})
    parallel = BayesianFixedDoseModel(
        sampler=GibbsSampler(config=parallel_config), spec=ModelSpec(rank=2),
    ).fit(observed)
    a, b = serial.inference_data_, parallel.inference_data_
    for name in a.posterior:
        np.testing.assert_array_equal(a.posterior[name], b.posterior[name])
    assert a.attrs["chain_seeds"] == b.attrs["chain_seeds"]
    assert a.attrs["chain_workers"] == 1 and b.attrs["chain_workers"] == 2
    assert "diverging" not in b.sample_stats
    assert "raw_postwarmup_divergences" not in b.attrs
    path = tmp_path / "gibbs.nc"
    b.to_netcdf(path, engine="h5netcdf")
    with closing(az.from_netcdf(path, engine="h5netcdf")) as loaded:
        loaded.load()
        validate_trace(loaded, parallel.training_data_, parallel.spec)
        assert loaded.attrs == b.attrs
        for name in b.posterior:
            np.testing.assert_array_equal(b.posterior[name], loaded.posterior[name])
        for name in b.sample_stats:
            np.testing.assert_array_equal(b.sample_stats[name], loaded.sample_stats[name])


def test_nuts_options_are_rejected():
    with pytest.raises(TypeError, match="SamplingConfig"):
        GibbsSampler(config=NUTSConfig())
