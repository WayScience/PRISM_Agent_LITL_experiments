"""PyMC model/selection contracts; short chains test mechanics, not convergence."""

from contextlib import closing

import arviz as az
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError
from scipy.stats import gamma, norm

from nbutils.batchie_mono.batchie import BayesianFixedDoseModel
from nbutils.batchie_mono.batchie_selector import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono.models import BayesianResponseModel
from nbutils.batchie_mono.data import EncodedObservations
from nbutils.batchie_mono.sampling import NUTSConfig
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.trace import validate_trace


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(12)
    return pd.DataFrame([
        (cell, f"drug-{drug:02}", 2.5, -drug / 3 + rng.normal(0, 0.1))
        for cell in ["background-a", "background-b", "target"]
        for drug in range(12)
    ], columns=["ModelID", "broad_id", "dose", "logfold_change"])


@pytest.fixture(scope="module")
def fitted(data):
    pytest.importorskip("pymc")
    from nbutils.batchie_mono.pymc_backend import PyMCSampler

    observed = data[~data.ModelID.eq("target") | data.broad_id.isin(["drug-00", "drug-01"])]
    model = BayesianFixedDoseModel(
        sampler=PyMCSampler(config=NUTSConfig(
            burnin=40, posterior_samples=10, thin=2, chains=2, max_treedepth=6,
        )),
        spec=ModelSpec(rank=2),
    )
    assert model.fit(observed) is model
    assert int(model.inference_data_.sample_stats.tree_depth.max()) <= 6
    return model, observed


def test_aligned_predictions(fitted, data):
    model, _ = fitted
    pairs = data[data.ModelID.eq("target")].iloc[[8, 2, 8]].drop(columns="logfold_change")
    pairs.index = [9, 4, 9]
    draws = model.posterior_draws(pairs)
    config = model.sampler.config
    assert draws.shape == (config.posterior_samples * config.chains, 3)
    np.testing.assert_array_equal(draws[:, 0], draws[:, 2])
    prediction = model.predict(pairs)
    pd.testing.assert_frame_equal(prediction, model.predict(pairs))
    assert prediction.index.tolist() == [9, 4, 9]
    assert np.isfinite(prediction.select_dtypes("number")).all().all()
    np.testing.assert_allclose(prediction.predicted_mean, draws.mean(axis=0))
    np.testing.assert_allclose(
        prediction.predictive_std ** 2,
        draws.var(axis=0, ddof=1) + model.observation_variance_draws().mean(),
    )
    assert (model.observation_variance_draws() > 0).all()


def test_candidate_validation(fitted, data):
    model, _ = fitted
    pairs = data.iloc[:2].drop(columns="logfold_change")
    with pytest.raises(KeyError, match="unseen"):
        model.predict(pairs.assign(broad_id="new"))
    with pytest.raises(ValueError, match="fixed dose"):
        model.predict(pairs.assign(dose=1.0))
    with pytest.raises(ValueError, match="missing columns"):
        model.predict(pairs.drop(columns="dose"))


def test_training_validation(data):
    pytest.importorskip("pymc")
    from nbutils.batchie_mono.pymc_backend import PyMCSampler

    model = BayesianFixedDoseModel(sampler=PyMCSampler())
    with pytest.raises(ValueError, match="max_treedepth"):
        NUTSConfig(max_treedepth=0)
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict(data)
    with pytest.raises(ValueError, match="duplicate"):
        model.fit(pd.concat([data, data.iloc[:1]]))
    with pytest.raises(ValueError, match="missing"):
        model.fit(data.assign(logfold_change=np.nan))
    with pytest.raises(ValueError, match="exactly one"):
        model.fit(data.assign(dose=np.arange(len(data))))
    with pytest.raises(ValueError, match="missing columns"):
        model.fit(data.drop(columns="broad_id"))


def test_pymc_sampler_config_validation():
    pytest.importorskip("pymc")
    from nbutils.batchie_mono.pymc_backend import PyMCSampler
    from nbutils.batchie_mono.sampling import SamplingConfig

    config = NUTSConfig(posterior_samples=3)
    sampler = PyMCSampler(config=config)
    assert isinstance(sampler, BaseModel)
    assert sampler.config is config
    assert PyMCSampler.model_validate_json(sampler.model_dump_json()) == sampler
    assert PyMCSampler(config={"posterior_samples": 3}).config == config
    with pytest.raises(ValidationError, match="frozen"):
        sampler.config = config
    for invalid in (SamplingConfig(), {"thin": 0}, {"draw_typo": 10}):
        with pytest.raises(ValidationError):
            PyMCSampler(config=invalid)


@pytest.mark.parametrize("selector_class", [PredictedLowestSelector, RandomSelector])
def test_adaptive_simulation(selector_class, data):
    pytest.importorskip("pymc")
    from nbutils.batchie_mono.pymc_backend import PyMCSampler

    background = data[~data.ModelID.eq("target")]
    target = data[data.ModelID.eq("target")]
    initial = ["drug-00", "drug-01"]
    observed_sizes = []

    class CheckedModel(BayesianFixedDoseModel):
        def fit(self, observed_data):
            observed_sizes.append(len(observed_data))
            assert not observed_data.duplicated(["ModelID", "broad_id", "dose"]).any()
            if len(observed_sizes) == 1:
                assert set(observed_data[observed_data.ModelID.eq("target")].broad_id) == set(initial)
            return super().fit(observed_data)

    class CheckedSelector(selector_class):
        def select(self, **kwargs):
            assert "logfold_change" not in kwargs["candidate_pairs"]
            assert "logfold_change" not in kwargs["evaluation_pairs"]
            return super().select(**kwargs)

    result = run_simulation(
        background=background, target_responses=target, initial_compounds=initial,
        model=CheckedModel(
            sampler=PyMCSampler(config=NUTSConfig(burnin=20, posterior_samples=5, thin=1)),
            spec=ModelSpec(rank=1),
        ),
        selector=CheckedSelector(), batch_size=2, iterations=2,
    )
    assert observed_sizes == [26, 28]
    assert result.compounds_seen.tolist() == [2, 4, 6]
    assert result.top_hits_total.tolist() == [2, 2, 2]
    assert result.top_hits_found.is_monotonic_increasing
    assert result.top_10pct_retrieval.between(0, 1).all()
    if selector_class is RandomSelector:
        remaining = sorted(set(target.broad_id) - set(initial))
        seen = set(initial)
        hits = set(target.nsmallest(2, "logfold_change").broad_id)
        rng = np.random.default_rng(0)
        expected = [len(seen & hits)]
        for _ in range(2):
            chosen = rng.choice(remaining, size=2, replace=False)
            seen.update(chosen)
            remaining = sorted(set(remaining) - set(chosen))
            expected.append(len(seen & hits))
        assert result.top_hits_found.tolist() == expected


def test_abstract_response_model_methods_raise_not_implemented():
    class MethodProbe(BayesianResponseModel):
        def fit(self, observed_data):
            return super().fit(observed_data)

        def predict(self, candidate_pairs):
            return super().predict(candidate_pairs)

        def posterior_draws(self, candidate_pairs):
            return super().posterior_draws(candidate_pairs)

        def observation_variance_draws(self):
            return super().observation_variance_draws()

    model = MethodProbe()
    with pytest.raises(NotImplementedError):
        model.fit(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.predict(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.posterior_draws(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.observation_variance_draws()


def test_centered_joint_density_matches_written_model(data):
    pytest.importorskip("pymc")
    from nbutils.batchie_mono.pymc_backend import build_model

    spec = ModelSpec(rank=2)
    encoded = EncodedObservations.from_frame(data)
    graph = build_model(encoded, spec)
    rng = np.random.default_rng(4)
    n_cells, n_compounds = len(encoded.models), len(encoded.compounds)
    values = {}
    precision_shapes = {
        "cell_offset_precision": (),
        "observation_precision": (),
        "dimension_precisions": (spec.rank,),
        "compound_precisions": (n_compounds,),
    }
    logp = 0.0
    for name, shape in precision_shapes.items():
        values[name] = rng.uniform(0.3, 2.0, size=shape)
        logp += gamma.logpdf(
            values[name], a=spec.prior_shape, scale=1 / spec.prior_rate
        ).sum()
    precisions = {
        "global_offset": spec.global_precision,
        "cell_offsets": values["cell_offset_precision"],
        "compound_offsets": values["compound_precisions"],
        "cell_embeddings": values["dimension_precisions"],
        "compound_embeddings": values["compound_precisions"][:, None],
    }
    for name, shape in {
        "global_offset": (),
        "cell_offsets": (n_cells,),
        "compound_offsets": (n_compounds,),
        "cell_embeddings": (n_cells, spec.rank),
        "compound_embeddings": (n_compounds, spec.rank),
    }.items():
        values[name] = rng.normal(size=shape)
        logp += norm.logpdf(values[name], scale=precisions[name] ** -0.5).sum()
    mu = (
        values["global_offset"]
        + values["cell_offsets"][encoded.model_ids]
        + values["compound_offsets"][encoded.compound_ids]
        + (
            values["cell_embeddings"][encoded.model_ids]
            * values["compound_embeddings"][encoded.compound_ids]
        ).sum(axis=1)
    )
    logp += norm.logpdf(
        encoded.responses, loc=mu, scale=values["observation_precision"] ** -0.5
    ).sum()
    point = {
        name + "_log__" if name in precision_shapes else name:
        np.log(value) if name in precision_shapes else value
        for name, value in values.items()
    }
    np.testing.assert_allclose(graph.compile_logp(jacobian=False)(point), logp, rtol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_real_sampler_roundtrip(data, tmp_path, parameterization):
    pytest.importorskip("pymc")
    pytest.importorskip("h5netcdf")
    from nbutils.batchie_mono.pymc_backend import PyMCSampler

    config = NUTSConfig(
        burnin=12, posterior_samples=6, thin=2, chains=2, cores=1,
        seed=8, parameterization=parameterization,
    )
    observed = data[data.ModelID.ne("target") & data.broad_id.isin(["drug-00", "drug-01"])]
    model = BayesianFixedDoseModel(
        sampler=PyMCSampler(config=config), spec=ModelSpec(rank=1)
    ).fit(observed)
    trace = model.inference_data_
    assert trace.posterior.sizes["chain"] == 2
    assert trace.posterior.sizes["draw"] == 6
    np.testing.assert_array_equal(trace.posterior.draw, np.arange(0, 12, 2))
    assert trace.attrs["raw_draws_per_chain"] == 12
    if parameterization == "noncentered":
        np.testing.assert_allclose(
            trace.posterior.cell_offsets,
            trace.posterior.z_cell_offsets / np.sqrt(trace.posterior.cell_offset_precision),
        )
    destination = tmp_path / "trace.nc"
    trace.to_netcdf(destination, engine="h5netcdf")
    with closing(az.from_netcdf(destination, engine="h5netcdf")) as loaded:
        loaded.load()
        validate_trace(loaded, model.training_data_, model.spec)
        assert loaded.attrs == trace.attrs
        for group in ("posterior", "sample_stats", "observed_data", "constant_data"):
            assert getattr(loaded, group).equals(getattr(trace, group))
