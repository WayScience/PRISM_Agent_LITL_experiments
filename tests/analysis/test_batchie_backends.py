"""Shared model/selection contracts; short chains test mechanics, not convergence."""


import numpy as np
import pandas as pd
import pytest

from nbutils.batchie_mono.batchie import BayesianFixedDoseModel
from nbutils.batchie_mono.batchie_selector import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono import model_utils
from nbutils.batchie_mono.models import BayesianResponseModel
from nbutils.batchie_mono.simulation import run_simulation


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(12)
    return pd.DataFrame([
        (cell, f"drug-{drug:02}", 2.5, -drug / 3 + rng.normal(0, 0.1))
        for cell in ["background-a", "background-b", "target"]
        for drug in range(12)
    ], columns=["ModelID", "broad_id", "dose", "logfold_change"])


@pytest.fixture(scope="module", params=["pymc"])
def fitted(request, data):
    if request.param == "pymc":
        pytest.importorskip("pymc")
    else:
        pytest.importorskip("nbutils.batchie_refactored.gibbs")
    observed = data[~data.ModelID.eq("target") | data.broad_id.isin(["drug-00", "drug-01"])]
    model = BayesianFixedDoseModel(
        backend=request.param, rank=2, burnin=40, posterior_samples=10, thin=2,
        chains=2 if request.param == "pymc" else 1,
    )
    assert model.fit(observed) is model
    return model, observed


def test_aligned_predictions(fitted, data):
    model, _ = fitted
    pairs = data[data.ModelID.eq("target")].iloc[[8, 2, 8]].drop(columns="logfold_change")
    pairs.index = [9, 4, 9]
    draws = model.posterior_draws(pairs)
    assert draws.shape == (model.posterior_samples * model.chains, 3)
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


@pytest.mark.parametrize("backend", ["pymc"])
def test_training_validation(backend, data):
    model = BayesianFixedDoseModel(backend=backend)
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


@pytest.mark.parametrize("backend", ["pymc"])
@pytest.mark.parametrize("selector_class", [PredictedLowestSelector, RandomSelector])
def test_adaptive_simulation(backend, selector_class, data):
    pytest.importorskip("pymc")
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
        model=CheckedModel(backend=backend, rank=1, burnin=20, posterior_samples=5, thin=1),
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


def test_model_utils_numerical_contracts():
    grouped = model_utils._group_row_indices(np.array([2, 0, 2]), group_count=4)
    assert [indices.tolist() for indices in grouped] == [[1], [], [0, 2], []]

    design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    target = np.array([1.0, 2.0, 2.0])
    penalty = 2.0
    expected_regularizer = np.diag([penalty * 0.1, penalty])
    expected = np.linalg.solve(
        design.T @ design + expected_regularizer,
        design.T @ target,
    )
    np.testing.assert_allclose(model_utils._ridge_solve(design, target, penalty), expected)

    means = model_utils._factor_mean(
        global_offset=1.0,
        cell_offsets=np.array([0.1, 0.2]),
        compound_offsets=np.array([-0.2, 0.3]),
        cell_embeddings=np.array([[1.0, 2.0], [3.0, 4.0]]),
        compound_embeddings=np.array([[0.5, 1.0], [2.0, -1.0]]),
        model_ids=np.array([0, 1, 0]),
        compound_ids=np.array([1, 0, 0]),
    )
    np.testing.assert_allclose(means, [1.4, 6.5, 3.4])

    precision = np.diag([2.0, 4.0])
    mean_part = np.array([2.0, 8.0])
    rng = np.random.default_rng(27)
    reference_rng = np.random.default_rng(27)
    expected_draw = np.array([1.0, 2.0]) + np.linalg.solve(
        np.linalg.cholesky(precision).T,
        reference_rng.normal(size=2),
    )
    np.testing.assert_allclose(
        model_utils._sample_from_precision(precision, mean_part, rng), expected_draw,
    )


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
