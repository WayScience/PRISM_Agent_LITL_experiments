"""Real Gibbs integration coverage for the committed experiment logger."""

import json
from math import ceil

import pandas as pd
import pytest
import xarray as xr

from nbutils.batchie_mono.factorization_model import BayesianFixedDoseModel
from nbutils.batchie_mono.gibbs import GibbsSampler
from nbutils.batchie_mono.logger import ExperimentLogger
from nbutils.batchie_mono.model_acquisition import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono.sampling import SamplingConfig
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.spec import ModelSpec


@pytest.mark.parametrize("policy", ["predicted_lowest", "random"])
def test_real_gibbs_logger_round_trip(tmp_path, policy):
    target = pd.DataFrame(
        [["t", f"d{i}", 1.0, -float(i + 1)] for i in range(5)],
        columns=["ModelID", "broad_id", "dose", "logfold_change"],
    )
    model = BayesianFixedDoseModel(
        sampler=GibbsSampler(config=SamplingConfig(
            burnin=2, posterior_samples=3, thin=2, chains=2, cores=1, blas_cores=1,
        )),
        spec=ModelSpec(rank=1),
        prediction_seed=0,
    )
    path = tmp_path / "screen.sqlite"
    with ExperimentLogger(path) as logger:
        original = run_simulation(
            background=target.assign(ModelID="b0"), target_responses=target,
            initial_compounds=["d0"], model=model,
            selector=RandomSelector(seed=0) if policy == "random" else PredictedLowestSelector(),
            batch_size=2, iterations=2, logger=logger, experiment_name="small",
        )
    live_trace = model.inference_data_
    trajectory_id = original.trajectory_id.iloc[0]

    # Reopen after closing the writer; action timestamps are not an ordering key.
    with ExperimentLogger(path) as logger:
        experiment = logger.one(
            "SELECT * FROM experiments WHERE experiment_id=?", (trajectory_id,),
        )
        assert experiment["status"] == "completed"
        config = json.loads(experiment["config"])
        actions = logger.rows(
            "SELECT a.* FROM actions a JOIN experiment_states s "
            "ON s.state_id=a.output_state_id WHERE a.experiment_id=? ORDER BY s.sequence",
            (trajectory_id,),
        )
        assert [a["kind"] for a in actions] == ["fit", "acquire", "fit", "acquire"]
        artifacts = [item for a in actions for item in logger.artifact_records(a["action_id"])]
        assert len(artifacts) == 2
        for artifact in artifacts:
            logger.artifacts.verify(artifact)  # Checks both SHA-256 and byte length.
        _, trace = logger.load_run(actions[-2]["action_id"])

        # Independently reconstruct hit membership from registered target truth.
        responses = logger.rows(
            "SELECT r.drug_id, r.value, m.role FROM split_memberships m "
            "JOIN responses r USING(response_id) WHERE m.split_id=? AND r.cell_id=? "
            "ORDER BY r.value, r.drug_id",
            (experiment["split_id"], "t"),
        )
        assert len(responses) == 5
        pool = {r["drug_id"] for r in responses}
        hits = {r["drug_id"] for r in responses[:max(1, ceil(0.1 * len(responses)))]}
        seen = {r["drug_id"] for r in responses if r["role"] == "initial"}
        assert seen == {"d0"}

        def retrieval(iteration):
            return {
                "iteration": iteration, "compounds_seen": len(seen),
                "top_hits_found": len(seen & hits), "top_hits_total": len(hits),
                "top_10pct_retrieval": len(seen & hits) / len(hits),
            }

        saved = [config["initial_metrics"]]
        replayed = [retrieval(0)]
        acquisitions = [a for a in actions if a["kind"] == "acquire"]
        for iteration, action in enumerate(acquisitions, start=1):
            saved.append({
                "selector": config["selector"],
                **{r["metric"]: r["value"] for r in logger.rows(
                    "SELECT metric, value FROM evaluations WHERE action_id=? AND scope='simulation'",
                    (action["action_id"],),
                )},
            })
            items = logger.rows(
                "SELECT r.drug_id, r.cell_id FROM acquisition_items i "
                "JOIN responses r USING(response_id) WHERE i.action_id=? ORDER BY i.rank",
                (action["action_id"],),
            )
            selected = {r["drug_id"] for r in items}
            assert len(items) == len(selected) == 2
            assert all(r["cell_id"] == "t" for r in items)
            assert selected <= pool - seen
            seen.update(selected)
            replayed.append(retrieval(iteration))

    assert trace.posterior.sizes["chain"] == 2
    assert trace.posterior.sizes["draw"] == 3
    assert trace.groups() == live_trace.groups()
    for group in trace.groups():
        xr.testing.assert_identical(trace[group], live_trace[group])

    reloaded = pd.DataFrame(saved).assign(trajectory_id=trajectory_id)
    pd.testing.assert_frame_equal(
        reloaded[original.columns], original, check_dtype=False, rtol=1e-12, atol=0,
    )
    counts = ["iteration", "compounds_seen", "top_hits_found", "top_hits_total"]
    values = [*counts, "top_10pct_retrieval"]
    for result in (reloaded, pd.DataFrame(replayed)):
        pd.testing.assert_frame_equal(
            result[counts], original[counts], check_dtype=False, check_exact=True,
        )
        pd.testing.assert_frame_equal(
            result[values], original[values], check_dtype=False, rtol=1e-12, atol=0,
        )
