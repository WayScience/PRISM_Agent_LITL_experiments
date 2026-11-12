"""Small real Gibbs screens exercise reload, selection and retrieval verification."""

import json
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from nbutils.batchie_mono.factorization_model import BayesianFixedDoseModel
from nbutils.batchie_mono.gibbs import GibbsSampler
from nbutils.batchie_mono.logged_results import load_logged_screen, load_logged_trace
from nbutils.batchie_mono.logger import ExperimentLogger, fingerprint_responses
from nbutils.batchie_mono.model_acquisition import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono.sampling import SamplingConfig
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.spec import ModelSpec


@pytest.fixture
def screen(tmp_path):
    target = pd.DataFrame(
        [["t", f"d{i}", 1.0, -float(i + 1)] for i in range(5)],
        columns=["ModelID", "broad_id", "dose", "logfold_change"],
    )
    background = pd.concat([target.assign(ModelID="b0"), target.assign(ModelID="b1")], ignore_index=True)
    path = tmp_path / "screen.sqlite"
    provenance = {
        "dataset": {"source_sha256": "synthetic-v1", "preparation": "test"},
        "target_model": "t", "background_counts": [1, 2], "backend": "gibbs",
    }
    sampling = SamplingConfig(burnin=2, posterior_samples=3, thin=2, chains=2, cores=1, blas_cores=1)

    def run(count, random=False):
        model = BayesianFixedDoseModel(sampler=GibbsSampler(config=sampling), spec=ModelSpec(rank=1), prediction_seed=0)
        with ExperimentLogger(path) as logger:
            metrics = run_simulation(
                background=background[background.ModelID.isin([f"b{i}" for i in range(count)])],
                target_responses=target, initial_compounds=["d0"], model=model,
                selector=RandomSelector(seed=0) if random else PredictedLowestSelector(),
                batch_size=3, iterations=4, logger=logger, experiment_name="small",
                metadata={**provenance, "background_cells": count},
            )
        return metrics, model.inference_data_

    runs = [run(1), run(2), run(1, random=True)]
    return path, run, runs


def test_real_gibbs_reload_replay_latest_and_explicit(screen):
    path, run, runs = screen
    ids = [metrics.trajectory_id.iloc[0] for metrics, _ in runs]
    # Inventory must not materialize posterior arrays.
    with patch.object(ExperimentLogger, "load_run", side_effect=AssertionError("inventory loaded a trace")):
        loaded = load_logged_screen(path, "small")
    assert set(loaded.configs) == {"predicted_lowest_b1", "predicted_lowest_b2", "random_b1"}
    assert len(loaded.inventory) == 6  # pool exhausted early, not four fixed rounds
    assert set(loaded.inventory.chains) == {2}
    assert set(loaded.inventory.posterior_samples) == {3}
    assert set(loaded.inventory.thin) == {2}
    assert loaded.inventory.wall_seconds.gt(0).all()
    assert loaded.inventory.chain_cpu_seconds.gt(0).all()
    assert loaded.inventory.action_timing_scope.str.contains("export").all()
    assert set(loaded.predictions) == {(f"predicted_lowest_b{c}", f) for c in [1, 2] for f in [1, 2]}
    assert set(loaded.acquisitions) == set(loaded.observations)
    for run_name in loaded.configs:
        before = loaded.observations[run_name, 1]
        after = loaded.observations[run_name, 2]
        increment = after[after.ModelID.eq("t") & ~after.broad_id.isin(before[before.ModelID.eq("t")].broad_id)]
        acquired = loaded.acquisitions[run_name, 1]
        pd.testing.assert_frame_equal(
            acquired.sort_values("broad_id").reset_index(drop=True),
            increment.sort_values("broad_id").reset_index(drop=True),
        )
        assert len(loaded.acquisitions[run_name, 2]) == 1  # final acquisition has no next fit
    for record in loaded.inventory.to_dict("records"):
        observed = loaded.observations[record["run"], record["fit"]]
        assert record["training_sha256"] == fingerprint_responses(observed.iloc[::-1])
        assert record["training_rows"] == len(observed)
        assert record["target_observations"] == (1 if record["fit"] == 1 else 4)
        with ExperimentLogger(path) as logger:
            inputs = logger.rows("SELECT r.drug_id, r.cell_id FROM fit_inputs f JOIN responses r USING(response_id) WHERE f.action_id=? ORDER BY f.position", (record["action_id"],))
        assert list(zip(observed.broad_id, observed.ModelID)) == [(r["drug_id"], r["cell_id"]) for r in inputs]
    columns = ["selector", "iteration", "compounds_seen", "top_hits_found", "top_hits_total", "top_10pct_retrieval", "trajectory_id"]
    for metrics, original in runs:
        replayed = loaded.retrieval[loaded.retrieval.trajectory_id.eq(metrics.trajectory_id.iloc[0])]
        replayed = replayed[columns].reset_index(drop=True)
        counts = ["compounds_seen", "top_hits_found", "top_hits_total"]
        pd.testing.assert_frame_equal(
            replayed[counts], metrics[counts], check_dtype=False, check_exact=True,
        )
        pd.testing.assert_frame_equal(
            replayed, metrics[columns], check_dtype=False, rtol=1e-12, atol=0,
        )
        record = loaded.inventory[loaded.inventory.trajectory_id.eq(metrics.trajectory_id.iloc[0])].iloc[-1]
        trace = load_logged_trace(record)
        assert trace.posterior.sizes["draw"] == 3
        assert trace.groups() == original.groups()
        for group in trace.groups():
            xr.testing.assert_identical(trace[group], original[group])
    for prediction in loaded.predictions.values():
        assert set(["ModelID", "broad_id", "dose", "predicted_mean", "posterior_std", "predictive_std"]) <= set(prediction)
        assert prediction.posterior_std.notna().all()
    newer, _ = run(1)
    latest = load_logged_screen(path, "small")
    assert set(latest.inventory.trajectory_id) == {newer.trajectory_id.iloc[0], *ids[1:]}
    explicit = load_logged_screen(path, "small", experiment_ids=ids)
    assert set(explicit.inventory.trajectory_id) == set(ids)
    with pytest.raises(ValueError, match="duplicate"):
        load_logged_screen(path, "small", experiment_ids=ids + ids[:1])
    with pytest.raises(ValueError, match="grouping"):
        load_logged_screen(path, "small", experiment_ids=ids[:2])


def test_reject_incompatible_cohort_and_retrieval_corruption(screen):
    path, run, runs = screen
    ids = [metrics.trajectory_id.iloc[0] for metrics, _ in runs]
    newer, _ = run(1)
    new_id = newer.trajectory_id.iloc[0]
    with sqlite3.connect(path) as connection:
        config = json.loads(connection.execute("SELECT config FROM experiments WHERE experiment_id=?", (new_id,)).fetchone()[0])
        config["provenance"]["dataset"]["source_sha256"] = "different-source"
        connection.execute("UPDATE experiments SET config=? WHERE experiment_id=?", (json.dumps(config), new_id))
    with pytest.raises(ValueError, match="grouping"):
        load_logged_screen(path, "small")  # cannot fill newer cohort from older runs
    with pytest.raises(ValueError, match="Incompatible"):
        load_logged_screen(path, "small", experiment_ids=[new_id, *ids[1:]])
    assert len(load_logged_screen(path, "small", experiment_ids=ids).inventory) == 6
    with sqlite3.connect(path) as connection:
        action = connection.execute("SELECT action_id FROM actions WHERE experiment_id=? AND kind='acquire' ORDER BY started_at LIMIT 1", (ids[0],)).fetchone()[0]
        for metric in ("compounds_seen", "top_hits_found", "top_hits_total", "top_10pct_retrieval", "iteration"):
            value = connection.execute("SELECT value FROM evaluations WHERE action_id=? AND metric=? AND scope='simulation'", (action, metric)).fetchone()[0]
            connection.execute("UPDATE evaluations SET value=? WHERE action_id=? AND metric=? AND scope='simulation'", (value + 1, action, metric))
            connection.commit()
            with pytest.raises(ValueError, match="Retrieval"):
                load_logged_screen(path, "small", experiment_ids=ids)
            connection.execute("UPDATE evaluations SET value=? WHERE action_id=? AND metric=? AND scope='simulation'", (value, action, metric))
            connection.commit()
        connection.execute("UPDATE acquisitions SET selected_size=0 WHERE action_id=?", (action,))
    with pytest.raises(ValueError, match="count mismatch"):
        load_logged_screen(path, "small", experiment_ids=ids)


def test_missing_and_corrupt_bundle(screen, tmp_path):
    path, _, _ = screen
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        load_logged_screen(missing, "small")
    assert not missing.exists()
    empty = tmp_path / "empty.sqlite"
    empty.touch()
    with pytest.raises(ValueError, match="v2"):
        load_logged_screen(empty, "small")
    loaded = load_logged_screen(path, "small")
    record = loaded.inventory.iloc[0]
    with ExperimentLogger(path) as logger:
        artifact = logger.artifact_records(record.action_id)[0]
        artifact_path = logger.artifacts.verify(artifact)
    data = artifact_path.read_bytes()
    artifact_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_logged_screen(path, "small")
    with pytest.raises(FileNotFoundError):
        load_logged_trace(record)
    artifact_path.write_bytes(data + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_logged_screen(path, "small")
    with pytest.raises(ValueError, match="checksum"):
        load_logged_trace(record)
