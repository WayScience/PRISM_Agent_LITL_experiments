"""Logger behavior and logged/unlogged simulation parity with controlled models."""

import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd

from nbutils.batchie_mono.logger import ExperimentLogger, fingerprint_responses
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.model_acquisition import (
    PredictedLowestSelector,
    RandomSelector,
)

TESTS = Path(__file__).resolve().parent
MANIFEST = {"schema_version": 1, "model_family": "test", "capabilities": []}


def setup(logger):
    dataset = logger.register_dataset(
        [
            {"cell_id": "c", "drug_id": str(i), "dose": 1, "value": float(-i)}
            for i in range(5)
        ]
    )
    ids = logger.response_ids(dataset, [("c", str(i), 1) for i in range(5)])
    split = logger.create_split(dataset, initial=ids[:1], pool=ids[1:4], test=ids[4:])
    exp = logger.start_experiment("test", split, model_family="test")
    return dataset, ids, split, exp


def fitted(logger, exp, ids):
    aid = logger.start_action(exp, "fit")
    logger.complete_fit(aid, ids, manifest=MANIFEST)
    return aid


class CoreTests(unittest.TestCase):
    def test_fingerprint_duplicate_identities_are_order_independent(self):
        for values in ([1.0, 2.0], [-0.0, 0.0]):
            with self.subTest(values=values):
                frame = pd.DataFrame(
                    {
                        "ModelID": ["c", "c"],
                        "broad_id": ["drug", "drug"],
                        "dose": [1.0, 1.0],
                        "logfold_change": values,
                    }
                )
                self.assertEqual(
                    fingerprint_responses(frame),
                    fingerprint_responses(frame.iloc[::-1]),
                )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "experiment.sqlite"
        self.log = ExperimentLogger(self.path)
        self.dataset, self.ids, self.split, self.exp = setup(self.log)

    def tearDown(self):
        self.log.close()
        self.tmp.cleanup()

    def test_response_ids_query_only_requested_keys_in_caller_order(self):
        keys = [("c", "3", 1), ("c", "1", 1), ("c", "3", 1)]
        with patch.object(self.log, "rows", wraps=self.log.rows) as rows:
            self.assertEqual(
                self.log.response_ids(self.dataset, keys),
                [self.ids[3], self.ids[1], self.ids[3]],
            )
            for call, key in zip(rows.call_args_list, keys):
                sql, parameters = call.args
                self.assertEqual(parameters, (self.dataset, *key))
                self.assertIn(
                    "WHERE dataset_id=? AND cell_id=? AND drug_id=? AND dose=?", sql
                )
            self.assertEqual(rows.call_count, len(keys))
            rows.reset_mock()
            self.assertEqual(self.log.response_ids(self.dataset, []), [])
            rows.assert_not_called()

    def test_trajectory_predictions_errors_and_reconstruction(self):
        initial = self.log.current_state(self.exp)
        fit = fitted(self.log, self.exp, self.ids[:1])
        after_fit = self.log.current_state(self.exp)
        self.assertEqual(initial["data_state_id"], after_fit["data_state_id"])
        pred = self.log.record_predictions(
            fit, [{"response_id": self.ids[1], "point_estimate": -0.5}]
        )
        acquire = self.log.start_action(self.exp, "acquire")
        self.log.complete_acquisition(
            acquire,
            self.ids[1:2],
            prediction_id=pred,
            policy="lowest",
            requested_size=1,
            objective="minimize",
        )
        after = self.log.current_state(self.exp)
        self.assertEqual(after["model_state_id"], after_fit["model_state_id"])
        self.assertEqual(
            self.log.revealed_ids(after["data_state_id"]), set(self.ids[:2])
        )
        self.assertEqual(
            self.log.revealed_ids(initial["data_state_id"]), set(self.ids[:1])
        )
        self.assertEqual(self.log.candidate_ids(self.exp), set(self.ids[2:4]))
        errors = self.log.rows("SELECT * FROM acquisition_errors")
        self.assertEqual(errors[0]["residual"], 0.5)
        self.assertEqual(
            self.log.one("SELECT value FROM evaluations WHERE metric='rmse'")["value"],
            0.5,
        )
        self.assertFalse(self.log.rows("PRAGMA foreign_key_check"))

    def test_invalid_updates_and_acquisitions_are_atomic(self):
        before = self.log.current_state(self.exp)
        fit = self.log.start_action(self.exp, "fit")
        with self.assertRaises(ValueError):
            self.log.complete_fit(fit, self.ids[:2], manifest=MANIFEST)
        self.assertEqual(self.log.current_state(self.exp), before)
        self.assertFalse(self.log.rows("SELECT * FROM model_states"))
        self.log.fail_action(fit, "expected")
        fitted(self.log, self.exp, self.ids[:1])
        for bad in [self.ids[:1], self.ids[4:], self.ids[1:2] * 2]:
            aid = self.log.start_action(self.exp, "acquire")
            with self.assertRaises(ValueError):
                self.log.complete_acquisition(
                    aid, bad, policy="random", requested_size=2
                )
            self.assertFalse(self.log.rows("SELECT * FROM acquisition_items"))
            self.log.fail_action(aid, "expected")

    def test_prediction_scope_and_stale_state(self):
        fit = fitted(self.log, self.exp, self.ids[:1])
        pred = self.log.record_predictions(
            fit, [{"response_id": i, "point_estimate": 0} for i in self.ids[1:3]]
        )
        aid = self.log.start_action(self.exp, "acquire")
        self.log.complete_acquisition(
            aid, self.ids[1:2], policy="random", requested_size=1
        )
        aid = self.log.start_action(self.exp, "acquire")
        with self.assertRaises(ValueError):
            self.log.complete_acquisition(
                aid,
                self.ids[2:3],
                prediction_id=pred,
                policy="lowest",
                requested_size=1,
            )
        self.assertEqual(
            self.log.one("SELECT status FROM actions WHERE action_id=?", (aid,))[
                "status"
            ],
            "running",
        )

    def test_logged_prediction_nan_extras_preserve_valid_values(self):
        from nbutils.batchie_mono.logger.integration import LoggedModel

        pairs = pd.DataFrame(
            [["c", "1", 1.0]], columns=["ModelID", "broad_id", "dose"]
        )
        extras = {
            "missing": float("nan"),
            "numpy_missing": np.float32("nan"),
            "finite": 1.25,
            "integer": 2,
            "boolean": False,
            "text": "NaN",
            "null": None,
            "list": [1, "two"],
            "object": {"value": 3},
        }
        result = pairs.assign(predicted_mean=-0.5)
        for key, value in extras.items():
            result[key] = pd.Series([value], dtype=object)
        before = result.copy(deep=True)
        model = FakeBayesianModel()
        wrapped = LoggedModel(model, self.log, "test")
        wrapped.dataset_id = self.dataset
        wrapped.last_run_id = fitted(self.log, self.exp, self.ids[:1])
        wrapped.family = "test"
        with patch.object(model, "predict", return_value=result) as predict:
            self.assertIs(wrapped.predict(pairs), result)
            predict.assert_called_once_with(pairs)
        saved = json.loads(self.log.one("SELECT extra FROM prediction_values")["extra"])
        self.assertEqual(saved, {**extras, "missing": None, "numpy_missing": None})
        pd.testing.assert_frame_equal(result, before)

    def test_strict_constraints_and_missingness(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.log.connection.execute("UPDATE responses SET value='not numeric'")
        with self.assertRaises(ValueError):
            self.log.register_dataset(
                [dict(cell_id="c", drug_id="d", dose=1, value=float("nan"))]
            )
        for key in [
            ("missing", "pair", 1),
            ("C", "1", 1),
            ("c", 1, 1),
            ("c", "1", "1"),
            ("c", "1", 1.0000000000000002),
        ]:
            with self.subTest(key=key), self.assertRaises(KeyError) as raised:
                self.log.response_ids(self.dataset, [key])
            self.assertEqual(raised.exception.args, (key,))
        with self.assertRaises(ValueError):
            self.log.create_split(self.dataset, initial=self.ids[:1], pool=self.ids[:1])
        self.assertEqual(
            self.log.create_split(
                self.dataset,
                initial=self.ids[:1],
                pool=self.ids[1:4],
                test=self.ids[4:],
            ),
            self.split,
        )

    def test_idempotent_registration_and_single_active_action(self):
        aid = self.log.start_action(self.exp, "fit", key="once")
        self.assertEqual(self.log.start_action(self.exp, "fit", key="once"), aid)
        with self.assertRaises(ValueError):
            self.log.start_action(self.exp, "acquire", key="once")
        with self.assertRaises(sqlite3.IntegrityError):
            self.log.start_action(self.exp, "fit")
        self.log.complete_fit(aid, self.ids[:1], manifest=MANIFEST)
        self.assertEqual(self.log.start_action(self.exp, "fit", key="once"), aid)
        with self.assertRaises(ValueError):
            self.log.complete_fit(aid, self.ids[:1], manifest=MANIFEST)

    def test_artifact_portability_and_corruption(self):
        artifact = self.log.artifacts.write(
            "checkpoint", "bin", lambda p: p.write_bytes(b"weights")
        )
        aid = self.log.start_action(self.exp, "fit")
        self.log.complete_fit(
            aid, self.ids[:1], manifest=MANIFEST, artifacts=[artifact]
        )
        self.assertFalse(Path(artifact.relative_path).is_absolute())
        dest = Path(self.tmp.name) / "copy"
        dest.mkdir()
        # Backup gives a consistent snapshot while the source uses WAL.
        with sqlite3.connect(dest / self.path.name) as target:
            self.log.connection.backup(target)
        shutil.copytree(self.path.parent / "artifacts", dest / "artifacts")
        with ExperimentLogger(dest / self.path.name) as copy:
            saved = copy.artifact_records(aid)[0]
            self.assertEqual(copy.artifacts.verify(saved).read_bytes(), b"weights")
            copy.artifacts.verify(saved).write_bytes(b"broken")
            with self.assertRaises(ValueError):
                copy.artifacts.verify(saved)

    def test_model_neutral_update(self):
        aid = self.log.start_action(self.exp, "reflect")
        artifact = self.log.artifacts.write(
            "requests", "jsonl", lambda p: p.write_text('{"answer":1}\n{"answer":2}\n')
        )
        self.log.complete_model_update(
            aid, self.ids[:1], manifest=MANIFEST, artifacts=[artifact]
        )
        self.assertEqual(self.log.get_run_metadata(aid)["manifest"], MANIFEST)

    def test_multiple_controller_connections(self):
        # Connections are created in their owning threads, as in independent folds.
        def worker(i):
            with ExperimentLogger(self.path) as log:
                exp = log.start_experiment(str(i), self.split, model_family="test")
                fitted(log, exp, self.ids[:1])
                log.finish_experiment(exp)

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(worker, range(12)))
        self.assertEqual(
            len(self.log.rows("SELECT * FROM experiments WHERE status='completed'")), 12
        )

    def test_independent_processes_and_stdlib_only_package_import(self):
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-S",
                    str(TESTS / "concurrent_writer.py"),
                    str(self.path),
                    self.split,
                    self.ids[0],
                    str(i),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for i in range(6)
        ]
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=40)
                self.assertEqual(process.returncode, 0, stderr)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            for process in processes:
                process.communicate()
        self.assertEqual(
            len(self.log.rows("SELECT * FROM experiments WHERE status='completed'")), 6
        )

    def test_reject_legacy_database(self):
        path = Path(self.tmp.name) / "old.db"
        with sqlite3.connect(path) as old:
            old.execute("PRAGMA user_version=1")
        with self.assertRaisesRegex(ValueError, "Legacy"):
            ExperimentLogger(path)

    def test_artifact_store_initialization_failure_closes_connection(self):
        connections = []
        connect = sqlite3.connect

        def capture_connection(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connections.append(connection)
            return connection

        error = OSError("artifact directory unavailable")
        with patch(
            "nbutils.batchie_mono.logger._storage.sqlite3.connect",
            wraps=capture_connection,
        ), patch(
            "nbutils.batchie_mono.logger.logger.ArtifactStore", side_effect=error
        ):
            with self.assertRaises(OSError) as raised:
                ExperimentLogger(self.path)
        self.assertIs(raised.exception, error)
        self.assertEqual(len(connections), 1)
        with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
            connections[0].execute("SELECT 1")


class FakeBayesianModel:
    """Real ArviZ serialization with deterministic, cheap model behavior."""

    prediction_seed = 0
    inference_data_ = None

    def __init__(self, fail=False):
        self.fit_calls = self.predict_calls = 0
        self.fit_frames = []
        self.fail = fail

    def fit(self, frame):
        import arviz as az

        self.fit_calls += 1
        self.fit_frames.append(frame.copy())
        if self.fail:
            raise RuntimeError("fit failed")
        self.inference_data_ = az.from_dict(
            posterior={"offset": np.arange(12, dtype=float).reshape(2, 6)},
            sample_stats={"diverging": np.zeros((2, 6), dtype=bool)},
        )
        self.inference_data_.attrs.update(
            model_config='{"rank":2}', backend="test", numeric_array=np.array([1.0, 2.0])
        )
        return self

    def predict(self, pairs):
        self.predict_calls += 1
        result = pairs.copy()
        result["predicted_mean"] = [-float(x[1:]) + 0.25 for x in pairs["broad_id"]]
        result["posterior_std"] = 0.2
        result["predictive_std"] = 0.3
        result["interval_lower"] = result.predicted_mean - 0.5
        result["interval_upper"] = result.predicted_mean + 0.5
        return result


def frames():
    background = pd.DataFrame(
        [["b", f"d{i}", 1.0, float(-i)] for i in range(5)],
        columns=["ModelID", "broad_id", "dose", "logfold_change"],
    )
    target = background.copy()
    target["ModelID"] = "t"
    return background, target


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("arviz", "h5netcdf", "h5py")),
    "ArviZ, h5netcdf, and h5py required for integration tests",
)
class IntegrationTests(unittest.TestCase):
    def test_logging_preserves_simulation_and_roundtrips_trace(self):
        background, target = frames()
        for selector_type in (PredictedLowestSelector, RandomSelector):
            with self.subTest(
                selector=selector_type.__name__
            ), tempfile.TemporaryDirectory() as directory:
                baseline, logged = FakeBayesianModel(), FakeBayesianModel()
                kwargs = dict(
                    background=background,
                    target_responses=target,
                    initial_compounds=["d0"],
                    batch_size=2,
                    iterations=2,
                )
                expected = run_simulation(
                    **kwargs, model=baseline, selector=selector_type()
                )
                with ExperimentLogger(Path(directory) / "db.sqlite") as log:
                    actual = run_simulation(
                        **kwargs,
                        model=logged,
                        selector=selector_type(),
                        logger=log,
                        experiment_name="comparison",
                    )
                    pd.testing.assert_frame_equal(
                        expected, actual.drop(columns="trajectory_id")
                    )
                    self.assertEqual(
                        (logged.fit_calls, logged.predict_calls),
                        (baseline.fit_calls, baseline.predict_calls),
                    )
                    for a, b in zip(baseline.fit_frames, logged.fit_frames):
                        pd.testing.assert_frame_equal(a, b)
                    fits = log.rows("SELECT action_id FROM actions WHERE kind='fit'")
                    self.assertEqual(len(fits), 2)
                    metadata, trace = log.load_run(fits[-1]["action_id"])
                    self.assertEqual(
                        metadata["manifest"]["trace_metadata"]["numeric_array"], [1.0, 2.0]
                    )
                    self.assertEqual(
                        metadata["manifest"]["trace_metadata"]["model_config"], {"rank": 2}
                    )
                    np.testing.assert_array_equal(
                        trace.attrs["numeric_array"],
                        logged.inference_data_.attrs["numeric_array"],
                    )
                    self.assertIsInstance(
                        logged.inference_data_.attrs["numeric_array"], np.ndarray
                    )
                    np.testing.assert_array_equal(
                        trace.posterior.offset, logged.inference_data_.posterior.offset
                    )
                    self.assertEqual(
                        trace.posterior.sizes, logged.inference_data_.posterior.sizes
                    )
                    exp = actual.trajectory_id.iloc[0]
                    self.assertEqual(log.candidate_ids(exp), set())
                    self.assertEqual(
                        log.one("SELECT status FROM experiments")["status"], "completed"
                    )
                    errors = log.rows("SELECT * FROM evaluations WHERE metric='rmse'")
                    self.assertEqual(
                        len(errors), 0 if selector_type is RandomSelector else 2
                    )
                    self.assertFalse(log.rows("PRAGMA foreign_key_check"))

    def test_selector_failure_keeps_completed_fit_without_reveal(self):
        class BrokenSelector:
            name = "broken"

            def select(self, **kwargs):
                raise RuntimeError("selector failed")

        background, target = frames()
        with tempfile.TemporaryDirectory() as directory, ExperimentLogger(
            Path(directory) / "db"
        ) as log:
            with self.assertRaisesRegex(RuntimeError, "selector failed"):
                run_simulation(
                    background=background,
                    target_responses=target,
                    initial_compounds=["d0"],
                    model=FakeBayesianModel(),
                    selector=BrokenSelector(),
                    iterations=1,
                    logger=log,
                    experiment_name="failure",
                )
            self.assertEqual(
                log.one("SELECT status FROM actions WHERE kind='fit'")["status"],
                "completed",
            )
            self.assertEqual(
                log.one("SELECT status FROM actions WHERE kind='acquire'")["status"],
                "failed",
            )
            self.assertFalse(log.rows("SELECT * FROM acquisition_items"))
            self.assertEqual(len(log.rows("SELECT * FROM data_states")), 1)

    def test_failed_fit_and_zero_rounds(self):
        background, target = frames()
        with tempfile.TemporaryDirectory() as directory, ExperimentLogger(
            Path(directory) / "db"
        ) as log:
            kwargs = dict(
                background=background,
                target_responses=target,
                initial_compounds=["d0"],
                selector=RandomSelector(),
                logger=log,
                experiment_name="test",
            )
            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                run_simulation(
                    **kwargs, model=FakeBayesianModel(fail=True), iterations=1
                )
            self.assertEqual(log.one("SELECT status FROM actions")["status"], "failed")
            self.assertEqual(
                log.one("SELECT status FROM experiments")["status"], "failed"
            )
            self.assertFalse(log.rows("SELECT * FROM model_states"))
            result = run_simulation(**kwargs, model=FakeBayesianModel(), iterations=0)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(log.rows("SELECT * FROM experiments")), 2)
            self.assertEqual(len(log.rows("SELECT * FROM actions")), 1)

            metadata = {"dataset": {"source": "test"}}
            kwargs["initial_compounds"] = target["broad_id"].tolist()
            model = FakeBayesianModel()
            result = run_simulation(
                **kwargs, model=model, iterations=1, metadata=metadata
            )
            exp = result.trajectory_id.iloc[0]
            config = json.loads(log.one(
                "SELECT config FROM experiments WHERE experiment_id=?", (exp,)
            )["config"])
            self.assertEqual(config["dataset"], {
                "background": fingerprint_responses(background),
                "target": fingerprint_responses(target),
                "background_rows": len(background),
                "target_rows": len(target),
            })
            self.assertNotEqual(
                config["dataset"]["background"], config["dataset"]["target"]
            )
            self.assertEqual(config["provenance"], metadata)
            self.assertEqual(metadata, {"dataset": {"source": "test"}})
            self.assertEqual(log.candidate_ids(exp), set())
            self.assertEqual((model.fit_calls, model.predict_calls), (0, 0))


if __name__ == "__main__":
    unittest.main()
