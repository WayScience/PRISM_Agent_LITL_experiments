"""Independent controller process used by test_logger (stdlib only)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "src"))
from nbutils.batchie_mono.logger import ExperimentLogger

with ExperimentLogger(sys.argv[1]) as log:
    exp = log.start_experiment(sys.argv[4], sys.argv[2], model_family="test")
    action = log.start_action(exp, "fit")
    log.complete_fit(
        action,
        [sys.argv[3]],
        manifest={"schema_version": 1, "model_family": "test", "capabilities": []},
    )
    log.finish_experiment(exp)
