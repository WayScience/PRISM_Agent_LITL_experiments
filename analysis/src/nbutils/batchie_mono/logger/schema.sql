CREATE TABLE experiment_runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    trajectory_id TEXT,
    batch_index INTEGER CHECK (batch_index >= 0),
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    backend TEXT,
    parameterization TEXT,
    seed INTEGER,
    raw_postwarmup_divergences INTEGER,
    raw_postwarmup_tree_depth_hits INTEGER,
    model_config TEXT NOT NULL,
    inference_config TEXT NOT NULL,
    trace_metadata TEXT NOT NULL,
    metadata TEXT NOT NULL,
    trace_netcdf BLOB NOT NULL
);
CREATE INDEX runs_by_experiment ON experiment_runs(experiment_name, backend);
CREATE INDEX runs_by_trajectory ON experiment_runs(trajectory_id, batch_index);
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    metadata TEXT NOT NULL,
    frame_json TEXT NOT NULL
);
CREATE TABLE batches (
    batch_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    metadata TEXT NOT NULL,
    selected_json TEXT NOT NULL,
    revealed_json TEXT
);
CREATE INDEX IF NOT EXISTS predictions_by_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS batches_by_run ON batches(run_id);
PRAGMA user_version = 1;
