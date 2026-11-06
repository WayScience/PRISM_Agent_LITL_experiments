-- Schema

-- drug response dataset
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    description TEXT NOT NULL CHECK(json_valid(description)),
    response_count INTEGER NOT NULL CHECK(response_count >= 0)
) STRICT;

-- Entities within the dataset, such as cells and drugs
CREATE TABLE entities (
    dataset_id TEXT NOT NULL REFERENCES datasets,
    axis TEXT NOT NULL CHECK(axis IN ('cell','drug')),
    entity_id TEXT NOT NULL,
    metadata TEXT NOT NULL CHECK(json_valid(metadata)),
    PRIMARY KEY(dataset_id, axis, entity_id)
) STRICT;

-- Individual responses linking cells and drugs within a dataset
CREATE TABLE responses (
    response_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets,
    cell_id TEXT NOT NULL,
    drug_id TEXT NOT NULL,
    dose REAL NOT NULL,
    value REAL NOT NULL,
    UNIQUE(dataset_id,cell_id,drug_id,dose)
) STRICT;

-- Data splits for cross-validation or train/test separation, related to a specific dataset
CREATE TABLE splits (
    split_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets,
    definition TEXT NOT NULL CHECK(json_valid(definition))
) STRICT;
CREATE TABLE split_memberships (
    split_id TEXT NOT NULL REFERENCES splits,
    response_id TEXT NOT NULL REFERENCES responses,
    role TEXT NOT NULL CHECK(role IN ('initial','pool','validation','test')),
    PRIMARY KEY(split_id,response_id)
) STRICT;

-- Experiments and their lifecycle, including current state and status
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    split_id TEXT NOT NULL REFERENCES splits,
    model_family TEXT NOT NULL,
    config TEXT NOT NULL CHECK(json_valid(config)),
    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),
    current_state_id TEXT REFERENCES experiment_states,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
) STRICT;
CREATE TABLE data_states (
    data_state_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments,
    parent_id TEXT REFERENCES data_states
) STRICT;
CREATE TABLE reveals (
    data_state_id TEXT NOT NULL REFERENCES data_states,
    response_id TEXT NOT NULL REFERENCES responses,
    PRIMARY KEY(data_state_id,response_id)
) STRICT;
CREATE TABLE model_states (
    model_state_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments,
    producing_action_id TEXT UNIQUE REFERENCES actions,
    manifest TEXT NOT NULL CHECK(json_valid(manifest))
) STRICT;
CREATE TABLE experiment_states (
    state_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments,
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    model_state_id TEXT REFERENCES model_states,
    data_state_id TEXT NOT NULL REFERENCES data_states,
    UNIQUE(experiment_id,sequence)
) STRICT;
CREATE TABLE actions (
    action_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments,
    kind TEXT NOT NULL,
    input_state_id TEXT NOT NULL REFERENCES experiment_states,
    output_state_id TEXT UNIQUE REFERENCES experiment_states,
    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),
    config TEXT NOT NULL CHECK(json_valid(config)),
    idempotency_key TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    UNIQUE(experiment_id,idempotency_key)
) STRICT;

-- Only one in-flight state transition in a trajectory. Different folds are independent.
CREATE UNIQUE INDEX active_action ON actions(experiment_id) WHERE status='running';

-- Actions that the model takes to alter the experiment state and or data state
CREATE TABLE fit_inputs (
    action_id TEXT NOT NULL REFERENCES actions,
    response_id TEXT NOT NULL REFERENCES responses,
    position INTEGER NOT NULL CHECK(position >= 0),
    UNIQUE(action_id,position),
    PRIMARY KEY(action_id,response_id)
) STRICT;
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES actions,
    role TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    format TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    metadata TEXT NOT NULL CHECK(json_valid(metadata))
) STRICT;
CREATE TABLE prediction_sets (
    prediction_id TEXT PRIMARY KEY,
    fit_action_id TEXT NOT NULL REFERENCES actions,
    state_id TEXT NOT NULL REFERENCES experiment_states,
    metadata TEXT NOT NULL CHECK(json_valid(metadata))
) STRICT;
CREATE TABLE prediction_values (
    prediction_id TEXT NOT NULL REFERENCES prediction_sets,
    response_id TEXT NOT NULL REFERENCES responses,
    position INTEGER NOT NULL CHECK(position >= 0),
    point_estimate REAL NOT NULL,
    posterior_std REAL CHECK(posterior_std >= 0),
    predictive_std REAL CHECK(predictive_std >= 0),
    interval_lower REAL,
    interval_upper REAL,
    extra TEXT NOT NULL CHECK(json_valid(extra)),
    PRIMARY KEY(prediction_id,response_id),
    UNIQUE(prediction_id,position),
    CHECK(interval_lower IS NULL OR interval_upper IS NULL OR interval_lower <= interval_upper)
) STRICT;
CREATE TABLE acquisitions (
    action_id TEXT PRIMARY KEY REFERENCES actions,
    prediction_id TEXT REFERENCES prediction_sets,
    policy TEXT NOT NULL,
    requested_size INTEGER NOT NULL CHECK(requested_size >= 0),
    selected_size INTEGER NOT NULL CHECK(selected_size >= 0),
    objective TEXT NOT NULL CHECK(objective IN ('minimize','maximize','none'))
) STRICT;
CREATE TABLE acquisition_items (
    action_id TEXT NOT NULL REFERENCES acquisitions,
    response_id TEXT NOT NULL REFERENCES responses,
    rank INTEGER NOT NULL CHECK(rank >= 0),
    PRIMARY KEY(action_id,response_id),
    UNIQUE(action_id,rank)
) STRICT;
CREATE TABLE evaluations (
    action_id TEXT NOT NULL REFERENCES actions,
    metric TEXT NOT NULL,
    scope TEXT NOT NULL,
    value REAL NOT NULL,
    n INTEGER NOT NULL CHECK(n >= 0),
    definition_version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(action_id,metric,scope)
) STRICT;
CREATE INDEX response_dataset ON responses(dataset_id);
CREATE INDEX action_experiment ON actions(experiment_id,started_at);
CREATE INDEX predictions_fit ON prediction_sets(fit_action_id);
CREATE INDEX artifacts_action ON artifacts(action_id);
CREATE VIEW acquisition_errors AS
SELECT i.action_id,i.response_id,i.rank,r.value AS actual,
       p.point_estimate,p.point_estimate-r.value AS residual,
       p.posterior_std,p.predictive_std,p.interval_lower,p.interval_upper
FROM acquisition_items i JOIN acquisitions a USING(action_id)
JOIN responses r USING(response_id)
LEFT JOIN prediction_values p ON p.prediction_id=a.prediction_id AND p.response_id=i.response_id;
PRAGMA user_version=2;
