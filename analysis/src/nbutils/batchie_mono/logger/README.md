# Adaptive-screen experiment logger

A self-contained SQLite logger for iterative drug screening. The core knows
responses, splits, states, actions, predictions and artifacts. It does not import
Batchie, PyMC, ArviZ, NumPy, pandas, Torch or DSPy. The optional `integration.py`
adapter translates the supplied simulation's DataFrames and Bayesian model.

## What is recorded

An experiment is one trajectory: one split, model family, configuration and
replicate. `name` groups comparable experiments. Dataset and split IDs are
content hashes, so identical inputs are reused within the same database;
experiment and action IDs are UUIDs, so replicates remain distinct.

An experiment state references a model snapshot and a data snapshot. A fit
changes only the model reference; acquisition changes only the data reference.
The newly revealed observations do not retroactively become inputs to the old fit.

```mermaid
flowchart TD
    START(["Start trajectory"]) --> INIT["Register dataset/split<br/>and initialize experiment"]
    INIT --> DATA[("Initial revealed data<br/>background + target seed")]
    INIT --> STATE["Current state<br/>model reference + data reference"]
    STATE --> CONTINUE{"More rounds<br/>and candidates?"}
    CONTINUE -->|No| FINISH["Mark experiment completed"]
    FINISH --> END(["End trajectory"])
    CONTINUE -->|Yes| FIT["Fit action<br/>use only revealed responses"]
    DATA -. "training input" .-> FIT
    FIT --> MODEL[("New model snapshot")]
    FIT --> TRACE[("Posterior artifact<br/>Bayesian fits")]
    FIT --> FITSTATE["Publish state after fit<br/>data reference unchanged"]
    MODEL -. "model reference" .-> FITSTATE
    FITSTATE --> ACQUIRE["Reserve acquire action"]
    ACQUIRE --> NEEDPRED{"Policy calls<br/>predict?"}
    NEEDPRED -->|Yes| PREDICT["Predict remaining candidates"]
    PREDICT --> PRED[("Frozen prediction set")]
    PRED --> SELECT["Select ordered batch"]
    NEEDPRED -->|"No: random policy"| SELECT
    SELECT --> ITEMS[("Ordered acquisition items")]
    ITEMS --> REVEAL["Reveal selected responses<br/>and record evaluations"]
    REVEAL --> DELTA[("New data snapshot<br/>parent-linked reveal delta")]
    REVEAL --> METRICS[("Evaluation records")]
    DELTA --> NEXT["Publish state after acquisition<br/>model reference unchanged"]
    NEXT --> STATE
    DELTA -. "reconstruct revealed data" .-> DATA

    classDef action fill:#e8f0fe,stroke:#2563eb,color:#172554;
    classDef product fill:#ecfdf5,stroke:#059669,color:#064e3b;
    classDef decision fill:#fffbeb,stroke:#d97706,color:#78350f;
    classDef state fill:#f3e8ff,stroke:#9333ea,color:#581c87;
    classDef terminal fill:#f1f5f9,stroke:#64748b,color:#0f172a;
    class INIT,FIT,ACQUIRE,PREDICT,SELECT,REVEAL,FINISH action;
    class DATA,MODEL,TRACE,PRED,ITEMS,DELTA,METRICS product;
    class CONTINUE,NEEDPRED decision;
    class STATE,FITSTATE,NEXT state;
    class START,END terminal;
```

| Table | Role |
|---|---|
| `datasets` | Immutable response definition/provenance and measured row count |
| `entities` | Cell and drug axes, with optional tissue/MOA metadata |
| `responses` | Measured cell–drug–dose entries and ground-truth values |
| `splits`, `split_memberships` | Explicit initial/pool/validation/test roles |
| `experiments` | Actual trajectories, configuration, status, current-state pointer |
| `experiment_states` | Ordered pairs of model/data references |
| `data_states`, `reveals` | Parent-linked reveal deltas; initial state includes background |
| `model_states` | Versioned manifest and producing action |
| `actions` | Input/output state, kind, configuration, timing, status and failure |
| `fit_inputs` | Exact input response IDs **and their order**, also usable by reflection |
| `artifacts` | Relative paths, sizes, checksums, roles and format metadata |
| `prediction_sets`, `prediction_values` | Frozen predictions linked to the producing model and data state |
| `acquisitions`, `acquisition_items` | Policy, requested/actual size and ordered selections |
| `evaluations` | Typed scalar results with scope, sample count and definition version |
| `acquisition_errors` (view) | Acquired truth, pre-reveal estimate, residual and uncertainty |
