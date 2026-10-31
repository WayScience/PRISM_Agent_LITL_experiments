# BATCHIE Experiment Logger

A SQLite writer for artifacts from completed model fits:
- Retained MCMC traces, including all supplied groups, coordinates, and attributes
- Optional prediction calls and selected/revealed acquisition records
- Fit configuration and caller-supplied provenance

Loading a trace helps with post-analysis of the bayesian sampling process.


## Storage design

The logger has separate tables for fits, prediction calls, and acquisition batches to match the structure of experimental artifacts.

```mermaid
flowchart TD
    DB["SQLite database"]
    DB --> R["experiment_runs: one row per logged fit"]
    DB --> P["predictions: one row per prediction call"]
    DB --> B["batches: one row per acquisition record"]
    P -->|"run_id foreign key"| R
    B -->|"run_id foreign key"| R
    R --> M["Metadata: experiment, trajectory, batch, configuration"]
    R --> N["trace_netcdf BLOB"]
    N --> X["ArviZ / xarray: posterior and sampler statistics"]
    X --> D["Named dimensions: chain, draw, cell, compound, factor"]
```
