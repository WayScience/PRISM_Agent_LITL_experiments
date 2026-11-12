#!/usr/bin/env python
# coding: utf-8

# # Miniature adaptive monotherapy screen — custom Gibbs
# 
# Repeat notebook 1.0 with `BayesianFixedDoseModel(sampler=GibbsSampler(...))`, preserving response preparation, random seed and RNG call order, nested 2/6/10 backgrounds, a supported pool capped at 256 compounds, 16 initial target observations, and ten rounds of eight acquisitions. More negative LFC means stronger killing. The fixed hit set is the pool's lowest-LFC ceil(10% × pool size), ties broken by compound ID; retrieval includes the initial measurements.
# 
# The current Gibbs configuration uses four chains, 200 warmup sweeps and 100 retained posterior samples per chain at thinning 2. These settings are not matched to notebook 1.0's lightweight NUTS settings and do not demonstrate convergence. Gibbs does not use `target_accept` or `max_treedepth`.
# 
# The v2 `ExperimentLogger` records dataset/split provenance, ordered fit inputs, state/action lineage, frozen selector predictions, ordered acquisitions, evaluations, and checksum-verified NetCDF traces. A separate bundle under `output/demo_sampler_comparison/gibbs/logger_v2` preserves both the historical Gibbs exports and notebook 1.0's bundle; reruns append fresh experiment IDs without deleting data.
# 
# The simulator and selectors are unchanged. Hidden target responses are used only for reveals and retrospective scoring, not model fitting or selection before they are revealed. Random selection still fits through the same interface, but never calls `predict`. There is no fit after the final acquisition: the last fit sees 88 target observations; final retrieval is scored at 96.
# 
# After fitting, reload this invocation's trajectories with `load_logged_screen`, inspect saved fit settings, and plot the reloaded retrieval metrics. The loader checks artifact checksums and reconstructs and validates retrieval from registered responses and acquisitions; serialization round-trip and metric-equality regression checks belong in the automated test suite rather than this demo. Reloading supports posterior analysis, not model reconstruction or resumed sampling. Historical CSV exports consumed by notebook 1.2 are not refreshed by this v2 workflow. This is a single-target demonstration, not general efficacy evidence.

# In[1]:


from pathlib import Path
import sys
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbutils.batchie_mono.factorization_model import BayesianFixedDoseModel
from nbutils.batchie_mono.gibbs import GibbsSampler
from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.model_acquisition import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono.sampling import SamplingConfig
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.logger import ExperimentLogger
from nbutils.batchie_mono.logged_results import load_logged_screen

# Scripts start from their own location; notebooks start from the working directory.
start_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
repo_root = next(path for path in [start_path, *start_path.parents]
                 if (path / "pyproject.toml").is_file()
                 and (path / "analysis").is_dir()
                 and (path / "data").is_dir())

backend = "gibbs"
model_spec = ModelSpec(rank=4)
sampling_config = SamplingConfig(
    burnin=200,
    posterior_samples=100,
    thin=2,
    chains=4,
    cores=16,
    blas_cores=1,
)
sampler_settings = sampling_config.model_dump()

output_dir = repo_root / "output/demo_sampler_comparison" / backend
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Backend: {backend}; Python: {sys.executable}; model: {model_spec.model_dump()}; settings: {sampler_settings}")


# In[2]:


response_file = repo_root / "data/secondary-screen-observed-response-long.parquet"
if not response_file.exists():
    raise FileNotFoundError(f"Response file does not exist: {response_file}")
keys = ["ModelID", "broad_id", "dose"]
responses = pd.read_parquet(response_file, columns=[*keys, "logfold_change"])
responses.head()


# In[3]:


# Identical response-blind selection and RNG call order to notebook 1.0.
seed = 0
rng = np.random.default_rng(seed)
initial_size, batch_size, iterations = 16, 8, 10
pool_size = 256
background_counts = [2, 6, 10]
fixed_dose = responses.groupby("dose").size().idxmax()
fixed = (
    responses.loc[responses["dose"].eq(fixed_dose)]
    .dropna(subset=[*keys, "logfold_change"])
    .groupby(keys, as_index=False)["logfold_change"].mean()
)
coverage = fixed.groupby("ModelID")["broad_id"].nunique()
eligible = coverage[coverage >= initial_size + batch_size * iterations].index
target_model = str(rng.choice(eligible))
other_models = rng.choice(coverage.index[coverage.index != target_model],
                          size=max(background_counts), replace=False)
smallest_background = fixed[fixed["ModelID"].isin(other_models[:2])]
target_responses = fixed[
    fixed["ModelID"].eq(target_model)
    & fixed["broad_id"].isin(smallest_background["broad_id"])
]
target_responses = target_responses.sample(
    n=min(pool_size, len(target_responses)), random_state=seed,
).reset_index(drop=True)
if len(target_responses) < initial_size + batch_size * iterations:
    raise ValueError("Not enough target responses to match the required pool size.")
backgrounds = {
    count: fixed[
        fixed["ModelID"].isin(other_models[:count])
        & fixed["broad_id"].isin(target_responses["broad_id"])
    ].copy()
    for count in background_counts
}
initial_compounds = target_responses.sample(n=initial_size, random_state=seed)["broad_id"].tolist()
print(f"Target: {target_model} | fixed dose: {fixed_dose:g}")
print(f"Background models: {other_models}")
print(f"Shared pool: {len(target_responses)} compounds (cap {pool_size}) | backgrounds: {background_counts}")
print(f"Shared initial observations: {initial_size} | acquisition: {iterations} × {batch_size}")


# In[4]:


def make_model():
    """Create an independent Gibbs model for one simulation trajectory."""
    return BayesianFixedDoseModel(
        sampler=GibbsSampler(config=sampling_config),
        spec=model_spec,
        prediction_seed=seed,
    )

# Hash the source once; run_simulation registers prepared responses and fit inputs.
source_hash = hashlib.sha256()
with response_file.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        source_hash.update(chunk)
provenance = {
    "dataset": {
        "source": str(response_file.relative_to(repo_root)),
        "source_sha256": source_hash.hexdigest(),
        "preparation": "miniature-background-ablation-v1",
        "recipe": [
            "most frequent dose; drop missing core values; mean per response key",
            "seeded eligible target then nested background sampling without replacement",
            "pool supported by first two backgrounds; seeded pool and initial sampling",
        ],
    },
    "data_seed": seed,
    "target_model": target_model,
    "fixed_dose": float(fixed_dose),
    "background_order": other_models.tolist(),
    "background_counts": background_counts,
    "pool_size": pool_size,
    "initial_size": initial_size,
    "backend": backend,
    "versions": {"numpy": np.__version__, "pandas": pd.__version__},
}
# Isolate v2 artifacts from historical exports and the PyMC bundle; never delete on rerun.
db_path = output_dir / "logger_v2" / "miniature_screen_gibbs_v2.sqlite"
experiment_name = "miniature-background-ablation-gibbs"
runs = []
random_background_count = min(background_counts)

with ExperimentLogger(db_path) as logger:
    for count in background_counts:
        result = run_simulation(
            background=backgrounds[count], target_responses=target_responses,
            initial_compounds=initial_compounds, model=make_model(),
            selector=PredictedLowestSelector(), batch_size=batch_size, iterations=iterations,
            logger=logger, experiment_name=experiment_name,
            metadata={**provenance, "background_cells": count},
        )
        result["background_cells"] = count
        result["background_measurements"] = len(backgrounds[count])
        runs.append(result)
        print(f"{count} background lines: retrieved {result.iloc[-1]['top_10pct_retrieval']:.1%} of top hits")

    random_baseline = run_simulation(
        background=backgrounds[random_background_count], target_responses=target_responses,
        initial_compounds=initial_compounds, model=make_model(),
        selector=RandomSelector(seed=seed), batch_size=batch_size, iterations=iterations,
        logger=logger, experiment_name=experiment_name,
        metadata={**provenance, "background_cells": random_background_count,
                  "selector_seed": seed},
    )

random_baseline["background_cells"] = random_background_count
random_baseline["background_measurements"] = len(backgrounds[random_background_count])
metrics = pd.concat([*runs, random_baseline], ignore_index=True)
print(f"Saved {metrics['trajectory_id'].nunique()} trajectories and retrieval metrics to {db_path}")
metrics.groupby(["selector", "background_cells"]).tail(1)


# In[5]:


# Reload only this invocation's trajectories; the loader validates artifacts and retrieval.
trajectory_ids = metrics["trajectory_id"].unique().tolist()
loaded = load_logged_screen(db_path, experiment_name, experiment_ids=trajectory_ids)
reloaded_metrics = loaded.retrieval

# Inspect saved fit settings without retaining live models or loading posterior arrays.
loaded.inventory[[
    "run", "fit", "target_observations", "chains", "posterior_samples", "thin",
]]


# In[6]:


# Plot persisted/reloaded trajectories rather than the original in-memory results.
fig, ax = plt.subplots(figsize=(9, 5))
markers = {2: "o", 6: "s", 10: "^"}
for count, trajectory in reloaded_metrics[reloaded_metrics.selector.eq("predicted_lowest")].groupby("background_cells"):
    ax.plot(trajectory.compounds_seen, trajectory.top_10pct_retrieval,
            marker=markers[count], label=f"Gibbs predicted lowest — {count} background lines")
reloaded_random = reloaded_metrics[reloaded_metrics.selector.eq("random")]
ax.plot(reloaded_random.compounds_seen, reloaded_random.top_10pct_retrieval,
        color="black", linestyle="--", marker="x", label="Random — shared baseline")
ax.set(xlabel=f"Target compounds observed (includes {initial_size} initial; +{batch_size} per round)",
       ylabel="Proportion of top-10% killing compounds retrieved",
       title=f"Gibbs | {target_model} | dose {fixed_dose:g} | {len(target_responses)}-compound pool",
       ylim=(0, 1.05), xticks=reloaded_random.compounds_seen)
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(db_path.parent / "retrieval.png", dpi=150)
plt.show()
