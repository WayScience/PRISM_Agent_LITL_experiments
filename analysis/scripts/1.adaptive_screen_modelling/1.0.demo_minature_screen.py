#!/usr/bin/env python
# coding: utf-8

# # Miniature adaptive monotherapy screen: background ablation
# 
# Compare **lowest predicted LFC** with **2, 6, or 10 background cell lines** against a shared **random acquisition** baseline. More negative LFC means stronger killing; we seek the lowest, not largest absolute, responses.
# - Pick the dose with the most response rows, average repeated screens, and randomly choose an eligible target cell line.
# - Sample 10 other cell lines once. Use nested backgrounds: the first 2, first 6, and all 10. Every candidate compound must occur in at least one of the **first two** background lines, keeping coverage fixed across comparisons.
# - Randomly limit this shared supported pool to 256 compounds. All runs use the same target, dose, pool, hit definition, and 16 initial target measurements, then reveal **8 compounds per round for 10 rounds**, refitting before each choice.
# - Random acquisition ignores predictions, so a single seeded random trajectory serves all three background sizes.
# - Hidden target responses are used only for revealing selected compounds and retrospective scoring. Background responses plus the initial 16 target responses train the first model; this is not a fit from only 16 total measurements.
# 
# **Retrieval metric:** fraction of the pool's lowest-LFC $\lceil 0.1N\rceil$ compounds discovered, including the initial 16. Ties are broken by compound ID. The hit set is fixed across runs and defined within this miniature pool, not all PRISM compounds. The x-axis counts target measurements only; historical background is excluded.
# 
# ## Note
# This is a sanity check demo that the minimal BATCHIE adaptation and implemenation works, to fully indicate superiority and utility we will need to place BATCHIE under stricter conditions like only seeing partial ground truth within each background cell lines and generalizing across tissue and cancer types. 

# In[1]:


from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbutils.batchie_mono.simulation import run_simulation
from nbutils.batchie_mono.factorization_model import BayesianFixedDoseModel
from nbutils.batchie_mono.model_acquisition import PredictedLowestSelector, RandomSelector
from nbutils.batchie_mono.spec import ModelSpec
from nbutils.batchie_mono.sampling import NUTSConfig
from nbutils.batchie_mono.pymc_backend import PyMCSampler
from nbutils.batchie_mono.logger import ExperimentLogger


# In[2]:


repo_root = next(
    path for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (path / "pyproject.toml").exists() and (path / "data").is_dir()
)
data_path = repo_root / "data"
response_file = data_path / "secondary-screen-observed-response-long.parquet"
if not response_file.exists():
    raise FileNotFoundError(f"Response file does not exist: {response_file}")

keys = ["ModelID", "broad_id", "dose"]
responses = pd.read_parquet(response_file, columns=[*keys, "logfold_change"])
responses.head()


# ## Experiment specifications

# In[3]:


# impacts both cell line/compound selection and the actual model fitting
# fine for demo purpose nontheless
# the former experiments will have separate deterministic datasplitting independent of model fitting
seed = 0
backend = "pymc"
model_spec = ModelSpec(
    rank=4,
    prior_shape=1.1,
    prior_rate=1.1,
    global_precision=0.01,
)

# Original lightweight defaults, suitable for checking the API migration.
# These are not a recommendation for reliable posterior inference.
sampling_config = NUTSConfig(
    burnin=40,
    posterior_samples=30,
    thin=2,
    chains=1,
    seed=seed,
    cores=8,
    blas_cores=1,
    target_accept=0.9,
    max_treedepth=10,
    parameterization="centered",
)

rng = np.random.default_rng(seed)

# initial size means the number of compounds initially tested for the target model
# batch size is the number of compounds acquired adaptively in each iteration
# iterations is the total number of acquisition rounds
# with 16, 8, 10, the total number of compounds whose experimental truth will be revealed
# to BATCHIE will be 16 + 8 * 10 = 96
initial_size, batch_size, iterations = 16, 8, 10

# how many shared drugs to pool across multiple cell lines
pool_size = 256
background_counts = [2, 6, 10]

# Choose the best-covered dose; average repeated screens into one response key.
fixed_dose = responses.groupby("dose").size().idxmax()
fixed = (
    responses.loc[responses["dose"].eq(fixed_dose)]
    .dropna(subset=[*keys, "logfold_change"])
    .groupby(keys, as_index=False)["logfold_change"].mean()
)
coverage = fixed.groupby("ModelID")["broad_id"].nunique()
eligible = coverage[coverage >= initial_size + batch_size * iterations].index

# pick one of all cell lines selected as the final evaluation target
target_model = str(rng.choice(eligible))
# everything else becomes part of the background models (training data)
other_models = rng.choice(
    coverage.index[coverage.index != target_model],
    size=max(background_counts), replace=False,
)

# Use nested backgrounds and one pool supported by even the first two lines.
smallest_background = fixed[fixed["ModelID"].isin(other_models[:2])]
target_responses = fixed[
    fixed["ModelID"].eq(target_model)
    & fixed["broad_id"].isin(smallest_background["broad_id"])
]

# sample the target responses to match the pool size
# ground truth y which the model will never see but willtry to predict
target_responses = target_responses.sample(
    n=min(pool_size, len(target_responses)), random_state=seed
).reset_index(drop=True)

if not (len(target_responses) >= initial_size + batch_size * iterations):
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
print(f"\tBackground models: {other_models[:max(background_counts)]}")
print(f"Shared pool: {len(target_responses)} compounds | backgrounds: {background_counts}")
print(f"Shared initial observations: {initial_size} | acquisition: {iterations} × {batch_size}")


# ## Fit PyMC NUTS and record each acquisition round
# 
# The lightweight configuration above uses one chain, 40 tuning steps, and 30 retained draws with `thin=2` (60 post-warmup transitions). These settings test mechanics, not posterior reliability.
# 
# One SQLite database stores each retained trace (including observed data and configuration), each actual prediction, and each selected/revealed batch with retrieval metrics. Random acquisition has no prediction records. Every simulation invocation gets a fresh trajectory ID; reruns append rather than overwrite. There is no extra fit after the final acquisition: fit index 9 has 88 target observations, and final retrieval is scored at 96.
# 
# Source data stay external. The source checksum and preparation recipe below accompany ordered background/target fingerprints computed by orchestration. Keep the source file and this preparation code to reconstruct and verify those inputs. Logging errors propagate; successful earlier writes remain committed.
# 
# **Rerun required:** the SQL integration has not been run on this full dataset. No historical CSV/NetCDF artifacts are imported automatically.

# In[4]:


def make_model():
    """Create an independent model for one simulation trajectory."""
    return BayesianFixedDoseModel(
        spec=model_spec,
        sampler=PyMCSampler(config=sampling_config),
        prediction_seed=seed,
    )

# Hash the external source once, not once per fit or comparison.
source_hash = hashlib.sha256()
with response_file.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        source_hash.update(chunk)
provenance = {
    "source": str(response_file.relative_to(repo_root)),
    "source_sha256": source_hash.hexdigest(),
    "preparation": "miniature-background-ablation-v1",
    "recipe": [
        "most frequent dose; drop missing core values; mean per response key",
        "seeded eligible target then nested background sampling without replacement",
        "pool supported by first two backgrounds; seeded pool and initial sampling",
    ],
    "data_seed": seed,
    "target_model": target_model,
    "fixed_dose": float(fixed_dose),
    "background_order": other_models.tolist(),
    "background_counts": background_counts,
    "pool_size": pool_size,
    "initial_size": initial_size,
    "versions": {"numpy": np.__version__, "pandas": pd.__version__},
}
db_path = repo_root / "output" / "miniature_screen.sqlite"
experiment_name = "miniature-background-ablation"
runs = []
random_background_count = min(background_counts)

with ExperimentLogger(db_path) as logger:
    for count in background_counts:
        result = run_simulation(
            background=backgrounds[count],
            target_responses=target_responses,
            initial_compounds=initial_compounds,
            model=make_model(),
            selector=PredictedLowestSelector(),
            batch_size=batch_size,
            iterations=iterations,
            logger=logger,
            experiment_name=experiment_name,
            metadata={**provenance, "background_cells": count},
        )
        result["background_cells"] = count
        result["background_measurements"] = len(backgrounds[count])
        runs.append(result)
        retrieval = result.iloc[-1]["top_10pct_retrieval"]
        print(f"{count} background lines: retrieved {retrieval:.1%} of top hits")

    random_baseline = run_simulation(
        background=backgrounds[random_background_count],
        target_responses=target_responses,
        initial_compounds=initial_compounds,
        model=make_model(),
        selector=RandomSelector(seed=seed),
        batch_size=batch_size,
        iterations=iterations,
        logger=logger,
        experiment_name=experiment_name,
        metadata={**provenance, "background_cells": random_background_count,
                  "selector_seed": seed},
    )
random_baseline["background_cells"] = random_background_count
random_baseline["background_measurements"] = len(backgrounds[random_background_count])
metrics = pd.concat([*runs, random_baseline], ignore_index=True)
metrics.groupby(["selector", "background_cells"]).tail(1)


# In[5]:


# Query this invocation, not every historical rerun of the same experiment.
with ExperimentLogger(db_path) as logger:
    saved_runs = logger.list_runs(trajectory_id=runs[0]["trajectory_id"].iloc[0])
    run_id = saved_runs.iloc[0]["run_id"]
    fit_metadata = logger.get_run_metadata(run_id)  # no trace BLOB read
    _, saved_trace = logger.load_run(run_id)  # explicit, potentially large read
    batch_id = logger.list_batches(run_id).iloc[0]["batch_id"]
    saved_batch = logger.load_batch(int(batch_id))
    prediction_id = saved_batch["metadata"]["last_prediction_id"]
    saved_prediction = (
        None if prediction_id is None else logger.load_prediction(prediction_id)
    )
print(fit_metadata["metadata"]["dataset"])
saved_runs


# ## Minimal sanity check results visualization

# The plot uses circles (2), squares (6), and triangles (10) for background size, and crosses for random acquisition. Differences reflect added background information for this particular nested set, not a general guarantee about the number of cell lines needed.

# In[6]:


fig, ax = plt.subplots(figsize=(9, 5))
markers = {2: "o", 6: "s", 10: "^"}
predicted_metrics = metrics[metrics["selector"].eq("predicted_lowest")]
for count, trajectory in predicted_metrics.groupby("background_cells"):
    ax.plot(
        trajectory["compounds_seen"], trajectory["top_10pct_retrieval"],
        marker=markers[count], label=f"Predicted lowest — {count} background lines",
    )
ax.plot(
    random_baseline["compounds_seen"], random_baseline["top_10pct_retrieval"],
    color="black", linestyle="--", marker="x", label="Random — shared baseline",
)
ax.set(
    xlabel=f"Target compounds observed (includes {initial_size} initial; +{batch_size} per round)",
    ylabel="Proportion of top-10% killing compounds retrieved",
    title=f"{target_model} | dose {fixed_dose:g} | {len(target_responses)}-compound pool",
    ylim=(0, 1.05),
    xticks=random_baseline["compounds_seen"],
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()
