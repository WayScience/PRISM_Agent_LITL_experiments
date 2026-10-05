#!/usr/bin/env python
# coding: utf-8

# # PRISM secondary-screen response table
# 
# This notebook converts the DepMap PRISM secondary-screen log-fold-change matrix into a tidy, analysis-ready table. It joins each measured response to compound/treatment annotations and DepMap model metadata, then writes the result as a Parquet file.
# 
# **Output:** `data/secondary-screen-observed-response-long.parquet` (relative to the repository root).

# In[1]:


from pathlib import Path

import pandas as pd


# In[2]:


# Scripts start from their own location; notebooks start from the working directory.
start_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
for repo_root in (start_path, *start_path.parents):
    # The nested analysis project also has a pyproject.toml; require the repo layout.
    if (
        (repo_root / "pyproject.toml").is_file()
        and (repo_root / "analysis").is_dir()
    ):
        break
else:
    raise FileNotFoundError(f"Could not locate the PRISM repository root from {start_path}")

data_path = repo_root / "data"
if not data_path.exists():
    raise FileNotFoundError(f"Data path does not exist: {data_path}")

lfc_file = data_path / "secondary-screen-replicate-collapsed-logfold-change.csv"
if not lfc_file.exists():
    raise FileNotFoundError(f"LFC file does not exist: {lfc_file}")
lfc_df = pd.read_csv(lfc_file)

trt_file = data_path / "secondary-screen-replicate-collapsed-treatment-info.csv"
if not trt_file.exists():
    raise FileNotFoundError(f"Treatment file does not exist: {trt_file}")
trt_df = pd.read_csv(trt_file)

model_file = data_path / "Model.csv"
if not model_file.exists():
    raise FileNotFoundError(f"Model file does not exist: {model_file}")
model_df = pd.read_csv(model_file)

out_file = data_path / "secondary-screen-observed-response-long.parquet"


# ## Build the long-form response table
# 
# The three inputs loaded above provide complementary pieces of information:
# 
# - **Log-fold-change matrix:** one row per DepMap model, with encoded treatment columns and replicate-collapsed response values.
# - **Treatment information:** maps each encoded column to a compound, dose, screen, and available drug annotations.
# - **Model metadata:** maps `ModelID` to readable cell-line names and disease, lineage, and model-type fields.
# 
# The transformation below follows these steps:
# 
# 1. Reshape the response matrix from wide to long format, producing one record per model and encoded treatment.
# 2. Exclude rows whose model identifier contains `FAILED` and remove records with missing log-fold-change values.
# 3. Join treatment annotations by the exact encoded treatment key, then enrich records with model metadata by `ModelID`. Both joins are checked as many-to-one; unmatched treatment rows are reported.
# 4. Extract `panel_identity` from the `_PR500` suffix in `column_name`, then map it to `cell_line_type`: `PR500` means an adherent cell line, while non-PR500 means a suspension cell line.
# 5. Average repeated measurements only within the same `ModelID`, compound (`broad_id`), dose, screen, and panel. Missing grouping values are retained during aggregation.
# 6. For identified model–compound–dose experiments, retain the highest-reliability screen (`MTS010 > MTS006 > MTS005 > HTS002`), then prefer `PR500` over `not PR500` within that screen. Unranked/missing screens rank below the named screens; ties are retained. Rows with missing model, compound, or dose keys are retained because their experiment identity cannot be safely compared. See [here](https://forum.depmap.org/t/question-about-duplicated-measures-in-prism-mts010-screen-auc-data/83?utm_source=chatgpt.com) for details.
# 7. Convert dose values to numeric where possible and sort the resulting table for easier inspection.
# 
# Keeping screen and panel in the within-screen aggregation key preserves separate measurements until priority selection. Missing/unparseable dose values and unmatched treatment metadata are retained.

# In[3]:


# Build one observed response per cell line x compound x dose from replicate-collapsed matrix.
cell_id_col = lfc_df.columns[0]

# 1) Matrix (wide) -> long observed responses.
obs_long = lfc_df.rename(columns={cell_id_col: "ModelID"}).melt(
    id_vars="ModelID",
    var_name="column_name",
    value_name="logfold_change",
)
# Preserve the panel marker and derive the corresponding cell-line growth type.
obs_long["panel_identity"] = (
    obs_long["column_name"]
    .str.extract(r"_(PR500)$", expand=False)
    .fillna("not PR500")
)
obs_long["cell_line_type"] = obs_long["panel_identity"].map(
    {"PR500": "adherent cell line", "not PR500": "suspension cell line"}
)

# Remove known failed profiles and missing responses.
obs_long = obs_long[~obs_long["ModelID"].astype(str).str.contains("FAILED", na=False)].copy()
obs_long = obs_long.dropna(subset=["logfold_change"])

# 2) Join treatment metadata using the exact encoded treatment key.
obs_with_treatment = obs_long.merge(
    trt_df,
    on="column_name",
    how="left",
    validate="many_to_one",
)

missing_treatment = obs_with_treatment["broad_id"].isna().sum()
if missing_treatment:
    print(f"Warning: {missing_treatment} rows are missing treatment metadata.")

# 3) Enrich with cell line disease/model metadata from Model.csv.
model_cols = [
    "ModelID",
    "CellLineName",
    "StrippedCellLineName",
    "DepmapModelType",
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "OncotreeCode",
    "TissueOrigin",
    "ModelType",
]
model_info = model_df[model_cols].drop_duplicates(subset=["ModelID"])

observed_response = obs_with_treatment.merge(
    model_info,
    on="ModelID",
    how="left",
    validate="many_to_one",
)

# 4) Average duplicates only within the same model/compound/dose/screen/panel.
screen_key_cols = ["ModelID", "broad_id", "dose", "screen_id", "panel_identity"]
experiment_key_cols = ["ModelID", "broad_id", "dose"]
annotation_agg = {
    "logfold_change": "mean",
    "column_name": "first",
    "cell_line_type": "first",
    "compound_plate": "first",
    "name": "first",
    "moa": "first",
    "target": "first",
    "disease.area": "first",
    "indication": "first",
    "smiles": "first",
    "phase": "first",
    "CellLineName": "first",
    "StrippedCellLineName": "first",
    "DepmapModelType": "first",
    "OncotreeLineage": "first",
    "OncotreePrimaryDisease": "first",
    "OncotreeSubtype": "first",
    "OncotreeCode": "first",
    "TissueOrigin": "first",
    "ModelType": "first",
}
within_screen_duplicates = observed_response.duplicated(
    subset=screen_key_cols,
    keep=False,
).sum()
if within_screen_duplicates:
    print(
        f"Averaging {within_screen_duplicates} duplicate rows within "
        f"{screen_key_cols}."
    )
observed_response = (
    observed_response
    .groupby(screen_key_cols, as_index=False, dropna=False)
    .agg(annotation_agg)
)

# 5) Select by screen reliability first, then prefer PR500 within the best screen.
screen_reliability = ["MTS010", "MTS006", "MTS005", "HTS002"]
screen_rank = {screen_id: rank for rank, screen_id in enumerate(screen_reliability)}
identified_experiment = observed_response[experiment_key_cols].notna().all(axis=1)
identified = observed_response.loc[identified_experiment].copy()
identified["_screen_rank"] = (
    identified["screen_id"].map(screen_rank).fillna(len(screen_reliability)).astype(int)
)
best_screen_rank = identified.groupby(
    experiment_key_cols,
    dropna=False,
)["_screen_rank"].transform("min")
best_screen = identified.loc[identified["_screen_rank"].eq(best_screen_rank)].copy()
best_screen["_panel_rank"] = best_screen["panel_identity"].ne("PR500").astype(int)
best_panel_rank = best_screen.groupby(
    experiment_key_cols,
    dropna=False,
)["_panel_rank"].transform("min")
selected_screens = best_screen.loc[
    best_screen["_panel_rank"].eq(best_panel_rank)
].drop(columns=["_screen_rank", "_panel_rank"])

# Keep rows with incomplete experiment keys; their compound/dose identity is ambiguous.
unidentified = observed_response.loc[~identified_experiment]
dropped_lower_priority = len(identified) - len(selected_screens)
observed_response = pd.concat(
    [selected_screens, unidentified],
    ignore_index=True,
)
if dropped_lower_priority:
    print(
        f"Dropped {dropped_lower_priority} lower-priority rows; priority is "
        f"screen {' > '.join(screen_reliability)}, then PR500 > not PR500."
    )

# Standardize ordering and types.
observed_response["dose"] = pd.to_numeric(observed_response["dose"], errors="coerce")
observed_response = observed_response.sort_values(["ModelID", "name", "dose", "screen_id"]).reset_index(drop=True)
key_cols = ["ModelID", "broad_id", "dose", "screen_id"]

print("Observed response table shape:", observed_response.shape)
print("Unique models:", observed_response["ModelID"].nunique())
print("Unique compounds:", observed_response["broad_id"].nunique())
print("Unique model x compound x dose x screen keys:", observed_response[key_cols].drop_duplicates().shape[0])


# ## Inspect the curated records
# 
# The summary printed above reports the output dimensions, numbers of unique models and compounds, and the count of distinct model–compound–dose–screen keys. Compare the last two counts as a quick uniqueness check: they should match after duplicate handling. Review any warning about missing treatment annotations, since those rows remain in the table but may have null compound metadata.
# 
# The preview below shows the first few sorted records. Check that the response, dose, compound, screen, and model-identification columns look plausible before saving.

# In[4]:


observed_response.head()


# ## Write the Parquet output
# 
# The final cell writes the complete `observed_response` table—not just the displayed preview—to the path defined in the setup cell. The output is saved without a pandas index and can be loaded later with `pandas.read_parquet`. Running the save cell again overwrites the existing output file.

# In[5]:


# drop the "panel_identity" column from the observed_response DataFrame before saving to parquet
# this helps with reducing the size of the saved parquet file so github is happier
observed_response.drop(columns=["panel_identity"], errors="ignore").to_parquet(out_file, index=False)

