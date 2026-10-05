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
# 4. Enforce one record per `ModelID`, compound (`broad_id`), dose, and screen. If repeated records share that key, their log-fold changes are averaged and the remaining annotations are taken from the first record in the group.
# 5. Convert dose values to numeric where possible and sort the resulting table for easier inspection.
# 
# Keeping `screen_id` in the key preserves separate measurements from different screens. Missing/unparseable dose values are retained as missing rather than discarded.

# In[3]:


# Build one observed response per cell line x compound x dose from replicate-collapsed matrix.
cell_id_col = lfc_df.columns[0]

# 1) Matrix (wide) -> long observed responses.
obs_long = lfc_df.rename(columns={cell_id_col: "ModelID"}).melt(
    id_vars="ModelID",
    var_name="column_name",
    value_name="logfold_change",
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

# 4) Keep one row per ModelID x compound x dose x screen.
key_cols = ["ModelID", "broad_id", "dose", "screen_id"]
dup_mask = observed_response.duplicated(subset=key_cols, keep=False)
if dup_mask.any():
    print(f"Found {dup_mask.sum()} duplicated rows for key {key_cols}; collapsing by mean LFC.")
    observed_response = (
        observed_response
        .groupby(key_cols, as_index=False)
        .agg({
            "logfold_change": "mean",
            "column_name": "first",
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
        })
    )
else:
    observed_response = observed_response.drop_duplicates(subset=key_cols).copy()

# Standardize ordering and types.
observed_response["dose"] = pd.to_numeric(observed_response["dose"], errors="coerce")
observed_response = observed_response.sort_values(["ModelID", "name", "dose", "screen_id"]).reset_index(drop=True)

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


observed_response.to_parquet(out_file, index=False)

