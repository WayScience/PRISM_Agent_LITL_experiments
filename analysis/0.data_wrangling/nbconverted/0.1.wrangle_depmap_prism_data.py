#!/usr/bin/env python
# coding: utf-8

# # DepMap PRISM Data Wrangling
# 
# This notebook preprocesses the **DepMap PRISM secondary drug repurposing dataset** to produce a clean, deduplicated table of drug-cell line-IC50 values.  
# The workflow includes:
# 
# 1. **Config validation** – Ensure required file paths are set correctly in `config.yml`.  
# 2. **Data loading** – Import cell line metadata and drug dose–response parameters.  
# 3. **Deduplication** – Resolve duplicate cell line–drug pairs within each screen (`HTS002`, `MTS010`) by preferring the highest quality curve fit (`r²`) or falling back to reproducible random selection.  
# 4. **Screen merging** – Combine both screens, prioritizing `MTS010` where overlaps occur.  
# 5. **QC checks** – Confirm no duplicate (cell line, drug) pairs remain.  
# 6. **Summary statistics and visualization** – Tabulate and plot dataset composition by tissue and cell line.  
# 7. **Export** – Save the cleaned dataset for downstream analysis.  
# 
# The output represents the **preferred IC50 values per unique (cell line, drug) combination**, ready to be used in downsteam agentic system experiments.
# 

# In[1]:


import pathlib
import yaml
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Config Validation

# In[2]:


IN_NOTEBOOK = False
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        print("Running in Jupyter Notebook")
        IN_NOTEBOOK = True
    else:
        print("Running in IPython shell")
except NameError:
    print("Running in standard Python shell")


# In[3]:


# --- Step 1: Locate repo root and config file ---
git_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()
config_path = pathlib.Path(git_root) / "config.yml"

if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at: {config_path}")

# --- Step 2: Load config.yml ---
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# --- Step 3: Validate data section ---
data_cfg = config.get("data")
if not data_cfg:
    raise ValueError("Missing 'data' section in config.yml")

# Required keys
required_keys = ["depmap_prism", "cell_line_info", "dose_response"]

# --- Step 4: Collect resolved paths ---
results = []
errors = []  # collect problems for later
for key in required_keys:
    value = data_cfg.get(key)
    if value is None:
        results.append((key, None, "Missing in config"))
        errors.append(f"Config key '{key}' is missing")
        continue

    # depmap_prism is a directory, the others are files inside it
    if key == "depmap_prism":
        full_path = pathlib.Path(value)
    else:
        full_path = pathlib.Path(data_cfg["depmap_prism"]) / value

    if full_path.exists():
        status = "Exists"
    else:
        status = "Not found"
        errors.append(f"Path for '{key}' does not exist: {full_path}")

    results.append((key, str(full_path), status))

# --- Step 5: Display summary nicely ---
config_df = pd.DataFrame(
    results, columns=["Config Key", "Resolved Path", "Status"])
config_df.set_index("Config Key", inplace=True)
print(config_df)

# --- Step 6: Fail if any errors were collected ---
if errors:
    raise FileNotFoundError(
        "Config validation failed:\n" + "\n".join(f"- {e}" for e in errors) +
        "\nPlease refer to /config.yml.template for correct specification."
    )


# ## Preprocessing

# ### Load depmap PRISM cell line and drug dose response 

# In[4]:


cell_line_info_df = pd.read_csv(
    config_df.loc['cell_line_info', 'Resolved Path'])
print(cell_line_info_df.head())


# In[5]:


dose_response_df = pd.read_csv(config_df.loc['dose_response', 'Resolved Path'])
print(dose_response_df.head())


# ### Perform deduplication and merge
# 
# The secondary PRISM repurposing dataset includes two screens: `HTS002` and `MTS010`.  
# Both contain overlapping **cell line–drug combinations**, and according to the official PRISM documentation (https://ndownloader.figshare.com/files/20238123), results from `MTS010` should be preferred when available.  
# 
# The documentation also states that both `'ccle_name'` and `'depmap_id'` can be used to identify cell lines. To ensure robustness, we include both identifiers in all grouping operations.  
# 
# An additional complication arises because `HTS002` contains duplicate cell line–drug entries that are only uniquely distinguishable when including the `broad_id` (batch–drug identifier). To avoid multiple IC50 values for the same combination—which would create ambiguity and downstream issues for agentic systems—we perform **per-screen deduplication** before merging.  
# 
# Deduplication is carried out as follows:
# - Within each screen, group by `(smiles, depmap_id, ccle_name)`.
# - If multiple entries exist for a group:
#   - Prefer the row with the highest dose–response curve fit quality (`r²` value), if available.  
#   - Otherwise, select a single row at random, using a fixed seed for reproducibility.  
# - `smiles` is treated as the unique identifier for each drug.  
# 
# Finally, the deduplicated screens are combined, giving **priority to `MTS010`**: if the same cell line–drug pair exists in both screens, the `MTS010` entry is retained.
# 

# In[6]:


DEDUP_SEED = 42
CELL_DRUG_COMBO_KEYS = ["smiles","depmap_id","ccle_name"]

# --- Step 0: Keep the two screens of interest; basic QC ---
df = dose_response_df.query("screen_id in ['HTS002','MTS010']").copy()
df = df.dropna(subset=CELL_DRUG_COMBO_KEYS + ["ic50"])  # ensure keys exist
df["smiles"] = df["smiles"].astype(str).str.strip() # these identify unique drug

if "convergence" in df.columns:
    df = df[df["convergence"].eq(True)]

# --- Step 1: Deduplicate MTS010 by (smiles, cell line) ---
mts = df[df["screen_id"] == "MTS010"].copy()
if "r2" in mts.columns:
    # If multiple rows per (SMILES, cell line) and r^2 is available,
    # pick the highest-r^2 row per (SMILES, cell line)
    # prefer the better dose-reponse curve fit
    idx_mts = mts.groupby(CELL_DRUG_COMBO_KEYS)["r2"].idxmax()
    print(
        f"Deduplicating MTS010 via highest r^2: picked {len(idx_mts)} "
        f"rows from {len(mts)} total")
    mts_dedup = mts.loc[idx_mts]
else:
    # No r^2 -> pick one random row per (SMILES, cell line)
    # seed ensures reproducibility
    mts_dedup = mts.groupby(
        CELL_DRUG_COMBO_KEYS, 
        group_keys=False).sample(n=1, random_state=DEDUP_SEED)
    print(f"Deduplicating MTS010: picked {len(mts_dedup)} "
          f"rows from {len(mts)} total")

# --- Step 2: Deduplicate HTS002 by (smiles, cell line) ---
hts = df[df["screen_id"] == "HTS002"].copy()
if "r2" in hts.columns and hts["r2"].notna().any():
    # similarly,
    # pick the highest-r^2 row per (SMILES, cell line) if available
    idx_hts = hts.groupby(CELL_DRUG_COMBO_KEYS)["r2"].idxmax()
    print(f"Deduplicating HTS002 via highest r^2: picked {len(idx_hts)} "
          f"rows from {len(hts)} total")
    hts_dedup = hts.loc[idx_hts]
else:
    # same fallback: pick one random row per (SMILES, cell line)
    hts_dedup = hts.groupby(
        CELL_DRUG_COMBO_KEYS,
        group_keys=False).sample(n=1, random_state=DEDUP_SEED)
    print(f"Deduplicating HTS002: picked {len(hts_dedup)} "
          f"rows from {len(hts)} total")

# --- Step 3: Combine with MTS010 preference ---
combined = pd.concat([mts_dedup, hts_dedup], ignore_index=True, copy=False)
combined = combined.drop_duplicates(
    subset=["smiles","depmap_id","ccle_name"], keep="first").copy()

# --- Step 4: attach tissue etc. without row blow-up if (many:1)---
cli = (cell_line_info_df[["depmap_id","ccle_name","primary_tissue"]]
         .drop_duplicates(subset=["depmap_id","ccle_name"]))
combined = combined.merge(
    cli, on=["depmap_id","ccle_name"], how="left", validate="m:1")

print(combined.head())


# ### Confirm no duplicate cell-drug combinations

# In[7]:


duplicates = combined.duplicated(subset=['ccle_name', 'name'], keep=False)
duplicate_counts = combined[duplicates].groupby(['ccle_name', 'name']).size().\
    reset_index(name='count')
duplicate_counts = duplicate_counts[duplicate_counts['count'] > 1]

if not duplicate_counts.empty:
    raise ValueError(
        f"Found {len(duplicate_counts)} duplicate (cell line, drug) "
        f"pairs:\n{duplicate_counts}"
    )


# ## Tabulate/visualize data

# ### primary tissue - cell line count

# In[8]:


grouped_counts = combined.groupby(['primary_tissue', 'ccle_name']).size().\
    reset_index(name='count')
print(grouped_counts.head(20))


# ### number of cell-drug combiantions in dataset, grouped by primary tissue

# In[9]:


if IN_NOTEBOOK:
    unique_counts = grouped_counts.groupby('primary_tissue')[
        'ccle_name'].nunique().reset_index(name='unique_ccle_count')

    # grouped_counts: columns = ['primary_tissue', 'ccle_name', 'count']
    # unique_counts:  columns = ['primary_tissue', 'unique_ccle_count']

    # 1) Pick a consistent tissue order
    order = (grouped_counts.groupby('primary_tissue')['count']
            .median().sort_values(ascending=False).index)

    # limit to top-N tissues to keep the x-axis readable
    TOP_N = 20
    if TOP_N is not None:
        keep = list(order[:TOP_N])
        grouped_counts = grouped_counts[
            grouped_counts['primary_tissue'].isin(keep)]
        unique_counts  = unique_counts[
            unique_counts['primary_tissue'].isin(keep)]
        order = [t for t in order if t in keep]

    # Ensure the bottom bar data follows the same order
    unique_counts = unique_counts.set_index('primary_tissue').reindex(order).\
        reset_index()

    # 2) Make vertically stacked subplots with a shared x-axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Top: distribution per tissue (box + dots) ---
    sns.boxplot(
        data=grouped_counts, 
        x='primary_tissue', 
        y='count', 
        order=order, 
        ax=ax_top)
    sns.stripplot(data=grouped_counts, x='primary_tissue', y='count',
                order=order, ax=ax_top, jitter=True, alpha=0.5)
    ax_top.set_xlabel('')
    ax_top.set_ylabel('# (molecule, cell line) combos')
    ax_top.set_title('Distribution of combos per cell line within each tissue')

    # --- Bottom: number of unique cell lines per tissue (bar) ---
    sns.barplot(data=unique_counts, x='primary_tissue', y='unique_ccle_count',
                order=order, ax=ax_bot)
    ax_bot.set_xlabel('Primary tissue')
    ax_bot.set_ylabel('# unique CCLE names')

    # Rotate x labels only on the bottom axis
    for label in ax_bot.get_xticklabels():
        label.set_rotation(90)

    plt.tight_layout()
    plt.show()
else:
    print("Skipping plotting since not in a notebook environment.")


# ## Export preprocessed data

# In[10]:


output_path = pathlib.Path(git_root) \
    / "data" / "processed" / "processed_depmap_prism_ic50.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_path, index=False)
