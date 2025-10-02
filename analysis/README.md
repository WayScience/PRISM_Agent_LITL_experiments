# Analysis & Experiments

This directory is for **experiments, configs, and results** for evaluating the
agentic system defined in [`agentic_system/`](../agentic_system/).

---

## Experiment/Analysis Overview

### `Configuration`

All notebooks require a `config.yml` file at the project root. 
This file specifies data paths and API credentials needed for the analysis.

Please refer to `config.yml.template` for guidance on how the file should be formatted.

### `0.1.WRANGLE_DEPMAP_PRISM_DATA.ipynb`

Preprocesses the raw DepMap PRISM secondary drug repurposing dataset to produce a clean, deduplicated table of drug-cell line-IC50 values. The script handles deduplication of overlapping entries between the HTS002 and MTS010 screens, prioritizing MTS010 results and highest-quality curve fits (r²).

**Output:** `data/processed/processed_depmap_prism_ic50.csv` - cleaned dataset with unique (cell line, drug) combinations ready for downstream analysis.

---

## Usage


> ⚙️ **Project Setup Note**
>
> To run the analysis notebooks or nbconverted scripts correctly, some project setup is required:
>
> - **Notebook mode (VS Code / Jupyter)**
>   - Ensure `.vscode/settings.json` contains a `python.envFile` pointing to a `.env` that sets the `PYTHONPATH` for both partitions:
>     ```json
>     {
>       "python.envFile": "${workspaceFolder}/.env",
>       "python.analysis.extraPaths": [
>         "agentic_system/src",
>         "analysis/src"
>       ],
>       "jupyter.notebookFileRoot": "${workspaceFolder}"
>     }
>     ```
>   - Example `.env` (at the repo root):
>     ```
>     PYTHONPATH=agentic_system/src:analysis/src:${PYTHONPATH}
>     ```
>
> - **Script mode (running nbconvert-generated `.py`)**
>   - Perform an **editable install** of the notebook utilities once:
>     ```bash
>     pip install -e ./analysis
>     # or: uv pip install -e ./analysis
>     ```
>   - This makes the `nbutils` package importable from the scripts:
>     ```python
>     from nbutils.pathing import project_file
>     config_path = project_file("config.yml")
>     ```
