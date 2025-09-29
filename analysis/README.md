# Analysis & Experiments

This directory is for **experiments, configs, and results** for evaluating the
agentic system defined in [`agentic_system/`](../agentic_system/).

---

## Experiment/Analysis Overview

**Details TBA**

---

## Usage


> ⚙️ **Project Setup Note**
>
> To run this notebook or its nbconverted script version correctly, some project setup is required:
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
> - **Script mode (running nbconvert-generated `.py` directly)**
>   - Perform an **editable install** of the notebook utilities once:
>     ```bash
>     pip install -e ./analysis
>     # or: uv pip install -e ./analysis
>     ```
>   - This makes the `nbutils` package importable everywhere:
>     ```python
>     from nbutils.pathing import project_file
>     config_path = project_file("config.yml")
>     ```
>
> With these two pieces in place, both the `.ipynb` (interactive) and the `.py` (script) versions of your notebooks will run consistently.
