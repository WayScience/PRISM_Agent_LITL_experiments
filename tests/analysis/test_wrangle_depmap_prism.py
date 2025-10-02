# test_wrangle_depmap_prism.py

import subprocess
import pathlib
import os

def test_wrangle_depmap_prism_script_runs():
    """
    Simple test to ensure preprocessing script runs without error.
    """
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    repo_root = pathlib.Path(repo_root)
    script_path = repo_root / "analysis" /\
        "scripts" /\
            "0.data_wrangling" /\
                "0.1.wrangle_depmap_prism_data.py"

    # Run from repo root so `git rev-parse --show-toplevel` and config.yml resolve
    result = subprocess.run(
        ["python", str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={**os.environ},  # inherit env
    )

    assert result.returncode == 0, (
        f"Script failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
