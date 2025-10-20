# test_wrangle_depmap_prism.py

import subprocess
import pathlib
import os


def test_wrangle_depmap_prism_script_runs(tmp_path):
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
    
    # Temp directory for binary outputs so that pre-commit won't complain
    # about modified binary files.
    out_dir = tmp_path / "test_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run from repo root so `git rev-parse --show-toplevel` and config.yml resolve
    result = subprocess.run(
        [
            "python", 
            str(script_path),
            "--out-dir", str(out_dir),
            "--overwrite"
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={**os.environ},  # inherit env
    )

    assert result.returncode == 0, (
        f"Script failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
