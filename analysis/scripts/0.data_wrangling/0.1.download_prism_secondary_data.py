#!/usr/bin/env python
# coding: utf-8

# # PRISM secondary data retrieval for multi-context screen analysis
# 
# This notebook retrieves PRISM secondary data for multi-context, screen-level analysis.
# 
# Our endpoint of interest is individual experiment-level **LFC** (log fold-change), rather than fitted dose-response summaries such as **IC50/EC50**.
# 
# It downloads all CSV resources specified by URL into the project `data/` directory and validates basic CSV readability.

# In[1]:


from pathlib import Path
import requests

import pandas as pd


# In[2]:


repo_root = Path.cwd().resolve()
while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
    repo_root = repo_root.parent

data_path = repo_root / "data"
data_path.mkdir(parents=True, exist_ok=True)

secondary_lfc_url = "https://ndownloader.figshare.com/files/20237757"
secondary_trt_info_url = "https://ndownloader.figshare.com/files/20237763"

model_url = "https://depmap.org/portal/data_page/?tab=allData&releasename=DepMap%20Public%2026Q1&filename=Model.csv"
download_targets = {
    "secondary-screen-replicate-collapsed-logfold-change.csv": secondary_lfc_url,
    "secondary-screen-replicate-collapsed-treatment-info.csv": secondary_trt_info_url,
    "Model.csv": model_url,
}

print(f"Using data path: {data_path}")


# In[3]:


DEPMAP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
    "Referer": "https://depmap.org/portal/download/",
}


def download_file(url: str, out_path: Path, chunk_size: int = 1 << 20) -> None:
    headers = DEPMAP_HEADERS if "depmap.org" in url else None

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=120,
        allow_redirects=True,
    ) as r:
        if r.status_code == 403 and "depmap.org" in url:
            raise RuntimeError(
                "DepMap blocks this request with bot verification (HTTP 403). "
                "Download Model.csv in browser and place it at data/Model.csv."
            )

        r.raise_for_status()

        # DepMap can return an anti-bot verification HTML page instead of CSV.
        if "depmap.org" in url:
            content_type = (r.headers.get("content-type") or "").lower()
            if "text/html" in content_type:
                preview = r.text[:5000].lower()
                if "verification" in preview and "enter depmap" in preview:
                    raise RuntimeError(
                        "DepMap verification blocked programmatic download. "
                        "Download Model.csv in browser and place it at data/Model.csv."
                    )

        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


# In[4]:


download_errors = {}

for fname, url in download_targets.items():
    out_file = data_path / fname
    try:
        print(f"Downloading {fname} ...")
        download_file(url, out_file)
        print(f"  [OK] saved to {out_file}")
    except Exception as e:
        download_errors[fname] = str(e)
        print(f"  [WARN] failed: {e}")

if download_errors:
    print("\nSome files were not downloaded programmatically:")
    for fname, err in download_errors.items():
        print(f"- {fname}: {err}")

    print(
        "\nIf Model.csv is blocked by verification, download it in browser and "
        "save it to data/Model.csv, then run the validation cell."
    )
else:
    print("\nAll download targets completed successfully.")


# In[5]:


for fname in download_targets:
    fpath = data_path / fname

    if not fpath.exists():
        print(f"[MISSING] {fname}: file not found")
        continue

    try:
        preview = pd.read_csv(fpath, nrows=5)
        print(f"[OK] {fname}: preview shape {preview.shape}")
    except Exception as e:
        print(f"[WARN] {fname}: failed CSV parse -> {e}")
