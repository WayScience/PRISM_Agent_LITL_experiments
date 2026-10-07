"""Figshare-only, version-pinned PRISM and DepMap downloads (Python >=3.10)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

API = "https://api.figshare.com/v2/articles"
CHUNK_SIZE = 1 << 20
TIMEOUT = 120

PRISM_RELEASE = (
    20564034, 1, "PRISM Repurposing 20Q2 Dataset",
    (
        "prism-repurposing-20q2-secondary-screen-replicate-collapsed-logfold-change.csv",
        "prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv",
    ),
)
# shared disease model metadata that applies to all depmap data as long as 
# the release version is up to date. 
# Because we analyze the secondary data which is updated last on 20Q2, 
# model release of 24Q4 would be sufficient. 

MODEL_RELEASE = (27993248, 1, "DepMap 24Q4 Public", ("Model.csv",))
@dataclass(frozen=True)
class DownloadRecord:
    filename: str
    source_filename: str
    article_id: int
    article_version: int
    release: str
    doi: str
    file_id: int
    url: str
    size: int
    md5: str


def _transfer(url: str, destination: Path | None = None) -> bytes | None:
    """Fetch metadata or stream a file to a staging path."""
    with urlopen(url, timeout=TIMEOUT) as response:
        if destination is None:
            return response.read()
        with destination.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
    return None


def resolve_records() -> list[DownloadRecord]:
    """Resolve exact filenames from pinned article versions; never use latest."""
    records = []
    for article_id, version, title, names in (PRISM_RELEASE, MODEL_RELEASE):
        article = json.loads(_transfer(f"{API}/{article_id}/versions/{version}"))
        for name in names:
            matches = [f for f in article["files"] if f["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name!r} in {title}")
            item = matches[0]
            md5 = (item.get("computed_md5") or "").lower()
            if not re.fullmatch(r"[0-9a-f]{32}", md5):
                raise ValueError(f"Missing published MD5 for {name}")
            if int(item["size"]) <= 0:
                raise ValueError(f"Invalid published size for {name}")
            # Preserve the existing secondary-screen filenames used by notebooks.
            filename = name.removeprefix("prism-repurposing-20q2-")
            records.append(DownloadRecord(
                filename, name, article_id, version, title, article["doi"],
                int(item["id"]), item["download_url"], int(item["size"]), md5,
            ))
    return records


def _validate(path: Path, record: DownloadRecord) -> None:
    """Confirm the bytes match the published Figshare checksum."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    if digest.hexdigest() != record.md5:
        raise ValueError(f"{path.name}: MD5 differs from the pinned Figshare file")


def download_prism_data(
    data_path: str | Path,
    *,
    overwrite: bool = False,
) -> list[dict]:
    """Download PRISM + 24Q4 Model.csv and return a list of provenance records.

    Existing files must match the selected release MD5. Invalid files raise;
    use overwrite=True explicitly to replace them. Each file is staged and
    verified before atomic replacement. Metadata requires network access on
    each call; verified files are not downloaded again.
    """
    records = resolve_records()
    root = Path(data_path)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        target = root / record.filename
        reuse = target.exists() and not overwrite
        if reuse:
            _validate(target, record)
        else:
            with tempfile.TemporaryDirectory(dir=root, prefix=".figshare-") as staging:
                staged = Path(staging) / record.filename
                _transfer(record.url, staged)
                _validate(staged, record)
                os.replace(staged, target)
        results.append({**asdict(record), "status": "reused" if reuse else "downloaded"})
        print(f"[{results[-1]['status'].upper()}] {record.filename}")
    manifest = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": results,
    }
    with tempfile.TemporaryDirectory(dir=root, prefix=".figshare-") as staging:
        staged = Path(staging) / "manifest.json"
        staged.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, root / "figshare_manifest.json")
    return results
