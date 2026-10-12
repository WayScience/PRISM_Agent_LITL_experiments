"""Focused, offline checks for pinned Figshare download helpers."""

import hashlib
import json

import pytest

from nbutils import prism_download as downloads


def make_record(filename="Model.csv", payload=b"verified bytes"):
    return downloads.DownloadRecord(
        filename=filename,
        source_filename=filename,
        article_id=27993248,
        article_version=1,
        release="DepMap 24Q4 Public",
        doi="10.0000/example",
        file_id=123,
        url="https://example.org/model",
        size=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )


def article_file(name, file_id, payload, url):
    return {
        "name": name,
        "id": file_id,
        "download_url": url,
        "computed_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        "size": len(payload),
    }


def test_resolve_records_uses_pinned_articles_and_preserves_names(monkeypatch):
    prism_names = downloads.PRISM_RELEASE[3]
    prism_article = {
        "doi": "10.0000/prism",
        "files": [
            article_file(prism_names[0], 11, b"primary", "https://example.org/primary"),
            article_file(prism_names[1], 12, b"treatment", "https://example.org/treatment"),
        ],
    }
    model_article = {
        "doi": "10.0000/model",
        "files": [article_file("Model.csv", 13, b"model", "https://example.org/model")],
    }
    articles = {downloads.PRISM_RELEASE[0]: prism_article, downloads.MODEL_RELEASE[0]: model_article}
    requested = []

    def transfer(url, destination=None):
        requested.append(url)
        article_id = int(url.split("/")[-3])
        return json.dumps(articles[article_id]).encode()

    monkeypatch.setattr(downloads, "_transfer", transfer)
    records = downloads.resolve_records()

    assert requested == [
        f"{downloads.API}/{article_id}/versions/{version}"
        for article_id, version, *_ in (downloads.PRISM_RELEASE, downloads.MODEL_RELEASE)
    ]
    assert [record.filename for record in records] == [
        name.removeprefix("prism-repurposing-20q2-") for name in prism_names
    ] + ["Model.csv"]
    assert [record.file_id for record in records] == [11, 12, 13]
    assert [record.release for record in records] == [
        downloads.PRISM_RELEASE[2], downloads.PRISM_RELEASE[2], downloads.MODEL_RELEASE[2],
    ]


def test_resolve_records_rejects_duplicate_pinned_filename(monkeypatch):
    prism_names = downloads.PRISM_RELEASE[3]
    articles = {
        downloads.PRISM_RELEASE[0]: {
            "doi": "10.0000/prism",
            "files": [article_file(name, i, b"x", f"https://example.org/{i}")
                      for i, name in enumerate(prism_names)],
        },
        downloads.MODEL_RELEASE[0]: {
            "doi": "10.0000/model",
            "files": [
                article_file("Model.csv", 13, b"model", "https://example.org/model"),
                article_file("Model.csv", 14, b"model", "https://example.org/duplicate"),
            ],
        },
    }
    monkeypatch.setattr(
        downloads, "_transfer",
        lambda url, destination=None: json.dumps(articles[int(url.split("/")[-3])]).encode(),
    )

    with pytest.raises(ValueError, match="exactly one 'Model.csv'"):
        downloads.resolve_records()


@pytest.mark.parametrize("overwrite", [False, True])
def test_download_reuses_or_overwrites_and_writes_manifest(tmp_path, monkeypatch, overwrite):
    payload = b"verified bytes"
    record = make_record(payload=payload)
    path = tmp_path / record.filename
    path.write_bytes(payload)
    transfers = []
    monkeypatch.setattr(downloads, "resolve_records", lambda: [record])

    def transfer(url, destination=None):
        transfers.append(url)
        destination.write_bytes(payload)

    monkeypatch.setattr(downloads, "_transfer", transfer)
    result = downloads.download_prism_data(tmp_path, overwrite=overwrite)

    assert len(transfers) == int(overwrite)
    assert result[0]["status"] == ("downloaded" if overwrite else "reused")
    assert path.read_bytes() == payload
    assert sorted(item.name for item in tmp_path.iterdir()) == ["Model.csv", "figshare_manifest.json"]
    manifest = json.loads((tmp_path / "figshare_manifest.json").read_text())
    assert manifest["files"] == result


def test_failed_checksum_preserves_existing_file_and_cleans_staging(tmp_path, monkeypatch):
    path = tmp_path / "Model.csv"
    path.write_bytes(b"existing verified file")
    record = make_record(payload=b"expected download")
    monkeypatch.setattr(downloads, "resolve_records", lambda: [record])
    monkeypatch.setattr(
        downloads, "_transfer",
        lambda url, destination=None: destination.write_bytes(b"bad bytes"),
    )

    with pytest.raises(ValueError, match="MD5 differs"):
        downloads.download_prism_data(tmp_path, overwrite=True)

    assert path.read_bytes() == b"existing verified file"
    assert list(tmp_path.iterdir()) == [path]
