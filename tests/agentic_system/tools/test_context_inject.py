"""Offline regressions for context injection wrappers."""

from unittest.mock import create_autospec

import pytest

from dspy_litl_agentic_system.tools.chembl_tools import context_inject as chembl
from dspy_litl_agentic_system.tools.pubchem_tools import context_inject as pubchem


@pytest.mark.parametrize("cell_line", [None, "TOV21G"])
def test_chembl_wrapper_keeps_compatible_signature(monkeypatch, cell_line):
    builder = create_autospec(chembl.build_drug_context, return_value="context")
    monkeypatch.setattr(chembl, "build_drug_context", builder)
    injector = chembl.ContextInjector(id_limit=3, activity_type="IC50")

    assert injector(drug_name="aspirin", cell_line=cell_line) == "context"
    builder.assert_called_once_with(
        drug_name="aspirin", id_limit=3, activity_type="IC50"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Found compound aspirin with CID 2244 for aspirin.", "2244"),
        ("Found compound with cid 2244.", "2244"),
        ("Found 2 compound(s): CIDs \n - 2244\n - 123", "2244"),
        ("Error searching for 'aspirin': HTTP 503", None),
        ("No compounds found for 'drug 123'.", None),
        ("Found 0 compound(s).", None),
        ("", None),
    ],
)
def test_pubchem_cid_extraction(text, expected):
    assert pubchem._extract_first_cid(text) == expected


def test_pubchem_error_skips_downstream_lookups(monkeypatch):
    monkeypatch.setattr(
        pubchem, "search_pubchem_cid", lambda *a, **kw: "HTTP 503 for drug 123"
    )

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("Unresolved CIDs must not trigger downstream lookups")

    for name in (
        "get_properties", "get_assay_summary", "get_safety_summary",
        "get_drug_summary", "find_similar_compounds",
    ):
        monkeypatch.setattr(pubchem, name, unexpected_lookup)

    context = pubchem.build_pubchem_context(query="drug 123", include_similar=True)
    assert "No PubChem CID could be resolved" in context