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
        ("Found compound 'name with CID 123' with CID 2244 for aspirin.", "2244"),
        ("Found 2 compound(s) matching 'drug CID 999': CIDs \n - 2244\n - 123", "2244"),
        ("Error searching for 'aspirin': HTTP 503", None),
        ("Error searching for compound 'CID 2244': HTTP 503\n - 123", None),
        ("No compounds found for 'drug 123'.", None),
        ("Found 0 compound(s).", None),
        ("- 2244\n - 123", None),
        ("Found compound with CID invalid for drug with CID 2244.", None),
        ("Found compound with CID 0 for drug with CID 2244.", None),
        ("Found 2 compound(s): CIDs\n - 2244\nHTTP 503", None),
        ("", None),
    ],
)
def test_pubchem_cid_extraction(text, expected):
    assert pubchem._extract_first_cid(text) == expected


@pytest.fixture(params=["builder", "injector"])
def context_builder(request):
    if request.param == "builder":
        return pubchem.build_pubchem_context

    def build(*, query, **kwargs):
        return pubchem.PubChemContextInjector(**kwargs)(query=query)

    return build


@pytest.mark.parametrize(
    "search_text",
    [
        "HTTP 503 for drug 123",
        "Error searching for compound 'CID 2244': HTTP 503\n - 123",
        "No compounds found for query 'drug 123'.",
        "",
    ],
)
def test_pubchem_error_skips_downstream_lookups(context_builder, search_text):
    calls = []

    class FailedSearchTools:
        def search_pubchem_cid(self, query, limit=5):
            calls.append((query, limit))
            return search_text

        def __getattr__(self, name):
            pytest.fail(f"Unresolved CIDs must not trigger lookup: {name}")

    context = context_builder(
        query="  drug 123  ", tools=FailedSearchTools(), cid_limit=3, include_similar=True
    )

    assert calls == [("drug 123", 3)]
    assert f"Call tool `search_pubchem_cid`:\n{search_text}\n" in context
    assert "No PubChem CID could be resolved" in context
    assert "Selected primary PubChem CID" not in context


def test_pubchem_injected_tools_receive_arguments(context_builder):
    tools = create_autospec(pubchem.for_agents, spec_set=True)
    tools.search_pubchem_cid.return_value = "Found compound with CID 2244."
    downstream = (
        "get_properties", "get_assay_summary", "get_safety_summary",
        "get_drug_summary", "find_similar_compounds",
    )
    for name in downstream:
        getattr(tools, name).return_value = f"Result from {name}"

    context = context_builder(
        query="aspirin", tools=tools, cid_limit=3, assay_limit=2,
        include_similar=True, similar_threshold=85, similar_limit=4,
    )

    tools.search_pubchem_cid.assert_called_once_with("aspirin", limit=3)
    tools.get_properties.assert_called_once_with("2244")
    tools.get_assay_summary.assert_called_once_with("2244", limit=2)
    tools.get_safety_summary.assert_called_once_with("2244")
    tools.get_drug_summary.assert_called_once_with("2244")
    tools.find_similar_compounds.assert_called_once_with("2244", threshold=85, limit=4)
    for name in downstream:
        assert f"Result from {name}" in context
