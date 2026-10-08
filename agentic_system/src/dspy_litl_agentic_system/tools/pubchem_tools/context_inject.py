# context_inject.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from . import for_agents


class PubChemTools(Protocol):
    """Agent-facing tools required by the context builder (modules also qualify)."""

    def search_pubchem_cid(self, query: str, limit: int = 5) -> str: ...

    def get_properties(self, cid: int | str) -> str: ...

    def get_assay_summary(self, cid: int | str, limit: int = 5) -> str: ...

    def get_safety_summary(self, cid: int | str) -> str: ...

    def get_drug_summary(self, cid: int | str) -> str: ...

    def find_similar_compounds(
        self, cid: int | str, threshold: int = 90, limit: int = 10
    ) -> str: ...


_SINGLE_RESULT_RE = re.compile(
    r"Found compound(?: '[^'\r\n]*'| (?:(?!\bwith\s+CID\b)[^\r\n])+?)?"
    r" with CID ([0-9]+)(?: for [^\r\n]+)?\.?",
    re.IGNORECASE,
)
_MULTI_RESULT_RE = re.compile(
    r"Found [1-9][0-9]* compound\(s\)(?: matching '[^\r\n]*')?: CIDs",
    re.IGNORECASE,
)
_CID_BULLET_RE = re.compile(r"-\s+([1-9][0-9]*)")


def _extract_first_cid(search_text: str) -> Optional[str]:
    """
    Extract a positive CID only from recognized successful search output.

    Expected patterns (from the agent-facing wrapper):
      - "Found compound ... with CID 12345 for <query>."
      - "Found N compound(s) matching ...: CIDs \n - 123\n - 456\n ..."

    Error messages, echoed query numbers, and standalone numeric bullets are
    not search results and must never trigger downstream lookups.
    """
    if not search_text:
        return None

    text = search_text.strip()
    match = _SINGLE_RESULT_RE.fullmatch(text)
    if match:
        cid = match.group(1)
        return cid if int(cid) > 0 else None

    lines = text.splitlines()
    if not lines or not _MULTI_RESULT_RE.fullmatch(lines[0].strip()):
        return None

    # Validate the result list, rather than mining arbitrary text for numbers.
    first_cid = None
    for line in lines[1:]:
        if not line.strip():
            continue
        match = _CID_BULLET_RE.fullmatch(line.strip())
        if not match:
            return None
        if first_cid is None:
            first_cid = match.group(1)
    return first_cid


def build_pubchem_context(
    *,
    query: str,
    cid_limit: int = 5,
    include_properties: bool = True,
    include_assays: bool = True,
    include_safety: bool = True,
    include_drug_med: bool = True,
    include_similar: bool = False,
    similar_threshold: int = 90,
    similar_limit: int = 5,
    assay_limit: int = 5,
    section_header: str = "Compound context (PubChem)",
    tools: PubChemTools | None = None,
) -> str:
    """
    Build a natural-language context block for a compound query using PubChem tools.

    Tool calls are spelled out verbatim so agents can avoid re-calling them.

    Parameters
    ----------
    query:
        Canonical compound name or synonym string.
    cid_limit:
        Maximum CIDs to return in search step (passed to search_pubchem_cid()).
    include_*:
        Toggle inclusion of downstream tool calls.
    include_similar:
        If True, call find_similar_compounds() for neighborhood context.
    similar_threshold / similar_limit:
        Parameters to find_similar_compounds().
    assay_limit:
        Parameter to get_assay_summary().
    tools:
        Optional tool provider; defaults to the agent-facing PubChem module.
        Returned error strings are preserved in context. Raised exceptions
        propagate unchanged. Provider methods are accessed only when needed.
    """
    query = (query or "").strip()
    if not query:
        return f"{section_header}:\nNo compound query provided."

    provider: PubChemTools = for_agents if tools is None else tools
    lines: list[str] = []
    lines.append(f"{section_header}:")
    lines.append(f"- Query compound: {query}")

    # 1) Search CID(s)
    lines.append("")
    lines.append("Call tool `search_pubchem_cid`:")
    search_text = provider.search_pubchem_cid(query, limit=cid_limit)
    lines.append(search_text)

    # 2) Pick a primary CID
    cid = _extract_first_cid(search_text)
    if not cid:
        lines.append("")
        lines.append("No PubChem CID could be resolved from search results; skipping downstream lookups.")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Selected primary PubChem CID: {cid} building context by running tools:")

    # 3) Downstream sections
    if include_properties:
        lines.append("")
        lines.append("Call tool `get_properties`:")
        lines.append(provider.get_properties(cid))

    if include_assays:
        lines.append("")
        lines.append("Call tool `get_assay_summary`:")
        lines.append(provider.get_assay_summary(cid, limit=assay_limit))

    if include_safety:
        lines.append("")
        lines.append("Call tool `get_safety_summary`:")
        lines.append(provider.get_safety_summary(cid))

    if include_drug_med:
        lines.append("")
        lines.append("Call tool `get_drug_summary`:")
        lines.append(provider.get_drug_summary(cid))

    if include_similar:
        lines.append("")
        lines.append("Call tool `find_similar_compounds`:")
        lines.append(
            provider.find_similar_compounds(
                cid,
                threshold=similar_threshold,
                limit=similar_limit,
            )
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class PubChemContextInjector:
    """
    Convenience callable wrapper, optionally sharing an injected tool provider.
    """
    cid_limit: int = 5
    assay_limit: int = 5
    include_similar: bool = False
    similar_threshold: int = 90
    similar_limit: int = 5
    include_properties: bool = True
    include_assays: bool = True
    include_safety: bool = True
    include_drug_med: bool = True
    section_header: str = "Compound context (PubChem)"
    tools: PubChemTools | None = None

    def __call__(self, *, query: str) -> str:
        return build_pubchem_context(
            query=query,
            cid_limit=self.cid_limit,
            assay_limit=self.assay_limit,
            include_similar=self.include_similar,
            similar_threshold=self.similar_threshold,
            similar_limit=self.similar_limit,
            include_properties=self.include_properties,
            include_assays=self.include_assays,
            include_safety=self.include_safety,
            include_drug_med=self.include_drug_med,
            section_header=self.section_header,
            tools=self.tools,
        )
