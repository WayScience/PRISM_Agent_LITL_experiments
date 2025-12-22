# context_inject.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

from .for_agents import (
    search_pubchem_cid,
    get_properties,
    get_assay_summary,
    get_safety_summary,
    get_drug_summary,
    find_similar_compounds,
)

_CID_RE = re.compile(r"\bCID\s+(\d+)\b", re.IGNORECASE)


def _extract_first_cid(search_text: str) -> Optional[str]:
    """
    Extract the first CID number from search_pubchem_cid() output.

    Expected patterns (from your wrapper):
      - "Found compound ... with CID 12345 for <query>."
      - "Found N compound(s) matching ...: CIDs \n - 123\n - 456\n ..."
    """
    if not search_text:
        return None

    # First preference: explicit "CID <num>" pattern
    m = _CID_RE.search(search_text)
    if m:
        return m.group(1)

    # Fallback: list style "- 12345" lines
    for line in (search_text or "").splitlines():
        line = line.strip()
        if line.startswith("-"):
            candidate = line.lstrip("-").strip()
            if candidate.isdigit():
                return candidate

    # Another fallback: any standalone integer token
    for tok in re.findall(r"\b\d+\b", search_text):
        return tok

    return None


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
    """
    query = (query or "").strip()
    if not query:
        return f"{section_header}:\nNo compound query provided."

    lines: list[str] = []
    lines.append(f"{section_header}:")
    lines.append(f"- Query compound: {query}")

    # 1) Search CID(s)
    lines.append("")
    lines.append("Call tool `search_pubchem_cid`:")
    search_text = search_pubchem_cid(query, limit=cid_limit)
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
        lines.append(get_properties(cid))

    if include_assays:
        lines.append("")
        lines.append("Call tool `get_assay_summary`:")
        lines.append(get_assay_summary(cid, limit=assay_limit))

    if include_safety:
        lines.append("")
        lines.append("Call tool `get_safety_summary`:")
        lines.append(get_safety_summary(cid))

    if include_drug_med:
        lines.append("")
        lines.append("Call tool `get_drug_summary`:")
        lines.append(get_drug_summary(cid))

    if include_similar:
        lines.append("")
        lines.append("Call tool `find_similar_compounds`:")
        lines.append(
            find_similar_compounds(
                cid,
                threshold=similar_threshold,
                limit=similar_limit,
            )
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class PubChemContextInjector:
    """
    Convenience callable wrapper.
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
        )
