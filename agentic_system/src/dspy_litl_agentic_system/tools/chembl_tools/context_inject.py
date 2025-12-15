# context_inject.py
from __future__ import annotations
import re


from dataclasses import dataclass
from typing import Optional

from .for_agents import (
    search_chembl_id,
    get_compound_properties,
    get_compound_activities,
    get_drug_approval_status,
    get_drug_moa,
    get_drug_indications,
)

_CHEMBL_ID_RE = re.compile(r"(CHEMBL\d+)")


def _extract_first_chembl_id(search_text: str) -> Optional[str]:
    """
    Extract the first CHEMBL<digits> token from search_chembl_id() output.
    Returns None if not found.
    """
    m = _CHEMBL_ID_RE.search(search_text or "")
    return m.group(1) if m else None


def build_drug_context(
    *,
    drug_name: str,
    id_limit: int = 5,
    include_properties: bool = True,
    include_activities: bool = True,
    include_moa: bool = True,
    include_indications: bool = True,
    include_approval: bool = False,
    activity_type: Optional[str] = None,  # e.g. "IC50"
    section_header: str = "Drug context (ChEMBL)",
) -> str:
    """
    Build a natural-language context block for a drug canonical name.

    :param drug_name: Canonical name of the drug.
    :param id_limit: Number of ChEMBL IDs to search for.
    :param include_approval: Include approval status section.
    :param include_moa: Include mechanism of action section.
    :param include_indications: Include indications section.
    :param include_properties: Include compound properties section.
    :param include_activities: Include bioactivity summary section.
    """
    drug_name = (drug_name or "").strip()
    if not drug_name:
        return f"{section_header}:\nNo drug name provided."

    lines: list[str] = []
    lines.append(f"{section_header}:")
    lines.append(f"- Query drug: {drug_name}")

    # 1) Search IDs (NL wrapper)
    id_search_text = search_chembl_id(drug_name, limit=id_limit)
    lines.append("")
    lines.append("Call tool `search_chembl_id`:")
    lines.append(id_search_text)

    # 2) Pick a primary ID
    chembl_id = _extract_first_chembl_id(id_search_text)
    if not chembl_id:
        lines.append("")
        lines.append("No ChEMBL ID could be resolved from search results; skipping downstream lookups.")
        return "\n".join(lines)

    # 3) Downstream detail sections (all NL wrappers)
    lines.append("")
    lines.append(f"Selected primary ChEMBL ID: {chembl_id} building context by running tools:")

    if include_approval:
        lines.append("")
        lines.append("Call tool `get_drug_approval_status`:")
        lines.append(get_drug_approval_status(chembl_id))

    if include_moa:
        lines.append("")
        lines.append("Call tool `get_drug_moa`:")
        lines.append(get_drug_moa(chembl_id, limit=5))

    if include_indications:
        lines.append("")
        lines.append("Call tool `get_drug_indications`:")
        lines.append(get_drug_indications(chembl_id, limit=5))

    if include_properties:
        lines.append("")
        lines.append("Call tool `get_compound_properties`:")
        lines.append(get_compound_properties(chembl_id, limit=5))

    if include_activities:
        lines.append("")
        lines.append("Call tool `get_compound_activities`:")
        lines.append(get_compound_activities(chembl_id, activity_type=activity_type, limit=5))

    return "\n".join(lines)


@dataclass(frozen=True)
class ContextInjector:
    """
    Convenience wrapper to keep your agent code clean.
    """
    id_limit: int = 5
    activity_type: Optional[str] = None

    def __call__(self, *, drug_name: str, cell_line: Optional[str] = None) -> str:
        return build_drug_context(
            drug_name=drug_name,
            cell_line=cell_line,
            id_limit=self.id_limit,
            activity_type=self.activity_type,
        )
