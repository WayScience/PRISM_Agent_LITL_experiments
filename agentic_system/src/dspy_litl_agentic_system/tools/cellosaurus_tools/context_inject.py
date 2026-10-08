# context_inject.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

from .for_agents import (
    search_cellosaurus_ac,
    get_cellosaurus_summary,
)

_AC_RE = re.compile(r"(CVCL_[A-Z0-9]{4,})")


def _extract_first_ac(search_text: str) -> Optional[str]:
    """
    Extract the first CVCL_XXXX token from search_cellosaurus_ac() output.
    """
    m = _AC_RE.search(search_text or "")
    return m.group(1) if m else None


def build_cell_context(
    *,
    cell_name: str,
    section_header: str = "Cell line context (Cellosaurus)",
) -> str:
    """
    Build a natural-language context block for a cell line name/synonym.

    Uses NL wrappers:
      - search_cellosaurus_ac(query) -> "Cellosaurus ACs found for '...': CVCL_...."
      - get_cellosaurus_summary(ac)  -> multi-line summary

    Returns
    -------
    str:
        Prompt-ready natural language context.
    """
    cell_name = (cell_name or "").strip()
    if not cell_name:
        return f"{section_header}:\nNo cell line name provided."

    lines: list[str] = []
    lines.append(f"{section_header}:")
    lines.append(f"- Query cell line: {cell_name}")

    # 1) Search ACs (NL wrapper)
    search_text = search_cellosaurus_ac(cell_name)
    lines.append("")
    lines.append("Cellosaurus accession search:")
    lines.append(search_text)

    # 2) Choose a primary AC
    ac = _extract_first_ac(search_text)
    if not ac:
        lines.append("")
        lines.append("No Cellosaurus accession could be resolved from search results; skipping summary retrieval.")
        return "\n".join(lines)

    # 3) Summary (NL wrapper)
    lines.append("")
    lines.append(f"Selected primary accession: {ac} building context by running tools:")
    lines.append("")
    lines.append("Call tool `get_cellosaurus_summary`:")
    lines.append(get_cellosaurus_summary(ac))

    return "\n".join(lines)


@dataclass(frozen=True)
class CellContextInjector:
    """
    Convenience callable so your dispatch code stays compact.
    """
    section_header: str = "Cell line context (Cellosaurus)"

    def __call__(self, *, cell_name: str) -> str:
        return build_cell_context(cell_name=cell_name, section_header=self.section_header)
