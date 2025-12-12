"""
This module provides functions to parse and 
    extract information from a cell line dictionary.
"""


from typing import Any, Callable, Dict, Iterable, List


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else ([] if x is None else [x])


def _first_present(d: Dict[str, Any], key: str, prefer_list: bool = True):
    """
    Return (actual_key_used, value) for either `key` or `key-list`.
    """
    k_list = f"{key}-list"
    order = (k_list, key) if prefer_list else (key, k_list)
    for k in order:
        if k in d and d[k] is not None:
            return k, d[k]
    return None, None


# -----------------
# Parsers
# -----------------

def parse_cell_line_list(cell_line_list: list) -> list:
    ac_list = []
    for c in cell_line_list:
        accession_list = c.get("accession-list", [])
        for a in accession_list:
            if a.get("type", "") == "primary" and "value" in a:
                ac_list.append(a.get("value", ""))

    return ac_list

def parse_site_list(site_list: list) -> list:
    return [
        f"{s['site'].get('value')} ({s['site'].get('site-type', 'N/A')})"
        for s in _as_list(site_list)
        if isinstance(s, dict)
        and s.get("site", {}).get("value")
    ]


def parse_disease_list(disease_list: list) -> list:
    return [
        d["label"]
        for d in _as_list(disease_list)
        if isinstance(d, dict) and d.get("label")
    ]


def parse_species_list(species_list: list) -> list:
    return [
        s["label"]
        for s in _as_list(species_list)
        if isinstance(s, dict) and s.get("label")
    ]


def parse_name_list(name_list: list) -> list:
    return [
        n["value"]
        for n in _as_list(name_list)
        if isinstance(n, dict) and n.get("type") == "identifier"
    ]


def parse_sequence_variation_list(sequence_variation_list: list) -> list:
    return [
        f"{s.get('mutation-description', '')} ({s.get('mutation-type')})"
        for s in _as_list(sequence_variation_list)
        if isinstance(s, dict)
        and s.get("variation-type") == "Mutation"
    ]


# -----------------
# Registry (base keys)
# -----------------

ParseFn = Callable[[Any], List[str]]

parse_registry: Dict[str, ParseFn] = {
    "derived-from-site": parse_site_list,
    "disease": parse_disease_list,
    "species": parse_species_list,
    "name": parse_name_list,
    "sequence-variation": parse_sequence_variation_list,
}


# -----------------
# Extraction
# -----------------

def extract_bio_summary(
    cell_line: Dict[str, Any],
    simple_keys: Iterable[str] = ("age", "category", "sex"),
    list_keys: Iterable[str] = (
        "derived-from-site",
        "disease",
        "species",
        "name",
        "sequence-variation",
    ),
    prefer_list: bool = True,
    normalize_keys: bool = True,
) -> Dict[str, Any]:
    """
    Minimal extractor operating on a constrained key set.

    - Accepts either `key` or `key-list`
    - If both exist, preference controlled by `prefer_list`
    - If normalize_keys=True, output uses base keys
      (e.g. 'disease' instead of 'disease-list')
    """
    out: Dict[str, Any] = {}

    # Scalars
    for k in simple_keys:
        if k in cell_line:
            out[k] = cell_line[k]

    # List-backed fields
    for base_key in list_keys:
        used_key, raw_val = _first_present(cell_line, base_key, prefer_list)
        if used_key is None:
            continue

        parser = parse_registry.get(base_key)
        if parser is None:
            continue

        parsed = parser(raw_val)
        if parsed:
            out_key = base_key if normalize_keys else used_key
            out[out_key] = parsed

    return out
