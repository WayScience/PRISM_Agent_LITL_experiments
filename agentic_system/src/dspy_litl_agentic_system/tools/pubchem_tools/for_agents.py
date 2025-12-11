"""
for_agents.py

Wrapper functions for PubChem tools, intended for use by agents.
"""

from typing import List, Union

from dspy_litl_agentic_system.tools.pubchem_tools.pcp_backend import (
    _search_pubchem_cid_cached,
    _get_cid_properties_cached,
    _get_assay_summary_cached,
    _get_ghs_classification_cached,
    _get_drug_med_info_cached,
    _get_similar_cids_cached,
    _compute_tanimoto_cached
)


def _str_with_markup_list(val) -> List[str]:
    out = []
    if isinstance(val, dict):
        for itm in val.get("StringWithMarkup", []) or []:
            s = itm.get("String")
            if s:
                out.append(s)
    return out


def search_pubchem_cid(query: str, limit: int = 5) -> str:
    """
    Search for PubChem CIDs matching a query string that is a name or synonym.

    Args:
        query (str): The compound name or synonym to search for.
        limit (int): Maximum number of CIDs to return (default 5).
    Returns:
        str: Summary of search results or error message.
    """
    limit = max(1, int(limit))

    result = _search_pubchem_cid_cached(query)
    if result["error"]:
        return f"Error searching for compound '{query}': {result['error']}"
    cids = result["cids"][:limit]

    if not cids:
        return f"No compounds found for query '{query}'."
    elif len(cids) == 1: # only enrich results for single hit
        properties = _get_cid_properties_cached(cids[0]).get("properties", {})

        if properties:
            name = properties.get("IUPACName") or properties.get("MolecularFormula")
            return f"Found compound '{name}' with CID {cids[0]} for {query}."
        else:
            return f"Found compound with CID {cids[0]} for {query}."
    else:
        return (
        f"Found {len(cids)} compound(s) matching '{query}': CIDs \n - "
        + "\n - ".join(map(str, cids))
    )


def get_properties(cid: Union[int, str]) -> str:
    """
    Using PubChem CID, fetch and summarize key molecular properties.
    Specifically: molecular formula, weight, lipophilicity (XLogP),
    TPSA, H-bond donors/acceptors, rotatable bonds, complexity, charge

    Args:
        cid (int | str): PubChem Compound ID.
    Returns:
        str: Summary of molecular properties or error message.
    """
    result = _get_cid_properties_cached(cid)
    if result["error"]:
        return f"Error fetching properties for CID {cid}: {result['error']}"

    props = result["properties"]
    if not props:
        return f"No properties found for CID {cid}."
    
    summary: List[str] = [f"Properties for CID {cid}:"]

    # IUPAC Name
    name = props.get("IUPACName")
    summary.append(f"IUPAC Name: {name}" if name else "IUPAC Name: N/A")

    # Molecular formula + weight
    formula = props.get("MolecularFormula")
    mw = props.get("MolecularWeight")
    if formula and mw:
        try:
            summary.append(f"molecular formula {formula} with MW {float(mw):.2f} g/mol")
        except Exception:
            summary.append(f"molecular formula {formula} with MW {mw} g/mol")

    # Lipophilicity
    xlogp = props.get("XLogP")
    if xlogp is not None:
        try:
            x = float(xlogp)
            lip = "hydrophilic" if x < 0 else ("lipophilic" if x > 3 else "moderate lipophilicity")
            summary.append(f"XLogP {x:.2f} ({lip})")
        except Exception:
            summary.append(f"XLogP {xlogp}")

    # TPSA
    tpsa = props.get("TPSA")
    if tpsa is not None:
        try:
            t = float(tpsa)
            perm = "good" if t < 90 else ("moderate" if t < 140 else "poor")
            summary.append(f"TPSA {t:.1f} Å² ({perm} permeability expected)")
        except Exception:
            summary.append(f"TPSA {tpsa} Å²")

    # H-bonding
    hbd = props.get("HBondDonorCount")
    hba = props.get("HBondAcceptorCount")
    if hbd is not None and hba is not None:
        summary.append(f"{hbd} H-bond donors and {hba} H-bond acceptors")

    # Flexibility
    rtb = props.get("RotatableBondCount")
    if rtb is not None:
        try:
            r = int(rtb)
            flex = "rigid" if r <= 3 else ("flexible" if r >= 7 else "moderate flexibility")
            summary.append(f"{r} rotatable bonds ({flex})")
        except Exception:
            summary.append(f"Rotatable bonds: {rtb}")

    # Complexity
    cplx = props.get("Complexity")
    if cplx is not None:
        try:
            c = float(cplx)
            desc = "simple" if c < 250 else ("complex" if c > 500 else "moderate complexity")
            summary.append(f"molecular complexity {c:.0f} ({desc})")
        except Exception:
            summary.append(f"molecular complexity {cplx}")

    # Charge
    chg = props.get("Charge")
    if chg is not None:
        try:
            ch = int(chg)
            if ch != 0:
                summary.append(f"formal charge {ch:+d} ({'cationic' if ch > 0 else 'anionic'})")
        except Exception:
            pass

    # Connectivity SMILES
    smiles = props.get("ConnectivitySMILES")
    if smiles:
        summary.append(f"Connectivity SMILES: {smiles}")

    return ". ".join(summary) + "."


def get_assay_summary(cid: Union[int, str], limit: int = 5) -> str:
    """
    Fetch and summarize biological assay activity data for a given PubChem CID.
    Summarizes the number of active, inactive, and inconclusive assays,
    and lists details of up to `limit` active assays.
    
    Args:
        cid (int | str): PubChem Compound ID.
        limit (int): Maximum number of active assays to list (default 5).
    Returns:
        str: Summary of assay activity or error message.
    """
    limit = max(0, int(limit))

    result = _get_assay_summary_cached(cid)
    if result["error"]:
        return f"Error fetching assay summary for CID {cid}: {result['error']}"

    table = result.get("table", {})    
    columns = table.get("Columns", {}).get("Column", [])
    rows = [row.get("Cell", []) for row in table.get("Row", []) if 'Cell' in row]
    if not columns or not rows:
        return f"No assay data found for CID {cid}."

    outcome_col_i = columns.index("Activity Outcome") if "Activity Outcome" in columns else None
    if not outcome_col_i:
        print("No 'Activity Outcome' column found")
    name_col_i = columns.index("Assay Name") if "Assay Name" in columns else None
    type_col_i = columns.index("Assay Type") if "Assay Type" in columns else None

    active, inactive, inconclusive = [], [], []
    for row in rows:
        if row[outcome_col_i].lower() == "active":
            active.append(row)
        elif row[outcome_col_i].lower() == "inactive":
            inactive.append(row)
        else:
            inconclusive.append(row)

    # Ugly nested if statements but works for now
    details = []
    for a in active:
        if name_col_i is not None:
            name = a[name_col_i]
            if name and len(name) > 200:
                name = name[:197] + "..."
                if type_col_i is not None:
                    atype = a[type_col_i]
                    details.append(f"{name} (Type: {atype})")
        if len(details) >= limit:
            break    

    if details:
        active_details = f"\nFirst {limit} active assays:\n" + "\n".join(f"• {d}" for d in details)

    return (
        f"Assay summary for CID {cid}:"
        f"- Active in {len(active)} assay(s)"
        f"- Inactive in {len(inactive)} assay(s)"
        f"- Inconclusive in {len(inconclusive)} assay(s)"
        + (active_details if details else "")
    )


def get_safety_summary(cid: Union[int, str]) -> str:
    """
    Fetch and summarize GHS safety classification data for a given PubChem CID.
    Args:
        cid (int | str): PubChem Compound ID.
    Returns:
        str: Summary of GHS classification or error message.
    """
    result = _get_ghs_classification_cached(cid)
    if result["error"]:
        return f"Error fetching GHS classification for CID {cid}: {result['error']}"

    record = result["record"]
    sections = record.get("Section", []) or []
    if not sections:
        return f"No GHS classification data found for CID {cid}."
    
    parts: List[str] = [f"Safety (GHS) classification for CID {cid}:"]
    added = False

    for section in sections:
        if section.get("TOCHeading") != "GHS Classification":
            continue
        for sub in section.get("Section", []) or []:
            for info in sub.get("Information", []) or []:
                name = info.get("Name", "")
                value = info.get("Value", {})
                # pictograms
                pictos = _str_with_markup_list(value) if "Pictogram" in str(value) else []
                if pictos:
                    parts.append(f"GHS Pictograms: {', '.join(pictos)}")
                    added = True
                    continue
                # signal word
                if "Signal" in str(value):
                    sw = _str_with_markup_list(value)
                    if sw:
                        parts.append(f"Signal word: {sw[0]}")
                        added = True
                        continue
                # hazards
                if "Hazard Statement" in name:
                    hazards = _str_with_markup_list(value)
                    if hazards:
                        parts.append(f"Hazard statements: {'; '.join(hazards[:3])}")
                        if len(hazards) > 3:
                            parts.append(f"  (and {len(hazards) - 3} more)")
                        added = True

    return "\n".join(parts) if added else f"Limited safety data available for CID {cid}"


def get_drug_summary(cid: Union[int, str]) -> str:
    """
    Fetch and summarize drug/medication information for a given PubChem CID.
    Specifically: therapeutic uses, drug classes, and FDA status.

    Args:
        cid (int | str): PubChem Compound ID.
    Returns:
        str: Summary of drug/medication information or error message.
    """
    result = _get_drug_med_info_cached(cid)
    if result["error"]:
        return f"Error fetching drug/medication info for CID {cid}: {result['error']}"

    info = result["info"]
    sections = info.get("Section", []) or []
    if not sections:
        return f"No drug/medication information found for CID {cid}."
    
    parts: List[str] = [f"Drug/Medication Information for CID {cid}:"]
    added = False

    for section in sections:
        for subsection in section.get("Section", []) or []:
            heading = subsection.get("TOCHeading", "") or ""
            infos = subsection.get("Information", []) or []

            if "Therapeutic Use" in heading:
                uses = []
                for info in infos:
                    val = info.get("Value", {})
                    for item in val.get("StringWithMarkup", []) or []:
                        s = item.get("String")
                        if s:
                            uses.append(s)
                if uses:
                    parts.append(f"Therapeutic uses: {', '.join(uses[:3])}")
                    if len(uses) > 3:
                        parts.append(f"  (and {len(uses) - 3} more)")
                    added = True

            elif "Drug Class" in heading:
                classes = []
                for info in infos:
                    val = info.get("Value", {})
                    for item in val.get("StringWithMarkup", []) or []:
                        s = item.get("String")
                        if s:
                            classes.append(s)
                if classes:
                    parts.append(f"Drug classes: {', '.join(classes[:2])}")
                    added = True

            elif "FDA" in heading:
                for info in infos:
                    name = info.get("Name", "")
                    val = info.get("Value", {})
                    s = ""
                    if isinstance(val, dict) and val.get("StringWithMarkup"):
                        s = val["StringWithMarkup"][0].get("String", "")
                    if s and "FDA" in name:
                        parts.append(f"{name}: {s}")
                        added = True

    return "\n".join(parts) if added else f"CID {cid} - no specific drug/medication information available"


def find_similar_compounds(
    cid: Union[int, str],
    threshold: int = 90,
    limit: int = 10
) -> str:
    """
    Find CIDs of compounds similar to a given PubChem CID based on Tanimoto similarity.
    Additionally fetches the IUPAC name and molecular formula for each similar compound.    

    Args:
        cid (int | str): PubChem Compound ID.
        threshold (int): Tanimoto similarity threshold percentage (default 90).
        limit (int): Maximum number of similar compounds to return (default 10).
    Returns:
        str: Summary of similar compounds or error message.
    """
    limit = max(1, int(limit))

    result = _get_similar_cids_cached(cid, threshold)
    if result["error"]:
        return f"Error fetching similar compounds for CID {cid}: {result['error']}"

    similar_cids = result["similar_cids"]
    if not similar_cids:
        return f"No similar compounds found for CID {cid} at ≥{threshold}% Tanimoto."

    parts: List[str] = [
        f"Compounds similar to CID {cid} (≥{threshold}% Tanimoto):",
        "cid | IUPAC Name | Molecular Formula"
    ]

    for similar_cid in similar_cids[:limit]:
        props_result = _get_cid_properties_cached(similar_cid)
        if props_result["error"]:
            parts.append(f"{similar_cid} | Error fetching properties | Error fetching properties")
        else:
            props = props_result.get("properties", {})
            iupac_name = props.get("IUPACName", "N/A")
            molecular_formula = props.get("MolecularFormula", "N/A")
            parts.append(f"{similar_cid} | {iupac_name} | {molecular_formula}")

    return "\n".join(parts)


def compute_tanimoto(cid1: Union[int, str], cid2: Union[int, str]) -> str:
    """
    Compute Tanimoto similarity between two PubChem CIDs based on their fingerprints.

    Args:
        cid1 (int | str): First PubChem Compound ID.
        cid2 (int | str): Second PubChem Compound ID.
    Returns:
        str: Tanimoto similarity score or error message.
    """
    result = _compute_tanimoto_cached(cid1, cid2)
    if result["error"]:
        return f"Error computing Tanimoto similarity between CID {cid1} and CID {cid2}: {result['error']}"

    tanimoto = result.get("tanimoto")
    if tanimoto is None:
        return f"Tanimoto similarity could not be computed between CID {cid1} and CID {cid2}."

    return f"Tanimoto similarity between CID {cid1} and CID {cid2} is {tanimoto:.4f}."
