"""
for_agents.py

Tools exposing ChEMBL API access methods to agentic systems.
Enables agents to look up drugs and targets based on canonical names and
    search for compound properties, activities, mechanisms of action,
    indications, and target activity summaries. 
All returns are natural language strings so agents make best use of them.
Note that these are thin wrappers around the cached backend methods and
    these exposed tools themselves are not cached.
Adapted from https://github.com/FibrolytixBio/cf-compound-selection-demo.
"""

from typing import Any, Dict, List

from .chembl_backend import (
    _search_chembl_id_global,
    _get_compound_properties_global,
    _get_compound_activities_global,
    _get_drug_info_global,
    _get_drug_moa_global,
    _get_drug_indications_global,
    _search_target_id_global,
    _get_target_activities_summary_global,
)


def search_chembl_id(query: str, limit: int = 5) -> str:
    result = _search_chembl_id_global(query)
    if result["error"]:
        return result["error"]
    shown = result["compounds"][:max(0, int(limit))]
    return (
        f"Found {len(shown)} compound(s) matching '{query}': \n - "
        + "\n - ".join(shown)
    )


def get_compound_properties(chembl_id: str) -> str:
    result = _get_compound_properties_global(chembl_id)
    if result["error"]:
        return result["error"]
    summary_parts = [f"Properties of {chembl_id}:"]
    props = result["properties"]

    # Add key properties with context
    mw = props.get("mw_freebase")
    if mw:
        try:
            mw_float = float(mw)
            summary_parts.append(f"molecular weight {mw_float:.1f} Da")
        except (ValueError, TypeError):
            summary_parts.append(f"molecular weight {mw} Da")

    logp = props.get("alogp")
    if logp is not None:
        try:
            logp_float = float(logp)
            lipophilicity = (
                "hydrophilic"
                if logp_float < 0
                else "lipophilic"
                if logp_float > 3
                else "moderate lipophilicity"
            )
            summary_parts.append(f"ALogP {logp_float:.2f} ({lipophilicity})")
        except (ValueError, TypeError):
            summary_parts.append(f"ALogP {logp}")

    tpsa = props.get("psa")
    if tpsa:
        try:
            tpsa_float = float(tpsa)
            permeability = (
                "good"
                if tpsa_float < 90
                else "moderate"
                if tpsa_float < 140
                else "poor"
            )
            summary_parts.append(
                f"TPSA {tpsa_float:.1f} Ų ({permeability} permeability expected)"
            )
        except (ValueError, TypeError):
            summary_parts.append(f"TPSA {tpsa} Ų")

    hbd = props.get("hbd")
    hba = props.get("hba")
    if hbd is not None and hba is not None:
        summary_parts.append(f"{hbd} H-bond donors and {hba} H-bond acceptors")

    rtb = props.get("rtb")
    if rtb is not None:
        flexibility = (
            "rigid" if rtb <= 3 
            else "flexible" if rtb >= 7 
            else "moderate flexibility"
        )
        summary_parts.append(f"{rtb} rotatable bonds ({flexibility})")

    ro5 = props.get("num_ro5_violations")
    if ro5 is not None:
        ro5_text = (
            "compliant with Lipinski's Rule of Five"
            if ro5 == 0
            else f"has {ro5} Ro5 violation(s)"
        )
        summary_parts.append(ro5_text)

    # Add molecular type
    mol_type = result["molecule"].get("molecule_type")
    if mol_type:
        summary_parts.append(f"classified as {mol_type}")

    return ". ".join(summary_parts) + "."


def get_compound_activities(
    chembl_id: str, 
    activity_type: str | None = None,
    limit: int = 5
) -> str:    
    result = _get_compound_activities_global(chembl_id, activity_type)
    if result["error"]:
        return result["error"]
    activities = result["activities"]
    target_activities: Dict[str, Dict[str, Any]] = {}
    
    for act in activities:
        target_name = act.get("target_pref_name", "Unknown target")
        target_id = act.get("target_chembl_id", "")
        if target_name not in target_activities:
            target_activities[target_name] = {"target_id": target_id, "activities": []}

        if act.get("standard_value") and act.get("standard_type"):
            try:
                val = float(act["standard_value"])
            except Exception:
                continue
            target_activities[target_name]["activities"].append(
                {
                    "type": act["standard_type"],
                    "value": val,
                    "units": act.get("standard_units", ""),
                    "relation": act.get("standard_relation", "="),
                }
            )

    if not target_activities:
        return f"No bioactivity data found for {chembl_id}"

    summary_parts = [f"Bioactivity summary for {chembl_id}:"]
    count = 0
    for target_name, data in sorted(
        target_activities.items(), key=lambda x: len(x[1]["activities"]), reverse=True
    ):
        if count >= max(0, int(limit)):
            break

        target_id = data["target_id"]
        acts = data["activities"]

        activity_summary: List[str] = []
        for act_type in {a["type"] for a in acts}:
            type_acts = [a for a in acts if a["type"] == act_type]
            if not type_acts:
                continue
            best = min(type_acts, key=lambda x: x["value"])
            v = best["value"]
            if v < 0.1:
                value_str = f"{v:.2e}"
            elif v < 1000:
                value_str = f"{v:.1f}"
            else:
                value_str = f"{v:.0f}"
            activity_summary.append(
                f"{best['type']} {best['relation']} {value_str} {best['units']}"
            )

        if activity_summary:
            summary_parts.append(f"\n• {target_name} ({target_id}): " + ", ".join(activity_summary))
            count += 1

    if len(target_activities) > limit:
        summary_parts.append(
            f"(Showing top {limit} of {len(target_activities)} targets with activity data)"
        )

    return "\n".join(summary_parts)


def get_drug_approval_status(chembl_id: str) -> str:
    result = _get_drug_info_global(chembl_id)
    if result["error"]:
        return result["error"]
    drug = result["info"][0]

    if drug.get("first_approval"):
        return (
            f"{chembl_id} is an approved drug (first approved: "
            f"{drug['first_approval']})"
        )
    else:
        return f"{chembl_id} is not an approved drug"
    

def get_drug_moa(chembl_id: str, limit: int = 5) -> str:
    result = _get_drug_moa_global(chembl_id)
    if result["error"]:
        return result["error"]
    mechanisms = result["moa"]

    if mechanisms:
        moa_summaries = []
        for mech in mechanisms[:limit]:
            moa = mech.get("mechanism_of_action", "")
            action_type = mech.get("action_type", "")
            target_id = mech.get("target_chembl_id", "")
            if moa:
                summary = f"{moa}"
                if action_type:
                    summary += f" ({action_type})"
                if target_id:
                    summary += f" targeting {target_id}"
                moa_summaries.append(summary)
        if moa_summaries:
            return "Mechanisms of action: " + "; ".join(moa_summaries)
    
    return f"No mechanism of action data found for {chembl_id}"


def get_drug_indications(chembl_id: str, limit: int = 5) -> str:
    result = _get_drug_indications_global(chembl_id)
    if result["error"]:
        return result["error"]
    indications = result["drug_indications"]
    if indications:
        indication_summaries = []
        for ind in indications[:limit]:
            term = ind.get("efo_term", "")
            phase = ind.get("max_phase_for_ind", "")
            mesh = ind.get("mesh_heading", "")
            if term:
                summary = term
                if phase:
                    summary += f" (Phase {phase})"
                if mesh and mesh != term:
                    summary += f" ({mesh})"
                indication_summaries.append(summary)
        if indication_summaries:
            return "Drug indications: " + ", ".join(indication_summaries)
    
    return f"No indication data found for {chembl_id}"


def search_target_id(query: str, limit: int = 5) -> str:
    result = _search_target_id_global(query)
    if result["error"]:
        return result["error"]    
    targets = result['targets']
    
    # Extract target IDs and names
    target_list = []
    for target in targets[:limit]:
        target_id = target.get("target_chembl_id", "Unknown")
        pref_name = target.get("pref_name", "No name")
        organism = target.get("organism", "")

        target_str = f"{target_id} ({pref_name}"
        if organism:
            target_str += f", {organism}"
        target_str += ")"

        target_list.append(target_str)

    return f"Found {len(target_list)} target(s) matching '{query}': " + ", ".join(
        target_list
    )


def get_target_activities_summary(
    target_chembl_id: str,
    activity_type: str | None = None,
    limit: int = 5
) -> str:
    result = _get_target_activities_summary_global(target_chembl_id)
    if result["error"]:
        return result["error"]
    activities = result["target_activities"]

    # Filter and sort by potency (lower standard_value is better for IC50/Ki)
    valid_activities = []
    for act in activities:
        if act.get("standard_value") and (
                activity_type is None or
                act.get("standard_type") == activity_type
            ):
            try:
                val = float(act["standard_value"])
                valid_activities.append((val, act))
            except (ValueError, TypeError):
                continue
    
    activity_type_str = activity_type if activity_type else "activity"
    if not valid_activities:
        return f"No valid {activity_type_str} data found for {target_chembl_id}"

    # Sort by potency (ascending value)
    valid_activities.sort(key=lambda x: x[0])
    top_activities = valid_activities[:limit]
    # Get target name from first activity
    target_name = top_activities[0][1].get("target_pref_name", target_chembl_id)

    summary_parts = [(
        f"Top {len(top_activities)} compounds with {activity_type_str} against "
        f"{target_name} ({target_chembl_id}):"
    )]

    for i, (val, act) in enumerate(top_activities, 1):
        mol_id = act.get("molecule_chembl_id", "Unknown")
        mol_name = act.get("molecule_pref_name", "No Preferred Name")
        if mol_name is None:
            mol_name = "No Preferred Name"
        units = act.get("standard_units", "")
        relation = act.get("standard_relation", "=")
        pchembl = act.get("pchembl_value")
        assay_desc = act.get("assay_description", "")

        # Format value
        if val < 0.1:
            val_str = f"{val:.2e}"
        elif val < 1000:
            val_str = f"{val:.1f}"
        else:
            val_str = f"{val:.0f}"

        compound_str = f"{mol_name} (CHEMBL ID: {mol_id})"
        activity_str = f"{activity_type} {relation} {val_str} {units}"
        if pchembl:
            activity_str += f" (pChEMBL value: {pchembl})"

        summary_parts.append(f"{i}. {compound_str}: {activity_str}")
        if assay_desc:
            assay_id = act.get("assay_chembl_id")
            year = act.get("document_year")
            organism = act.get("target_organism")
            assay_info = f"Assay: {assay_desc}"
            details = []
            if assay_id:
                details.append(f"ID: {assay_id}")
            if year:
                details.append(f"Year: {year}")
            if organism:
                details.append(f"Organism: {organism}")
            if details:
                assay_info += f" ({', '.join(details)})"
            summary_parts.append(f"   {assay_info}")

    return "\n".join(summary_parts)
