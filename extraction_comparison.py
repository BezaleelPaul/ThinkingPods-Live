"""
Extraction comparison — observation-only audit of rule vs LLM extraction.

For every mentor turn this module computes what the deterministic
rule-based extractor *would* have captured vs what the LLM extractor
*actually* produced, projected onto the ProjectState field level.

The comparison is purely observational:

  * It NEVER changes what is applied to state.
  * It NEVER feeds a decision path (objective, checklist, lifecycle, reply).
  * It NEVER replaces or skips the LLM extractor.
  * It only enriches the Developer Console and supports the offline audit
    summary over the golden conversation fixtures.

Basis of comparison is the ProjectState-field key space — the space that
actually mutates state and drives the Objective Engine. Rule extractor
output is mapped to StateField keys using the same legacy->StateField
mapping that ``merge_extracted_to_state`` applies, so "rules captured X"
is directly comparable with "LLM captured X".
"""

from memory_extractor import MessageType, StateField

# Mirror of the canonical legacy->StateField mapping applied by
# ``merge_extracted_to_state`` (extraction_pipeline.py). Must stay in sync:
# the rule extractor's legacy output keys are mapped here to the StateField
# each value would land on.
_RULE_LEGACY_TO_STATE: dict[str, StateField] = {
    "target_audience": StateField.PERSONAS,
    "pain_point": StateField.PROBLEMS,
    "motivation": StateField.PAIN_POINTS,
    "existing_solution": StateField.CURRENT_SOLUTIONS,
    "evidence": StateField.EVIDENCE,
    "impact": StateField.IMPACTS,
    "frequency": StateField.FREQUENCY,
}

_FIELD_LABELS: dict[str, str] = {
    StateField.PERSONAS.value: "Personas",
    StateField.PROBLEMS.value: "Problems",
    StateField.CURRENT_SOLUTIONS.value: "Current Solutions",
    StateField.PAIN_POINTS.value: "Pain Points",
    StateField.EVIDENCE.value: "Evidence",
    StateField.IMPACTS.value: "Impacts",
    StateField.FREQUENCY.value: "Frequency",
}


def field_label(field_value: str) -> str:
    """Human-readable label for a StateField value string."""
    return _FIELD_LABELS.get(field_value, str(field_value))


def _clean(v):
    """Mirror of ``extraction_pipeline.clean_val`` (empty -> None)."""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("null", "none", ""):
        return None
    return s


def rule_fields(rule_extracted: dict) -> set[str]:
    """StateField values the rule extractor would have populated."""
    fields: set[str] = set()
    for legacy, sf in _RULE_LEGACY_TO_STATE.items():
        if _clean(rule_extracted.get(legacy)):
            fields.add(sf.value)
    return fields


def llm_fields(llm_result) -> set[str]:
    """StateField values the LLM extractor produced (MEANINGFUL updates)."""
    if llm_result is None or llm_result.message_type != MessageType.MEANINGFUL:
        return set()
    return {u.field.value for u in llm_result.updates}


def compare_extraction(rule_extracted: dict, llm_result, applied: bool) -> dict:
    """Per-turn comparison record. JSON-serialisable, read-only."""
    r_fields = rule_fields(rule_extracted or {})
    l_fields = llm_fields(llm_result)
    matching = sorted(r_fields & l_fields)
    rule_only = sorted(r_fields - l_fields)
    llm_only = sorted(l_fields - r_fields)

    if llm_only:
        decision = "Rule sufficient: No"
        reason = "Missing: " + ", ".join(field_label(f) for f in llm_only)
    elif r_fields:
        decision = "Rule sufficient: Yes"
        reason = "LLM added nothing."
    else:
        decision = "Rule sufficient: Yes"
        reason = "Nothing to extract."

    return {
        "rule_meaningful": bool(r_fields),
        "llm_meaningful": bool(l_fields),
        "llm_applied": bool(applied),
        "rule_fields": sorted(r_fields),
        "llm_fields": sorted(l_fields),
        "matching_fields": matching,
        "rule_only_fields": rule_only,
        "llm_only_fields": llm_only,
        "rule_sufficient": not bool(llm_only),
        "llm_added_value": bool(llm_only),
        "decision": decision,
        "reason": reason,
    }


def summarize_comparisons(records: list[dict]) -> dict:
    """Aggregate per-turn comparison records into audit statistics.

    A turn is "informative" when either extractor produced at least one
    ProjectState field; greeting/empty turns (both empty) are excluded from
    the sufficiency percentages so they do not skew them upwards.
    """
    total = len(records)
    informative = [r for r in records if r.get("rule_meaningful") or r.get("llm_meaningful")]
    n_info = len(informative)

    def _pct(count: int, denom: int) -> float:
        return round(count / denom * 100, 1) if denom else 0.0

    llm_only_counts: dict[str, int] = {}
    rule_only_counts: dict[str, int] = {}
    for r in informative:
        for f in r.get("llm_only_fields", []):
            llm_only_counts[f] = llm_only_counts.get(f, 0) + 1
        for f in r.get("rule_only_fields", []):
            rule_only_counts[f] = rule_only_counts.get(f, 0) + 1

    return {
        "total_turns": total,
        "informative_turns": n_info,
        "rule_sufficient_turns": sum(1 for r in informative if r.get("rule_sufficient")),
        "rule_sufficient_pct": _pct(
            sum(1 for r in informative if r.get("rule_sufficient")), n_info
        ),
        "llm_added_value_turns": sum(1 for r in informative if r.get("llm_added_value")),
        "llm_added_value_pct": _pct(
            sum(1 for r in informative if r.get("llm_added_value")), n_info
        ),
        "llm_only_field_counts": dict(
            sorted(llm_only_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "rule_only_field_counts": dict(
            sorted(rule_only_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }
