"""
hybrid_extraction — objective-aware decision layer for extraction.

Sits between the deterministic rule extractor and the LLM extractor. For
the current user message it decides — using ONLY the existing
``ObjectiveEngine``, ``SufficiencyChecker`` and ``ProjectState`` — whether
the deterministic extraction already satisfies the CURRENT objective (the
objective the mentor is pursuing this turn, derived from the pre-extraction
state):

  * satisfied  → the LLM extraction is skipped and the deterministic
                 extraction is applied through the existing StateManager
                 merge path (``merge_extracted_to_state``).
  * not        → the pipeline runs the LLM extraction exactly as before.

The decision is recorded for the Developer Console. This module owns no
state, no lifecycle, no prompt, and no extraction logic — it is the
objective-aware decision layer only. No new completeness logic is added:
sufficiency is the existing ``SufficiencyChecker`` classification, and the
"current objective" is the existing ``ObjectiveEngine`` output.
"""

import copy

from memory_extractor import LIST_FIELDS, ProjectState, StateField
from module3 import ObjectiveContext, ObjectiveEngine, SufficiencyChecker
from session_manager import SessionData
from extraction_pipeline import DEFAULT_CONFIDENCE, clean_val, merge_extracted_to_state
from extraction_comparison import _RULE_LEGACY_TO_STATE

__all__ = ["decide_hybrid_extraction", "rule_update_dicts"]


def _objective_context(session_data) -> ObjectiveContext:
    """Read-only conversation context matching ``mentor._determine_objective``
    so the decision sees the same objective the pipeline would."""
    return ObjectiveContext(
        user_messages=tuple(
            m.get("content", "")
            for m in session_data.conversation_history
            if m.get("role") == "user"
        ),
        asked_families=frozenset(session_data.asked_question_families),
    )


def rule_update_dicts(rule_observation: dict) -> list[dict]:
    """Diagnostics view of what the deterministic output would apply to
    state — mirrors the updates ``merge_extracted_to_state`` would build
    (same legacy->StateField mapping, same ADD/SET operations)."""
    out = []
    confidences = (rule_observation or {}).get("confidences", {}) or {}
    for legacy, sf in _RULE_LEGACY_TO_STATE.items():
        val = clean_val((rule_observation or {}).get(legacy))
        if val:
            out.append(
                {
                    "operation": "ADD" if sf in LIST_FIELDS else "SET",
                    "field": sf.value,
                    "value": val,
                    "confidence": confidences.get(legacy, DEFAULT_CONFIDENCE),
                }
            )
    return out


def decide_hybrid_extraction(
    rule_observation: dict,
    state_before: ProjectState,
    session_data,
    objective_engine: ObjectiveEngine | None = None,
    user_message: str | None = None,
) -> dict:
    """Decide whether the current objective is satisfied by deterministic
    extraction alone.

    ``state_before`` is the ``ProjectState`` *before* this turn's extraction
    (the current objective is derived from it). Returns a JSON-serialisable
    decision record:

      rule_output       — what the deterministic extraction would apply
      objective         — the current objective being pursued (value)
      objective_field   — the StateField that objective targets (or None)
      satisfied         — whether the rules satisfied that objective
      llm_invoked       — whether the LLM extraction must still run
      complexity        — semantic complexity classification (advisory)
      complexity_reason — human-readable complexity justification
      gate_enabled      — whether the Semantic Complexity Gate affected the
                          decision
      reason            — human-readable justification
    """
    from complexity_gate import enabled as _gate_enabled
    from complexity_gate import classify_complexity

    gate_enabled = _gate_enabled()

    engine = objective_engine or ObjectiveEngine()
    current = engine.determine_next(state_before, context=_objective_context(session_data))
    target = current.targeted_field()

    complexity_record = classify_complexity(user_message, rule_observation)

    if target is None:
        # No active field objective (e.g. WRAP_UP / transition). There is
        # nothing for the deterministic extraction to satisfy, so the LLM
        # runs exactly as before.
        return {
            "rule_output": rule_update_dicts(rule_observation),
            "objective": current.objective.value,
            "objective_field": None,
            "satisfied": False,
            "llm_invoked": True,
            "complexity": complexity_record["complexity"],
            "complexity_reason": "; ".join(complexity_record["reasons"]),
            "complexity_decision": complexity_record["decision"],
            "gate_enabled": gate_enabled,
            "reason": "No active field objective to satisfy; preserving LLM-first behavior.",
        }

    # Simulate applying the deterministic extraction to a throwaway copy of
    # the state (through the existing StateManager merge path) and classify
    # the target field with the existing sufficiency logic.
    sim_state = copy.deepcopy(state_before)
    merge_extracted_to_state(sim_state, SessionData(), rule_observation or {})
    satisfied = SufficiencyChecker.evaluate(sim_state).is_satisfied(target)

    llm_invoked = not bool(satisfied)
    if satisfied and gate_enabled and complexity_record["decision"] == "RUN_LLM":
        # The Semantic Complexity Gate: rules satisfy the current objective,
        # but the message is complex enough that skipping the LLM risks
        # information loss, so the LLM still runs.
        llm_invoked = True

    return {
        "rule_output": rule_update_dicts(rule_observation),
        "objective": current.objective.value,
        "objective_field": target.value,
        "satisfied": bool(satisfied),
        "llm_invoked": bool(llm_invoked),
        "complexity": complexity_record["complexity"],
        "complexity_reason": "; ".join(complexity_record["reasons"]),
        "complexity_decision": complexity_record["decision"],
        "gate_enabled": gate_enabled,
        "reason": (
            "Current objective satisfied by deterministic extraction."
            if satisfied and not llm_invoked
            else (
                "Current objective satisfied by deterministic extraction, but "
                "message complexity requires the LLM extractor."
                if satisfied and llm_invoked
                else "Current objective still unresolved."
            )
        ),
    }
