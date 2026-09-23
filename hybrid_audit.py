"""
hybrid_audit — measurement-only shadow audit of hybrid extraction skips.

When the objective-aware hybrid decision layer (``hybrid_extraction``) skips
the LLM extractor because the deterministic rules satisfied the current
objective, this module can (opt-in via ``HYBRID_AUDIT=true``) ALSO run the
LLM extractor in *shadow mode* and compare what the rules applied vs what
the LLM would have extracted.

The shadow result is used ONLY for measurement:

  * it is NEVER applied to ``ProjectState``
  * it NEVER affects objectives, lifecycle, prompts, or replies
  * it only appears in Developer Console diagnostics and a session-level
    aggregate summary (``HybridAudit`` / ``HybridSummary``)

Risk levels (per skip):

  NONE   — shadow LLM produced nothing useful beyond the rules.
  LOW    — shadow LLM produced only duplicate / already-known facts
           (would not change ``ProjectState``).
  MEDIUM — shadow LLM produced new facts for non-current objectives
           (would change state but not the mentor's next objective).
  HIGH   — shadow LLM produced facts that would have changed the current
           objective or materially improved the conversation.

This module owns no decision logic, no extraction, and no prompts. It
deliberately imports only cycle-free leaves of the dependency graph.
"""

import copy
import os

from memory_extractor import MessageType, ProjectState  # noqa: F401
from state_manager import StateBatchValidationError, StateManager
from module3 import ObjectiveContext, ObjectiveEngine
from extraction_comparison import field_label, llm_fields, rule_fields

__all__ = [
    "HybridAuditSummary",
    "RISK_HIGH",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_NONE",
    "build_audit_record",
    "enabled",
]

RISK_NONE = "NONE"
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"

_HIGH_RISK_EXAMPLE_CAP = 5

_OBJECTIVE_ENGINE = ObjectiveEngine()


def enabled() -> bool:
    """Whether the shadow audit is switched on (``HYBRID_AUDIT=true``).

    When disabled, the mentor pipeline is byte-identical to a non-audited
    run: no shadow extraction, no audit diagnostics, no summary.
    """
    return os.getenv("HYBRID_AUDIT", "false").lower() == "true"


def _objective_context(session_data) -> ObjectiveContext:
    """Read-only conversation context mirroring ``mentor._determine_objective``
    so the objective comparison uses the same inputs the pipeline would."""
    history = getattr(session_data, "conversation_history", None) or []
    asked = getattr(session_data, "asked_question_families", None) or []
    return ObjectiveContext(
        user_messages=tuple(
            m.get("content", "")
            for m in history
            if m.get("role") == "user"
        ),
        asked_families=frozenset(asked),
    )


def _differing_fields(state_after_rules: ProjectState, sim_shadow: ProjectState) -> set:
    """Fields whose value differs between the rules-applied state and the
    state the shadow LLM extraction would have produced."""
    r = state_after_rules.to_state_dict()
    s = sim_shadow.to_state_dict()
    return {k for k in s if s.get(k) != r.get(k)}


def build_audit_record(
    rule_observation: dict,
    shadow,
    state_before: ProjectState,
    state_after_rules: ProjectState,
    session_data=None,
    previous_assistant_message=None,
    previous_user_message=None,
) -> dict:
    """Compare rule-applied vs shadow-LLM extraction for one skipped turn.

    ``state_before`` is the pre-extraction ``ProjectState``;
    ``state_after_rules`` is the actual post-rule state (the shadow was not
    applied). The shadow is simulated on a throwaway copy only.

    Returns a JSON-serialisable record. Pure measurement — mutates nothing
    except the simulated copy.
    """
    shadow_proposed = llm_fields(shadow)

    sim_shadow = copy.deepcopy(state_before)
    applied = False
    if shadow is not None and shadow.message_type == MessageType.MEANINGFUL and shadow.updates:
        try:
            StateManager(sim_shadow).apply_extraction(
                shadow,
                previous_assistant_message=previous_assistant_message,
                previous_user_message=previous_user_message,
            )
            applied = True
        except StateBatchValidationError:
            # An invalid shadow batch would not have applied either.
            applied = False

    differing = set()
    if applied:
        differing = _differing_fields(state_after_rules, sim_shadow)
    # "Additional fields" = the shadow's proposed fields that would have
    # changed ProjectState vs the rules-applied state (a new field OR a
    # different value than the rules captured). Empty when the shadow batch
    # would not have applied (validation error).
    additional = sorted(shadow_proposed & differing)

    would_change_state = bool(applied and differing)

    would_change_objective = False
    if applied and session_data is not None:
        context = _objective_context(session_data)
        try:
            obj_rules = _OBJECTIVE_ENGINE.determine_next(
                state_after_rules, context=context
            )
            obj_shadow = _OBJECTIVE_ENGINE.determine_next(
                sim_shadow, context=context
            )
            would_change_objective = (
                obj_rules.targeted_field() != obj_shadow.targeted_field()
            )
        except Exception:
            would_change_objective = False

    return {
        "hybrid_decision": "skip",
        "llm_skipped": True,
        "shadow_llm_fields": sorted(shadow_proposed),
        "additional_fields": additional,
        "would_change_state": bool(would_change_state),
        "would_change_objective": bool(would_change_objective),
        "risk_level": _classify_risk(
            shadow_proposed, would_change_state, would_change_objective
        ),
    }


def _classify_risk(shadow_fields, would_change_state, would_change_objective) -> str:
    """Risk classification per the module docstring."""
    if not shadow_fields:
        return RISK_NONE
    if not would_change_state:
        return RISK_LOW
    if not would_change_objective:
        return RISK_MEDIUM
    return RISK_HIGH


class HybridAuditSummary:
    """Append-only aggregate of per-skip audit records for a session.

    Persists with the session (JSON-safe). Never read by any decision path.
    """

    def __init__(
        self,
        llm_skipped: int = 0,
        no_difference: int = 0,
        low_risk: int = 0,
        medium_risk: int = 0,
        high_risk: int = 0,
        additional_field_counts: dict | None = None,
        high_risk_examples: list | None = None,
    ):
        self.llm_skipped = llm_skipped
        self.no_difference = no_difference
        self.low_risk = low_risk
        self.medium_risk = medium_risk
        self.high_risk = high_risk
        self.additional_field_counts: dict[str, int] = dict(
            additional_field_counts or {}
        )
        self.high_risk_examples: list[dict] = list(high_risk_examples or [])

    def add(self, record: dict) -> None:
        self.llm_skipped += 1
        risk = record.get("risk_level", RISK_NONE)
        if risk == RISK_LOW:
            self.low_risk += 1
        elif risk == RISK_MEDIUM:
            self.medium_risk += 1
        elif risk == RISK_HIGH:
            self.high_risk += 1
        else:
            self.no_difference += 1
        for field in record.get("additional_fields", []):
            self.additional_field_counts[field] = (
                self.additional_field_counts.get(field, 0) + 1
            )
        if risk == RISK_HIGH and len(self.high_risk_examples) < _HIGH_RISK_EXAMPLE_CAP:
            self.high_risk_examples.append(
                {
                    "fields": list(record.get("additional_fields", [])),
                    "would_change_objective": bool(
                        record.get("would_change_objective", False)
                    ),
                }
            )

    def to_dict(self) -> dict:
        return {
            "llm_skipped": self.llm_skipped,
            "no_difference": self.no_difference,
            "low_risk": self.low_risk,
            "medium_risk": self.medium_risk,
            "high_risk": self.high_risk,
            "additional_field_counts": dict(self.additional_field_counts),
            "high_risk_examples": [dict(e) for e in self.high_risk_examples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HybridAuditSummary":
        d = d or {}
        return cls(
            llm_skipped=int(d.get("llm_skipped", 0)),
            no_difference=int(d.get("no_difference", 0)),
            low_risk=int(d.get("low_risk", 0)),
            medium_risk=int(d.get("medium_risk", 0)),
            high_risk=int(d.get("high_risk", 0)),
            additional_field_counts=dict(d.get("additional_field_counts", {}) or {}),
            high_risk_examples=list(d.get("high_risk_examples", []) or []),
        )

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        return {
            "LLM skipped": self.llm_skipped,
            "No difference": self.no_difference,
            "Low risk": self.low_risk,
            "Medium risk": self.medium_risk,
            "High risk": self.high_risk,
            "Most commonly missed": {
                field_label(f): count
                for f, count in sorted(
                    self.additional_field_counts.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            },
            "High-risk examples": [dict(e) for e in self.high_risk_examples],
        }
