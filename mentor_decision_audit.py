"""
mentor_decision_audit — measurement-only audit of the mentor's conversational
decisions.

For every turn this module classifies, in a fully deterministic way:

  * **Objective appropriateness** — did the mentor pursue the field the user
    just answered (or a topic the user opened up earlier), and was the chosen
    objective the best next conversational move given the user's answer?
  * **Conversation continuity** — did the question naturally follow the user's
    last answer (GOOD), return to an earlier thread or re-ask from a fresh
    angle (PARTIAL), or restart an already-answered topic / jump to an
    unrelated field (POOR)?
  * **Transition type** — what kind of transition to the next objective was
    this? Direct (natural progression), Connected (bridging the user’s answer to the next), Summary Bridge (the transition relies on presenting a summary), or Topic Restart (restarting a previously discussed topic).
  * **Question quality** — was the question Natural, Slightly Repetitive
    (same field, fresh family), Repeated (same family re-asked), or
    Questionnaire-like (a rigid checklist question that ignores the user's
    answer)?
  * **Acknowledgment** — did the mentor build on / move past the information
    the user just provided?
  * **Restart** — did the mentor re-ask something that was already sufficiently
    answered?

Measurement ONLY. This module:

  * NEVER changes objectives, lifecycle, prompts, extraction, state, or
    replies,
  * only appears in Developer Console diagnostics and a session-level
    aggregate (``MentorDecision`` / ``MentorDecisionSummary``),
  * is a pure function of the pipeline's existing per-turn record.

The audit deliberately imports only cycle-free leaves of the dependency graph
(``memory_extractor``, ``module3``) and accepts plain data — never ``mentor``
or ``session_manager``.
"""

from __future__ import annotations

from memory_extractor import LIST_FIELDS, REQUIRED_FIELDS, StateField
from module3 import QuestionFamily, SufficiencyChecker, field_of

__all__ = [
    "CONTINUITY_GOOD",
    "CONTINUITY_PARTIAL",
    "CONTINUITY_POOR",
    "QUALITY_NATURAL",
    "QUALITY_SLIGHTLY_REPETITIVE",
    "QUALITY_REPEATED",
    "QUALITY_QUESTIONNAIRE",
    "TRANSITION_TYPE_DIRECT",
    "TRANSITION_TYPE_CONNECTED",
    "TRANSITION_TYPE_SUMMARY_BRIDGE",
    "TRANSITION_TYPE_TOPIC_RESTART",
    "MentorDecisionSummary",
    "assess_mentor_decision",
]

CONTINUITY_GOOD = "GOOD"
CONTINUITY_PARTIAL = "PARTIAL"
CONTINUITY_POOR = "POOR"

QUALITY_NATURAL = "Natural"
QUALITY_SLIGHTLY_REPETITIVE = "Slightly Repetitive"
QUALITY_REPEATED = "Repeated"
QUALITY_QUESTIONNAIRE = "Questionnaire-like"

_WEIGHTS = {
    CONTINUITY_GOOD: 1.0,
    CONTINUITY_PARTIAL: 0.5,
    CONTINUITY_POOR: 0.0,
    QUALITY_NATURAL: 1.0,
    QUALITY_SLIGHTLY_REPETITIVE: 0.6,
    QUALITY_REPEATED: 0.3,
    QUALITY_QUESTIONNAIRE: 0.0,
}

_STATE_FIELD_KEYS = tuple(sf.value for sf in StateField)
_LIST_KEYS = frozenset(sf.value for sf in LIST_FIELDS)
_REQUIRED_FIELD_VALUES = frozenset(sf.value for sf in REQUIRED_FIELDS)

_SUMMARY_LIFECYCLES = (
    "READY_FOR_SUMMARY",
    "WAITING_FOR_CONFIRMATION",
    "READY_FOR_TRANSITION",
)

TRANSITION_TYPE_DIRECT = "Direct"
TRANSITION_TYPE_CONNECTED = "Connected"
TRANSITION_TYPE_SUMMARY_BRIDGE = "Summary Bridge"
TRANSITION_TYPE_TOPIC_RESTART = "Topic Restart"

_TRANSITION_WEIGHTS = {
    TRANSITION_TYPE_DIRECT: 1.0,
    TRANSITION_TYPE_CONNECTED: 0.7,
    TRANSITION_TYPE_SUMMARY_BRIDGE: 0.3,
    TRANSITION_TYPE_TOPIC_RESTART: 0.0,
}

_REQUIRED_FIELD_ORDER = [sf.value for sf in REQUIRED_FIELDS]


def _changed_fields(before: dict, after: dict) -> set:
    """StateField values whose content actually changed between snapshots."""
    changed: set = set()
    for key in _STATE_FIELD_KEYS:
        b = before.get(key)
        a = after.get(key)
        if key in _LIST_KEYS:
            new_items = [i for i in (a or []) if i not in (b or [])]
            if new_items:
                changed.add(key)
        else:
            bv = str(b).strip() if b else ""
            av = str(a).strip() if a else ""
            if av and av != bv:
                changed.add(key)
    return changed


def _satisfied(field_value: str, state: dict) -> bool:
    try:
        return SufficiencyChecker.evaluate_dict(state).is_satisfied(
            StateField(field_value)
        )
    except (ValueError, KeyError):
        return False


def _newly_resolved(before: dict, after: dict) -> set:
    """Required fields that reached a usable answer this turn."""
    return {
        sf.value
        for sf in REQUIRED_FIELDS
        if _satisfied(sf.value, after) and not _satisfied(sf.value, before)
    }


def _family_field_value(family_value) -> str | None:
    """Field value string targeted by a question family (None on garbage)."""
    if not family_value:
        return None
    try:
        sf = field_of(QuestionFamily(family_value))
        return sf.value
    except (ValueError, AttributeError, TypeError):
        return None


def _memory_field_set(memory: dict, key: str) -> set:
    return {r.get("field") for r in (memory or {}).get(key, []) if r.get("field")}


def _objective_reason(
    objective_value, objective_confidence, selection_trace, appropriate, target_value, changed, newly_resolved
) -> str:
    if objective_value == "WRAP_UP":
        return "All required Empathize fields are populated — deterministic WRAP_UP, full confidence."
    trace_reason = ""
    if selection_trace and selection_trace.get("reason_selected"):
        trace_reason = selection_trace["reason_selected"]
    if appropriate:
        if target_value in changed:
            return f"Appropriate: the user answered {target_value}, the objective's target. {trace_reason}".strip()
        if newly_resolved:
            return (
                f"Appropriate: the objective advanced past {sorted(newly_resolved)}. "
                f"{trace_reason}".strip()
            )
        if not changed:
            return f"Appropriate: no new field info to react to; objective continues. {trace_reason}".strip()
        return f"Appropriate: returns to a topic the user opened earlier. {trace_reason}".strip()
    return (
        f"Questionable: the user's answer opened {sorted(changed)} but the objective "
        f"pursued {target_value}. {trace_reason}".strip()
    )


def _classify_continuity(
    *,
    q_field_value,
    changed,
    open_fields,
    deferred_fields,
    state_before,
    guard_blocked,
    recovery_category,
    family_plan_reask,
    newly_resolved,
) -> tuple[str, str]:
    if q_field_value is None:
        return CONTINUITY_GOOD, "Stage-appropriate: no new question this turn (summary / confirmation)."
    if _satisfied(q_field_value, state_before):
        return (
            CONTINUITY_POOR,
            f"Restarts {q_field_value}: it already held a sufficient answer before this turn.",
        )
    if guard_blocked:
        return (
            CONTINUITY_POOR,
            "The produced question repeated an already-asked family; the guard replaced it with a fallback.",
        )
    if q_field_value in changed:
        return CONTINUITY_GOOD, f"Follows the user's answer on {q_field_value}."
    if q_field_value in deferred_fields:
        return (
            CONTINUITY_PARTIAL,
            f"Returns to the user's earlier topic {q_field_value} (deferred).",
        )
    if q_field_value in open_fields:
        return (
            CONTINUITY_PARTIAL,
            f"Continues the open thread on {q_field_value} (asked earlier, still unmet).",
        )
    if family_plan_reask:
        return (
            CONTINUITY_PARTIAL,
            f"Re-asks {q_field_value} from a fresh angle — every family for it was already tried.",
        )
    if recovery_category in ("DONT_KNOW", "UNCERTAIN_ANSWER"):
        return (
            CONTINUITY_PARTIAL,
            f"User could not answer; the mentor re-asks {q_field_value} from a fresh family.",
        )
    # The user's answer changed fields that are still unmet.
    # If the question targets an unrelated field, the mentor
    # ignored the user's lead.
    still_unmet_changed = changed - newly_resolved
    if still_unmet_changed and q_field_value not in still_unmet_changed:
        return (
            CONTINUITY_POOR,
            "The question about "
            + (q_field_value or "a topic")
            + " is unrelated to the user's answer "
            + f"(which supplied {sorted(still_unmet_changed)})."
        )
    # POOR: the question ignores the user's answer entirely
    # (questionnaire-like pattern). The user touched fields
    # but the question targets a completely different field.
    if changed and q_field_value not in changed and q_field_value not in deferred_fields and q_field_value not in open_fields:
        return (
            CONTINUITY_POOR,
            "The question ignores the user's answer "
            "and proceeds through a fixed checklist.",
        )
    # The user's answer satisfied the target field (or another
    # field) — the objective advanced naturally.
    if newly_resolved:
        return (
            CONTINUITY_GOOD,
            f"The objective advanced past {sorted(newly_resolved)}; "
            "the next question targets an unmet field.",
        )
    if not changed:
        return CONTINUITY_PARTIAL, f"Continues the planned objective {q_field_value}."
    # changed fields were all resolved; question targets a new
    # field — natural progression.
    return CONTINUITY_GOOD, f"The objective advanced; the next question targets {q_field_value}."


def _classify_quality(
    *,
    question_family,
    q_field_value,
    asked_families,
    asked_fields,
    family_plan_reask,
    guard_blocked,
    changed,
    deferred_fields,
    open_fields,
    continuity,
) -> tuple[str, str]:
    if question_family is None:
        return QUALITY_NATURAL, "No new question this turn."
    # Questionnaire-like: the question ignores the user's
    # answer entirely — it targets a field the user did not
    # touch and is not a deferred/open thread.
    if (
        changed
        and q_field_value not in changed
        and q_field_value not in deferred_fields
        and q_field_value not in open_fields
    ):
        return (
            QUALITY_QUESTIONNAIRE,
            "The question ignores the user's answer and "
            "proceeds through a fixed checklist.",
        )
    if family_plan_reask:
        return (
            QUALITY_SLIGHTLY_REPETITIVE,
            f"Re-asks {q_field_value} from a fresh family — all families already tried.",
        )
    if guard_blocked:
        return (
            QUALITY_REPEATED,
            "The produced question repeated a family already asked; the guard replaced it with a fallback.",
        )
    if question_family in asked_families:
        return QUALITY_REPEATED, f"Family {question_family} was already asked this session."
    if q_field_value in asked_fields:
        return (
            QUALITY_SLIGHTLY_REPETITIVE,
            f"Re-approaches {q_field_value} from a different angle; another family for it was already asked.",
        )
    return QUALITY_NATURAL, "Fresh family; builds on the conversation."


def _classify_transition_type(
    *,
    q_field_value,
    changed,
    open_fields,
    deferred_fields,
    state_before,
    guard_blocked,
    lifecycle_decision,
    newly_resolved,
) -> tuple[str, str]:
    if lifecycle_decision in _SUMMARY_LIFECYCLES:
        return TRANSITION_TYPE_SUMMARY_BRIDGE, "Transition relies on presenting a summary."
    if q_field_value is None:
        return TRANSITION_TYPE_DIRECT, "No new question this turn."
    if guard_blocked:
        return TRANSITION_TYPE_TOPIC_RESTART, "The produced question repeated an already-asked family; restarts topic with fallback."
    if _satisfied(q_field_value, state_before):
        return TRANSITION_TYPE_TOPIC_RESTART, f"Re-asks {q_field_value}, which was already sufficiently answered before this turn."
    if q_field_value in changed:
        return TRANSITION_TYPE_DIRECT, f"Direct transition following user answer on {q_field_value}."
    if q_field_value in open_fields:
        return TRANSITION_TYPE_CONNECTED, f"Connected transition continuing open thread on {q_field_value}."
    if q_field_value in deferred_fields:
        return TRANSITION_TYPE_CONNECTED, f"Connected transition returning to deferred topic {q_field_value}."
    if newly_resolved:
        return TRANSITION_TYPE_CONNECTED, f"Connected transition advanced past {sorted(newly_resolved)}; next question targets unmet field."
    return TRANSITION_TYPE_CONNECTED, f"Connected transition advances to new field {q_field_value}."


def _better_alternative(
    *,
    target_value,
    changed,
    newly_resolved,
    deferred_fields,
    open_fields,
    state_after,
) -> dict | None:
    """A field the user just answered (or opened earlier) that the mentor did
    NOT pursue and that is still unmet — pursuing it next would likely have
    produced a smoother conversation."""
    if target_value in changed:
        return None
    candidates = []
    for f in _REQUIRED_FIELD_ORDER:
        if f == target_value:
            continue
        if f not in changed and f not in deferred_fields and f not in open_fields:
            continue
        if f in newly_resolved:
            continue
        if _satisfied(f, state_after):
            continue
        candidates.append(f)
    if not candidates:
        return None
    best = candidates[0]
    source = "answer" if best in changed else "earlier topic"
    return {
        "objective": best.upper(),
        "reason": (
            f"The user's {source} opened {best}, which is still unmet — pursuing it "
            f"next could have produced a smoother conversation than {target_value or 'the chosen objective'}."
        ),
    }


def assess_mentor_decision(
    *,
    turn_index: int = 1,
    user_message: str = "",
    generated_question: str = "",
    objective_value: str = "PERSONAS",
    objective_confidence: float = 1.0,
    objective_advancement: str = "STAYED_ACTIVE",
    selection_trace: dict | None = None,
    question_family: str | None = None,
    family_plan: dict | None = None,
    guard_blocked: bool = False,
    state_before: dict | None = None,
    state_after: dict | None = None,
    recovery: dict | None = None,
    memory: dict | None = None,
    lifecycle_decision: str = "CONTINUE",
    response_strategy: str = "ASK_QUESTION",
) -> dict:
    """Deterministic per-turn mentor-decision assessment.

    Pure function of the pipeline's existing per-turn record; mutates nothing.
    Returns a JSON-serialisable record for the Developer Console and the
    session-level ``MentorDecisionSummary``.
    """
    family_plan = family_plan or {}
    recovery = recovery or {}
    memory = memory or {}
    state_before = dict(state_before or {})
    state_after = dict(state_after or {})

    changed = _changed_fields(state_before, state_after)
    newly_resolved = _newly_resolved(state_before, state_after)
    q_field_value = _family_field_value(question_family)
    target_value = None if objective_value == "WRAP_UP" else objective_value

    open_fields = _memory_field_set(memory, "open_threads")
    deferred_fields = _memory_field_set(memory, "deferred_topics")
    resolved_fields = _memory_field_set(memory, "resolved_threads")

    asked_families = set(family_plan.get("asked_families") or [])
    asked_fields = {
        fv
        for f in asked_families
        for fv in [_family_field_value(f)]
        if fv
    }

    # --- 1. objective appropriateness --------------------------------------
    if objective_value == "WRAP_UP":
        appropriate = True
    elif target_value in changed:
        appropriate = True
    elif newly_resolved:
        appropriate = True
    elif not changed:
        appropriate = True
    elif target_value in deferred_fields or target_value in open_fields:
        appropriate = True
    else:
        appropriate = False

    # --- 2. conversation continuity ----------------------------------------
    continuity, continuity_reason = _classify_continuity(
        q_field_value=q_field_value,
        changed=changed,
        open_fields=open_fields,
        deferred_fields=deferred_fields,
        state_before=state_before,
        guard_blocked=guard_blocked,
        recovery_category=recovery.get("Category"),
        family_plan_reask=bool(family_plan.get("reask")),
        newly_resolved=newly_resolved,
    )

    # --- 2.5. transition type ---------------------------------------------
    transition_type, transition_reason = _classify_transition_type(
        q_field_value=q_field_value,
        changed=changed,
        open_fields=open_fields,
        deferred_fields=deferred_fields,
        state_before=state_before,
        guard_blocked=guard_blocked,
        lifecycle_decision=lifecycle_decision,
        newly_resolved=newly_resolved,
    )

    # --- 3. question quality ------------------------------------------------
    question_quality, quality_reason = _classify_quality(
        question_family=question_family,
        q_field_value=q_field_value,
        asked_families=asked_families,
        asked_fields=asked_fields,
        family_plan_reask=bool(family_plan.get("reask")),
        guard_blocked=guard_blocked,
        changed=changed,
        deferred_fields=deferred_fields,
        open_fields=open_fields,
        continuity=continuity,
    )

    # --- 4. acknowledgment & restart ----------------------------------------
    if not changed:
        acknowledged = True
    else:
        acknowledged = (
            q_field_value in changed
            or bool(newly_resolved)
            or lifecycle_decision in _SUMMARY_LIFECYCLES
            or response_strategy == "GENERATE_SUMMARY"
        )
    restart_reason = ""
    if q_field_value and _satisfied(q_field_value, state_before):
        restarted = True
        restart_reason = f"Re-asks {q_field_value}, which was already sufficiently answered before this turn."
    elif guard_blocked:
        restarted = True
        restart_reason = "The produced question repeated an already-asked family (semantic duplicate)."
    else:
        restarted = False

    # --- 5. better alternative ----------------------------------------------
    better = _better_alternative(
        target_value=target_value,
        changed=changed,
        newly_resolved=newly_resolved,
        deferred_fields=deferred_fields,
        open_fields=open_fields,
        state_after=state_after,
    )

    alternative_objectives = []
    if selection_trace:
        for c in selection_trace.get("candidates", []):
            if c.get("objective") == objective_value:
                continue
            alternative_objectives.append(
                {
                    "objective": c.get("objective"),
                    "score": round(float(c.get("score", 0.0)), 4),
                    "discussed": bool(c.get("discussed")),
                    "repeated": bool(c.get("repeated")),
                }
            )

    return {
        "turn_index": turn_index,
        "user_message": user_message,
        "generated_question": generated_question,
        "objective": objective_value,
        "objective_confidence": round(float(objective_confidence), 2),
        "objective_advancement": objective_advancement,
        "alternative_objectives": alternative_objectives,
        "objective_appropriate": appropriate,
        "objective_reason": _objective_reason(
            objective_value,
            objective_confidence,
            selection_trace,
            appropriate,
            target_value,
            changed,
            newly_resolved,
        ),
        "continuity": continuity,
        "continuity_reason": continuity_reason,
        "transition_type": transition_type,
        "transition_reason": transition_reason,
        "question_quality": question_quality,
        "quality_reason": quality_reason,
        "acknowledged_information": acknowledged,
        "restarted_topic": restarted,
        "restart_reason": restart_reason,
        "better_alternative": better,
        "question_family": question_family,
        "question_field": q_field_value,
        "changed_fields": sorted(changed),
        "newly_resolved": sorted(newly_resolved),
        "state_before": dict(state_before or {}),
        "state_after": dict(state_after or {}),
        "memory": {
            "open_threads": list(open_fields),
            "deferred_topics": list(deferred_fields),
            "resolved_threads": list(resolved_fields),
        },
        "lifecycle_decision": lifecycle_decision,
        "response_strategy": response_strategy,
    }


class MentorDecisionSummary:
    """Append-only aggregate of per-turn mentor-decision assessments for a
    session. Persists with the session (JSON-safe). Never read by any decision
    path — Developer Console + offline report only."""

    def __init__(
        self,
        objective_counts: dict | None = None,
        continuity_counts: dict | None = None,
        quality_counts: dict | None = None,
        transition_counts: dict | None = None,
        appropriate_count: int = 0,
        acknowledged_count: int = 0,
        restarted_count: int = 0,
        turn_count: int = 0,
        weak_transitions: dict | None = None,
        better_alternative_turns: list | None = None,
        examples: list | None = None,
    ):
        self.objective_counts: dict[str, int] = dict(objective_counts or {})
        self.continuity_counts: dict[str, int] = dict(
            continuity_counts
            or {CONTINUITY_GOOD: 0, CONTINUITY_PARTIAL: 0, CONTINUITY_POOR: 0}
        )
        self.quality_counts: dict[str, int] = dict(
            quality_counts
            or {
                QUALITY_NATURAL: 0,
                QUALITY_SLIGHTLY_REPETITIVE: 0,
                QUALITY_REPEATED: 0,
                QUALITY_QUESTIONNAIRE: 0,
            }
        )
        self.transition_counts: dict[str, int] = dict(
            transition_counts
            or {
                TRANSITION_TYPE_DIRECT: 0,
                TRANSITION_TYPE_CONNECTED: 0,
                TRANSITION_TYPE_SUMMARY_BRIDGE: 0,
                TRANSITION_TYPE_TOPIC_RESTART: 0,
            }
        )
        self.appropriate_count = appropriate_count
        self.acknowledged_count = acknowledged_count
        self.restarted_count = restarted_count
        self.turn_count = turn_count
        self.weak_transitions: dict[str, int] = dict(weak_transitions or {})
        self.better_alternative_turns: list = list(better_alternative_turns or [])
        self.examples: list = list(examples or [])
        self._prev_objective: str | None = None

    # ---- aggregation -------------------------------------------------------

    def add_record(self, record: dict) -> None:
        self.turn_count += 1
        objective = record.get("objective")
        self.objective_counts[objective] = self.objective_counts.get(objective, 0) + 1
        self.continuity_counts[record.get("continuity")] = (
            self.continuity_counts.get(record.get("continuity"), 0) + 1
        )
        self.quality_counts[record.get("question_quality")] = (
            self.quality_counts.get(record.get("question_quality"), 0) + 1
        )
        self.transition_counts[record.get("transition_type")] = (
            self.transition_counts.get(record.get("transition_type"), 0) + 1
        )
        if record.get("objective_appropriate"):
            self.appropriate_count += 1
        if record.get("acknowledged_information"):
            self.acknowledged_count += 1
        if record.get("restarted_topic"):
            self.restarted_count += 1

        weak = (
            record.get("continuity") == CONTINUITY_POOR
            or record.get("question_quality")
            in (QUALITY_REPEATED, QUALITY_QUESTIONNAIRE)
        )
        if weak and self._prev_objective is not None:
            key = f"{self._prev_objective} → {objective}"
            self.weak_transitions[key] = self.weak_transitions.get(key, 0) + 1

        better = record.get("better_alternative")
        if better:
            self.better_alternative_turns.append(
                {
                    "turn": record.get("turn_index"),
                    "objective": objective,
                    "better_alternative": better.get("objective"),
                    "reason": better.get("reason", ""),
                    "message": (record.get("user_message") or "")[:160],
                    "question": (record.get("generated_question") or "")[:160],
                }
            )
            if len(self.better_alternative_turns) > 5:
                self.better_alternative_turns = self.better_alternative_turns[-5:]

        if record.get("question_quality") in (
            QUALITY_REPEATED,
            QUALITY_QUESTIONNAIRE,
        ):
            self.examples.append(
                {
                    "turn": record.get("turn_index"),
                    "objective": objective,
                    "quality": record.get("question_quality"),
                    "reason": record.get("quality_reason", ""),
                    "question": (record.get("generated_question") or "")[:160],
                }
            )
            if len(self.examples) > 5:
                self.examples = self.examples[-5:]

        self._prev_objective = objective

    def merge(self, other: "MentorDecisionSummary") -> None:
        self.turn_count += other.turn_count
        self.appropriate_count += other.appropriate_count
        self.acknowledged_count += other.acknowledged_count
        self.restarted_count += other.restarted_count
        for k, v in other.objective_counts.items():
            self.objective_counts[k] = self.objective_counts.get(k, 0) + v
        for k, v in other.continuity_counts.items():
            self.continuity_counts[k] = self.continuity_counts.get(k, 0) + v
        for k, v in other.quality_counts.items():
            self.quality_counts[k] = self.quality_counts.get(k, 0) + v
        for k, v in other.transition_counts.items():
            self.transition_counts[k] = self.transition_counts.get(k, 0) + v
        for k, v in other.weak_transitions.items():
            self.weak_transitions[k] = self.weak_transitions.get(k, 0) + v
        self.better_alternative_turns.extend(other.better_alternative_turns)
        self.better_alternative_turns = self.better_alternative_turns[-5:]
        self.examples.extend(other.examples)
        self.examples = self.examples[-5:]
        self._prev_objective = other._prev_objective

    # ---- scoring -----------------------------------------------------------

    def continuity_score(self) -> float:
        """Weighted continuity score in ``[0, 100]`` (GOOD=1, PARTIAL=0.5, POOR=0)."""
        total = self.turn_count
        if not total:
            return 0.0
        return round(
            sum(
                _WEIGHTS.get(k, 0.0) * v
                for k, v in self.continuity_counts.items()
            )
            / total
            * 100,
            1,
        )

    def quality_score(self) -> float:
        """Weighted question-quality score in ``[0, 100]`` (Natural=1, …)."""
        total = self.turn_count
        if not total:
            return 0.0
        return round(
            sum(
                _WEIGHTS.get(k, 0.0) * v
                for k, v in self.quality_counts.items()
            )
            / total
            * 100,
            1,
        )

    def repeated_objectives(self) -> dict[str, int]:
        """Objectives pursued on more than one turn (re-asked / stayed active)."""
        return {
            obj: n
            for obj, n in sorted(
                self.objective_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if n > 1
        }

    def most_common_weak_transitions(self, limit: int = 8) -> dict[str, int]:
        return dict(
            sorted(self.weak_transitions.items(), key=lambda kv: (-kv[1], kv[0]))[
                :limit
            ]
        )

    # ---- persistence / display ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "objective_counts": dict(self.objective_counts),
            "continuity_counts": dict(self.continuity_counts),
            "quality_counts": dict(self.quality_counts),
            "transition_counts": dict(self.transition_counts),
            "appropriate_count": self.appropriate_count,
            "acknowledged_count": self.acknowledged_count,
            "restarted_count": self.restarted_count,
            "weak_transitions": dict(self.weak_transitions),
            "better_alternative_turns": [dict(e) for e in self.better_alternative_turns],
            "examples": [dict(e) for e in self.examples],
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "MentorDecisionSummary":
        d = d or {}
        obj = cls(
            objective_counts=d.get("objective_counts", {}),
            continuity_counts=d.get("continuity_counts", {}),
            quality_counts=d.get("quality_counts", {}),
            transition_counts=d.get("transition_counts", {}),
            appropriate_count=int(d.get("appropriate_count", 0)),
            acknowledged_count=int(d.get("acknowledged_count", 0)),
            restarted_count=int(d.get("restarted_count", 0)),
            turn_count=int(d.get("turn_count", 0)),
            weak_transitions=d.get("weak_transitions", {}),
            better_alternative_turns=list(d.get("better_alternative_turns", []) or []),
            examples=list(d.get("examples", []) or []),
        )
        return obj

    def transition_score(self) -> float:
        """Weighted transition score in ``[0, 100]`` (Direct=1, Connected=0.7, Summary Bridge=0.3, Topic Restart=0)."""
        total = self.turn_count
        if not total:
            return 0.0
        return round(
            sum(
                _TRANSITION_WEIGHTS.get(k, 0.0) * v
                for k, v in self.transition_counts.items()
            )
            / total
            * 100,
            1,
        )

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        return {
            "Objective Distribution": dict(
                sorted(
                    self.objective_counts.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "Repeated Objectives": self.repeated_objectives(),
            "Conversation Continuity": dict(self.continuity_counts),
            "Continuity Score": f"{self.continuity_score():.1f}% (GOOD-weighted)",
            "Transition Type": dict(
                sorted(
                    self.transition_counts.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "Transition Score": f"{self.transition_score():.1f}% (Direct-weighted)",
            "Question Quality": dict(self.quality_counts),
            "Question Quality Score": (
                f"{self.quality_score():.1f}% (Natural-weighted)"
            ),
            "Objective Appropriateness": (
                f"{self.appropriate_count}/{self.turn_count} appropriate"
            ),
            "Acknowledged Information": (
                f"{self.acknowledged_count}/{self.turn_count} turns"
            ),
            "Restarted Topics": self.restarted_count,
            "Most Common Weak Transitions": self.most_common_weak_transitions(),
            "Better-Alternative Turns": [dict(e) for e in self.better_alternative_turns],
            "Examples": [dict(e) for e in self.examples],
        }
