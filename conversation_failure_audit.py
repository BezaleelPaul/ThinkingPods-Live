"""Observation-only Conversation Failure Audit for ReqGPT.

Deterministic, per-turn classification of *failed conversations*. A "failure"
is a moment where the mentor and the user are no longer moving in sync toward
a usable Empathize state — the mentor misread the user, re-asked something the
user already resolved, sprinted ahead, ignored the state it has, or delivered
a reply that does not match the move it selected.

Observation-only contract (mirrors :mod:`conversation_style_audit` /
:mod:`product_experience_audit`):
    * pure standard library (no ``mentor`` / ``app`` / numpy / streamlit);
    * reads existing runtime data only — never writes SharedState fields,
      never calls any planner/extractor/validator, never changes prompts;
    * computed AFTER the reply is finalised and only mounted into the
      Developer Console, so it can never influence the reply/objective/
      extraction/lifecycle/memory/recovery decisions;
    * produces a JSON-safe ``ConversationFailureRecord`` plus an append-only
      ``ConversationFailureSummary`` that persists with the session.

Failure categories (any number can apply per turn; the first matched becomes
the *primary* failure, the rest are *secondary*):

    META_INTENT                User turn was meta/non-substantive (greeting,
                               trust, validation), not a design contribution.
    REPEATED_INFORMATION       User re-offered content for a field the mentor
                               had already acknowledged / summarised.
    MISUNDERSTOOD_RESPONSE     Mentor (re)asked a field that already has a
                               usable value in project state.
    ABRUPT_TRANSITION          Mentor moved to a new topic / summarised /
                               closed without resolving or acknowledging the
                               pursued thread.
    OVER_EXPLORATION           Mentor kept exploring a field that is already
                               resolved (unnecessary re-expansion).
    UNSUPPORTED_INFERENCE      Reply implied a magnitude/frequency without any
                               supporting evidence in state.
    MISSED_INSIGHT             Project state holds a usable value that the
                               mentor's conversational memory never recorded.
    MEMORY_FAILURE             The conversational-memory layer self-contradicts
                               (e.g. a field is both acknowledged and open).
    SOCIAL_FAILURE             Reply reprimanded the user outside an explicit
                               challenge move.
    MOVE_SELECTION             The chosen ConversationMove does not match the
                               generated reply (e.g. ELICIT_INFORMATION with
                               no question), or no move was recorded.
    NONE                       No failure detected this turn.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------


class ConversationFailure(str, Enum):
    """Deterministic observation-only failure classification."""

    NONE = "NONE"
    META_INTENT = "META_INTENT"
    REPEATED_INFORMATION = "REPEATED_INFORMATION"
    MISUNDERSTOOD_RESPONSE = "MISUNDERSTOOD_RESPONSE"
    ABRUPT_TRANSITION = "ABRUPT_TRANSITION"
    OVER_EXPLORATION = "OVER_EXPLORATION"
    UNSUPPORTED_INFERENCE = "UNSUPPORTED_INFERENCE"
    MISSED_INSIGHT = "MISSED_INSIGHT"
    MEMORY_FAILURE = "MEMORY_FAILURE"
    SOCIAL_FAILURE = "SOCIAL_FAILURE"
    MOVE_SELECTION = "MOVE_SELECTION"


# Human-friendly labels for the Developer Console.
FAILURE_LABELS: Dict[ConversationFailure, str] = {
    ConversationFailure.NONE: "No failure",
    ConversationFailure.META_INTENT: "Meta intent",
    ConversationFailure.REPEATED_INFORMATION: "Repeated information",
    ConversationFailure.MISUNDERSTOOD_RESPONSE: "Misunderstood response",
    ConversationFailure.ABRUPT_TRANSITION: "Abrupt transition",
    ConversationFailure.OVER_EXPLORATION: "Over-exploration",
    ConversationFailure.UNSUPPORTED_INFERENCE: "Unsupported inference",
    ConversationFailure.MISSED_INSIGHT: "Missed insight",
    ConversationFailure.MEMORY_FAILURE: "Memory failure",
    ConversationFailure.SOCIAL_FAILURE: "Social failure",
    ConversationFailure.MOVE_SELECTION: "Move selection",
}


def failure_label(value: Any) -> str:
    try:
        return FAILURE_LABELS[ConversationFailure(value)]
    except (ValueError, KeyError):
        return str(value)


# ---------------------------------------------------------------------------
# Canonical project-state fields (mirror StateField values / mentor.py
# ``_FIELD_DISPLAY_LABELS``). Kept local so the module stays dependency-free.
# ---------------------------------------------------------------------------

_FIELD_KEYS: Tuple[str, ...] = (
    "personas",
    "problems",
    "current_solutions",
    "pain_points",
    "evidence",
    "impacts",
    "frequency",
)

# Scalar fields (SET-overwrite semantics) vs list fields (append semantics).
_SCALAR_FIELDS: frozenset = frozenset({"frequency"})
_LIST_FIELDS: frozenset = frozenset(_FIELD_KEYS) - _SCALAR_FIELDS

field_label_map: Dict[str, str] = {
    "personas": "Personas",
    "problems": "Problems",
    "current_solutions": "Current Solutions",
    "pain_points": "Pain Points",
    "evidence": "Evidence",
    "impacts": "Impacts",
    "frequency": "Frequency",
}


def _field_label(field_value: str) -> str:
    """Display label for a raw state/field key (fallback to the key itself)."""
    return field_label_map.get(field_value, field_value)


def _family_field(family_value: Any) -> Optional[str]:
    """Map a question-family value (e.g. ``PERSONAS_WHO``) to a state field.

    Replicates ``module3.question_families.FIELD_OF_FAMILY`` prefix rules
    without importing that module, keeping this audit a dependency-free leaf.
    """
    upper = str(family_value or "").upper()
    for prefix, field_key in (
        ("PERSONAS", "personas"),
        ("PROBLEMS", "problems"),
        ("SOLUTIONS_", "current_solutions"),
        ("SOLUTION", "current_solutions"),
        ("PAIN_", "pain_points"),
        ("PAIN", "pain_points"),
        ("EVIDENCE", "evidence"),
        ("FREQUENCY", "frequency"),
        ("IMPACTS_", "impacts"),
        ("IMPACT_", "impacts"),
    ):
        if upper.startswith(prefix):
            return field_key
    return None


# ---------------------------------------------------------------------------
# Text helpers (pure standard library)
# ---------------------------------------------------------------------------

_IGNORED_RESPONSES: Tuple[str, ...] = (
    "i don't know",
    "i dont know",
    "i don''t know",
    "don't know",
    "dont know",
    "not sure",
    "unsure",
    "no idea",
    "i have no idea",
    "i can't think",
    "i cant think",
    "n/a",
    "na",
    "none",
    "nothing",
    "not applicable",
)

_META_STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "so", "then", "well",
        "oh", "hi", "hello", "hey", "ok", "okay", "thanks", "thank", "you",
        "your", "that", "this", "those", "these", "it", "its", "i", "me",
        "my", "to", "of", "in", "on", "for", "with", "at", "by", "as", "is",
        "are", "was", "were", "be", "been", "do", "does", "did", "have",
        "has", "had", "can", "could", "will", "would", "should", "just",
        "really", "very", "much", "many", "some", "any",
    }
)

_TRUST_MARKERS: Tuple[str, ...] = (
    "trust you",
    "you're right",
    "you are right",
    "i believe you",
    "you know best",
    "do what you think",
    "whatever you think",
)

_META_START_MARKERS: Tuple[str, ...] = (
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "thanks for", "thank you for",
    "how do you work", "what do you do", "can you help me with",
)

_UNCERTAINTY_PHRASES: Tuple[str, ...] = (
    "i don't know",
    "i dont know",
    "not sure",
    "unsure",
    "i dunno",
    "i'm not sure",
    "im not sure",
    "can't say",
    "cant say",
)

_HEDGE_WORDS: frozenset = frozenset(
    {
        "think", "guess", "suppose", "maybe", "probably", "possibly",
        "approximately", "roughly", "about", "around", "kind of", "sort of",
    }
)

# Vague / deferring non-answers. These are NOT "meta" (the user is genuinely
# uncertain, not talking about the conversation itself) — the deterministic
# recovery machinery already handles SET/append for them, so the audit should
# stay silent rather than mislabel them META_INTENT.
_DEFERRING_PHRASES: Tuple[str, ...] = (
    "it depends",
    "depends on",
    "it varies",
    "varies",
    "case by case",
    "different for",
    "hard to say",
    "difficult to say",
    "depends on the situation",
)

# Object / description trigger words: any of these content words in the reply
# strongly implies the reply is targeting that field.
_FIELD_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "personas": (
        "person", "people", "user", "users", "audience", "customer",
        "customers", "who", "demographic", "age", "role", "cashier",
        "cashiers", "staff", "worker", "workers", "employee", "employees",
        "student", "students", "parent", "parents", "teacher", "teachers",
        "owner", "owners", "shopkeeper", "client", "clients", "buyer",
        "buyers", "shopper", "shoppers", "child", "children", "retiree",
        "which people", "who else",
    ),
    "problems": (
        "problem", "problem of", "difficulty", "difficulties", "frustrat",
        "challenge", "struggle", "pain", "issue", "hard", "hardest",
    ),
    "current_solutions": (
        "solution", "solutions", "workaround", "workarounds", "currently",
        "current", "today", "now", "handle", "manage", "existing", "way",
    ),
    "pain_points": (
        "pain", "frustrat", "frustration", "impact", "affects", "affect",
        "annoy", "bother", "why", "because",
    ),
    "evidence": (
        "evidence", "proof", "observation", "observations", "example",
        "examples", "seen", "notice", "notice", "data", "verify", "told",
    ),
    "impacts": (
        "impact", "impacts", "effect", "effects", "consequence",
        "consequences", "affect", "affects", "cost", "loss",
    ),
    "frequency": (
        "often", "frequency", "frequently", "daily", "weekly", "monthly",
        "how often", "how many", "times", "usual", "every", "percentage",
    ),
}

# Magnitude qualifiers offered as UNSUPPORTED_INFERENCE when no evidence exists.
_UNSUPPORTED_MAGNITUDE_TERMS: Tuple[str, ...] = (
    "daily",
    "every day",
    "every single day",
    "always",
    "everyone",
    "most people",
    "everybody",
    "all the time",
    "majority",
)

_WORDS_RE = re.compile(r"[a-z0-9']+")


def _content_words(text: str) -> Set[str]:
    """Lowercased non-stopword tokens from ``text``."""
    if not text:
        return set()
    tokens = _WORDS_RE.findall(text.lower())
    return {t for t in tokens if t not in _META_STOPWORDS and len(t) > 1}


def _lower(text: str) -> str:
    return (text or "").lower()


def _contains_any(text: str, markers: Tuple[str, ...]) -> bool:
    lowered = _lower(text)
    return any(m in lowered for m in markers)


# ---------------------------------------------------------------------------
# Project-state value helpers
# ---------------------------------------------------------------------------


def _value_items(value: Any) -> List[str]:
    """Normalise a state field value to a list of meaningful strings.

    ``None`` / empty / placeholder / "i don't know" values collapse to ``[]``
    so they are never counted as a usable answer.
    """
    if value is None:
        return []
    if isinstance(value, list):
        items: List[str] = []
        for entry in value:
            text = str(entry).strip()
            if not text:
                continue
            lowered = _lower(text)
            if lowered in _IGNORED_RESPONSES:
                continue
            items.append(text)
        return items
    text = str(value).strip()
    if not text:
        return []
    if _lower(text) in _IGNORED_RESPONSES:
        return []
    return [text]


def _has_value(state: Optional[dict], key: str) -> bool:
    if not isinstance(state, dict):
        return False
    return bool(_value_items(state.get(key)))


def _changed_fields(
    before: Optional[dict], after: Optional[dict]
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Return ``(added, updated)`` across all canonical fields.

    ``added``  : field key -> list of new list items (or the new scalar) that
                 were not present before.
    ``updated``: set of scalar-ish fields whose value changed to another
                 non-empty value.
    """
    added: Dict[str, List[str]] = {}
    updated: Set[str] = set()
    if not isinstance(before, dict) or not isinstance(after, dict):
        return added, updated
    for key in _FIELD_KEYS:
        b_items = _value_items(before.get(key))
        a_items = _value_items(after.get(key))
        if not b_items and a_items:
            added[key] = a_items
        elif b_items and a_items:
            if key in _SCALAR_FIELDS:
                if a_items != b_items:
                    updated.add(key)
            else:
                new_items = [v for v in a_items if v not in b_items]
                if new_items:
                    added[key] = new_items
        elif b_items and not a_items and key in _SCALAR_FIELDS:
            # not counted (a reset is an update, but we stay observation-only
            # and only flag forward movement)
            continue
    return added, updated


def _target_field(reply: str) -> Optional[str]:
    """Which field the mentor reply appears to be targeting (keyword-based).

    Returns ``None`` when the reply does not clearly target any single field.
    """
    if not reply:
        return None
    reply_words = _content_words(reply)
    if not reply_words:
        return None
    lower_reply = _lower(reply)
    best_field: Optional[str] = None
    best_score = 1  # require at least one keyword hit
    for key in _FIELD_KEYS:
        score = 0
        for kw in _FIELD_KEYWORDS.get(key, ()):
            kw_stripped = kw.strip()
            if kw_stripped in reply_words:
                score += 1
            elif kw_stripped in lower_reply:
                score += 1
        if score > best_score:
            best_field = key
            best_score = score
    return best_field


# ---------------------------------------------------------------------------
# Reply-shape helpers
# ---------------------------------------------------------------------------

_QUESTION_STARTS: Tuple[str, ...] = (
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "could you", "would you", "can you", "do you", "does any",
)

_QUESTION_MID_TERMS: Tuple[str, ...] = (
    " could you", " can you", " would you", " please describe",
    " give me", " tell me", " say more", " talk to me", " how often",
    " what do", " what is", " what are", " what did", " what does",
    " why do", " who are", " who is", " an example", "?",
)


def _is_question(reply: str) -> bool:
    """Cheap deterministic question detection (mirrors style-audit intent).

    A reply is a question if it starts with a question word, uses a
    requestive phrase, or ends in ``?``.
    """
    text = (reply or "").strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    lowered = _lower(text)
    for start in _QUESTION_STARTS:
        if lowered.startswith(start):
            return True
    for mid in _QUESTION_MID_TERMS:
        if mid in lowered:
            return True
    return False


# A reply is "leading"/"open" when it answers-the-question-itself before
# asking — i.e. it states a candidate the user must confirm.
_LEADING_MARKERS: Tuple[str, ...] = (
    "you might", "you could", "maybe you", "perhaps you", "you probably",
    "e.g.", "for example", "such as", "like perhaps", "would that",
)


# ---------------------------------------------------------------------------
# ConversationFailureRecord
# ---------------------------------------------------------------------------


@dataclass
class ConversationFailureRecord:
    """One observation-only failure classification for a single turn."""

    primary_failure: ConversationFailure = ConversationFailure.NONE
    secondary_failures: List[ConversationFailure] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_failure": self.primary_failure.value if isinstance(
                self.primary_failure, ConversationFailure
            ) else str(self.primary_failure),
            "secondary_failures": [
                f.value if isinstance(f, ConversationFailure) else str(f)
                for f in self.secondary_failures
            ],
            "confidence": round(float(self.confidence or 0.0), 2),
            "reason": self.reason,
            "signals": list(self.signals),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ConversationFailureRecord":
        d = d or {}
        primary = ConversationFailure.NONE
        try:
            primary = ConversationFailure(d.get("primary_failure") or "NONE")
        except (ValueError, KeyError):
            primary = ConversationFailure.NONE
        secondary: List[ConversationFailure] = []
        for value in list(d.get("secondary_failures", []) or []):
            try:
                secondary.append(ConversationFailure(str(value)))
            except (ValueError, KeyError):
                continue
        return cls(
            primary_failure=primary,
            secondary_failures=secondary,
            confidence=round(float(d.get("confidence", 0.0) or 0.0), 2),
            reason=str(d.get("reason", "") or ""),
            signals=list(d.get("signals", []) or []),
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _confidence_for(primary: ConversationFailure, **ctx: Any) -> float:
    move = ctx.get("move")
    target_field = ctx.get("target_field")
    updated = bool(ctx.get("updated_fields"))
    if primary == ConversationFailure.NONE:
        return 0.98
    if primary == ConversationFailure.MOVE_SELECTION:
        return 0.62
    if primary in (
        ConversationFailure.REPEATED_INFORMATION,
        ConversationFailure.MISUNDERSTOOD_RESPONSE,
        ConversationFailure.ABRUPT_TRANSITION,
    ):
        return 0.60 + (0.25 if updated else 0.0)
    if primary in (
        ConversationFailure.OVER_EXPLORATION,
        ConversationFailure.MISSED_INSIGHT,
    ):
        return 0.65 + (0.20 if target_field else 0.0)
    if primary == ConversationFailure.UNSUPPORTED_INFERENCE:
        return 0.80
    if primary in (
        ConversationFailure.MEMORY_FAILURE,
        ConversationFailure.SOCIAL_FAILURE,
    ):
        return 0.70
    if primary == ConversationFailure.META_INTENT:
        return 0.72
    return 0.60


def analyze_conversation_failure(
    *,
    user_message: str = "",
    mentor_reply: str = "",
    objective: str = "",
    project_state_before: Optional[dict] = None,
    project_state_after: Optional[dict] = None,
    conversation_memory: Optional[dict] = None,
    conversation_move: Optional[str] = None,
    coaching_strategy: Optional[str] = None,
    question_family: Any = None,
    style_audit: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> ConversationFailureRecord:
    """Deterministic, observation-only conversation-failure classification.

    Takes the runtime data the mentor already produced for this turn and
    returns a ``ConversationFailureRecord``. It mutates nothing and is safe
    to call on the golden fixtures / ReplayLab without changing any other
    output.
    """
    user_text = (user_message or "").strip()
    reply_text = (mentor_reply or "").strip()
    moves_ = (conversation_move or "").strip()
    move = moves_.upper() if moves_ and moves_.upper() not in {
        "PLACEHOLDER", "NONE", "UNKNOWN"
    } else moves_

    memory = conversation_memory or {}
    mem_open = {r["field"] for r in memory.get("open_threads", []) or []
                if isinstance(r, dict) and r.get("field")}
    mem_resolved = {r["field"] for r in memory.get("resolved_threads", []) or []
                    if isinstance(r, dict) and r.get("field")}
    mem_deferred = {r["field"] for r in memory.get("deferred_topics", []) or []
                    if isinstance(r, dict) and r.get("field")}
    mem_acknowledged = {r["field"] for r in memory.get("acknowledged_facts", []) or []
                        if isinstance(r, dict) and r.get("field")}
    mem_summarized = {r["field"] for r in memory.get("summarized_facts", []) or []
                      if isinstance(r, dict) and r.get("field")}

    added, updated_fields = _changed_fields(project_state_before, project_state_after)
    known_before = {k for k in _FIELD_KEYS if _has_value(project_state_before, k)}
    known_after = {k for k in _FIELD_KEYS if _has_value(project_state_after, k)}

    current_field = _family_field(question_family)
    target_field = _target_field(reply_text)

    user_words = _content_words(user_text)
    reply_words = _content_words(reply_text)
    lower_user = _lower(user_text)
    lower_reply = _lower(reply_text)

    question = _is_question(reply_text)
    answered = bool(added or updated_fields)

    user_uncertain = _contains_any(user_text, _UNCERTAINTY_PHRASES)
    user_hedged = bool(user_uncertain or (user_words & _HEDGE_WORDS))
    user_deferring = _contains_any(user_text, _DEFERRING_PHRASES)
    user_substantive_but_vague = user_hedged or user_deferring

    signals: List[str] = []
    candidates: List[ConversationFailure] = []

    # -- 1. META_INTENT -----------------------------------------------------
    if not user_text:
        signals.append("User turn was empty.")
        candidates.append(ConversationFailure.META_INTENT)
    elif len(user_words) < 3 and not answered and not user_substantive_but_vague:
        # A very short turn that added no project-state value is a validator /
        # acknowledgement ("thanks", "OK"), not a design contribution. A
        # hedged / deferring vague answer is a genuine (if weak) answer, not
        # meta — the recovery machinery already handles those.
        signals.append(
            "User turn was meta/non-substantive (short, no project-state value added)."
        )
        candidates.append(ConversationFailure.META_INTENT)
    elif _contains_any(user_text, _TRUST_MARKERS):
        # "I trust you", "you're right" — the user deferred rather than giving
        # design content. Deferral signals the mentor may have over-stepped.
        signals.append("User deferred to the mentor (trust/validation marker).")
        candidates.append(ConversationFailure.META_INTENT)

    # -- 2. MOVE_SELECTION --------------------------------------------------
    if not moves_:
        signals.append("No conversation move was recorded for this turn.")
        candidates.append(ConversationFailure.MOVE_SELECTION)
    elif move not in {
        "ELICIT_INFORMATION", "EXPAND_IDEA", "CONNECT_INFORMATION",
        "VALIDATE_DISCOVERY", "CHALLENGE_ASSUMPTION", "SUMMARIZE_PROGRESS",
        "TRANSITION_TOPIC", "CLOSE_TOPIC",
    }:
        signals.append(f"Unrecognised conversation move: {conversation_move!r}.")
        candidates.append(ConversationFailure.MOVE_SELECTION)
    elif move == "ELICIT_INFORMATION" and not question:
        # Chose to elicit information but shipped no question.
        signals.append(
            "ELICIT_INFORMATION move selected but the reply contains no question."
        )
        candidates.append(ConversationFailure.MOVE_SELECTION)

    # -- 3. REPEATED_INFORMATION ---------------------------------------------
    for acknowledged_field in sorted(mem_acknowledged):
        if acknowledged_field in added:
            signals.append(
                f"User re-offered content for acknowledged field "
                f"'{acknowledged_field}' after it was acknowledged."
            )
            candidates.append(ConversationFailure.REPEATED_INFORMATION)
    for summarized_field in sorted(mem_summarized):
        if summarized_field in added:
            signals.append(
                f"User re-offered content for summarized field "
                f"'{summarized_field}' after the summary."
            )
            candidates.append(ConversationFailure.REPEATED_INFORMATION)

    # -- 4. MEMORY_FAILURE ----------------------------------------------------
    contradictory = sorted(mem_acknowledged & mem_open)
    if contradictory:
        signals.append(
            "Field(s) acknowledged as ready-to-transition yet still listed as "
            "open threads: " + ", ".join(_field_label(f) for f in contradictory) + "."
        )
        candidates.append(ConversationFailure.MEMORY_FAILURE)
    drifted_resolved = sorted(
        f for f in mem_resolved
        if isinstance(f, str) and f in _FIELD_KEYS and not _has_value(project_state_after, f)
    )
    if drifted_resolved:
        signals.append(
            "Field(s) marked resolved in memory but absent from project state: "
            + ", ".join(_field_label(f) for f in drifted_resolved) + "."
        )
        candidates.append(ConversationFailure.MEMORY_FAILURE)

    # -- 5. MISUNDERSTOOD_RESPONSE -------------------------------------------
    # Mentor (re)asks for the same first-time information a field already has.
    # A field the memory already resolved is handled by OVER_EXPLORATION (the
    # mentor should have moved on), not by MISUNDERSTOOD.
    if move in {"ELICIT_INFORMATION", "CHALLENGE_ASSUMPTION"}:
        if question and target_field and target_field in known_after and \
                target_field not in mem_resolved:
            signals.append(
                f"Mentor re-asks field '{target_field}' that already has a "
                "usable value in project state."
            )
            candidates.append(ConversationFailure.MISUNDERSTOOD_RESPONSE)

    # -- 6. OVER_EXPLORATION -------------------------------------------------
    # Mentor keeps exploring a field the conversational memory already marked
    # resolved (it should have moved on to a new thread).
    if move in {"ELICIT_INFORMATION", "EXPAND_IDEA"}:
        if target_field and target_field in mem_resolved:
            signals.append(
                f"Mentor re-explores field '{target_field}' that is already resolved."
            )
            candidates.append(ConversationFailure.OVER_EXPLORATION)

    # -- 7. UNSUPPORTED_INFERENCE ---------------------------------------------
    if not user_uncertain:
        if "frequency" not in known_before and "frequency" not in known_after:
            if _contains_any(reply_text, _UNSUPPORTED_MAGNITUDE_TERMS):
                if current_field != "frequency" and target_field != "frequency":
                    signals.append(
                        "Mentor implied a magnitude/frequency with no supporting "
                        "evidence in project state."
                    )
                    candidates.append(ConversationFailure.UNSUPPORTED_INFERENCE)

    # -- 8. ABRUPT_TRANSITION --------------------------------------------------
    if move == "TRANSITION_TOPIC":
        if current_field and current_field not in mem_acknowledged and \
                current_field not in mem_resolved:
            signals.append(
                "Mentor moved to a new topic before acknowledging/resolving the "
                "previously pursued field."
            )
            candidates.append(ConversationFailure.ABRUPT_TRANSITION)
        elif not answered and not mem_acknowledged:
            signals.append(
                "Mentor changed topic without an answer or acknowledgment for the "
                "current thread."
            )
            candidates.append(ConversationFailure.ABRUPT_TRANSITION)
    elif move in {"SUMMARIZE_PROGRESS", "CLOSE_TOPIC"}:
        if not mem_acknowledged and not mem_resolved and not mem_summarized:
            signals.append(
                "Mentor summarised/closed the session with no acknowledged, "
                "resolved, or summarized content."
            )
            candidates.append(ConversationFailure.ABRUPT_TRANSITION)

    # -- 9. MISSED_INSIGHT -----------------------------------------------------
    if current_field and current_field in known_after and current_field not in mem_acknowledged:
        if not mem_open and not mem_resolved:
            signals.append(
                "Project state holds a usable value for "
                f"'{current_field}' but conversational memory recorded no "
                "acknowledged/resolved/open thread for it."
            )
            candidates.append(ConversationFailure.MISSED_INSIGHT)

    # -- 10. SOCIAL_FAILURE -----------------------------------------------------
    if move != "CHALLENGE_ASSUMPTION":
        if _contains_any(
            reply_text,
            (
                "you are wrong", "you're wrong", "that is wrong",
                "that's wrong", "that is incorrect", "that's incorrect",
                "you are incorrect", "that's not right", "that is not right",
                "dont confuse", "don't confuse", "that doesn't matter",
                "that does not matter",
            ),
        ):
            signals.append(
                "Mentor reprimanded the user (explicit negative judgment) "
                "outside an intentional challenge turn."
            )
            candidates.append(ConversationFailure.SOCIAL_FAILURE)

    # -- Primary selection ------------------------------------------------------
    primary = candidates[0] if candidates else ConversationFailure.NONE
    secondary: List[ConversationFailure] = []
    for cat in candidates[1:]:
        if cat != primary and cat not in secondary:
            secondary.append(cat)

    reasons: List[str] = []
    if primary == ConversationFailure.NONE:
        reasons.append("No failure observed this turn.")
    else:
        reasons.append(f"{failure_label(primary)} detected.")
        if secondary:
            reasons.append(
                "Also: " + ", ".join(failure_label(c) for c in secondary) + "."
            )
        if not signals:
            signals.append("Signal list empty (see primary/secondary).")

    confidence = _confidence_for(
        primary,
        move=move,
        target_field=target_field,
        updated_fields=updated_fields,
    )

    return ConversationFailureRecord(
        primary_failure=primary,
        secondary_failures=secondary,
        confidence=round(float(confidence), 2),
        reason=" ".join(reasons),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Developer Console per-turn section
# ---------------------------------------------------------------------------


def conversation_failure_diagnostics_section(record: Optional[
    "ConversationFailureRecord"
]) -> dict:
    """Developer Console projection of one turn's failure record.

    Returns an empty dict when no record is available this turn.
    """
    if not record:
        return {}
    primary = record.primary_failure
    primary_label = (
        failure_label(primary)
        if primary != ConversationFailure.NONE
        else "None"
    )
    secondary = record.secondary_failures
    section: Dict[str, Any] = {
        "Primary Failure": primary_label,
        "Secondary Failures": (
            ", ".join(failure_label(c) for c in secondary) if secondary else "None"
        ),
        "Confidence": f"{record.confidence:.2f}",
        "Reason": record.reason,
    }
    if record.signals:
        section["Signals"] = record.signals
    return section


# ---------------------------------------------------------------------------
# ConversationFailureSummary
# ---------------------------------------------------------------------------


@dataclass
class ConversationFailureSummary:
    """Append-only aggregate of per-turn conversation-failure records.

    Persists with the session (JSON-safe). Never read by any decision path —
    Developer Console + offline report only.
    """

    turn_records: List[dict] = field(default_factory=list)

    # ---- aggregation -------------------------------------------------------

    def add_record(self, record: Any) -> None:
        """Append a record (ConversationFailureRecord or dict / None)."""
        if not record:
            return
        if isinstance(record, ConversationFailureRecord):
            self.turn_records.append(record.to_dict())
        elif isinstance(record, dict):
            self.turn_records.append(ConversationFailureRecord.from_dict(record).to_dict())
        else:
            # tolerate unknown types by refusing to store non-JSON values
            return

    def merge(self, other: "ConversationFailureSummary") -> None:
        self.turn_records.extend(other.turn_records)

    # ---- metrics -----------------------------------------------------------

    def total_turns(self) -> int:
        return len(self.turn_records)

    def total_failures(self) -> int:
        return sum(
            1
            for r in self.turn_records
            if not isinstance(r, dict)
            or ConversationFailureRecord.from_dict(r).primary_failure
            != ConversationFailure.NONE
        )

    def failure_counts(self) -> Dict[str, int]:
        """Counts across every failure category that fired (primary + secondary),
        excluding NONE."""
        counter: Counter = Counter()
        for r in self.turn_records:
            rec = ConversationFailureRecord.from_dict(r)
            categories = [rec.primary_failure] + list(rec.secondary_failures)
            for cat in set(categories):
                if cat != ConversationFailure.NONE:
                    counter[cat.value] += 1
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    def failure_rate(self) -> float:
        turns = self.total_turns()
        if not turns:
            return 0.0
        return round(self.total_failures() / turns, 2)

    def most_common_failure(self) -> Optional[str]:
        counts = self.failure_counts()
        if not counts:
            return None
        top = max(counts.items(), key=lambda kv: kv[1])[0]
        try:
            return failure_label(ConversationFailure(top))
        except (ValueError, KeyError):
            return top

    def per_failure_percentages(self) -> Dict[str, float]:
        turns = self.total_turns()
        counts = self.failure_counts()
        if not turns:
            return {}
        return {
            category: round((count / turns) * 100.0, 1)
            for category, count in counts.items()
        }

    # ---- persistence / display ----------------------------------------------

    def to_dict(self) -> dict:
        return {"turn_records": list(self.turn_records)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ConversationFailureSummary":
        d = d or {}
        return cls(
            turn_records=list(d.get("turn_records", []) or []),
        )

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        counts = self.failure_counts()
        percents = self.per_failure_percentages()
        display: Dict[str, Any] = {
            "Turns Analyzed": self.total_turns(),
            "Turns with Failures": self.total_failures(),
            "Turn Failure Rate": f"{self.failure_rate():.0%}",
            "Most Common Failure": self.most_common_failure() or "None",
        }
        if counts:
            display["Failure Category Counts"] = dict(counts)
        if percents:
            display["Failure Category Percentages"] = dict(percents)
        return display