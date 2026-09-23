"""
extraction_accuracy — measurement-only audit of LLM extraction accuracy.

For every turn where the LLM extractor actually ran, this module classifies
each proposed ``ExtractionUpdate`` into exactly one label:

  * ``CORRECT``          — a new, non-duplicate fact placed in the right
                           StateField with a specific-enough value.
  * ``PARTIALLY_CORRECT`` — a real fact, but imperfect: the value is too
                           generic, or the value was already known
                           (``Duplicate``).
  * ``INCORRECT``         — the value belongs to a different StateField than
                           the one proposed. The reason names the intended
                           field when it is one of the audited targets.
  * ``MISSED_OPPORTUNITY`` — the update captured only part of the message:
                           the user's message carried another fact for the
                           same field (explicit or implicit) that this
                           update, and every other update for that field,
                           failed to capture.

Measurement ONLY. This module:

  * NEVER changes what is applied to ``ProjectState``,
  * NEVER affects objectives, lifecycle, prompts, or replies,
  * only appears in Developer Console diagnostics and a session-level
    aggregate (``ExtractionAccuracy`` / ``ExtractionAccuracySummary``).

The audit reuses the deterministic rule extractor
(``RuleBasedExtractor.extract``) as a *proxy* for "facts the user message
explicitly contains", then checks whether the LLM's updates for each field
covered them. It is a heuristic for measurement, never a decision input.

This module deliberately imports only cycle-free leaves of the dependency
graph (``memory_extractor``, ``extraction_pipeline``,
``extraction_comparison``) — never ``mentor``.
"""

import re

from memory_extractor import LIST_FIELDS, MessageType
from extraction_comparison import _RULE_LEGACY_TO_STATE, field_label

__all__ = [
    "CORRECT",
    "PARTIALLY_CORRECT",
    "INCORRECT",
    "MISSED_OPPORTUNITY",
    "REASON_WRONG_FIELD",
    "REASON_TOO_GENERIC",
    "REASON_DUPLICATE",
    "REASON_MISSED_EXPLICIT",
    "REASON_MISSED_IMPLICIT",
    "ExtractionAccuracySummary",
    "assess_extraction_accuracy",
    "classify_update",
]

CORRECT = "Correct"
PARTIALLY_CORRECT = "Partially Correct"
INCORRECT = "Incorrect"
MISSED_OPPORTUNITY = "Missed Opportunity"

# Allowed reason keys (see PR scope).
REASON_WRONG_FIELD = "Wrong field"
REASON_TOO_GENERIC = "Too generic"
REASON_SHOULD_EVIDENCE = "Should have been evidence"
REASON_SHOULD_IMPACT = "Should have been impact"
REASON_SHOULD_MOTIVATION = "Should have been motivation"
REASON_SHOULD_PAIN_POINT = "Should have been pain point"
REASON_DUPLICATE = "Duplicate"
REASON_MISSED_EXPLICIT = "Missed explicit fact"
REASON_MISSED_IMPLICIT = "Missed implicit fact"

# Report buckets for per-field precision (the 8 labels required by the PR).
# ``Motivation`` is a sub-bucket of ``pain_points``: updates whose value
# carries builder-motivation signals ("I want to help …" → pain_points).
_REPORT_FIELDS: tuple[tuple[str, str], ...] = (
    ("personas", "Audience"),
    ("problems", "Problems"),
    ("pain_points", "Pain"),
    ("pain_points_motivation", "Motivation"),
    ("evidence", "Evidence"),
    ("frequency", "Frequency"),
    ("current_solutions", "Current Solution"),
    ("impacts", "Impact"),
)


# ---------------------------------------------------------------------------
# Keyword signal sets — deterministic proxies for "this text is about X".
# ---------------------------------------------------------------------------

_EVIDENCE_SIGNALS = (
    "i saw", "i have seen", "have seen", "i observed", "observed", "witnessed",
    "i interviewed", "interviewed", "interview", "research shows", "research",
    "study shows", "studies show", "studies", "data shows", "data", "statistics",
    "statistic", "survey", "surveys", "personally experienced", "my grandmother",
    "my grandfather", "my father", "my mother", "my friend", "my sister",
    "my brother", "my aunt", "my uncle", "percent", "%", "in my experience",
)

_PAIN_SIGNALS = (
    "stress", "stressed", "stressful", "anxious", "anxiety", "frustrated",
    "frustration", "frustrating", "overwhelmed", "overwhelming", "worried",
    "worry", "worries", "scared", "scary", "fear", "afraid", "tired of",
    "exhausted", "hate", "hates", "burnout", "burned out", "guilt", "guilty",
    "embarrassed", "embarrassment", "helpless", "hopeless", "demotivated",
    "painful", "panic", "dread", "anxious", "nervous", "upset", "angry",
    "feel", "feels", "feeling", "felt", "struggling", "fed up",
)

_IMPACT_SIGNALS = (
    "causes", "caused", "cause", "leads to", "lead to", "led to", "result in",
    "results in", "resulted in", "consequence", "consequences", "impact",
    "affects", "affected", "as a result", "lost marks", "lose marks",
    "losing marks", "missed deadlines", "miss deadlines", "wasted time",
    "wastes time", "health risk", "health risks", "medical", "costs money",
    "costs them", "dropped out", "fail classes", "failing classes",
)

_MOTIVATION_SIGNALS = (
    "want to", "wants to", "would like to", "hoping to", "hope to", "hopes to",
    "goal is", "my goal", "our goal", "aim", "aiming to", "trying to",
    "wish to", "i want", "i'd love", "would love to", "purpose", "mission",
    "help people", "help them", "helps people", "make people", "build a",
    "build an", "create a", "develop a", "design a", "for them to",
)

_PROBLEM_SIGNALS = (
    "forget", "forgets", "forgetting", "forgot", "miss", "misses", "missing",
    "late", "deadline", "deadlines", "difficulty", "difficult", "can't",
    "cannot", "unable", "struggle", "struggles", "struggling", "problem",
    "problems", "challenge", "challenges", "issue", "issues", "fail", "fails",
    "failing", "failed", "procrastinat", "overload", "buried", "overwhelming",
    "non-adherence", "doesn't work", "does not work", "stop taking",
    "skipping", "skips", "fall behind", "falling behind",
)

_PERSONA_SIGNALS = (
    "students", "student", "elderly", "seniors", "senior citizens", "old people",
    "people", "children", "kids", "doctors", "doctor", "nurses", "teachers",
    "teacher", "parents", "workers", "employees", "teenagers", "caregivers",
    "freshmen", "college", "patients", "customer", "customers", "users",
    "user", "clients", "grandmother", "grandparents", "families", "family",
)

_SOLUTION_SIGNALS = (
    "use", "uses", "using", "used to", "sticky notes", "calendar", "alarm",
    "alarms", "reminder", "reminders", "notebook", "excel", "spreadsheet",
    "app", "tool", "tools", "workaround", "workarounds", "manual", "paper",
    "whatsapp", "notion", "trello", "post-it", "post it", "phone", "planner",
    "planners", "schedule", "scheduling", "checkbox", "notepad", "google",
)

_FREQUENCY_SIGNALS = (
    "daily", "weekly", "monthly", "yearly", "often", "rarely", "always",
    "never", "sometimes", "every day", "every single day", "every morning",
    "every week", "every month", "once a", "twice a", "multiple times",
    "several times", "constantly", "occasionally", "frequently", "regularly",
    "all the time", "most of the time",
)

# Values/patterns that are too generic to be a useful capture.
_GENERIC_VALUES = frozenset({
    "problem", "problems", "the problem", "the problem is", "issue", "issues",
    "help", "helps", "helping", "people", "things", "thing", "something",
    "everything", "stuff", "it", "it is hard", "it's hard", "it is difficult",
    "it's difficult", "difficult", "hard", "bad", "worse", "struggle",
    "struggles", "a lot", "more", "less", "etc", "etc.", "solution",
    "solutions", "the solution", "a tool", "an app", "app", "system",
    "general", "overall", "it doesn't work", "doesn't work", "no idea",
})

_GENERIC_PATTERNS = (
    re.compile(r"^(the |a |an )?(problem|issue|challenge|struggle)s?$"),
    re.compile(r"^(the |a |an )?(solution|workaround)s?$"),
    re.compile(r"^(it('s| is| was)? )?(hard|difficult|tough|bad|worse|stressful)s?$"),
    re.compile(r"^(they|people|everyone|some people|many people)s?$"),
    re.compile(r"^(helps?|support(s|ed)?|assist(s|ed)?)"),
    re.compile(r"^(make|making|want|need|has|have|get|getting|find|found)"),
    re.compile(r"^etc(etera)?\.?$"),
    re.compile(r"^\w{1,2}$"),
)

# Evidence/pain signals that mark an observation phrase as a specific field
# when they appear in an extracted *value* (used by ``_detect_field_from_value``).
_SPECIFIC_FIELD_SIGNALS = {
    "evidence": _EVIDENCE_SIGNALS,
    "pain_points": _PAIN_SIGNALS,
    "impacts": _IMPACT_SIGNALS,
    "current_solutions": _SOLUTION_SIGNALS,
    "problems": _PROBLEM_SIGNALS,
    "personas": _PERSONA_SIGNALS,
    "frequency": _FREQUENCY_SIGNALS,
}


def _norm(s: str) -> str:
    """Normalize a string for comparison: lower, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _clean(v):
    """Mirror of ``extraction_pipeline.clean_val`` (empty -> None). Kept local
    so this module never imports ``extraction_pipeline`` (which imports
    ``session_manager`` and would create an import cycle)."""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("null", "none", ""):
        return None
    return s


def _signal_hits(text: str, signals) -> int:
    """Number of distinct keyword signals present in ``text`` (word-boundary
    matched, so ``student`` does not count inside ``students``)."""
    t = text.lower()
    n = 0
    for sig in signals:
        if sig not in t:
            continue
        rx = _SIGNAL_PATTERNS.get(sig)
        if rx is None:
            rx = re.compile(r"\b" + re.escape(sig) + r"\b")
            _SIGNAL_PATTERNS[sig] = rx
        if rx.search(t):
            n += 1
    return n


_SIGNAL_PATTERNS: dict[str, re.Pattern] = {}


# ---------------------------------------------------------------------------
# Per-update classifier
# ---------------------------------------------------------------------------


def _is_duplicate(update, state_before: dict) -> bool:
    """True when the same normalized value is already in ``state_before``."""
    field = update.field.value
    val = _norm(update.value)
    if not val:
        return False
    existing = state_before.get(field) if isinstance(state_before, dict) else None
    if update.field in LIST_FIELDS:
        return any(_norm(v) == val for v in (existing or []))
    return _norm(existing or "") == val


def _detect_field_from_value(value: str):
    """Return ``(true_field_value_or_None, reason_or_None)`` for a value that
    clearly signals a different StateField than the one it was placed under.

    ``reason`` names the intended field when it is one of the audited targets
    (evidence / impact / motivation / pain point); otherwise ``Wrong field``.
    Returns ``(None, None)`` when the value is ambiguous (tie or no signal).
    """
    mot_hits = _signal_hits(value, _MOTIVATION_SIGNALS)
    hits: dict[str, int] = {}
    for field, signals in _SPECIFIC_FIELD_SIGNALS.items():
        n = _signal_hits(value, signals)
        if n:
            hits[field] = n

    # A motivation-flavoured value ("I want to help …") belongs to
    # pain_points per the extraction contract — unless a stronger concrete
    # field signal (evidence/impact/solution/…) identifies it instead.
    if mot_hits and "pain_points" not in hits:
        strong = {f: n for f, n in hits.items() if f != "personas"}
        if not strong:
            return "pain_points", REASON_SHOULD_MOTIVATION

    if not hits:
        return None, None
    best_field = max(hits, key=lambda f: (hits[f], -list(_SPECIFIC_FIELD_SIGNALS).index(f)))
    best_count = hits[best_field]
    ties = [f for f, n in hits.items() if n == best_count]
    if len(ties) > 1:
        # Multiple fields tie -> ambiguous, do not call it wrong.
        return None, None
    return best_field, _reason_for_true_field(best_field, value)


def _reason_for_true_field(true_field: str, value: str) -> str:
    if true_field == "evidence":
        return REASON_SHOULD_EVIDENCE
    if true_field == "impacts":
        return REASON_SHOULD_IMPACT
    if true_field == "pain_points":
        # A motivation-flavoured value that landed elsewhere was a "should
        # have been motivation" error; an emotional-state value was a "should
        # have been pain point" error.
        if _signal_hits(value, _MOTIVATION_SIGNALS) > 0:
            return REASON_SHOULD_MOTIVATION
        return REASON_SHOULD_PAIN_POINT
    return REASON_WRONG_FIELD


def _is_generic(value: str) -> bool:
    v = _norm(value)
    if v in _GENERIC_VALUES:
        return True
    return any(p.search(v) for p in _GENERIC_PATTERNS)


def _is_motivation_domain(value: str) -> bool:
    """Whether a pain_points-domain value reads as builder motivation."""
    return _signal_hits(value, _MOTIVATION_SIGNALS) > 0


def classify_update(update, state_before: dict) -> tuple[str, list[str]]:
    """Deterministic per-update classification.

    Returns ``(label, [reasons])``. Pure function — reads only ``update``
    and ``state_before``; mutates nothing.
    """
    if _is_duplicate(update, state_before):
        return PARTIALLY_CORRECT, [REASON_DUPLICATE]

    true_field, reason = _detect_field_from_value(update.value)
    if true_field is not None and true_field != update.field.value:
        return INCORRECT, [reason or REASON_WRONG_FIELD]

    if _is_generic(update.value):
        return PARTIALLY_CORRECT, [REASON_TOO_GENERIC]

    return CORRECT, []


# ---------------------------------------------------------------------------
# Message-level "missed fact" detection
# ---------------------------------------------------------------------------


def _rule_facts(rule_observation: dict) -> dict[str, list[str]]:
    """Map the deterministic rule extractor's output onto StateField values
    (the same legacy->StateField mapping ``merge_extracted_to_state`` uses).
    Used ONLY as a measurement proxy for facts the message explicitly states."""
    out: dict[str, list[str]] = {}
    for legacy, sf in _RULE_LEGACY_TO_STATE.items():
        val = _clean((rule_observation or {}).get(legacy))
        if val:
            out.setdefault(sf.value, []).append(val)
    return out


def _tokens(s: str) -> set[str]:
    return {w for w in re.split(r"\W+", _norm(s)) if w}


def _covers(fact_value: str, llm_values: list[str]) -> bool:
    """Whether any LLM value for the same field plausibly covers ``fact_value``
    (normalized equality, substring containment either way, or 50% token overlap)."""
    f = _norm(fact_value)
    if not f:
        return False
    ft = _tokens(f)
    for raw in llm_values:
        v = _norm(raw)
        if not v:
            continue
        if v == f or v in f or f in v:
            return True
        vt = _tokens(v)
        if not vt:
            continue
        jac = len(ft & vt) / len(ft | vt)
        if jac >= 0.5:
            return True
    return False


def _detect_missed_facts(
    user_message: str,
    rule_observation: dict,
    field_values: dict[str, list[str]],
) -> list[dict]:
    """Facts the message carries (per the deterministic rule extractor proxy)
    that the LLM updates did not capture.

    ``field_values`` maps a StateField value string to the list of LLM
    extracted values for that field. Returns a list of
    ``{"field", "fact", "kind"}`` records. ``kind`` is ``explicit`` for facts
    the rule extractor surfaced directly and ``implicit`` for signal-derived
    facts (frequency / personas / pain / evidence) the rules missed.
    """
    missed: list[dict] = []
    llm_by_field: dict[str, list[str]] = dict(field_values or {})

    covered_fields = {f for f, vals in llm_by_field.items()}

    # Explicit facts from the deterministic rule extractor.
    for field, facts in _rule_facts(rule_observation).items():
        for fact in facts:
            if not _covers(fact, llm_by_field.get(field, [])):
                missed.append({"field": field, "fact": fact, "kind": "explicit"})

    # Implicit facts from keyword signals, only for fields the rules did not
    # already flag (avoid double-reporting the same miss).
    explicit_fields = {m["field"] for m in missed}
    lower = _norm(user_message)

    implicit_rules = (
        ("frequency", _FREQUENCY_SIGNALS),
        ("personas", _PERSONA_SIGNALS),
        ("pain_points", _PAIN_SIGNALS),
        ("evidence", _EVIDENCE_SIGNALS),
    )
    for field, signals in implicit_rules:
        if field in explicit_fields or field in covered_fields:
            continue
        if _signal_hits(lower, signals) > 0:
            missed.append({"field": field, "fact": field, "kind": "implicit"})

    return missed


# ---------------------------------------------------------------------------
# Per-turn assessment
# ---------------------------------------------------------------------------


def assess_extraction_accuracy(
    user_message: str,
    objective: str | None,
    extraction_result,
    state_before: dict,
    rule_observation: dict | None = None,
) -> dict:
    """Classify every update of one LLM extraction result.

    Parameters
    ----------
    user_message : the user's raw message (kept in the record for the console).
    objective : the objective value the mentor was pursuing at decision time
                (``hybrid_decision["objective"]``) — recorded, never used to
                classify.
    extraction_result : the typed ``ExtractionResult`` the LLM produced.
    state_before : pre-extraction state dict (for duplicate detection).
    rule_observation : the deterministic rule extractor's output for the same
                       message (used as the explicit-fact proxy). Optional.

    Returns a JSON-serialisable per-turn record:
    ``{llm_ran, message_type, objective, user_message, state_before,
        state_after (empty until the caller fills it), updates,
        missed_opportunities}``
    """
    updates = []
    if (
        extraction_result is not None
        and extraction_result.message_type == MessageType.MEANINGFUL
        and extraction_result.updates
    ):
        for u in extraction_result.updates:
            label, reasons = classify_update(u, state_before or {})
            entry = u.to_dict()
            entry["label"] = label
            entry["reasons"] = reasons
            if label == CORRECT and _is_motivation_domain(u.value):
                entry["domain"] = "motivation"
            updates.append(entry)

    missed: list[dict] = []
    if updates:
        field_values: dict[str, list[str]] = {}
        for e in updates:
            field_values.setdefault(e["field"], []).append(e["value"])
        missed = _detect_missed_facts(user_message, rule_observation or {}, field_values)

    # Attach each unattached missed fact to the first Correct update of the
    # same field (one Missed Opportunity label per missed-fact-field, keeping
    # the summary counts honest). Facts with no matching update stay turn-level.
    missed_opportunities: list[dict] = []
    for field_facts in _group_by_field(missed):
        field = field_facts["field"]
        fact_records = field_facts["facts"]
        candidate = next(
            (i for i, e in enumerate(updates)
             if e["field"] == field and e["label"] == CORRECT),
            None,
        )
        reasons = [REASON_MISSED_EXPLICIT if f["kind"] == "explicit" else REASON_MISSED_IMPLICIT
                   for f in fact_records]
        reason = (REASON_MISSED_EXPLICIT if any(f["kind"] == "explicit" for f in fact_records)
                  else REASON_MISSED_IMPLICIT)
        if candidate is not None:
            updates[candidate]["label"] = MISSED_OPPORTUNITY
            updates[candidate]["reasons"] = [reason]
            missed_opportunities.append({
                "field": field,
                "fact": [f["fact"] for f in fact_records],
                "kind": sorted({f["kind"] for f in fact_records}),
                "attached_to_update": candidate,
            })
        else:
            for f in fact_records:
                missed_opportunities.append({
                    "field": field,
                    "fact": f["fact"],
                    "kind": [f["kind"]],
                    "attached_to_update": None,
                })

    return {
        "llm_ran": True,
        "message_type": (
            extraction_result.message_type.value
            if extraction_result is not None else None
        ),
        "objective": objective,
        "user_message": user_message,
        "state_before": dict(state_before or {}),
        "state_after": {},
        "updates": updates,
        "missed_opportunities": missed_opportunities,
    }


def _group_by_field(missed: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for m in missed:
        grouped.setdefault(m["field"], []).append(m)
    return [
        {"field": f, "facts": facts}
        for f, facts in sorted(grouped.items())
    ]


# ---------------------------------------------------------------------------
# Session-level aggregate
# ---------------------------------------------------------------------------


class ExtractionAccuracySummary:
    """Append-only aggregate of per-turn accuracy assessments for a session.

    Persists with the session (JSON-safe). Never read by any decision path.
    Mirrors ``HybridAuditSummary`` in style and lifecycle.
    """

    def __init__(
        self,
        correct: int = 0,
        partially_correct: int = 0,
        incorrect: int = 0,
        missed: int = 0,
        field_counts: dict | None = None,
        field_correct: dict | None = None,
        reason_counts: dict | None = None,
        unattached_misses: int = 0,
        examples: list | None = None,
    ):
        self.correct = correct
        self.partially_correct = partially_correct
        self.incorrect = incorrect
        self.missed = missed
        self.field_counts: dict[str, int] = dict(field_counts or {})
        self.field_correct: dict[str, int] = dict(field_correct or {})
        self.reason_counts: dict[str, int] = dict(reason_counts or {})
        self.unattached_misses = unattached_misses
        self.examples: list[dict] = list(examples or [])

    def total(self) -> int:
        return self.correct + self.partially_correct + self.incorrect + self.missed

    def merge(self, other: "ExtractionAccuracySummary") -> None:
        """Combine another summary's counters into this one (used by the
        offline report runner to fold per-fixture session summaries)."""
        self.correct += other.correct
        self.partially_correct += other.partially_correct
        self.incorrect += other.incorrect
        self.missed += other.missed
        self.unattached_misses += other.unattached_misses
        for field, n in other.field_counts.items():
            self.field_counts[field] = self.field_counts.get(field, 0) + n
        for field, n in other.field_correct.items():
            self.field_correct[field] = self.field_correct.get(field, 0) + n
        for reason, n in other.reason_counts.items():
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + n
        self.examples.extend(other.examples)
        if len(self.examples) > 5:
            self.examples = self.examples[-5:]

    def add_record(self, record: dict) -> None:
        """Aggregate one per-turn accuracy record."""
        for entry in record.get("updates", []):
            self._add_entry(entry)
        self.unattached_misses += sum(
            1 for mo in record.get("missed_opportunities", [])
            if mo.get("attached_to_update") is None
        )
        if record.get("missed_opportunities"):
            self.examples.append({
                "message": (record.get("user_message") or "")[:200],
                "missed": [
                    {"field": m.get("field"), "fact": m.get("fact"),
                     "kind": m.get("kind")}
                    for m in record["missed_opportunities"][:3]
                ],
            })
            if len(self.examples) > 5:
                self.examples = self.examples[-5:]

    def _add_entry(self, entry: dict) -> None:
        label = entry.get("label", CORRECT)
        field = entry.get("field")
        self.field_counts[field] = self.field_counts.get(field, 0) + 1
        if entry.get("domain") == "motivation":
            self.field_counts["pain_points_motivation"] = (
                self.field_counts.get("pain_points_motivation", 0) + 1
            )

        if label == CORRECT:
            self.correct += 1
            self.field_correct[field] = self.field_correct.get(field, 0) + 1
            if entry.get("domain") == "motivation":
                self.field_correct["pain_points_motivation"] = (
                    self.field_correct.get("pain_points_motivation", 0) + 1
                )
        elif label == PARTIALLY_CORRECT:
            self.partially_correct += 1
        elif label == INCORRECT:
            self.incorrect += 1
        elif label == MISSED_OPPORTUNITY:
            self.missed += 1

        for reason in entry.get("reasons", []):
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

    def per_field_precision(self) -> dict[str, dict]:
        """Precision per report bucket: ``{label: {classified, correct, pct}}``."""
        out: dict[str, dict] = {}
        for field, label in _REPORT_FIELDS:
            classified = self.field_counts.get(field, 0)
            correct = self.field_correct.get(field, 0)
            out[label] = {
                "classified": classified,
                "correct": correct,
                "pct": round(correct / classified * 100, 1) if classified else 0.0,
            }
        return out

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "incorrect": self.incorrect,
            "missed": self.missed,
            "field_counts": dict(self.field_counts),
            "field_correct": dict(self.field_correct),
            "reason_counts": dict(self.reason_counts),
            "unattached_misses": self.unattached_misses,
            "examples": [dict(e) for e in self.examples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionAccuracySummary":
        d = d or {}
        return cls(
            correct=int(d.get("correct", 0)),
            partially_correct=int(d.get("partially_correct", 0)),
            incorrect=int(d.get("incorrect", 0)),
            missed=int(d.get("missed", 0)),
            field_counts=dict(d.get("field_counts", {}) or {}),
            field_correct=dict(d.get("field_correct", {}) or {}),
            reason_counts=dict(d.get("reason_counts", {}) or {}),
            unattached_misses=int(d.get("unattached_misses", 0)),
            examples=list(d.get("examples", []) or []),
        )

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        precision = self.per_field_precision()
        return {
            "Correct": self.correct,
            "Partially Correct": self.partially_correct,
            "Incorrect": self.incorrect,
            "Missed Opportunity": self.missed,
            "Turn-level missed facts": self.unattached_misses,
            "Per-field precision": {
                label: f"{info['correct']}/{info['classified']} "
                       f"({info['pct']}%)"
                for label, info in precision.items()
            },
            "Most common mistakes": dict(
                sorted(self.reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "Examples": [dict(e) for e in self.examples],
        }
