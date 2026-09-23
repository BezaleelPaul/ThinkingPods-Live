"""
conversation_context.py — deterministic Conversational Context Gate.

Architecture role (Phase 1: OBSERVATION-ONLY)
---------------------------------------------
Classifies what is happening in the conversation RIGHT NOW, before the
Objective Engine decides what Design Thinking question to pursue next.
Phase 1 changes NOTHING: the classifier result is recorded for the
Developer Console only. Phase 2 will let strong non-NORMAL modes
temporarily pause DT question steering via ``pause_dt`` /
``follow_user_lead``.

    user message ──► classify_conversation_context() ──► ConversationContext
                              │                          (observation only)
    existing pipeline ◄───────┘ unchanged

Design principles
-----------------
* Detect STRONG signals deterministically; preserve ambiguity otherwise.
* Signal-based detection (frames, combinations, context guards), NOT a
  giant phrase dictionary. Ordinary answers containing words such as
  "actually", "no", "what", "help", "don't", or "problem" stay NORMAL_DT
  unless a specific signal frame matches.
* Never force an interpretation: weak/odd input maps to AMBIGUOUS with
  LOW confidence rather than a speculative strong mode.
* Pure function: same inputs -> identical output. No LLM, no network,
  no filesystem, no mutation of ProjectState / ConversationMemory /
  ObjectiveContext.

Modes
-----
NORMAL_DT       Ordinary Design Thinking exchange (default).
CORRECTION      User rejects/corrects the mentor's premise or a prior read.
TOPIC_SHIFT     User explicitly abandons or replaces the current topic.
DIRECT_QUESTION User asks the mentor for help/advice/a solution.
CONFUSED        User cannot answer or does not understand the question.
HYPOTHETICAL    Explicit hypothetical/fictional/roleplay framing.
UNSAFE          Clearly unsafe request (conservative boundary signal only).
AMBIGUOUS       Odd/uninterpretable input — ambiguity preserved (LOW).

Reused infrastructure
---------------------
* ``module3.question_families.classify_question`` — identifies which
  family the previous mentor question belonged to (correction context).
* ``recovery_monitor._DONT_KNOW_PATTERNS`` — the existing deterministic
  "cannot answer" vocabulary (single source of truth; not duplicated).
* ``memory_extractor.StateField`` — field keys for the asked-field
  keyword combination used by the correction combo signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from memory_extractor import StateField
from module3 import classify_question, field_of
from recovery_monitor import _DONT_KNOW_PATTERNS

__all__ = [
    "ContextMode",
    "ContextConfidence",
    "ConversationContext",
    "classify_conversation_context",
    "conversation_context_diagnostics_section",
    "is_dt_paused",
    "context_instruction_bullet",
    "acknowledge_first_instruction_bullet",
    "paused_turn_fallback",
    "SAFETY_REFUSAL_REPLY",
    "is_unsafe_request",
    "strip_pivot_language",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContextMode(str, Enum):
    """The conversational situation detected for the current turn."""

    NORMAL_DT = "NORMAL_DT"
    """Ordinary Design Thinking exchange; no non-normal signal fired."""

    CORRECTION = "CORRECTION"
    """User rejects or corrects the mentor's premise / prior reading."""

    TOPIC_SHIFT = "TOPIC_SHIFT"
    """User explicitly abandons or replaces the current topic."""

    DIRECT_QUESTION = "DIRECT_QUESTION"
    """User directly asks the mentor for help / advice / a solution."""

    CONFUSED = "CONFUSED"
    """User cannot answer (DONT_KNOW) or does not understand the question."""

    HYPOTHETICAL = "HYPOTHETICAL"
    """Explicit hypothetical / fictional / roleplay framing."""

    UNSAFE = "UNSAFE"
    """Clearly unsafe request. Conservative boundary signal only — the
    classifier performs NO safety reasoning beyond recognition."""

    AMBIGUOUS = "AMBIGUOUS"
    """Odd or uninterpretable input; ambiguity deliberately preserved."""


class ContextConfidence(str, Enum):
    """Deterministic strength label for the detected mode.

    Same value vocabulary as ``insight_detection.InsightConfidence`` —
    a qualitative level, NOT a probability.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# ConversationContext — frozen decision record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationContext:
    """
    Immutable, JSON-safe record of the detected conversational situation.

    Fields
    ------
    mode:
        The strongest non-NORMAL signal's mode (NORMAL_DT when none fired).
    confidence:
        Qualitative strength of that classification.
    signals:
        Human-readable descriptions of every matched signal for the chosen
        mode (Developer Console trace; deterministic ordering).
    pause_dt:
        Phase 2 hook — True when the evidence justifies pausing DT question
        steering THIS turn (strong, unambiguous non-normal mode). Phase 1
        computes it but no component consumes it.
    follow_user_lead:
        Phase 2 hook — True when the reply should engage the user's actual
        move before (or instead of) pursuing the pending objective. Any
        detected non-normal signal sets it; Phase 1 ignores it.
    acknowledge_first:
        Phase 6 hook — True for MEDIUM-confidence conversational signals:
        the turn remains a fully normal DT turn (extraction, objective,
        family planning, enforcement all canonical), but Module 4 receives
        one extra framing instruction to briefly acknowledge the possible
        correction/clarification WITHOUT assuming what the user meant.
        Never True for NORMAL_DT or LOW-confidence ambiguity.

    The classifier NEVER mutates ProjectState or ConversationMemory; this
    object is the only output.
    """

    mode: ContextMode = ContextMode.NORMAL_DT
    confidence: ContextConfidence = ContextConfidence.HIGH
    signals: Tuple[str, ...] = ()
    pause_dt: bool = False
    follow_user_lead: bool = False
    acknowledge_first: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe projection using the repo's Title-Cased key convention."""
        return {
            "Mode": self.mode.value,
            "Confidence": self.confidence.value,
            "Signals": list(self.signals),
            "Pause DT": self.pause_dt,
            "Follow User Lead": self.follow_user_lead,
            "Acknowledge First": self.acknowledge_first,
        }


def _derive_pause_dt(mode: ContextMode, confidence: ContextConfidence) -> bool:
    """Strong, unambiguous non-normal modes justify pausing DT steering."""
    if mode in (ContextMode.NORMAL_DT, ContextMode.AMBIGUOUS):
        return False
    return confidence is ContextConfidence.HIGH


# ---------------------------------------------------------------------------
# Phase 2 activation surface
# ---------------------------------------------------------------------------
#
# The gate PAUSES DT question steering only for these modes, and only at
# HIGH confidence. HYPOTHETICAL / AMBIGUOUS / UNSAFE are deliberately NOT
# consumable yet (Phase 1 computes their pause_dt for diagnostics only;
# Phase 3+ will activate them behind explicit boundaries).

_PAUSABLE_MODES: frozenset = frozenset(
    {
        ContextMode.CORRECTION,
        ContextMode.TOPIC_SHIFT,
        ContextMode.DIRECT_QUESTION,
        ContextMode.CONFUSED,
        # Phase 3 activation: hypothetical framing engages conversationally
        # without becoming factual ProjectState information.
        ContextMode.HYPOTHETICAL,
    }
)


def is_dt_paused(conversation_context: Optional["ConversationContext"]) -> bool:
    """True when THIS turn should pause Design Thinking question steering.

    Single source of truth consumed by the pipeline. Requires BOTH the
    classifier's strong-evidence flag (``pause_dt``) AND an explicitly
    activated mode, so HYPOTHETICAL/AMBIGUOUS/UNSAFE stay observation-only.
    The pause is per-turn only: the Objective Engine remains canonical and
    resumes on the next turn.
    """
    if conversation_context is None:
        return False
    return bool(
        conversation_context.pause_dt
        and conversation_context.mode in _PAUSABLE_MODES
    )


# One concise GUIDANCE bullet per pausable mode (never a response template).
_CONTEXT_INSTRUCTIONS: Dict[ContextMode, str] = {
    ContextMode.CORRECTION: (
        "The user appears to be correcting your previous interpretation. "
        "Address the correction first and do not continue the previous "
        "Design Thinking question until their meaning is clear."
    ),
    ContextMode.TOPIC_SHIFT: (
        "The user appears to be intentionally changing topics. Follow the "
        "new topic naturally rather than forcing the previous Design "
        "Thinking objective into this reply."
    ),
    ContextMode.DIRECT_QUESTION: (
        "The user is directly asking for help. Acknowledge the request, "
        "but stay within the mentor role and first understand their "
        "situation rather than immediately prescribing a solution."
    ),
    ContextMode.CONFUSED: (
        "The user appears confused or uncertain. Clarify what they are "
        "asking about in simple language rather than pushing the planned "
        "Design Thinking question."
    ),
    ContextMode.HYPOTHETICAL: (
        "The user is exploring a hypothetical or fictional scenario. "
        "Respond to the hypothetical on its own terms without treating "
        "it as factual information about their project."
    ),
}


def context_instruction_bullet(mode: ContextMode) -> Optional[str]:
    """Guidance instruction bullet for a paused conversational turn.

    Rendered through the existing extra-bullets mechanism. ``None`` for
    modes without a Phase 2 instruction so normal turns stay byte-identical.
    """
    guidance = _CONTEXT_INSTRUCTIONS.get(mode)
    if guidance is None:
        return None
    return f"Conversational context: {guidance}"


# ---------------------------------------------------------------------------
# Phase 6 — MEDIUM-confidence acknowledge-first framing
# ---------------------------------------------------------------------------
#
# MEDIUM signals keep the turn fully canonical (extraction, objective,
# family planning, enforcement all unchanged); Module 4 receives ONE extra
# framing bullet communicating uncertainty without inventing meaning. The
# planned DT question stays the canonical question of the reply.

_ACKNOWLEDGE_INSTRUCTIONS: Dict[ContextMode, str] = {
    ContextMode.CORRECTION: (
        "Briefly acknowledge that the user may be correcting or clarifying "
        "something. Do not assume what they meant. Continue with the "
        "planned question."
    ),
    ContextMode.CONFUSED: (
        "Briefly acknowledge that the previous exchange may have been "
        "unclear. Do not assume what the user meant. Continue with the "
        "planned question."
    ),
}


def acknowledge_first_instruction_bullet(mode: ContextMode) -> Optional[str]:
    """Single framing bullet for an acknowledge-first (MEDIUM) turn.

    ``None`` for modes without mapping so no other path changes.
    """
    guidance = _ACKNOWLEDGE_INSTRUCTIONS.get(mode)
    if guidance is None:
        return None
    return f"Conversational framing: {guidance}"


# Deterministic replies used ONLY when the LLM is unavailable on a paused
# conversational turn (same role the family-aware templates play on normal
# turns). Each satisfies enforce_mentor_reply constraints: <=55 words,
# at most one '?', no advice markers — and statements are permitted.
_PAUSE_FALLBACK_REPLIES: Dict[ContextMode, str] = {
    ContextMode.CORRECTION: (
        "Got it - thanks for clarifying. Tell me a bit more about what you "
        "have in mind whenever you're ready."
    ),
    ContextMode.TOPIC_SHIFT: (
        "Understood - let's switch to the new direction. Tell me about what "
        "you're working on now."
    ),
    ContextMode.DIRECT_QUESTION: (
        "I hear you - before jumping ahead, tell me a bit more about where "
        "things stand right now."
    ),
    ContextMode.CONFUSED: (
        "Sorry about that - let me put it differently. What part felt unclear?"
    ),
    ContextMode.HYPOTHETICAL: (
        "Happy to explore that scenario with you. Tell me more about the "
        "situation you have in mind."
    ),
}


def paused_turn_fallback(mode: ContextMode) -> Optional[str]:
    """Deterministic reply for a paused conversational turn when the LLM
    fails. ``None`` when the mode has no activated fallback."""
    return _PAUSE_FALLBACK_REPLIES.get(mode)


# ---------------------------------------------------------------------------
# Safety boundary (Phase 3)
# ---------------------------------------------------------------------------
#
# Single source of truth with the UNSAFE classifier frames above. The
# boundary is intentionally CONSERVATIVE (precision over recall): benign
# discussion containing a dangerous word — movies, games, history,
# research, definitions, safety education — must NOT be intercepted.
# No LLM, no extraction, no state access; pure text frames only.

SAFETY_REFUSAL_REPLY: str = (
    "I'm not able to help with that. If you're working on a design "
    "project, I'm happy to explore the problem with you instead."
)


def is_unsafe_request(user_message: str) -> bool:
    """True iff the message matches a conservative unsafe-intent frame.

    Reuses the exact compiled frames that drive ``ContextMode.UNSAFE`` so
    the safety boundary can never diverge from the classifier. Pure
    function; microsecond cost; no LLM.
    """
    text = (user_message or "").strip()
    if not text:
        return False
    return bool(_matches_all(text.lower(), _COMPILED_UNSAFE))


# ---------------------------------------------------------------------------
# Pivot-language stripping (Phase 5, Problem 4)
# ---------------------------------------------------------------------------
#
# Conversational pivot language ("forget that...", "let's switch...") can be
# mis-ingested by generic extractor keyword fallbacks (e.g. "forget" ->
# pain point). This stripper removes ONLY clause-initial pivot heads so the
# remaining text can be probed for genuine Design Thinking content before
# deciding whether extraction is suppressed on a paused conversational turn.
# Subject-led factual statements ("I forget things daily") never match,
# because the pivot token must sit at the very start of the message or
# directly after punctuation.

_PIVOT_HEAD_RULES: Tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:^|(?<=[.;!?]\s)|(?<=,\s))"
        r"(?:forget|scratch(?:\s+that)?|drop\s+(?:it|that)"
        r"|ignore\s+(?:that|this|what\s+i\s+said)|never\s*mind)\b[,]?",
        r"(?:^|(?<=[.;!?]\s)|(?<=,\s))"
        r"let'?s\s+(?:talk\s+about|switch(?:\s+to)?|move\s+on(?:\s+to)?)\b[,]?",
    )
)


def strip_pivot_language(text: str) -> str:
    """Remove clause-initial conversational pivot heads from ``text``.

    Pure function. Genuine factual clauses survive; only pivot heads
    ("forget that,", "let's switch to", ...) disappear, so downstream
    probing sees the user's actual content.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for rx in _PIVOT_HEAD_RULES:
        cleaned = rx.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" \t,.;")


def _derive_follow_user_lead(mode: ContextMode) -> bool:
    """Any detected non-normal signal favours engaging the user's move."""
    return mode is not ContextMode.NORMAL_DT


def _derive_acknowledge_first(
    mode: ContextMode, confidence: ContextConfidence
) -> bool:
    """MEDIUM conversational signal -> frame the reply, never pause DT.

    NORMAL_DT and LOW-confidence ambiguity stay untouched; HIGH goes
    through pause_dt instead. Purely response framing.
    """
    return (
        mode is not ContextMode.NORMAL_DT
        and confidence is ContextConfidence.MEDIUM
    )


# ---------------------------------------------------------------------------
# Signal frames (compiled once at import; pure constants)
# ---------------------------------------------------------------------------

# --- UNSAFE — conservative boundary frames only ---------------------------
# Precision over recall: each frame requires intent/volition language plus
# an ACTION verb around a harm noun, so game-design, research, movie, and
# awareness-campaign wording does not fire.
_UNSAFE_FRAMES: Tuple[Tuple[str, str], ...] = (
    (
        "first-person volition toward a weapon/explosive",
        r"\b(?:i|we)\s+(?:really\s+|just\s+)?(?:want\s+to|wanna|need\s+to|plan\s+to|intend\s+to|going\s+to|gonna)\s+"
        r"(?:make|build|create|construct|assemble|get|acquire|buy|obtain|use|set\s+off|detonate|deploy)\b"
        r".{0,50}?\b(bombs?|explosives?|explosive\s+devices?|pipe\s+bombs?|ieds?|molotovs?|weapons?|poisons?|nerve\s+agents?)\b",
    ),
    (
        "how-to construction request for a weapon/explosive",
        r"\bhow\s+(?:do|can|to|does\s+(?:one|someone))\b.{0,40}?"
        r"\b(?:make|build|create|construct|assemble|acquire|get)\b.{0,30}?"
        r"\b(bombs?|explosives?|explosive\s+devices?|pipe\s+bombs?|weapons?|poisons?)\b",
    ),
    (
        "instructions/ways request for a weapon/explosive",
        r"\b(?:ways?|steps?|instructions?|methods?)\s+(?:to|for|of)\s+"
        r"(?:making|building|creating|constructing|assembling|getting|make|build|create|construct|assemble|get)\b"
        r".{0,30}?\b(bombs?|explosives?|weapons?|poisons?)\b",
    ),
    (
        "first-person volition to harm others",
        r"\b(?:i|we)\s+(?:want\s+to|wanna|need\s+to|plan\s+to|going\s+to)\b"
        r".{0,50}?\b(kill|murder|hurt|harm|attack|poison)\b.{0,25}?\b(someone|somebody|people|them|him|her)\b",
    ),
    (
        "self-harm intent",
        r"\b(?:kill|hurt|harm)\s+myself\b",
    ),
)

# --- HYPOTHETICAL — explicit framing markers only --------------------------
_HYPOTHETICAL_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("suppose/supposing", r"\bsuppos(?:e|ing)\b"),
    ("imagine", r"\bimagine\b"),
    ("what-if", r"\bwhat\s+if\b"),
    ("hypothetical(ly)", r"\bhypothetic[a-z]*\b"),
    ("pretend", r"\bpretend\b"),
    ("roleplay(ing)", r"\brole-?play(?:ing)?\b"),
    ("let's-say", r"\blet'?s\s+say\b"),
    ("fictional frame", r"\b(?:in\s+a\s+)?fictional\b"),
    ("joking frame", r"\b(?:i(?:'m|\sam)\s+(?:just\s+)?|just\s+)kidding\b"),
)

# --- CONFUSED ---------------------------------------------------------------
# Comprehension markers (strong, HIGH). Bare "don't understand" is excluded
# because "you don't understand me" is correction-shaped, not confusion.
_CONFUSED_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("I don't understand", r"\bi\s+(?:do\s+n[o']t\s+|don'?t\s+)understand\b"),
    ("I don't get it", r"\bi\s+don'?t\s+get\s+it\b"),
    ("I don't get what/how/why", r"\bi\s+don'?t\s+get\s+(?:what|where|how|why|your)\b"),
    ("can't figure this out", r"\bi\s+can'?t\s+figure\s+(?:this|it)\s+out\b"),
    ("didn't catch/follow/get that", r"\bi\s+didn'?t\s+(?:catch|follow|get)\s+(?:that|it|you)\b"),
    ("I'm confused", r"\bi\s*(?:am|'m)\s+confused\b"),
    ("I'm lost", r"\bi\s*(?:am|'m)\s+lost\b"),
    ("what does that mean", r"\bwhat\s+does\s+that\s+mean\b"),
    ("what do you mean", r"\bwhat\s+do\s+you\s+mean\b"),
    ("supposed to mean", r"\bwhat'?s\s+that\s+supposed\s+to\s+mean\b"),
    ("come again", r"\bcome\s+again\b"),
    # Phase 8: comprehension failure frames (HIGH)
    ("I'm not following", r"\bi\s*(?:am|'m)\s+not\s+following\b"),
    ("I'm not following you", r"\bi\s*(?:am|'m)\s+not\s+following\s+you\b"),
    ("I'm having trouble following", r"\bi\s*(?:am|'m)\s+having\s+trouble\s+following\b"),
    ("I can't follow what you mean", r"\bi\s+can'?t\s+follow\s+what\s+you\s+mean\b"),
    # Comprehension-directed uncertainty (context-dependent, handled separately)
    ("I'm not sure what you're asking", r"\bi\s*(?:am|'m)\s+not\s+sure\s+what\s+you'?re\s+asking\b"),
    ("I'm not sure what you're looking for", r"\bi\s*(?:am|'m)\s+not\s+sure\s+what\s+you'?re\s+looking\s+for\b"),
    ("I'm not sure what you're looking for", r"\bi\s*(?:am|'m)\s+not\s+sure\s+what\s+you'?re\s+looking\s+for\b"),
)
# Interrogative explanation request aimed at the mentor. CONTEXT-DEPENDENT:
# HIGH only when a previous mentor message exists (comprehension-directed);
# otherwise it may be a bare topic request, so it degrades to MEDIUM.
_EXPLAIN_REQUEST_RE = re.compile(
    r"\b(?:can|could)\s+you\s+explain\b|\bplease\s+explain\b"
    r"|\bwhat\s+do\s+you\s+mean\s+by\s+that\b"
    r"|\bwhat\s+do\s+you\s+mean\s+by\s+this\b",
    re.IGNORECASE
)
# Reuse the existing recovery "cannot answer" vocabulary as the single
# source of truth, but PERSON-SCOPE it here: "X don't know" statements about
# OTHER people are project facts ("users don't know where to start"), not
# the user failing to understand. Patterns whose text contains "know"
# require a first-person pronoun in the message; inherently self-directed
# idioms ("no idea", "no clue", ...) stay self-scoped without one.
_COMPILED_DONT_KNOW = [re.compile(p, re.IGNORECASE) for p in _DONT_KNOW_PATTERNS]
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'?m|i am|me|my|myself)\b", re.IGNORECASE)
_KNOW_FAMILY_DK = re.compile(r"\bknow\b", re.IGNORECASE)


def _dont_know_signals(lowered: str) -> bool:
    """Person-scoped reuse of the recovery DONT_KNOW lexicon."""
    for compiled, source in zip(_COMPILED_DONT_KNOW, _DONT_KNOW_PATTERNS):
        if not compiled.search(lowered):
            continue
        if _KNOW_FAMILY_DK.search(source) and not _FIRST_PERSON_RE.search(lowered):
            continue  # third-person "don't know" -> project fact, not confusion
        return True
    return False

# --- CORRECTION --------------------------------------------------------------
# Explicit meta-correction markers (strong, HIGH).
_CORRECTION_META_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("not-what-I-meant", r"\bnot\s+what\s+i\s+(?:meant|said|asked)\b"),
    ("you-misunderstand", r"\b(?:you|u)\s*(?:'re|are|r)?\s*(?:misunderstanding|misreading|misinterpreting)\b"),
    ("getting-me-wrong", r"\b(?:you|u)\s*(?:'re|are|r)?\s*getting\s+(?:me|it|this)\s+(?:wrong|all\s+wrong)\b"),
    ("you-confusing-me", r"\b(?:you|u)\s*(?:'re|are|r)?\s*confusing\s+(?:me|it|this|things)\b"),
    ("that's-wrong", r"\bthat'?s\s+(?:wrong|incorrect|not\s+right)\b"),
    ("I-didn't-say/mean-that", r"\bi\s+(?:didn'?t|did\s+not|don'?t|do\s+not|never)\s+(?:say|mean|talk\s+about|mention)\b"),
    ("not-what-I'm-talking-about", r"\bnot\s+what\s+i(?:'m|\sam)?\s*(?:talking|asking|referring)\s+about\b"),
    ("self-reframe-correction", r"\b(?:no|nope|nah)\b[,.!]?\s*i\s+mean\b.{0,40}?\b(?:my\s+own|myself|personally|me\b)"),
    ("I'm-not-saying", r"\bi'?m\s+not\s+saying\b"),
    ("that's-not-the-point", r"\bthat'?s\s+not\s+(?:the\s+point|it|i\s+said)\b"),
    # Phase 8: misunderstanding frames (HIGH)
    ("misunderstood-me", r"\b(?:i\s+think\s+)?(?:you|u)\s*(?:'re|are|r)?\s*misunderstood\s+(?:me|what\s+i\s+(?:mean|said))\b"),
    ("misunderstood-what-i-say", r"\b(?:i\s+think\s+)?(?:you|u)\s*(?:'re|are|r)?\s*misunderstood\s+what\s+i\s+(?:mean|said)\b"),
    # Phase 8: "not quite what I was trying to say" variants (HIGH)
    ("not-quite-what-i-was-trying", r"\bnot\s+quite\s+what\s+i\s+was\s+trying\s+to\s+say\b"),
    ("not-really-what-i-was-trying", r"\bnot\s+really\s+what\s+i\s+was\s+trying\s+to\s+say\b"),
    ("not-what-i-was-trying", r"\bnot\s+what\s+i\s+was\s+trying\s+to\s+say\b"),
    # Phase 8: self-reframe corrections (HIGH)
    ("my-situation-is-different", r"\b(?:actually\s+)?(?:my\s+situation|the\s+situation)\s+is\s+different\b"),
    ("meant-personal-problem", r"\bi\s+meant\s+the\s+problem\s+i'?m\s+(?:personally\s+)?having\b"),
    # Phase 8: self-reframe "No, I mean my own..." (HIGH)
    ("self-reframe-own", r"\b(?:no|nope|nah)\b[,.!]?\s*i\s+mean\b.{0,40}?\b(?:my\s+own|myself|personally|me\b)"),
    # Phase 8: "not quite right" / "not really right" (HIGH)
    ("not-quite-right", r"\bthat'?s\s+not\s+(?:quite|really)\s+right\b"),
)
# Premise-rejection frames (strong, HIGH): negate the implied project frame
# itself. Specific enough that ordinary answers never match.
_PREMISE_REJECTION_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("not-designing-for-anyone", r"\bnot\s+(?:designing|building|making|creating)\s+(?:this|it|that|anything)?\s*(?:for|about)\b"),
    ("not-for-anyone", r"\bnot\s+for\s+(?:anyone|anybody|users|people)\b"),
    ("no-one-would-use-it", r"\b(?:no\s+one|nobody)\s+(?:would|will|is\s+going\s+to)?\s*(?:use|benefit|need)\b"),
    ("there-is-no-target-audience", r"\bthere(?:'s|\s+is)\s+no\s+(?:target|audience|users?|customers?|beneficiar\w+)\b"),
    ("I-have-no-target-audience", r"\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:a\s+)?(?:target|audience|specific\s+users?)\b"),
    ("not-targeting-anyone", r"\bnot\s+(?:targeting|aimed\s+at|aiming\s+(?:at|for))\b"),
)
# Weak combination (MEDIUM): "actually"-style pivot + an explicit negation.
_ACTUALLY_NEGATION = re.compile(
    r"\bactually\b.{0,60}?\b(?:not|never|nothing|no\s+one|nobody)\b", re.IGNORECASE
)
# Weak combination (MEDIUM): negation token near a keyword belonging to the
# field the previous mentor question asked about (context-guarded).
_NEGATION_TOKEN = re.compile(r"\b(?:no|not|never|nothing|nobody|no\s+one|none)\b", re.IGNORECASE)

_FIELD_TOPIC_KEYWORDS: Dict[StateField, Tuple[str, ...]] = {
    StateField.PERSONAS: (
        "design", "audience", "user", "users", "people", "person", "anyone",
        "anybody", "target", "who",
    ),
    StateField.PROBLEMS: ("problem", "issue", "challenge", "struggle", "difficult"),
    StateField.CURRENT_SOLUTIONS: ("use", "using", "tool", "workaround", "currently", "solution"),
    StateField.PAIN_POINTS: ("stress", "frustrat", "hurt", "worry", "pain", "annoy"),
    StateField.EVIDENCE: ("seen", "observed", "research", "survey", "interview", "data"),
    StateField.FREQUENCY: ("often", "daily", "weekly", "frequency", "times", "every"),
}

# --- TOPIC_SHIFT ------------------------------------------------------------
_ABANDON_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("forget-the-X", r"\bforget\s+(?:the\s+|about\s+|that\s+)?(?:old\s+)?(?:coding|project|idea|thing|app|website|topic|game|it|that)\b"),
    ("scratch-that", r"\bscratch\s+that\b"),
    ("never-mind", r"\bnever\s+mind(?:\s+that|\s+the\s+\w+)?\b"),
    ("scrap-that", r"\bscrap\s+(?:that|the\s+\w+)\b"),
    ("drop-that", r"\bdrop\s+(?:that|the\s+old)\b"),
    ("ignore-what-i-said", r"\bignore\s+(?:that|everything\s+i\s+said|what\s+i\s+said)\b"),
    # Phase 8: additional abandonment frames (HIGH)
    ("forget-previous-problem", r"\bforget\s+(?:the\s+|about\s+|that\s+)?(?:previous|old)\s+(?:problem|topic)\b"),
    ("forget-previous-topic", r"\bforget\s+(?:the\s+|about\s+|that\s+)?(?:previous|old)\s+topic\b"),
    ("scratch-old-idea", r"\bscratch\s+the\s+old\s+idea\b"),
    ("drop-old-idea", r"\bdrop\s+the\s+old\s+idea\b"),
    ("let's-leave-that-aside", r"\blet'?s\s+leave\s+that\s+aside(?:\s+for\s+now)?\b"),
    ("let's-leave-that-aside", r"\blet'?s\s+leave\s+that\s+aside\b"),
    # Phase 8: additional abandonment frames
    ("let's-leave-that-aside-for-now", r"\blet'?s\s+leave\s+that\s+aside\s+for\s+now\b"),
    ("forget-previous-problem", r"\bforget\s+the\s+previous\s+problem\b"),
    ("forget-old-problem", r"\bforget\s+the\s+old\s+problem\b"),
    ("forget-previous-topic", r"\bforget\s+the\s+previous\s+topic\b"),
    ("scratch-old-idea", r"\bscratch\s+the\s+old\s+idea\b"),
    ("drop-old-idea", r"\bdrop\s+the\s+old\s+idea\b"),
    ("let's-leave-that-aside-for-now", r"\blet'?s\s+leave\s+that\s+aside\s+for\s+now\b"),
)
_TRANSITION_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("let's-talk-about", r"\blet'?s\s+talk\s+about\b"),
    ("let's-switch-to/topics/things", r"\blet'?s\s+switch\s+(?:to\b|topics\b|things\b|gears?\b)"),
    ("switching-to-different", r"\bswitch(?:ing)?\s+to\s+(?:a\s+|an\s+)?(?:completely\s+)?(?:different|new|another)\b"),
    ("moving-on-to", r"\bmoving\s+on\s+to\b"),
    ("instead-working-on", r"\binstead,?\s+i'?m\s+(?:working|thinking)\b"),
    ("working-on-something-different", r"\b(?:working|thinking)\s+on\s+something\s+(?:completely\s+)?different\b"),
    ("changed-my/the-project", r"\bchanged\s+(?:my|the)\s+(?:project|topic|idea|direction)\b(?:\s+(?:completely|entirely|totally))?"),
    ("completely-different-project", r"\b(?:completely|totally|entirely)\s+different\s+(?:project|topic|thing|idea|direction)\b"),
    ("talk-about-something-else", r"\b(?:want\s+to|wanna|let'?s)\s+talk\s+about\s+(?:something\s+else|other\s+things)\b"),
    # Phase 8: explicit topic replacement frames (HIGH)
    ("moved-on-to-different", r"\bi'?ve\s+moved\s+on\s+to\s+(?:a\s+)?(?:different|another)\s+(?:project|topic)\b"),
    ("moved-on-to-something-else", r"\bi'?ve\s+moved\s+on\s+to\s+something\s+else\b"),
    ("let's-leave-that-aside", r"\blet'?s\s+leave\s+that\s+aside\b"),
    ("let's-leave-that-aside-for-now", r"\blet'?s\s+leave\s+that\s+aside\s+for\s+now\b"),
    ("forget-previous-problem", r"\bforget\s+the\s+previous\s+problem\b"),
    ("forget-old-problem", r"\bforget\s+the\s+old\s+problem\b"),
    ("forget-previous-topic", r"\bforget\s+the\s+previous\s+topic\b"),
    ("scratch-old-idea", r"\bscratch\s+the\s+old\s+idea\b"),
    ("drop-old-idea", r"\bdrop\s+the\s+old\s+idea\b"),
    ("discuss-something-completely-different", r"\bi\s+(?:want\s+to|wanna|let'?s)\s+discuss\s+something\s+completely\s+different\b"),
    ("let's-discuss-something-completely-different", r"\blet'?s\s+discuss\s+something\s+completely\s+different\b"),
    ("want-to-talk-about-something-completely-different", r"\bi\s+want\s+to\s+talk\s+about\s+something\s+completely\s+different\b"),
    ("changed-project-completely", r"\bi'?ve\s+changed\s+the\s+project\s+completely\b"),
    ("changed-topic-completely", r"\bi'?ve\s+changed\s+the\s+topic\s+completely\b"),
    # Phase 8: additional explicit topic shift frames (HIGH)
    ("let's-switch-topics", r"\blet'?s\s+switch\s+topics\b"),
    ("let's-switch-topics", r"\blet'?s\s+switch\s+topics\b"),
    ("let's-switch-to-something-else", r"\blet'?s\s+switch\s+to\s+something\s+else\b"),
    ("i-want-to-switch-topics", r"\bi\s+want\s+to\s+switch\s+topics\b"),
    ("i-want-to-change-the-topic", r"\bi\s+want\s+to\s+change\s+the\s+topic\b"),
    ("i-want-to-change-the-subject", r"\bi\s+want\s+to\s+change\s+the\s+subject\b"),
    ("let's-change-the-topic", r"\blet'?s\s+change\s+the\s+topic\b"),
    ("let's-change-the-subject", r"\blet'?s\s+change\s+the\s+subject\b"),
    ("i-want-to-talk-about-something-else", r"\bi\s+want\s+to\s+talk\s+about\s+something\s+else\b"),
    ("i-want-to-discuss-something-else", r"\bi\s+want\s+to\s+discuss\s+something\s+else\b"),
)

# --- DIRECT_QUESTION --------------------------------------------------------
_ADVICE_FRAMES: Tuple[Tuple[str, str], ...] = (
    ("what-should-i-do", r"\bwhat\s+(?:should|shall|would)\s+i\s+do\b"),
    ("what-do-i-do", r"\bwhat\s+do\s+i\s+do\b"),
    ("how-do-i-solve/start/fix", r"\bhow\s+(?:do|can|should|would)\s+i\s+(?:solve|fix|start|proceed|approach|overcome|learn|handle)\b"),
    ("can-you-help", r"\bcan\s+you\s+(?:help|assist|suggest|advise)\b"),
    ("could-you-help", r"\bcould\s+you\s+(?:help|assist|suggest|advise)\b"),
    ("help-me", r"\bhelp\s+me\b"),
    ("any-advice", r"\bany\s+(?:advice|suggestions?|tips?)\b"),
    ("what-would-you-do", r"\bwhat\s+would\s+you\s+do\b"),
    ("give-me-advice", r"\b(?:give|need)\s+(?:me\s+)?(?:some\s+)?(?:advice|solutions?|guidance|ideas)\b"),
    # Phase 8: mentor-directed questions (HIGH)
    ("how-would-you-approach", r"\bhow\s+would\s+you\s+(?:approach|solve|handle|fix)\b"),
    ("how-would-you-solve", r"\bhow\s+would\s+you\s+solve\b"),
    ("how-would-you-handle", r"\bhow\s+would\s+you\s+handle\b"),
    ("what-do-you-think-i-should-try", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+(?:try|do)\b"),
    ("what-do-you-think-i-should-do", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+do\b"),
    ("what-would-you-suggest", r"\bwhat\s+would\s+you\s+suggest\b"),
    ("what-would-you-recommend", r"\bwhat\s+would\s+you\s+recommend\b"),
    ("why-do-you-need-to-know", r"\bwhy\s+do\s+you\s+need\s+to\s+know\b"),
    # Phase 8: additional mentor-directed questions (HIGH)
    ("what-would-you-suggest", r"\bwhat\s+would\s+you\s+suggest\b"),
    ("what-would-you-recommend", r"\bwhat\s+would\s+you\s+recommend\b"),
    ("why-do-you-need-to-know", r"\bwhy\s+do\s+you\s+need\s+to\s+know\b"),
    ("how-would-you-approach", r"\bhow\s+would\s+you\s+(?:approach|solve|handle|fix)\b"),
    ("how-would-you-solve", r"\bhow\s+would\s+you\s+solve\b"),
    ("how-would-you-handle", r"\bhow\s+would\s+you\s+handle\b"),
    ("what-do-you-think-i-should-try", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+(?:try|do)\b"),
    ("what-do-you-think-i-should-do", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+do\b"),
    ("what-would-you-suggest", r"\bwhat\s+would\s+you\s+suggest\b"),
    ("what-would-you-recommend", r"\bwhat\s+would\s+you\s+recommend\b"),
    ("why-do-you-need-to-know", r"\bwhy\s+do\s+you\s+need\s+to\s+know\b"),
    ("how-would-you-approach", r"\bhow\s+would\s+you\s+(?:approach|solve|handle|fix)\b"),
    ("how-would-you-solve", r"\bhow\s+would\s+you\s+solve\b"),
    ("how-would-you-handle", r"\bhow\s+would\s+you\s+handle\b"),
    ("what-do-you-think-i-should-try", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+(?:try|do)\b"),
    ("what-do-you-think-i-should-do", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+do\b"),
    ("what-would-you-suggest", r"\bwhat\s+would\s+you\s+suggest\b"),
    ("what-would-you-recommend", r"\bwhat\s+would\s+you\s+recommend\b"),
    ("why-do-you-need-to-know", r"\bwhy\s+do\s+you\s+need\s+to\s+know\b"),
    ("how-would-you-approach", r"\bhow\s+would\s+you\s+(?:approach|solve|handle|fix)\b"),
    ("how-would-you-solve", r"\bhow\s+would\s+you\s+solve\b"),
    ("how-would-you-handle", r"\bhow\s+would\s+you\s+handle\b"),
    ("what-do-you-think-i-should-try", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+(?:try|do)\b"),
    ("what-do-you-think-i-should-do", r"\bwhat\s+do\s+you\s+think\s+i\s+should\s+do\b"),
    ("what-would-you-suggest", r"\bwhat\s+would\s+you\s+suggest\b"),
    ("what-would-you-recommend", r"\bwhat\s+would\s+you\s+recommend\b"),
    ("why-do-you-need-to-know", r"\bwhy\s+do\s+you\s+need\s+to\s+know\b"),
)

# --- AMBIGUOUS — unusual first-person identity assertion --------------------
# "I'm a dog" is preserved as ambiguous (literal? fictional? joking?). A
# stakeholder-plausible predicate noun keeps ordinary self-descriptions
# ("I'm a student") out of this bucket.
_IDENTITY_ASSERTION = re.compile(
    r"\bi\s*(?:am|'m)\s+(?:just\s+|only\s+|simply\s+|really\s+)?"
    r"(?:a|an|the)\s+([a-z][a-z'-]{2,}(?:\s+[a-z]{3,})?)\b",
    re.IGNORECASE,
)
_PERSONA_PLAUSIBLE_NOUNS = frozenset(
    {
        "student", "teacher", "parent", "developer", "programmer", "engineer",
        "designer", "founder", "entrepreneur", "manager", "doctor", "nurse",
        "driver", "worker", "employee", "employer", "owner", "customer",
        "client", "user", "beginner", "learner", "novice", "hobbyist",
        "freelancer", "consultant", "analyst", "researcher", "creator",
        "builder", "adult", "teenager", "kid", "child", "human", "person",
        "professional", "volunteer", "organizer", "trader", "farmer",
        "artist", "writer", "chef", "athlete",
    }
)
_PERSONA_PLAUSIBLE_SUFFIXES = ("person", "people", "lover", "enthusiast", "expert", "advocate")


def _compile(frames: Tuple[Tuple[str, str], ...]):
    return [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in frames]


_COMPILED_UNSAFE = _compile(_UNSAFE_FRAMES)
_COMPILED_HYPOTHETICAL = _compile(_HYPOTHETICAL_FRAMES)
_COMPILED_CONFUSED = _compile(_CONFUSED_FRAMES)
_COMPILED_CORRECTION_META = _compile(_CORRECTION_META_FRAMES)
_COMPILED_PREMISE_REJECTION = _compile(_PREMISE_REJECTION_FRAMES)
_COMPILED_ABANDON = _compile(_ABANDON_FRAMES)
_COMPILED_TRANSITION = _compile(_TRANSITION_FRAMES)
_COMPILED_ADVICE = _compile(_ADVICE_FRAMES)


# ---------------------------------------------------------------------------
# Detection helpers (pure)
# ---------------------------------------------------------------------------


def _matches_all(text: str, compiled) -> Tuple[str, ...]:
    """Names of every frame that matched, in declared order."""
    return tuple(name for name, rx in compiled if rx.search(text))


def _identity_is_implausible(lowered: str) -> Optional[str]:
    """Return the captured predicate noun when a first-person identity
    assertion uses a non-stakeholder noun (ambiguity-preserving AMBIGUOUS
    signal), else ``None``."""
    m = _IDENTITY_ASSERTION.search(lowered)
    if m is None:
        return None
    captured = m.group(1).lower()
    words = captured.split()
    # Plausible when ANY captured word is a stakeholder noun or carries a
    # stakeholder suffix — so "developer building" stays normal while
    # "animal lover" remains plausible too.
    for w in words:
        if w in _PERSONA_PLAUSIBLE_NOUNS:
            return None
        if w.endswith(_PERSONA_PLAUSIBLE_SUFFIXES):
            return None
    return captured


def _asked_field_of_previous(previous_assistant_message: Optional[str]) -> Optional[StateField]:
    """StateField the previous mentor question targeted, via the existing
    question-family classifier; ``None`` when unknown."""
    if not previous_assistant_message:
        return None
    family = classify_question(previous_assistant_message)
    if family is None:
        return None
    return field_of(family)


def _correction_combo_signals(
    lowered: str,
    asked_field: Optional[StateField],
) -> Tuple[str, ...]:
    """Weak (MEDIUM) correction combinations requiring context agreement."""
    signals: Tuple[str, ...] = ()
    if _ACTUALLY_NEGATION.search(lowered):
        signals += ("'actually'-pivot with negation",)
    if asked_field is not None:
        keywords = _FIELD_TOPIC_KEYWORDS.get(asked_field, ())
        if keywords and _NEGATION_TOKEN.search(lowered) and any(kw in lowered for kw in keywords):
            signals += (
                f"negation near '{asked_field.value}' wording while the "
                "previous question asked about "
                f"{asked_field.value}",
            )
    return signals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def classify_conversation_context(
    *,
    user_message: str,
    previous_assistant_message: Optional[str] = None,
    recent_user_messages: Sequence[str] = (),
    extraction_message_type: Optional[str] = None,
    extraction_updates: Sequence[dict] = (),
    asked_families: Sequence[str] = (),
) -> ConversationContext:
    """Classify the conversational situation of one turn. PURE FUNCTION.

    Parameters
    ----------
    user_message:
        The raw latest user message (required).
    previous_assistant_message:
        The mentor's immediately preceding reply, used to identify which
        question family the user is responding to (via the existing
        ``module3.question_families.classify_question``).
    recent_user_messages:
        Reserved forward-compatibility input (mirrors the repo convention of
        accepting inputs a layer will grow into, cf. LifecycleManager's
        unused ``response_strategy``). Not consulted in Phase 1.
    extraction_message_type:
        Reserved forward-compatibility input (MEANINGFUL / NO_UPDATE /
        AMBIGUOUS / END). Not consulted in Phase 1.
    extraction_updates:
        Serialized extraction updates for this turn. Used ONLY as a
        context guard: substantive extracted facts suppress the weak
        unusual-identity AMBIGUOUS signal ("I'm a night owl who forgets
        medication" stays NORMAL_DT; bare "I'm a dog" stays AMBIGUOUS).
    asked_families:
        Reserved forward-compatibility input. Not consulted in Phase 1.

    Returns
    -------
    ConversationContext
        Frozen record. Priority order (first detector with a match wins):
        UNSAFE > HYPOTHETICAL > CONFUSED > CORRECTION > TOPIC_SHIFT >
        identity-AMBIGUOUS > DIRECT_QUESTION > NORMAL_DT.
    """
    text = (user_message or "").strip()
    if not text:
        return ConversationContext(
            mode=ContextMode.NORMAL_DT,
            confidence=ContextConfidence.HIGH,
            signals=("empty or whitespace-only message",),
        )

    lowered = text.lower()

    # 0. Shared context: which field did the previous question target?
    asked_field = _asked_field_of_previous(previous_assistant_message)

    # 1. UNSAFE (conservative boundary; precision over recall)
    unsafe = _matches_all(lowered, _COMPILED_UNSAFE)
    if unsafe:
        return _result(ContextMode.UNSAFE, ContextConfidence.HIGH, unsafe)

    # 2. HYPOTHETICAL (explicit framing only)
    hypothetical = _matches_all(lowered, _COMPILED_HYPOTHETICAL)
    if hypothetical:
        return _result(ContextMode.HYPOTHETICAL, ContextConfidence.HIGH, hypothetical)

    # 3. CONFUSED — comprehension markers (HIGH), then the person-scoped,
    #    reused recovery DONT_KNOW vocabulary (MEDIUM: may be a hedged
    #    answer). Third-person "X don't know" statements are project facts.
    confused = _matches_all(lowered, _COMPILED_CONFUSED)
    if confused:
        return _result(ContextMode.CONFUSED, ContextConfidence.HIGH, confused)
    if _EXPLAIN_REQUEST_RE.search(lowered):
        conf = (
            ContextConfidence.HIGH
            if previous_assistant_message
            else ContextConfidence.MEDIUM
        )
        return _result(
            ContextMode.CONFUSED,
            conf,
            ("explanation request"
             + (" following a mentor turn" if previous_assistant_message else "")),
        )
    if _dont_know_signals(lowered):
        return _result(
            ContextMode.CONFUSED,
            ContextConfidence.MEDIUM,
            ("person-scoped recovery DONT_KNOW vocabulary",),
        )

    # 4. CORRECTION — meta markers / premise rejection (HIGH), context
    #    combinations (MEDIUM).
    meta = _matches_all(lowered, _COMPILED_CORRECTION_META)
    if meta:
        return _result(ContextMode.CORRECTION, ContextConfidence.HIGH, meta)
    premise = _matches_all(lowered, _COMPILED_PREMISE_REJECTION)
    if premise:
        return _result(ContextMode.CORRECTION, ContextConfidence.HIGH, premise)
    combo = _correction_combo_signals(lowered, asked_field)
    if combo:
        return _result(ContextMode.CORRECTION, ContextConfidence.MEDIUM, combo)

    # 5. TOPIC_SHIFT — abandonment (HIGH) or explicit transition (HIGH).
    abandon = _matches_all(lowered, _COMPILED_ABANDON)
    if abandon:
        return _result(ContextMode.TOPIC_SHIFT, ContextConfidence.HIGH, abandon)
    transition = _matches_all(lowered, _COMPILED_TRANSITION)
    if transition:
        return _result(ContextMode.TOPIC_SHIFT, ContextConfidence.HIGH, transition)

    # 6. Identity-AMBIGUOUS (LOW) — preserve ambiguity for odd first-person
    #    identity claims; suppressed when real facts were extracted this turn.
    implausible = _identity_is_implausible(lowered)
    if implausible is not None and not extraction_updates:
        return _result(
            ContextMode.AMBIGUOUS,
            ContextConfidence.LOW,
            (
                f"unusual first-person identity statement ('I am a "
                f"{implausible}'); could be literal, fictional, hypothetical, "
                "or joking — ambiguity preserved",
            ),
        )

    # 7. DIRECT_QUESTION — explicit advice/solution-seeking frames.
    advice = _matches_all(lowered, _COMPILED_ADVICE)
    if advice:
        return _result(ContextMode.DIRECT_QUESTION, ContextConfidence.HIGH, advice)

    # 8. Default — ordinary Design Thinking exchange.
    return _result(ContextMode.NORMAL_DT, ContextConfidence.HIGH, ())


def _result(
    mode: ContextMode,
    confidence: ContextConfidence,
    signals: Tuple[str, ...],
) -> ConversationContext:
    """Construct the frozen record with derived Phase-2/6 hooks."""
    return ConversationContext(
        mode=mode,
        confidence=confidence,
        signals=tuple(signals),
        pause_dt=_derive_pause_dt(mode, confidence),
        follow_user_lead=_derive_follow_user_lead(mode),
        acknowledge_first=_derive_acknowledge_first(mode, confidence),
    )


# ---------------------------------------------------------------------------
# Developer Console projection (observation-only consumer pattern)
# ---------------------------------------------------------------------------


def conversation_context_diagnostics_section(capture: dict) -> dict:
    """Developer Console view of this turn's conversational context.

    Read-only projection following the same consumer pattern as
    ``coaching_diagnostics_section`` / ``insight_diagnostics_section``:
    returns ``{}`` when the layer produced nothing, so diagnostics stay
    byte-compatible for callers that never ran the classifier.
    """
    ctx = (capture or {}).get("conversation_context")
    if ctx is None:
        return {}
    if isinstance(ctx, ConversationContext):
        return ctx.to_dict()
    return dict(ctx)
