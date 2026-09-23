"""
conversation_style_audit — measurement-only audit of the mentor's reply
*style* every turn.

Whereas ``mentor_decision_audit`` scores *decisions* (objective choice,
continuity, question quality), this audit scores the *surface of the
generated reply text* — a pure function of the reply string plus minimal
deterministic context (the previous assistant message and the session's
already-known facts):

  * **Opening phrase** — the first few words of the reply.
  * **Reply length** — word and sentence counts.
  * **Begins with acknowledgment** — does the reply start by validating the
    user rather than jumping straight in?
  * **Observation / reflection** — does the reply state a reflective takeaway
    about the user's input?
  * **Bridge** — does it connect to previous information (acknowledge the
    prior turn's content)?
  * **Immediately asks a question** — does the first sentence end in ``?``?
  * **Number of questions** — raw ``?`` count.
  * **Multiple unrelated questions** — two+ questions that share no content
    word (a shotgun questionnaire) vs. a natural follow-up.
  * **Contains a summary** — recap/wrap-up markers.
  * **References prior info** — overlaps the session's known facts / prior
    reply.
  * **Restarts a topic** — re-asks a question whose focus already appears in
    the known facts.

Measurement ONLY. This module:

  * NEVER changes replies, prompts, extraction, objectives, lifecycle, or
    state,
  * only appears in Developer Console diagnostics and a session-level
    aggregate (``ConversationStyle`` / ``ConversationStyleSummary``),
  * is a pure function of the pipeline's existing per-turn record.

It imports only the standard library so it stays at a cycle-free leaf of the
dependency graph.
"""

from __future__ import annotations

import re

__all__ = [
    "ConversationStyleSummary",
    "analyze_reply",
    "conversation_style_diagnostics_section",
]

_OPENING_WORDS_DEFAULT = 3

# --- sentence splitting ------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


_WORD_RE = re.compile(r"[A-Za-z0-9']+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


# --- acknowledgment detection ----------------------------------------------
_ACK_START_PHRASES = (
    "thank you", "thanks", "thanks for", "thank you for", "i see",
    "i see -", "i see —", "got it", "understood", "that's", "that’s",
    "good to know", "of course", "glad", "noted", "appreciate it",
    "i appreciate", "absolutely", "exactly", "sure", "okay,", "ok,",
    "yes,", "yep,", "thanks so much",
)
_ACK_FIRST_WORDS = {
    "thanks", "thank", "great", "good", "perfect", "excellent", "awesome",
    "nice", "absolutely", "yes", "yep", "right", "sure", "glad", "okay",
    "ok", "understood", "appreciate", "appreciated", "noted", "exactly",
}


def _begins_with_acknowledgment(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    for phrase in _ACK_START_PHRASES:
        if lowered.startswith(phrase):
            return True
    first = _words(lowered)[0] if _words(lowered) else ""
    return first in _ACK_FIRST_WORDS


# --- content-signal markers ------------------------------------------------
_OBSERVATION_MARKERS = (
    "it sounds", "sounds like", "sound like", "that suggests",
    "that suggests that", "it seems", "seems like", "you're",
    "you are", "i notice", "i noticed", "noticing", "interesting",
    "it reflects", "reflects", "that tells me", "it looks", "looks like",
    "so you", "it sounds like", "i can see", "makes me think",
    "that's a", "thats a", "which tells me", "suggests that",
)

_BRIDGE_MARKERS = (
    "as you mentioned", "as mentioned", "you mentioned", "you said",
    "earlier you", "you told me", "building on", "building from",
    "following up", "to connect", "that connects", "related to what",
    "as we discussed", "you brought up", "you shared", "to tie back",
    "tie back", "back to what", "as part of", "linked to", "as you shared",
    "returning to", "which you", "that you mentioned", "you earlier",
    "going back", "as you noted",
)

_SUMMARY_MARKERS = (
    "here is", "here's", "heres", "so far", "let's recap", "lets recap",
    "to summarize", "let me summarize", "wrap up", "to wrap up",
    "summary so far", "what we know", "recap", "let's summarize",
    "lets summarize", "here is a summary", "here's a summary",
    "that's a great summary", "thats a great summary",
)

_QUESTION_WORDS = {
    "who", "what", "where", "when", "why", "how", "which", "whom", "whose",
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
    "would", "should", "will", "wont", "wouldnt", "couldnt", "dont",
    "doesnt", "not", "it", "this", "that", "a", "an", "the", "you", "your",
    "they", "their", "them", "we", "our", "us", "i", "me", "my", "to", "of",
    "for", "on", "at", "in", "with", "and", "or", "but", "about", "some",
}


def _content_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in _words(text or "")
        if w.lower() not in _QUESTION_WORDS and len(w) > 2
    }


def _focus_words(question: str) -> set[str]:
    return _content_words(question)


# --- helper: does text contain any of markers? ------------------------------
def _contains_any(text: str, markers) -> bool:
    lowered = (text or "").lower()
    return any(m in lowered for m in markers)


# ---------------------------------------------------------------------------
# Per-turn analysis
# ---------------------------------------------------------------------------


def analyze_reply(
    *,
    turn_index: int = 1,
    reply: str = "",
    previous_reply: str = "",
    known_facts: list | None = None,
    opening_words: int = _OPENING_WORDS_DEFAULT,
) -> dict:
    """Deterministic per-turn reply-style assessment.

    Pure function of the reply text (plus optional prior-reply / known-facts
    context for bridge and prior-info detection). Mutates nothing. Returns a
    JSON-serialisable record for the Developer Console and the session-level
    ``ConversationStyleSummary``.
    """
    known_facts = list(known_facts or [])
    reply = reply or ""
    previous_reply = previous_reply or ""

    sentences = _split_sentences(reply)
    tokens = _words(reply)

    question_count = reply.count("?")
    question_sentences = [s for s in sentences if s.strip().endswith("?")]
    immediately_asks = bool(sentences) and sentences[0].rstrip().endswith("?")

    # multiple unrelated questions: 2+ questions, no shared content focus.
    multiple_unrelated = False
    if len(question_sentences) >= 2:
        foci = [_focus_words(q) for q in question_sentences]
        base = set.intersection(*foci) if foci else set()
        multiple_unrelated = not base

    fact_words = set()
    for fact in known_facts:
        fact_words |= _content_words(str(fact))
    prev_words = _content_words(previous_reply)

    references_prior_info = bool(fact_words & set(_content_words(reply)))

    # bridge = explicit bridge marker OR reuses a distinctive prior-turn word.
    has_bridge = _contains_any(reply, _BRIDGE_MARKERS) or bool(
        prev_words and (_content_words(reply) & prev_words)
    )

    # restarts a topic = asks about content that is already a known fact.
    restarts_topic = False
    if references_prior_info and question_sentences:
        if any(_focus_words(q) & fact_words for q in question_sentences):
            restarts_topic = True

    opening_tokens = tokens[:opening_words] or ["(empty)"]
    opening = " ".join(opening_tokens)

    return {
        "turn_index": turn_index,
        "reply": reply,
        "opening": opening,
        "sentence_count": len(sentences),
        "word_count": len(tokens),
        "begins_with_acknowledgment": _begins_with_acknowledgment(reply),
        "has_observation": _contains_any(reply, _OBSERVATION_MARKERS),
        "has_bridge": has_bridge,
        "immediately_asks_question": immediately_asks,
        "question_count": question_count,
        "multiple_unrelated_questions": multiple_unrelated,
        "has_summary": _contains_any(reply, _SUMMARY_MARKERS),
        "references_prior_info": references_prior_info,
        "restarts_topic": restarts_topic,
    }


# --- Developer Console per-turn section --------------------------------------


def conversation_style_diagnostics_section(record: dict | None) -> dict:
    """Developer Console projection of one turn's reply-style assessment.

    Returns an empty dict when no record exists this turn (action-less /
    prior-to-any-assistant-msg turns).
    """
    if not record:
        return {}
    return {
        "Opening": record.get("opening") or "—",
        "Reply Length": (
            f"{record.get('word_count', 0)} words / "
            f"{record.get('sentence_count', 0)} sentences"
        ),
        "Begins with Acknowledgment": (
            "Yes" if record.get("begins_with_acknowledgment") else "No"
        ),
        "Observation/Reflection": (
            "Yes" if record.get("has_observation") else "No"
        ),
        "Bridge to Prior Info": "Yes" if record.get("has_bridge") else "No",
        "Immediately Asks Question": (
            "Yes" if record.get("immediately_asks_question") else "No"
        ),
        "Questions": record.get("question_count", 0),
        "Multiple Unrelated Questions": (
            "Yes" if record.get("multiple_unrelated_questions") else "No"
        ),
        "Contains Summary": "Yes" if record.get("has_summary") else "No",
        "References Prior Info": (
            "Yes" if record.get("references_prior_info") else "No"
        ),
        "Restarts Topic": "Yes" if record.get("restarts_topic") else "No",
    }


# ---------------------------------------------------------------------------
# Session-level aggregate
# ---------------------------------------------------------------------------


class ConversationStyleSummary:
    """Append-only aggregate of per-turn reply-style assessments for a
    session. Persists with the session (JSON-safe). Never read by any
    decision path — Developer Console + offline report only."""

    def __init__(
        self,
        turn_count: int = 0,
        opening_counts: dict | None = None,
        total_words: int = 0,
        total_sentences: int = 0,
        acknowledgment_count: int = 0,
        reflection_count: int = 0,
        summary_count: int = 0,
        bridge_count: int = 0,
        questionnaire_count: int = 0,
        restart_count: int = 0,
        multiple_question_count: int = 0,
    ):
        self.turn_count: int = turn_count
        self.opening_counts: dict[str, int] = dict(opening_counts or {})
        self.total_words: int = total_words
        self.total_sentences: int = total_sentences
        self.acknowledgment_count: int = acknowledgment_count
        self.reflection_count: int = reflection_count
        self.summary_count: int = summary_count
        self.bridge_count: int = bridge_count
        self.questionnaire_count: int = questionnaire_count
        self.restart_count: int = restart_count
        self.multiple_question_count: int = multiple_question_count

    # ---- aggregation -------------------------------------------------------

    def add_record(self, record: dict | None) -> None:
        if not record:
            return
        self.turn_count += 1
        opening = record.get("opening") or "(empty)"
        self.opening_counts[opening] = self.opening_counts.get(opening, 0) + 1
        self.total_words += int(record.get("word_count", 0))
        self.total_sentences += int(record.get("sentence_count", 0))
        if record.get("begins_with_acknowledgment"):
            self.acknowledgment_count += 1
        if record.get("has_observation"):
            self.reflection_count += 1
        if record.get("has_summary"):
            self.summary_count += 1
        if record.get("has_bridge"):
            self.bridge_count += 1
        if record.get("immediately_asks_question"):
            self.questionnaire_count += 1
        if record.get("restarts_topic"):
            self.restart_count += 1
        if record.get("multiple_unrelated_questions"):
            self.multiple_question_count += 1

    def merge(self, other: "ConversationStyleSummary") -> None:
        self.turn_count += other.turn_count
        self.total_words += other.total_words
        self.total_sentences += other.total_sentences
        self.acknowledgment_count += other.acknowledgment_count
        self.reflection_count += other.reflection_count
        self.summary_count += other.summary_count
        self.bridge_count += other.bridge_count
        self.questionnaire_count += other.questionnaire_count
        self.restart_count += other.restart_count
        self.multiple_question_count += other.multiple_question_count
        for k, v in other.opening_counts.items():
            self.opening_counts[k] = self.opening_counts.get(k, 0) + v

    # ---- metrics -----------------------------------------------------------

    def average_reply_length(self) -> float:
        """Mean words-per-reply (0.0 when no turns)."""
        if not self.turn_count:
            return 0.0
        return round(self.total_words / self.turn_count, 2)

    def _rate(self, numerator: int) -> float:
        if not self.turn_count:
            return 0.0
        return round(numerator / self.turn_count, 3)

    def reflection_rate(self) -> float:
        return self._rate(self.reflection_count)

    def summary_rate(self) -> float:
        return self._rate(self.summary_count)

    def bridge_rate(self) -> float:
        return self._rate(self.bridge_count)

    def questionnaire_rate(self) -> float:
        """Fraction of replies that open with a question (no preamble)."""
        return self._rate(self.questionnaire_count)

    def topic_restart_rate(self) -> float:
        return self._rate(self.restart_count)

    def multiple_question_rate(self) -> float:
        return self._rate(self.multiple_question_count)

    def acknowledgment_rate(self) -> float:
        return self._rate(self.acknowledgment_count)

    def most_common_openings(self, limit: int = 5) -> list[tuple[str, int]]:
        """Most frequent reply openings as ``(opening, count)`` pairs."""
        return sorted(
            self.opening_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[:limit]

    def opening_diversity_score(self) -> float:
        """How varied the openings are, in ``[0, 1]`` (1 = all distinct;
        0 = every reply opens identically)."""
        if not self.turn_count:
            return 0.0
        most = max(self.opening_counts.values())
        return round(1.0 - (most / self.turn_count), 3)

    # ---- persistence / display ---------------------------------------------

    def to_dict(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "opening_counts": dict(self.opening_counts),
            "total_words": self.total_words,
            "total_sentences": self.total_sentences,
            "acknowledgment_count": self.acknowledgment_count,
            "reflection_count": self.reflection_count,
            "summary_count": self.summary_count,
            "bridge_count": self.bridge_count,
            "questionnaire_count": self.questionnaire_count,
            "restart_count": self.restart_count,
            "multiple_question_count": self.multiple_question_count,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "ConversationStyleSummary":
        d = d or {}
        return cls(
            turn_count=int(d.get("turn_count", 0)),
            opening_counts=dict(d.get("opening_counts", {})),
            total_words=int(d.get("total_words", 0)),
            total_sentences=int(d.get("total_sentences", 0)),
            acknowledgment_count=int(d.get("acknowledgment_count", 0)),
            reflection_count=int(d.get("reflection_count", 0)),
            summary_count=int(d.get("summary_count", 0)),
            bridge_count=int(d.get("bridge_count", 0)),
            questionnaire_count=int(d.get("questionnaire_count", 0)),
            restart_count=int(d.get("restart_count", 0)),
            multiple_question_count=int(d.get("multiple_question_count", 0)),
        )

    @staticmethod
    def _summary_line(label: str, numerator: int, turn_count: int) -> str:
        rate = (numerator / turn_count * 100) if turn_count else 0.0
        return f"{numerator}/{turn_count} ({rate:.1f}%) {label}"

    def _display_rate(self, label: str, numerator: int) -> str:
        return self.__class__._summary_line(label, numerator, self.turn_count)

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        return {
            "Turns Analyzed": self.turn_count,
            "Most Common Openings": dict(self.most_common_openings()),
            "Opening Diversity Score": f"{self.opening_diversity_score():.2f} / 1.00",
            "Average Reply Length": (
                f"{self.average_reply_length():.1f} words/reply "
                f"({self.total_sentences} sentences total)"
            ),
            "Reflection Rate": self._display_rate("reflection", self.reflection_count),
            "Summary Rate": self._display_rate("summary", self.summary_count),
            "Bridge Rate": self._display_rate("bridge", self.bridge_count),
            "Questionnaire Rate": self._display_rate(
                "immediate question", self.questionnaire_count
            ),
            "Topic Restart Rate": self._display_rate(
                "restart", self.restart_count
            ),
            "Multiple-Question Rate": self._display_rate(
                "multiple unrelated", self.multiple_question_count
            ),
            "Begins with Acknowledgment": self._display_rate(
                "acknowledgment", self.acknowledgment_count
            ),
        }