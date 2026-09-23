"""
developer_console.py — deterministic builder + inspector-panel logic for the
Developer Console right sidebar.

Pure, standard-library-only (no Streamlit) so the whole panel can be
unit-tested in isolation. The Streamlit frontend only executes the UI
primitives; this module produces the *render plan* and owns every decision:

  * section enumeration — preserves the backend diagnostics order, the exact
    section dictionaries, and every existing section (``Prompt`` is pulled
    into its own dedicated block),
  * turn collection + selection — every assistant turn with diagnostics
    stays available and individually selectable,
  * prompt rendering — the ``Prompt`` section becomes its own expandable
    block with the exact prompt text,
  * collapsed/expanded state — ``CollapseState`` tracks, deterministically,
    which sections are expanded and which are collapsed,
  * timing rendering — the per-turn performance block (labels, extraction
    breakdown) moved here unchanged from the frontend,
  * conversation tree — ``build_tree_plan`` turns the flat per-turn plan into
    a VS Code-style tree (``Conversation → Turn N → Category → Section``) with
    instant search, per-category grouping, pinning, and expand/collapse-all
    modelling.

The module NEVER:

  * produces or mutates diagnostics (it only *reads* the dicts the pipeline
    already emits — each ``ConsoleSection.items`` is the original object;
    ``build_tree_plan`` and search only *filter/read*, never modify),
  * changes conversation behavior, extraction, objectives, prompts, or any
    diagnostic producer,
  * imports streamlit, staying a cycle-free leaf.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

__all__ = [
    "DIAG_SECTION_DISPLAY",
    "DEFAULT_EXPANDED_SECTIONS",
    "CollapseState",
    "ConsoleSection",
    "TurnDiagnostics",
    "collect_turns",
    "build_render_plan",
    "build_turn",
    "prompt_text",
    "render_section_text",
    "render_timing_lines",
    "prompt_stats",
    "highlight_matches",
    "prompt_viewer_stats",
    "highlight_prompt",
    "select_turn",
    "turn_label",
    "SECTION_CATEGORIES",
    "DEFAULT_CATEGORY",
    "CATEGORY_ORDER",
    "section_category",
    "category_order",
    "SectionNode",
    "CategoryNode",
    "TurnNode",
    "TreePlan",
    "build_tree_plan",
    "SectionStatus",
    "STATUS_ICONS",
    "STATUS_LABELS",
    "status_icon",
    "status_label",
    "section_status",
    "section_subtitle",
    "TimelineStage",
    "build_timeline",
]

# ---------------------------------------------------------------------------
# Timing rendering (moved unchanged from the Streamlit frontend)
# ---------------------------------------------------------------------------

TIMING_STAGE_LABELS = [
    ("LLM", "llm_ms"),
    ("Prompt", "prompt_ms"),
    ("Extraction", "extraction_ms"),
    ("Prepare", "prepare_ms"),
    ("Objective", "objective_ms"),
    ("Lifecycle", "lifecycle_ms"),
    ("Finalize", "finalize_ms"),
]

EXTRACTION_BREAKDOWN_ROWS = [
    ("preprocessing", "preprocessing_ms"),
    ("rules", "rules_ms"),
    ("llm", "llm_ms"),
    ("merge", "merge_ms"),
    ("validation", "validation_ms"),
    ("persistence", "persistence_ms"),
]


def format_duration(ms) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{round(ms)} ms"


def render_extraction_breakdown(breakdown) -> List[str]:
    """Render the per-sub-stage extraction timing tree. Mirrors the original
    frontend output exactly (elapsed ms + percentage of extraction wall time,
    plus the live prompt token count when the LLM path ran)."""
    total = breakdown.get("total_ms", 0) or 0
    lines = ["  Extraction"]
    prompt_tokens = breakdown.get("prompt_tokens") or 0
    if prompt_tokens:
        baseline = breakdown.get("prompt_baseline_tokens") or 0
        pct = breakdown.get("prompt_reduction_pct") or 0
        suffix = f" (−{pct}% vs baseline {baseline})" if pct else ""
        lines.append(f"  ├── prompt        {prompt_tokens} tok{suffix}")
    for label, key in EXTRACTION_BREAKDOWN_ROWS:
        ms = breakdown.get(key, 0) or 0
        pct = (ms / total * 100) if total else 0.0
        lines.append(f"  ├── {label:<14} {format_duration(ms):>9} {pct:5.1f}%")
    lines.append(f"  └── {'total':<14} {format_duration(total):>9} 100.0%")
    return lines


def render_timing_lines(timing: Optional[dict]) -> str:
    """Per-turn performance text block (total + per-stage durations)."""
    timing = timing or {}
    lines = [f"  {'Total':<12} {format_duration(timing.get('total_ms', 0))}"]
    for label, key in TIMING_STAGE_LABELS:
        val = timing.get(key)
        if val is not None:
            lines.append(f"  {label:<12} {format_duration(val)}")
    breakdown = timing.get("extraction_breakdown")
    if breakdown:
        lines.append("")
        lines.extend(render_extraction_breakdown(breakdown))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section display labels (preserves the existing map — UI naming only)
# ---------------------------------------------------------------------------

DIAG_SECTION_DISPLAY = {
    "ProjectState": "Project State",
    "StateChanges": "State Changes",
    "ObjectiveTrace": "Objective Trace",
    "Extraction": "Raw Extraction",
    "ExtractionComparison": "Extraction Comparison",
    "ExtractionAccuracy": "Extraction Accuracy",
    "ExtractionAccuracySummary": "Extraction Accuracy Summary",
    "MentorDecision": "Mentor Decision",
    "MentorDecisionSummary": "Mentor Decision Summary",
    "ConversationStyle": "Conversation Style",
    "ConversationStyleSummary": "Conversation Style Summary",
    "ProductExperience": "Product Experience",
    "ProductExperienceSummary": "Product Experience Summary",
    "ConversationFailure": "Conversation Failure",
    "ConversationFailureSummary": "Conversation Failure Summary",
    "HybridExtraction": "Hybrid Extraction",
    "SemanticComplexity": "Semantic Complexity",
    "HybridAudit": "Hybrid Audit",
    "HybridSummary": "Hybrid Summary",
    "Prompt": "Prompt Inspector",
    "QuestionHistory": "Question History",
    "QuestionFamilies": "Question Families",
    "Recovery": "Recovery Analysis",
    "Memory": "Conversation Memory",
    "CoachingStrategy": "Coaching Strategy",
    "InsightDetection": "Insight Detection",
    "ConversationMove": "Conversation Move",
    "PromptEffectiveness": "Prompt Effectiveness",
    "ChecklistReasoning": "Checklist Reasoning",
    "ConversationMetrics": "Conversation Metrics",
}

# Sections that start expanded in the inspector (everything else collapses).
DEFAULT_EXPANDED_SECTIONS = ("Performance", "Prompt")

#: Text shown when a diagnostic section is present but empty (same as the
#: original console).
_EMPTY_SECTION_TEXT = "No state changes"


# ---------------------------------------------------------------------------
# Collapse / expand state
# ---------------------------------------------------------------------------


class CollapseState:
    """Deterministic, JSON-safe track of which console sections are
    expanded vs collapsed.

    ``defaults`` provide the initial expanded state for keys that the user
    has never toggled; ``explicit`` overrides (expanded True/False) always
    win over defaults. The Streamlit frontend uses keyed expanders for live
    persistence; this class is the testable model of the same state and is
    what the render plan reflects.
    """

    def __init__(
        self,
        explicit: Optional[Dict[str, bool]] = None,
        defaults: Optional[Dict[str, bool]] = None,
        expanded: Optional[List[str]] = None,
    ):
        # ``expanded`` is kept as a convenience alias for constructing a
        # state where the listed keys are explicitly expanded.
        self._explicit: Dict[str, bool] = dict(explicit or {})
        for key in expanded or []:
            self._explicit[key] = True
        self._defaults: Dict[str, bool] = dict(defaults or {})

    def is_expanded(self, key: str, default: Optional[bool] = None) -> bool:
        if key in self._explicit:
            return self._explicit[key]
        if default is not None:
            return default
        return self._defaults.get(key, False)

    def toggle(self, key: str) -> None:
        self._explicit[key] = not self.is_expanded(key)

    def set_expanded(self, key: str, expanded: bool) -> None:
        self._explicit[key] = expanded

    def expanded_keys(self) -> List[str]:
        return sorted(k for k, v in self._explicit.items() if v)

    def collapsed_keys(self) -> List[str]:
        return sorted(k for k, v in self._explicit.items() if not v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explicit": dict(self._explicit),
            "defaults": dict(self._defaults),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CollapseState":
        d = d or {}
        return cls(
            explicit=dict(d.get("explicit", {}) or {}),
            defaults=dict(d.get("defaults", {}) or {}),
        )


# ---------------------------------------------------------------------------
# Per-turn console model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsoleSection:
    """One diagnostic section of one turn.

    ``items`` is the ORIGINAL dictionary/list from the diagnostics dict —
    never copied or transformed — so the console can never drift from what
    the pipeline produced.
    """

    key: str
    display_name: str
    items: Any
    is_prompt: bool = False

    def __eq__(self, other):
        if not isinstance(other, ConsoleSection):
            return NotImplemented
        return (
            self.key == other.key
            and self.display_name == other.display_name
            and self.is_prompt == other.is_prompt
            and self.items is other.items
        )

    def __hash__(self):
        return hash((self.key, self.display_name, self.is_prompt))


@dataclass
class TurnDiagnostics:
    """One assistant turn's console payload.

    ``diagnostics`` holds the EXACT original dict emitted by the pipeline;
    ``turn_index`` is the message's index in the chat history (stable for
    turn switching); ``ordinal`` is the 1-based display number.
    """

    turn_index: int
    ordinal: int
    role: str
    content_preview: str
    timing: Optional[dict]
    diagnostics: Optional[dict]
    sections: List[ConsoleSection] = field(default_factory=list)
    has_data: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot. The diagnostics dict is kept by reference via
        ``section keys``; the underlying dicts are never re-serialised here."""
        return {
            "turn_index": self.turn_index,
            "ordinal": self.ordinal,
            "role": self.role,
            "content_preview": self.content_preview,
            "section_keys": [s.key for s in self.sections],
            "has_data": self.has_data,
        }


def build_turn(
    *,
    turn_index: int,
    ordinal: int,
    role: str,
    content: str = "",
    timing: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> TurnDiagnostics:
    """Build a ``TurnDiagnostics`` from a message's raw parts. Diagnostics is
    stored by reference — nothing is copied or mutated."""
    content = content or ""
    preview = content if len(content) <= 60 else content[:60] + "…"
    sections: List[ConsoleSection] = []
    for key, items in (diagnostics or {}).items():
        sections.append(
            ConsoleSection(
                key=key,
                display_name=DIAG_SECTION_DISPLAY.get(key, key),
                items=items,
                is_prompt=(key == "Prompt"),
            )
        )
    return TurnDiagnostics(
        turn_index=turn_index,
        ordinal=ordinal,
        role=role,
        content_preview=preview,
        timing=timing,
        diagnostics=diagnostics,
        sections=sections,
        has_data=bool(diagnostics or timing),
    )


def collect_turns(messages: List[dict]) -> List[TurnDiagnostics]:
    """Extract every assistant turn that carries diagnostics/timing, in chat
    order. ``turn_index`` is the message's index in ``messages``."""
    turns: List[TurnDiagnostics] = []
    for index, message in enumerate(messages or []):
        if message.get("role") != "assistant":
            continue
        timing = message.get("timing")
        diagnostics = message.get("diagnostics")
        if not diagnostics and not timing:
            continue
        turns.append(
            build_turn(
                turn_index=index,
                ordinal=len(turns) + 1,
                role="assistant",
                content=message.get("content", ""),
                timing=timing,
                diagnostics=diagnostics,
            )
        )
    return turns


def select_turn(turns: List[TurnDiagnostics], selected_index: Optional[int]):
    """Return the turn whose ``turn_index`` equals ``selected_index``, or
    ``None`` when out of range / unknown. The caller falls back to the last
    turn (or a "no diagnostics" state) when this returns ``None``."""
    for turn in turns:
        if turn.turn_index == selected_index:
            return turn
    return None


def turn_label(turn: TurnDiagnostics) -> str:
    """Stable tree label for a turn — ``Turn N`` only. Never assistant text,
    so a turn's meaning never depends on whatever it happened to say."""
    return f"Turn {turn.ordinal}"


# ---------------------------------------------------------------------------
# Rendering helpers (text only — no Streamlit)
# ---------------------------------------------------------------------------


def render_section_text(section: ConsoleSection) -> str:
    """Rendered display text for one diagnostic section (reuses the generic
    ``diag_render`` renderer so the appearance is byte-identical)."""
    from diag_render import render_diag_items

    if section.is_prompt:
        return render_prompt_text(section.items)
    if not section.items:
        return _EMPTY_SECTION_TEXT
    return "\n".join(render_diag_items(section.items))


def render_prompt_text(prompt: Any) -> str:
    """The prompt inspector block text. Strings render verbatim; structured
    prompt dicts render through the generic renderer."""
    from diag_render import render_diag_items

    if isinstance(prompt, str):
        return prompt
    if prompt is None:
        return _EMPTY_SECTION_TEXT
    return "\n".join(render_diag_items(prompt))


def prompt_stats(text) -> Dict[str, int]:
    """Character and word counts for a rendered prompt string. Pure helper
    for the Prompt inspector's statistics row (monospace block header)."""
    text = text or ""
    return {"characters": len(text), "words": len(text.split())}


def highlight_matches(text: Any, query: str) -> str:
    """HTML-escape ``text`` and wrap each case-insensitive occurrence of every
    whitespace-separated term in ``query`` in ``<mark class="dc-search-hit">``.

    Pure, read-only helper for the inspector's live search highlight — the
    returned string is safe to pass into ``st.markdown(..., unsafe_allow_html=True)``.
    With an empty query (search cleared) the text is returned escaped but
    otherwise unchanged, so a cleared search renders byte-identically.
    """
    source = str(text)
    if not query:
        return html.escape(source)
    terms = [t for t in query.split() if t]
    if not terms:
        return html.escape(source)
    lowered = source.lower()

    ranges: List[tuple] = []
    for term in terms:
        term_lower = term.lower()
        start = 0
        while True:
            idx = lowered.find(term_lower, start)
            if idx == -1:
                break
            ranges.append((idx, idx + len(term)))
            start = idx + len(term)
    if not ranges:
        return html.escape(source)

    ranges.sort()
    merged: List[tuple] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(html.escape(source[cursor:start]))
        parts.append(f'<mark class="dc-search-hit">{html.escape(source[start:end])}</mark>')
        cursor = end
    if cursor < len(source):
        parts.append(html.escape(source[cursor:]))
    return "".join(parts)


def prompt_viewer_stats(text: Any) -> Dict[str, int]:
    """Character / word / line / byte counts for the Prompt Viewer toolbar.

    ``characters`` is the raw string length (whitespace included); ``words``
    counts whitespace-separated tokens; ``lines`` is the number of visible
    text lines (``str.splitlines()``); ``bytes`` is the UTF-8 size. Pure,
    deterministic helper for the editor-style viewer header row.
    """
    text = text or ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
        "bytes": len(text.encode("utf-8")),
    }


_PROMPT_TOKEN_RE = re.compile(r"\{[^{}\n]*\}")


def highlight_prompt(text: Any) -> str:
    """Returns the prompt as an HTML-escaped, IDE-style syntax-highlighted
    string ready for ``st.html``/markdown ``unsafe_allow_html``.

    Line-oriented, deterministic and offline: role/header banners, divider
    lines, list bullets and ``{placeholder}`` tokens get their own CSS class
    (``.pc-h``, ``.pc-hline``, ``.pc-list``, ``.pc-tok``); plain lines are
    escaped unchanged. Never mutates the input prompt — it only produces a
    presentation copy. Lines are joined with ``\\n`` to preserve layout.
    """
    source = text or ""
    if not isinstance(source, str):
        source = str(source)
    out_lines: List[str] = []
    for raw in source.split("\n"):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        kind = None
        if re.match(r"^(role|system|user|assistant|human|bot)\s*:", line, re.I) or re.match(
            r"^#{1,6}\s", stripped
        ) or re.match(r"^\[[A-Z0-9 _.\-/]+\]\s*$", stripped):
            kind = "pc-h"
        elif re.match(r"^[=\-]{3,}\s*$", stripped):
            kind = "pc-hline"
        elif re.match(r"^\s*(?:[-•*]|\d{1,3}[.)])\s+", line):
            kind = "pc-list"
        elif _PROMPT_TOKEN_RE.search(line):
            kind = "pc-tokline"
        escaped = html.escape(line, quote=True)
        if kind in ("pc-tokline", None):
            escaped = _PROMPT_TOKEN_RE.sub(
                r'<span class="pc-tok">\g<0></span>', escaped
            )
        if kind:
            escaped = f'<span class="{kind}">{escaped}</span>'
        out_lines.append(escaped)
    return "\n".join(out_lines)


def prompt_text(turn: TurnDiagnostics) -> str:
    """Exact prompt string for a turn (``""`` when absent)."""
    if not turn or not turn.diagnostics:
        return ""
    prompt = turn.diagnostics.get("Prompt")
    if isinstance(prompt, str):
        return prompt
    return render_prompt_text(prompt)


def build_render_plan(
    turn: Optional[TurnDiagnostics],
    collapse_state: Optional[CollapseState] = None,
) -> List[Dict[str, Any]]:
    """Deterministic render plan for one selected turn.

    Each entry is::

        {
          "kind": "performance" | "prompt" | "section",
          "key": str,
          "display_name": str,
          "text": str,
          "expanded": bool,
        }

    The prompt is its own block; every other diagnostic section keeps its
    backend order and original items. ``expanded`` follows
    ``DEFAULT_EXPANDED_SECTIONS`` unless ``collapse_state`` overrides.
    """
    if not turn:
        return []
    default_expanded = {k: True for k in DEFAULT_EXPANDED_SECTIONS}
    if collapse_state is None:
        collapse_state = CollapseState(defaults=dict(default_expanded))
    else:
        # Seed module defaults for keys the caller never toggled, so behaviour
        # is identical with or without a persisted CollapseState.
        merged_defaults = dict(collapse_state._defaults)
        for key in DEFAULT_EXPANDED_SECTIONS:
            merged_defaults.setdefault(key, True)
        collapse_state = CollapseState(
            explicit=dict(collapse_state._explicit),
            defaults=merged_defaults,
        )
    plan: List[Dict[str, Any]] = []

    performance_text = render_timing_lines(turn.timing)
    if performance_text:
        plan.append(
            {
                "kind": "performance",
                "key": "Performance",
                "display_name": "Performance",
                "text": performance_text,
                "expanded": collapse_state.is_expanded("Performance"),
            }
        )

    for section in turn.sections:
        if section.is_prompt:
            plan.append(
                {
                    "kind": "prompt",
                    "key": section.key,
                    "display_name": section.display_name,
                    "text": render_section_text(section),
                    "expanded": collapse_state.is_expanded(section.key),
                }
            )
            continue
        plan.append(
            {
                "kind": "section",
                "key": section.key,
                "display_name": section.display_name,
                "text": render_section_text(section),
                "expanded": collapse_state.is_expanded(section.key),
            }
        )
    return plan


# ---------------------------------------------------------------------------
# Section categories (UI grouping only — never touches diagnostics)
# ---------------------------------------------------------------------------

#: Maps a section key to its VS Code-style inspector category. ``Performance``
#: is the synthetic timing block. Keys not listed fall into ``DEFAULT_CATEGORY``.
SECTION_CATEGORIES: Dict[str, str] = {
    "Performance": "Core Pipeline",
    "Pipeline": "Core Pipeline",
    "ObjectiveTrace": "Core Pipeline",
    "Extraction": "Core Pipeline",
    "HybridExtraction": "Core Pipeline",
    "InsightDetection": "Core Pipeline",
    "ProjectState": "Conversation",
    "StateChanges": "Conversation",
    "Memory": "Conversation",
    "Recovery": "Conversation",
    "QuestionHistory": "Conversation",
    "QuestionFamilies": "Conversation",
    "MentorDecision": "Conversation",
    "MentorDecisionSummary": "Conversation",
    "ChecklistReasoning": "Conversation",
    "CoachingStrategy": "Conversation",
    "ConversationMove": "Conversation",
    "ExtractionComparison": "Quality Audits",
    "ExtractionAccuracy": "Quality Audits",
    "ExtractionAccuracySummary": "Quality Audits",
    "ConversationStyle": "Quality Audits",
    "ConversationStyleSummary": "Quality Audits",
    "ProductExperience": "Quality Audits",
    "ProductExperienceSummary": "Quality Audits",
    "ConversationFailure": "Quality Audits",
    "ConversationFailureSummary": "Quality Audits",
    "SemanticComplexity": "Quality Audits",
    "HybridAudit": "Quality Audits",
    "HybridSummary": "Quality Audits",
    "ConversationMetrics": "Quality Audits",
    "Prompt": "Prompt",
    "PromptEffectiveness": "Prompt",
}

#: Category used when a section key has no explicit mapping.
DEFAULT_CATEGORY = "Conversation"

#: Stable display order of the inspector categories.
CATEGORY_ORDER: tuple = (
    "Core Pipeline",
    "Conversation",
    "Quality Audits",
    "Prompt",
    "Visual Design",
)


def section_category(key: str) -> str:
    """VS Code-style category a section belongs to (``DEFAULT_CATEGORY`` for
    unknown keys). Pure mapping — never inspects or mutates the section."""
    return SECTION_CATEGORIES.get(key, DEFAULT_CATEGORY)


def category_order() -> List[str]:
    """Ordered copy of ``CATEGORY_ORDER`` (includes empty categories)."""
    return list(CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# Section health status (visual-only — never touches diagnostics)
# ---------------------------------------------------------------------------
#
# Every section gets a derived status so the inspector can surface "which part
# of the pipeline needs attention" at a glance. The classifier is pure and
# read-only: it consumes the EXACT dicts the pipeline already emitted and never
# mutates or removes anything. Unknown / absent sections default to HEALTHY.


class SectionStatus(str, Enum):
    """Three-level visual health state for a console section."""

    HEALTHY = "healthy"
    WARNING = "warning"
    PROBLEM = "problem"


STATUS_ICONS: Dict[SectionStatus, str] = {
    SectionStatus.HEALTHY: "🟢",
    SectionStatus.WARNING: "🟡",
    SectionStatus.PROBLEM: "🔴",
}

STATUS_LABELS: Dict[SectionStatus, str] = {
    SectionStatus.HEALTHY: "Healthy",
    SectionStatus.WARNING: "Warning",
    SectionStatus.PROBLEM: "Problem",
}


def status_icon(status: SectionStatus) -> str:
    """Traffic-light emoji for a status (UI presentation only)."""
    return STATUS_ICONS.get(status, STATUS_ICONS[SectionStatus.HEALTHY])


def status_label(status: SectionStatus) -> str:
    """Human label for a status (``"Healthy"`` / ``"Warning"`` / ``"Problem"``)."""
    return STATUS_LABELS.get(status, STATUS_LABELS[SectionStatus.HEALTHY])


def _flag(value: Any) -> bool:
    """Interpret ``"Yes"`` / ``True`` / ``"true"`` style values as boolean."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "true", "1")


def _rate_count(value: Any) -> int:
    """Pull the leading ``N`` out of a ``"N/M (X.X%)"`` summary string."""
    match = re.match(r"(\d+)\s*/", str(value or ""))
    return int(match.group(1)) if match else 0


def section_status(
    key: str,
    items: Any = None,
    *,
    timing: Optional[dict] = None,
) -> tuple:
    """Deterministic health status + short reason for one console section.

    ``items`` is the section's ORIGINAL diagnostics value (dict/list/scalar);
    ``timing`` is the turn's ``timing`` dict and is used for the synthetic
    ``Performance`` block. Returns ``(SectionStatus, reason)`` where ``reason``
    is a short human string (``""`` for healthy sections).

    Pure read-only classification — no diagnostics are ever modified, dropped,
    or re-ordered. Unknown sections default to ``(HEALTHY, "")``.
    """
    if key == "Performance":
        return _performance_status(timing)
    items = items if items is not None else {}
    if not isinstance(items, dict):
        return _sequence_status(key, items)
    handler = _SECTION_STATUS_RULES.get(key)
    if handler is None:
        return SectionStatus.HEALTHY, ""
    return handler(items)


# Memory section keys whose list lengths make up the "facts" summary.
_MEMORY_SUMMARY_KEYS = (
    "Resolved Threads",
    "Acknowledged Facts",
    "Summarized Facts",
)


def section_subtitle(
    key: str,
    items: Any = None,
    *,
    timing: Optional[dict] = None,
) -> str:
    """Short one-line subtitle for a console section's navigation row.

    Complements ``section_status`` (which explains *problems*): this derives a
    compact neutral summary for healthy sections — e.g. ``"84.50 s"`` for
    Performance, ``"4 fact(s)"`` for Memory, ``"12 lines · 40 words"`` for the
    Prompt. Unknown / unhelpful sections return ``""`` (the UI simply hides the
    subtitle). Pure, read-only and defensive — it only inspects the original
    diagnostic value and never mutates anything.
    """
    if key == "Performance":
        total = timing.get("total_ms") if isinstance(timing, dict) else None
        if isinstance(total, (int, float)) and total:
            return format_duration(total)
        return ""
    items = items if items is not None else {}
    if key == "Memory" and isinstance(items, dict):
        facts = sum(len(items.get(k) or []) for k in _MEMORY_SUMMARY_KEYS)
        if facts:
            return f"{facts} fact(s)"
        open_threads = len(items.get("Open Threads") or [])
        deferred = len(items.get("Deferred Topics") or [])
        if open_threads or deferred:
            return f"{open_threads} open · {deferred} deferred"
        return ""
    if key == "Prompt":
        stats = prompt_viewer_stats(render_prompt_text(items))
        return f"{stats['lines']} lines · {stats['words']} words"
    if key == "ProjectState" and isinstance(items, dict):
        count = sum(1 for v in items.values() if v)
        return f"{count} field(s)" if count else "empty"
    if key == "Pipeline" and isinstance(items, dict):
        return str(items.get("Objective") or items.get("Stage") or "")
    if key == "QuestionFamilies" and isinstance(items, dict):
        family = items.get("Question Family")
        return str(family) if family else ""
    if key == "ConversationFailure" and isinstance(items, dict):
        primary = items.get("Primary Failure")
        if primary and primary not in ("None", "No failure", ""):
            return str(primary)
        return "No failures"
    if key == "Recovery" and isinstance(items, dict):
        category = items.get("Category")
        return str(category) if category and category != "NONE" else ""
    return ""


def _performance_status(timing: Optional[dict]) -> tuple:
    """Performance status from the turn's timing block (ms thresholds)."""
    if not isinstance(timing, dict):
        return SectionStatus.HEALTHY, ""
    total = timing.get("total_ms")
    llm = timing.get("llm_ms")
    if isinstance(total, (int, float)) and total > 10000:
        return SectionStatus.PROBLEM, f"Slow turn ({format_duration(total)})"
    if isinstance(llm, (int, float)) and llm > 8000:
        return SectionStatus.PROBLEM, f"Slow LLM call ({format_duration(llm)})"
    if isinstance(total, (int, float)) and total > 3000:
        return SectionStatus.WARNING, f"Slow turn ({format_duration(total)})"
    return SectionStatus.HEALTHY, ""


def _sequence_status(key: str, items: Any) -> tuple:
    """Status for non-dict sections (lists/scalars)."""
    if key == "QuestionHistory" and isinstance(items, list):
        if len(items) > 10 and len(set(items)) < len(items):
            return SectionStatus.WARNING, "Repeated questions in history"
    return SectionStatus.HEALTHY, ""


def _status_project_state(items: dict) -> tuple:
    if not items or all(not v for v in items.values()):
        return SectionStatus.WARNING, "No state captured yet"
    return SectionStatus.HEALTHY, ""


def _status_state_changes(items: dict) -> tuple:
    updated = items.get("Updated") or {}
    if isinstance(updated, dict) and updated:
        if "Frequency" in updated:
            return SectionStatus.WARNING, "Frequency overwritten"
        return SectionStatus.WARNING, "State overwritten"
    return SectionStatus.HEALTHY, ""


def _status_objective_trace(items: dict) -> tuple:
    missing = items.get("Missing Fields") or []
    verdict = (items.get("Advancement") or {}).get("Verdict")
    if missing and verdict == "WRAP_UP":
        return SectionStatus.PROBLEM, (
            f"Wrapped up with {len(missing)} missing field(s)"
        )
    if missing:
        return SectionStatus.WARNING, f"{len(missing)} missing field(s)"
    confidence = items.get("Confidence")
    if isinstance(confidence, (int, float)) and confidence < 1.0:
        return SectionStatus.WARNING, "Low objective confidence"
    return SectionStatus.HEALTHY, ""


def _status_extraction(items: dict) -> tuple:
    if items.get("message_type") == "MEANINGFUL" and not (items.get("updates") or []):
        return SectionStatus.WARNING, "No updates extracted from a meaningful message"
    return SectionStatus.HEALTHY, ""


def _status_extraction_comparison(items: dict) -> tuple:
    decision = items.get("Decision") or {}
    if decision.get("Rule sufficient") == "No":
        return SectionStatus.WARNING, "Rule/LLM disagreement"
    if (items.get("Fields Only Found by LLM") or []) or (
        items.get("Fields Only Found by Rules") or []
    ):
        return SectionStatus.WARNING, "Rule/LLM disagreement"
    if decision.get("LLM Applied") == "No" and (items.get("LLM Extractor") or []):
        return SectionStatus.WARNING, "LLM output discarded"
    return SectionStatus.HEALTHY, ""


def _status_extraction_accuracy(items: dict) -> tuple:
    updates = items.get("Extracted Updates") or []
    if any(u.get("label") == "Incorrect" for u in updates):
        return SectionStatus.PROBLEM, "Incorrect extraction"
    misses = items.get("Missed Opportunities") or []
    if any(m.get("attached") == "No" for m in misses):
        return SectionStatus.PROBLEM, f"{len(misses)} missed fact(s) dropped"
    if misses:
        return SectionStatus.WARNING, f"{len(misses)} missed opportunity/opportunities"
    if any(u.get("label") == "Partially Correct" for u in updates):
        return SectionStatus.WARNING, "Partially correct extraction(s)"
    return SectionStatus.HEALTHY, ""


def _status_extraction_accuracy_summary(items: dict) -> tuple:
    incorrect = items.get("Incorrect") or 0
    missed = items.get("Missed Opportunity") or 0
    dropped = items.get("Turn-level missed facts") or 0
    if incorrect or missed or dropped:
        return SectionStatus.PROBLEM, (
            f"{incorrect} incorrect · {missed} missed · {dropped} dropped"
        )
    if (items.get("Partially Correct") or 0) > 0:
        return SectionStatus.WARNING, "Partially correct extractions"
    return SectionStatus.HEALTHY, ""


def _status_mentor_decision(items: dict) -> tuple:
    if items.get("Restarted Topic") == "Yes":
        return SectionStatus.PROBLEM, "Topic restarted"
    if items.get("Objective Appropriate") == "No":
        return SectionStatus.PROBLEM, "Objective not appropriate"
    if items.get("Acknowledged Information") == "No":
        return SectionStatus.WARNING, "User information not acknowledged"
    if items.get("Better Alternative") not in (None, ""):
        return SectionStatus.WARNING, "Better alternative objective existed"
    if str(items.get("Question Quality", "")).strip().lower() == "generic":
        return SectionStatus.WARNING, "Generic question"
    return SectionStatus.HEALTHY, ""


def _status_mentor_decision_summary(items: dict) -> tuple:
    if (items.get("Restarted Topics") or 0) > 0:
        return SectionStatus.PROBLEM, f"{items['Restarted Topics']} restarted topic(s)"
    if items.get("Better-Alternative Turns"):
        return SectionStatus.WARNING, "Better-alternative turns found"
    return SectionStatus.HEALTHY, ""


def _status_conversation_style(items: dict) -> tuple:
    if items.get("Restarts Topic") == "Yes":
        return SectionStatus.PROBLEM, "Topic restarted"
    if items.get("Multiple Unrelated Questions") == "Yes":
        return SectionStatus.PROBLEM, "Multiple unrelated questions"
    if (items.get("Questions") or 0) > 1:
        return SectionStatus.WARNING, "Asks multiple questions"
    if items.get("Immediately Asks Question") == "Yes":
        return SectionStatus.WARNING, "Questionnaire-style reply"
    if items.get("Begins with Acknowledgment") == "No":
        return SectionStatus.WARNING, "No acknowledgment"
    return SectionStatus.HEALTHY, ""


def _status_conversation_style_summary(items: dict) -> tuple:
    if _rate_count(items.get("Topic Restart Rate")) > 0:
        return SectionStatus.PROBLEM, "Topic restarts recorded"
    if _rate_count(items.get("Multiple-Question Rate")) > 0:
        return SectionStatus.PROBLEM, "Multiple-question replies recorded"
    if _rate_count(items.get("Questionnaire Rate")) > 0:
        return SectionStatus.WARNING, "Questionnaire-style replies recorded"
    diversity = items.get("Opening Diversity Score")
    if isinstance(diversity, str) and "/" in diversity:
        try:
            if float(diversity.split("/")[0].strip()) < 0.5:
                return SectionStatus.WARNING, "Low opening diversity"
        except ValueError:
            pass
    return SectionStatus.HEALTHY, ""


def _status_conversation_failure(items: dict) -> tuple:
    primary = items.get("Primary Failure")
    if primary and primary not in ("None", "No failure", ""):
        return SectionStatus.PROBLEM, f"{primary} detected"
    secondary = items.get("Secondary Failures")
    if secondary not in (None, "", "None"):
        return SectionStatus.WARNING, "Secondary failure(s) present"
    return SectionStatus.HEALTHY, ""


def _status_conversation_failure_summary(items: dict) -> tuple:
    with_failures = items.get("Turns with Failures") or 0
    if with_failures > 0:
        rate = items.get("Turn Failure Rate", "")
        suffix = f" ({rate})" if rate else ""
        return SectionStatus.PROBLEM, f"{with_failures} turn(s) with failures{suffix}"
    return SectionStatus.HEALTHY, ""


def _status_recovery(items: dict) -> tuple:
    category = items.get("Category")
    inconsistency = items.get("Detected Inconsistency")
    if category == "CONTRADICTION":
        return SectionStatus.PROBLEM, inconsistency or "Contradiction detected"
    if category in ("UNCERTAIN_ANSWER", "DONT_KNOW"):
        return SectionStatus.WARNING, "Uncertain or unanswered"
    if category == "TOPIC_CHANGE":
        return SectionStatus.WARNING, "Topic changed"
    if category == "MULTIPLE_UNRELATED_FACTS":
        return SectionStatus.WARNING, "Multiple unrelated facts"
    if inconsistency:
        return SectionStatus.PROBLEM, inconsistency
    return SectionStatus.HEALTHY, ""


def _status_memory(items: dict) -> tuple:
    open_threads = items.get("Open Threads") or []
    partial = items.get("Partially Answered Objectives") or []
    deferred = items.get("Deferred Topics") or []
    if open_threads:
        return SectionStatus.WARNING, f"{len(open_threads)} open thread(s)"
    if partial:
        return SectionStatus.WARNING, (
            f"{len(partial)} partially answered objective(s)"
        )
    if deferred:
        return SectionStatus.WARNING, f"{len(deferred)} deferred topic(s)"
    return SectionStatus.HEALTHY, ""


def _status_insight_detection(items: dict) -> tuple:
    if items.get("Insight Type") == "CONTRADICTION":
        return SectionStatus.PROBLEM, "Contradictory insight"
    if items.get("Insight Type") and items.get("Confidence") == "LOW":
        return SectionStatus.WARNING, "Low-confidence insight"
    return SectionStatus.HEALTHY, ""


def _status_conversation_move(items: dict) -> tuple:
    move = items.get("Move")
    if move in (None, "", "NONE", "UNKNOWN", "PLACEHOLDER"):
        return SectionStatus.WARNING, "No move selected"
    return SectionStatus.HEALTHY, ""


def _status_checklist(items: dict) -> tuple:
    incomplete = [
        k for k, v in items.items()
        if isinstance(v, dict) and not v.get("complete")
    ]
    if incomplete:
        return SectionStatus.WARNING, f"{len(incomplete)} checklist item(s) incomplete"
    return SectionStatus.HEALTHY, ""


def _status_conversation_metrics(items: dict) -> tuple:
    if (items.get("Repeated Question Attempts") or 0) > 0:
        return SectionStatus.WARNING, "Repeated question attempts"
    return SectionStatus.HEALTHY, ""


def _status_question_families(items: dict) -> tuple:
    if items.get("Re-ask Sanctioned"):
        return SectionStatus.WARNING, "Re-ask sanctioned"
    return SectionStatus.HEALTHY, ""


def _status_hybrid_extraction(items: dict) -> tuple:
    if items.get("Objective Satisfied By Rules") == "No" and (
        items.get("LLM Invoked") == "No"
    ):
        return SectionStatus.PROBLEM, "Objective unsatisfied and LLM skipped"
    if items.get("LLM Invoked") == "No":
        return SectionStatus.WARNING, "LLM skipped (rules-only)"
    return SectionStatus.HEALTHY, ""


def _status_semantic_complexity(items: dict) -> tuple:
    if items.get("Complexity") == "HIGH" and items.get("Decision") == "SKIP_LLM":
        return SectionStatus.PROBLEM, "High complexity but LLM skipped"
    if items.get("Decision") == "SKIP_LLM":
        return SectionStatus.WARNING, "LLM skipped"
    return SectionStatus.HEALTHY, ""


def _status_hybrid_audit(items: dict) -> tuple:
    risk = items.get("Risk Level")
    if risk == "HIGH":
        return SectionStatus.PROBLEM, "High risk (would change objective)"
    if risk == "MEDIUM":
        return SectionStatus.WARNING, "Medium risk"
    return SectionStatus.HEALTHY, ""


def _status_hybrid_summary(items: dict) -> tuple:
    if (items.get("High risk") or 0) > 0:
        return SectionStatus.PROBLEM, f"{items['High risk']} high-risk LLM skip(s)"
    if (items.get("Medium risk") or 0) > 0:
        return SectionStatus.WARNING, "Medium-risk LLM skip(s)"
    return SectionStatus.HEALTHY, ""


def _status_prompt(items: Any) -> tuple:
    if isinstance(items, str) and len(items) > 8000:
        return SectionStatus.WARNING, "Very large prompt"
    return SectionStatus.HEALTHY, ""


_SECTION_STATUS_RULES: Dict[str, Any] = {
    "ProjectState": _status_project_state,
    "StateChanges": _status_state_changes,
    "ObjectiveTrace": _status_objective_trace,
    "Extraction": _status_extraction,
    "ExtractionComparison": _status_extraction_comparison,
    "ExtractionAccuracy": _status_extraction_accuracy,
    "ExtractionAccuracySummary": _status_extraction_accuracy_summary,
    "MentorDecision": _status_mentor_decision,
    "MentorDecisionSummary": _status_mentor_decision_summary,
    "ConversationStyle": _status_conversation_style,
    "ConversationStyleSummary": _status_conversation_style_summary,
    "ConversationFailure": _status_conversation_failure,
    "ConversationFailureSummary": _status_conversation_failure_summary,
    "Recovery": _status_recovery,
    "Memory": _status_memory,
    "InsightDetection": _status_insight_detection,
    "ConversationMove": _status_conversation_move,
    "ChecklistReasoning": _status_checklist,
    "ConversationMetrics": _status_conversation_metrics,
    "QuestionFamilies": _status_question_families,
    "HybridExtraction": _status_hybrid_extraction,
    "SemanticComplexity": _status_semantic_complexity,
    "HybridAudit": _status_hybrid_audit,
    "HybridSummary": _status_hybrid_summary,
    "Prompt": _status_prompt,
}


# ---------------------------------------------------------------------------
# Conversation tree (turn → category → section)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionNode:
    """One section inside a turn's tree. ``text`` is the exact rendered block
    (identical to ``build_render_plan`` for the same section); ``expanded``
    already folds in pinning + per-turn collapse state. ``status`` /
    ``status_reason`` are the derived visual health indicators (read-only
    classification of the section's own diagnostics — never a mutation)."""
    kind: str
    key: str
    category: str
    display_name: str
    text: str
    expanded: bool
    pinned: bool
    visible: bool
    search_matched: bool
    status: str = "healthy"
    status_reason: str = ""


@dataclass(frozen=True)
class CategoryNode:
    """A category group of ``SectionNode`` children (order-preserving)."""
    name: str
    sections: List[SectionNode]
    visible: bool


@dataclass(frozen=True)
class TurnNode:
    """One turn in the Conversation tree. ``categories`` always carries the
    turn's full materialized structure (every turn, searched or not); the UI
    only renders children for the selected turn. ``visible`` folds in search,
    and ``selected`` turns are always visible so the inspector never goes
    blank."""
    turn_index: int
    ordinal: int
    label: str
    selected: bool
    visible: bool
    search_matched: bool
    section_count: int
    categories: List[CategoryNode]


@dataclass(frozen=True)
class TreePlan:
    """Deterministic, search-filtered Conversation tree for the inspector."""
    turns: List[TurnNode]
    total_turns: int
    visible_turns: int
    visible_sections: int
    query: str
    categories: List[str] = field(default_factory=category_order)


def _as_int(value) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _normalize_collapse(collapse: Optional[dict]) -> Dict[Any, Dict[str, bool]]:
    """Normalise the persisted ``{turn_index: {section_key: bool}}`` state into
    an ``{int_turn_index: {section_key: bool}}`` map (accepting JSON-stringify
    survival)."""
    out: Dict[Any, Dict[str, bool]] = {}
    for turn_key, per_turn in (collapse or {}).items():
        per_turn = per_turn or {}
        if not isinstance(per_turn, dict):
            continue
        ti = _as_int(turn_key)
        bucket = out.setdefault(ti, {})
        for section_key, expanded in per_turn.items():
            if isinstance(expanded, (bool, int)):
                bucket[str(section_key)] = bool(expanded)
    return out


def _search_matches(query: str, *fields) -> bool:
    """True when ``query`` (case-insensitive substring) appears in any field.
    With an empty query every section/turn is visible and nothing is flagged
    'matched' by search."""
    if not query:
        return False
    q = query.lower()
    return any(q in str(f).lower() for f in fields if f is not None)


def _section_expanded(
    key: str,
    collapse: Dict[Any, Dict[str, bool]],
    pins: Set[str],
    turn_index: int,
    *,
    search_active: bool = False,
    matched: bool = False,
) -> bool:
    """Effective expanded state for a section.

    Pinned sections always win. When a search query is active (``search_active``)
    the expansion follows the search result: matching sections auto-expand and
    non-matching sections collapse, so the user lands directly on the hits. When
    the search is cleared, per-turn explicit state wins, then the module default
    (``DEFAULT_EXPANDED_SECTIONS``).
    """
    if key in pins:
        return True
    if search_active:
        return bool(matched)
    per_turn = collapse.get(turn_index) or {}
    if key in per_turn:
        return per_turn[key]
    return key in DEFAULT_EXPANDED_SECTIONS


def _build_section_nodes(
    turn: TurnDiagnostics,
    collapse: Dict[Any, Dict[str, bool]],
    pins: Set[str],
    query: str,
) -> List[SectionNode]:
    nodes: List[SectionNode] = []
    search_active = bool(query)
    performance_text = render_timing_lines(turn.timing)
    if performance_text:
        matched = _search_matches(query, "Performance", "Core Pipeline", performance_text)
        perf_status, perf_reason = section_status("Performance", timing=turn.timing)
        nodes.append(
            SectionNode(
                kind="performance",
                key="Performance",
                category=section_category("Performance"),
                display_name="Performance",
                text=performance_text,
                expanded=_section_expanded(
                    "Performance", collapse, pins, turn.turn_index,
                    search_active=search_active, matched=matched,
                ),
                pinned="Performance" in pins,
                visible=bool(not query or matched),
                search_matched=matched,
                status=perf_status.value,
                status_reason=perf_reason,
            )
        )
    for section in turn.sections:
        category = section_category(section.key)
        text = render_section_text(section)
        matched = _search_matches(query, section.key, section.display_name, category, text)
        status, reason = section_status(section.key, section.items)
        nodes.append(
            SectionNode(
                kind="prompt" if section.is_prompt else "section",
                key=section.key,
                category=category,
                display_name=section.display_name,
                text=text,
                expanded=_section_expanded(
                    section.key, collapse, pins, turn.turn_index,
                    search_active=search_active, matched=matched,
                ),
                pinned=section.key in pins,
                visible=bool(not query or matched),
                search_matched=matched,
                status=status.value,
                status_reason=reason,
            )
        )
    return nodes


def _build_turn_node(
    turn: TurnDiagnostics,
    selected: bool,
    collapse: Dict[Any, Dict[str, bool]],
    pins: Set[str],
    query: str,
) -> TurnNode:
    section_nodes = _build_section_nodes(turn, collapse, pins, query)
    categories: List[CategoryNode] = []
    for name in CATEGORY_ORDER:
        in_cat = [s for s in section_nodes if s.category == name]
        if in_cat:
            categories.append(
                CategoryNode(
                    name=name,
                    sections=in_cat,
                    visible=any(s.visible for s in in_cat),
                )
            )
    turn_matched = _search_matches(query, f"Turn {turn.ordinal}", turn.ordinal)
    visible = selected or turn_matched or any(s.visible for s in section_nodes)
    return TurnNode(
        turn_index=turn.turn_index,
        ordinal=turn.ordinal,
        label=turn_label(turn),
        selected=selected,
        visible=visible,
        search_matched=turn_matched,
        section_count=len(section_nodes),
        categories=categories,
    )


def build_tree_plan(
    turns: Optional[List[TurnDiagnostics]],
    *,
    selected_index: Optional[int] = None,
    collapse: Optional[Dict[Any, Dict[str, bool]]] = None,
    pins: Optional[Iterable[str]] = (),
    query: str = "",
) -> TreePlan:
    """Build the search-filtered Conversation tree for the inspector.

    ``turns`` are the collected ``TurnDiagnostics`` (as produced by
    ``collect_turns``). ``selected_index`` picks the highlighted turn (falls
    back to the last turn); ``collapse`` is the persisted per-turn
    ``{turn_index: {section_key: bool}}`` state; ``pins`` are globally pinned
    section keys (pinned sections render expanded in every turn); ``query`` is
    the instant-search string.

    Search matches turns (label/ordinal — never assistant text), section names
    (``display_name``), keys, category names, and the section's rendered text
    (content search — e.g. searching inside the ``Prompt`` block works). It
    only *filters* the tree; the diagnostic dictionaries and their rendered
    text are never mutated.

    This is a pure, deterministic model — it reads diagnostics by reference and
    returns shallow structural nodes, exactly like ``build_render_plan``.
    """
    turns = list(turns or [])
    if not turns:
        return TreePlan(
            turns=[],
            total_turns=0,
            visible_turns=0,
            visible_sections=0,
            query=(query or "").strip(),
        )
    collapse = _normalize_collapse(collapse)
    pins = set(pins or ())
    query = (query or "").strip()

    selected: Optional[TurnDiagnostics] = select_turn(turns, selected_index)
    if selected is None:
        selected = turns[-1]

    nodes = [
        _build_turn_node(t, t.turn_index == selected.turn_index, collapse, pins, query)
        for t in turns
    ]
    visible_turns = sum(1 for n in nodes if n.visible)
    visible_sections = sum(
        1 for n in nodes for c in n.categories for s in c.sections if s.visible
    )
    return TreePlan(
        turns=nodes,
        total_turns=len(nodes),
        visible_turns=visible_turns,
        visible_sections=visible_sections,
        query=query,
        categories=category_order(),
    )


# ---------------------------------------------------------------------------
# Pipeline Timeline (visual stage view — read-only)
# ---------------------------------------------------------------------------
#
# A fixed, ordered visualisation of the mentor pipeline for one turn: every
# stage carries its execution time (where measured), a derived status, curated
# metadata, and the rendered diagnostics of the sections that stage produced.
# Pure and read-only: it consumes the same timing + diagnostics dicts the tree
# uses and never mutates them. Timing-stage mapping (mentor.py):
#
#   prepare_ms      session prepare (before extraction)
#   extraction_ms   memory extraction
#   objective_ms    objective engine
#   lifecycle_ms    recovery + question family + lifecycle (shared)
#   prompt_ms       prompt builder   (inside reply generation)
#   llm_ms          LLM call        (inside reply generation)
#   finalize_ms     response finalize (session persist + reply assembly)


@dataclass(frozen=True)
class TimelineStage:
    """One visual stage of the pipeline timeline.

    ``duration_ms`` is the measured wall time for the stage (``None`` when no
    dedicated timing exists — e.g. recovery/question-family share the lifecycle
    window); ``status``/``status_reason`` mirror the section status model;
    ``metadata`` is an ordered list of ``(label, value)`` display pairs and
    ``detail`` is the rendered diagnostics text for the stage (click-to-open)."""

    key: str
    label: str
    icon: str
    duration_ms: Optional[float]
    status: str
    status_reason: str
    metadata: List[tuple]
    detail: str


def _timing_value(timing: Optional[dict], key: str) -> Optional[float]:
    if not isinstance(timing, dict):
        return None
    value = timing.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _stage_section_texts(turn: TurnDiagnostics, keys: Iterable[str]) -> List[str]:
    """Rendered text for the given diagnostic sections, prefixed by their
    display names, in the order they appear in the turn."""
    by_key = {s.key: s for s in turn.sections}
    out: List[str] = []
    for key in keys:
        section = by_key.get(key)
        if section is None:
            continue
        text = render_section_text(section)
        out.append(f"{section.display_name}\n{text}")
    return out


def _stage_status(*statuses) -> SectionStatus:
    """Worst of the given statuses (HEALTHY < WARNING < PROBLEM).

    Accepts either ``SectionStatus`` members or ``(status, reason)`` tuples
    as returned by ``section_status``."""
    rank = {
        SectionStatus.HEALTHY: 0,
        SectionStatus.WARNING: 1,
        SectionStatus.PROBLEM: 2,
    }
    worst = SectionStatus.HEALTHY
    for entry in statuses:
        status = entry[0] if isinstance(entry, tuple) else entry
        if rank[status] > rank[worst]:
            worst = status
    return worst


def _duration_status(ms: Optional[float], *, llm: bool = False) -> SectionStatus:
    if ms is None:
        return SectionStatus.HEALTHY
    if llm:
        if ms > 8000:
            return SectionStatus.PROBLEM
        if ms > 3000:
            return SectionStatus.WARNING
        return SectionStatus.HEALTHY
    if ms > 10000:
        return SectionStatus.PROBLEM
    if ms > 3000:
        return SectionStatus.WARNING
    return SectionStatus.HEALTHY


def build_timeline(turn: Optional[TurnDiagnostics]) -> List[TimelineStage]:
    """Build the ordered pipeline timeline for one assistant turn.

    Consumes ``turn.timing`` (per-stage wall times) and ``turn.diagnostics``
    (the EXACT dicts the pipeline emitted — by reference, never copied or
    mutated). Each stage maps to the sections that produced it, so clicking a
    stage reveals that stage's detailed diagnostics. Pure and deterministic.
    """
    if turn is None:
        return []
    timing = turn.timing or {}
    sections = {s.key: s for s in turn.sections}
    total_ms = _timing_value(timing, "total_ms")
    breakdown = timing.get("extraction_breakdown") or {}

    stages: List[TimelineStage] = []

    # 1. User Message
    stages.append(
        TimelineStage(
            key="user_message",
            label="User Message",
            icon="👤",
            duration_ms=None,
            status=SectionStatus.HEALTHY.value,
            status_reason="",
            metadata=[("message", turn.content_preview)],
            detail=turn.content_preview,
        )
    )

    # 2. Memory Extraction
    extraction_ms = _timing_value(timing, "extraction_ms")
    extraction_status = _stage_status(
        section_status("Extraction", sections["Extraction"].items if "Extraction" in sections else {}),
        section_status("ExtractionComparison", sections["ExtractionComparison"].items if "ExtractionComparison" in sections else {}),
        section_status("ExtractionAccuracy", sections["ExtractionAccuracy"].items if "ExtractionAccuracy" in sections else {}),
        _duration_status(extraction_ms),
    )
    extraction_meta: List[tuple] = []
    if "Extraction" in sections:
        items = sections["Extraction"].items
        if isinstance(items, dict) and items.get("message_type"):
            extraction_meta.append(("type", items["message_type"]))
            extraction_meta.append(("updates", len(items.get("updates") or [])))
    if "ExtractionComparison" in sections:
        decision = sections["ExtractionComparison"].items.get("Decision") or {}
        if decision.get("Rule sufficient"):
            extraction_meta.append(("rule sufficient", decision["Rule sufficient"]))
    if breakdown.get("prompt_tokens"):
        extraction_meta.append(("tokens", f"{breakdown['prompt_tokens']} (est.)"))
    stages.append(
        TimelineStage(
            key="memory_extraction",
            label="Memory Extraction",
            icon="🧠",
            duration_ms=extraction_ms,
            status=extraction_status.value,
            status_reason="",
            metadata=extraction_meta,
            detail="\n\n".join(
                _stage_section_texts(turn, ("Extraction", "ExtractionComparison", "ExtractionAccuracy", "HybridExtraction", "SemanticComplexity"))
            ),
        )
    )

    # 3. Objective Engine
    objective_ms = _timing_value(timing, "objective_ms")
    objective_section_status, objective_reason = section_status(
        "ObjectiveTrace", sections["ObjectiveTrace"].items if "ObjectiveTrace" in sections else {}
    )
    objective_status = _stage_status(objective_section_status, _duration_status(objective_ms))
    objective_meta: List[tuple] = []
    if "ObjectiveTrace" in sections:
        items = sections["ObjectiveTrace"].items
        if isinstance(items, dict):
            if items.get("Objective"):
                objective_meta.append(("objective", items["Objective"]))
            if items.get("Confidence") is not None:
                objective_meta.append(("confidence", items["Confidence"]))
            verdict = (items.get("Advancement") or {}).get("Verdict")
            if verdict:
                objective_meta.append(("verdict", verdict))
    stages.append(
        TimelineStage(
            key="objective_engine",
            label="Objective Engine",
            icon="🎯",
            duration_ms=objective_ms,
            status=objective_status.value,
            status_reason=objective_reason,
            metadata=objective_meta,
            detail="\n\n".join(_stage_section_texts(turn, ("ObjectiveTrace", "MentorDecision"))),
        )
    )

    # 4. Recovery
    recovery_status, recovery_reason = section_status(
        "Recovery", sections["Recovery"].items if "Recovery" in sections else {}
    )
    recovery_meta: List[tuple] = []
    if "Recovery" in sections:
        items = sections["Recovery"].items
        if isinstance(items, dict):
            if items.get("Category") and items["Category"] != "NONE":
                recovery_meta.append(("category", items["Category"]))
            if items.get("Recovery Strategy") and items["Recovery Strategy"] != "NONE":
                recovery_meta.append(("strategy", items["Recovery Strategy"]))
    stages.append(
        TimelineStage(
            key="recovery",
            label="Recovery",
            icon="🔄",
            duration_ms=None,
            status=recovery_status.value,
            status_reason=recovery_reason,
            metadata=recovery_meta,
            detail="\n\n".join(_stage_section_texts(turn, ("Recovery",))),
        )
    )

    # 5. Question Family
    family_status, family_reason = section_status(
        "QuestionFamilies", sections["QuestionFamilies"].items if "QuestionFamilies" in sections else {}
    )
    family_meta: List[tuple] = []
    if "QuestionFamilies" in sections:
        items = sections["QuestionFamilies"].items
        if isinstance(items, dict):
            if items.get("Question Family"):
                family_meta.append(("family", items["Question Family"]))
            if items.get("Re-ask Sanctioned"):
                family_meta.append(("re-ask", "sanctioned"))
    stages.append(
        TimelineStage(
            key="question_family",
            label="Question Family",
            icon="💬",
            duration_ms=None,
            status=family_status.value,
            status_reason=family_reason,
            metadata=family_meta,
            detail="\n\n".join(_stage_section_texts(turn, ("QuestionFamilies", "QuestionHistory"))),
        )
    )

    # 6. Prompt Builder
    prompt_ms = _timing_value(timing, "prompt_ms")
    prompt_section_status, prompt_reason = section_status(
        "Prompt", sections["Prompt"].items if "Prompt" in sections else {}
    )
    prompt_status = _stage_status(prompt_section_status, _duration_status(prompt_ms))
    prompt_meta: List[tuple] = []
    if "Prompt" in sections:
        stats = prompt_viewer_stats(render_prompt_text(sections["Prompt"].items))
        prompt_meta.append(("chars", stats["characters"]))
        prompt_meta.append(("words", stats["words"]))
        prompt_meta.append(("lines", stats["lines"]))
    stages.append(
        TimelineStage(
            key="prompt_builder",
            label="Prompt Builder",
            icon="📜",
            duration_ms=prompt_ms,
            status=prompt_status.value,
            status_reason=prompt_reason,
            metadata=prompt_meta,
            detail="\n\n".join(_stage_section_texts(turn, ("Prompt",))),
        )
    )

    # 7. LLM
    llm_ms = _timing_value(timing, "llm_ms")
    llm_status = _duration_status(llm_ms, llm=True)
    llm_reason = ""
    if llm_ms == 0 and (total_ms or 0) > 0:
        llm_status = _stage_status(llm_status, SectionStatus.WARNING)
        llm_reason = "LLM not invoked (rules-only path)"
    llm_meta: List[tuple] = []
    if "Pipeline" in sections:
        items = sections["Pipeline"].items
        if isinstance(items, dict) and items.get("Model"):
            llm_meta.append(("model", items["Model"]))
    stages.append(
        TimelineStage(
            key="llm",
            label="LLM",
            icon="🤖",
            duration_ms=llm_ms,
            status=llm_status.value,
            status_reason=llm_reason,
            metadata=llm_meta,
            detail="\n\n".join(_stage_section_texts(turn, ())),
        )
    )

    # 8. Response
    finalize_ms = _timing_value(timing, "finalize_ms")
    response_status = _stage_status(
        section_status("ConversationStyle", sections["ConversationStyle"].items if "ConversationStyle" in sections else {}),
        section_status("ConversationMove", sections["ConversationMove"].items if "ConversationMove" in sections else {}),
        _duration_status(finalize_ms),
    )
    response_meta: List[tuple] = [("words", len(turn.content_preview.split()))]
    if "Pipeline" in sections:
        items = sections["Pipeline"].items
        if isinstance(items, dict) and items.get("Response Strategy"):
            response_meta.append(("strategy", items["Response Strategy"]))
    if finalize_ms is not None:
        response_meta.append(("finalize", format_duration(finalize_ms)))
    stages.append(
        TimelineStage(
            key="response",
            label="Response",
            icon="✍️",
            duration_ms=finalize_ms,
            status=response_status.value,
            status_reason="",
            metadata=response_meta,
            detail="\n\n".join(_stage_section_texts(turn, ("ConversationStyle", "ConversationMove", "ConversationMetrics"))),
        )
    )

    # 9. Diagnostics
    diagnostics_status = SectionStatus.HEALTHY
    problems = 0
    warnings = 0
    for section in turn.sections:
        status, _ = section_status(section.key, section.items)
        diagnostics_status = _stage_status(diagnostics_status, status)
        if status is SectionStatus.PROBLEM:
            problems += 1
        elif status is SectionStatus.WARNING:
            warnings += 1
    diagnostics_meta: List[tuple] = []
    if total_ms is not None:
        diagnostics_meta.append(("total", format_duration(total_ms)))
    diagnostics_meta.append(("sections", len(turn.sections)))
    if problems:
        diagnostics_meta.append(("problems", problems))
    if warnings:
        diagnostics_meta.append(("warnings", warnings))
    stages.append(
        TimelineStage(
            key="diagnostics",
            label="Diagnostics",
            icon="🧾",
            duration_ms=total_ms,
            status=diagnostics_status.value,
            status_reason="" if diagnostics_status is SectionStatus.HEALTHY else diagnostics_status.value,
            metadata=diagnostics_meta,
            detail="",
        )
    )

    return stages
