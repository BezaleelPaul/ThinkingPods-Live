"""
product_experience_audit.py — measurement-only audit of the end-to-end
product experience of the Streamlit application.

Follows the exact philosophy of the existing audits (``conversation_style_audit``,
``mentor_decision_audit``, ``prompt_effectiveness``, Replay Lab): it is
STRICTLY observational.

The audit NEVER:

  * changes application behavior
  * changes the UI / Streamlit layout
  * affects conversation flow, objectives, extraction, lifecycle,
    PromptBuilder, or any decision path
  * mutates ProjectState, SessionData, or any prompt

It only *reads* data the pipeline already produces — the Developer Console
``diagnostics`` dict, the per-turn ``TurnTiming``, and ``conversation_history``
— and projects it into a deterministic, JSON-safe record.

Areas audited
-------------

A. Startup
   The first observed call into the mentor pipeline is the earliest
   measurable "application start" signal from the backend. We record the
   wall-clock time of that first turn, the session/state preparation time
   (mentor creation), and the first message processing time.

B. Session
   Whether the session was freshly seeded or restored, its id, the message
   count present at start, whether a prior conversation was loaded, and an
   observation-only "refresh required" flag.

C. Developer Console
   Per-turn shape/cost of the diagnostics block: section count, total leaf
   entries, the largest section, prompt char size, nesting depth, and an
   estimate of the rendered lines.

D. Conversation Export
   Rough size of what a user would export: the number of chat messages, the
   estimated exported characters (header + content), the diagnostics size,
   and the combined export total.

E. Performance Timeline
   Per-turn wall-clock breakdown: extraction, merge, objective, prompt build,
   LLM, render (finalize as the closest signal), and total turn time.

The module imports only the standard library so it stays a cycle-free leaf of
the dependency graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "ProductExperienceSummary",
    "analyze_product_experience",
    "product_experience_diagnostics_section",
]

# Per-message formatting overhead assumed for the export estimate (role label
# + markdown separators + blank line); purely an observational heuristic.
_MESSAGE_OVERHEAD_CHARS = 12


# ---------------------------------------------------------------------------
# Deterministic measurement helpers
# ---------------------------------------------------------------------------


def _json_chars(value: Any) -> int:
    """Number of characters a value occupies once JSON-serialised."""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _entry_count(value: Any) -> int:
    """Count scalar leaf entries in a nested dict/list (each leaf = 1)."""
    if isinstance(value, dict):
        return sum(_entry_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_entry_count(v) for v in value)
    return 1


def _max_depth(value: Any, depth: int = 1) -> int:
    """Maximum nesting depth of a nested dict/list structure."""
    if isinstance(value, dict):
        if not value:
            return depth
        return max(_max_depth(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        if not value:
            return depth
        return max(_max_depth(v, depth + 1) for v in value)
    return depth


def _estimate_rendered_lines(diagnostics: dict | None) -> int:
    """Rough count of lines the Developer Console would render for a
    diagnostics block — one line per scalar leaf entry."""
    return _entry_count(diagnostics or {})


def _largest_section(diagnostics: dict | None) -> Optional[list]:
    """Return ``[section_key, char_size]`` of the largest section by JSON
    size, or ``None`` for an empty diagnostics block."""
    best_key: Optional[str] = None
    best_size = -1
    for key, value in (diagnostics or {}).items():
        size = _json_chars(value)
        if size > best_size:
            best_size = size
            best_key = key
    if best_key is None:
        return None
    return [best_key, best_size]


def _estimate_export_chars(history: list | None) -> int:
    """Rough markdown-exported character count for a message history."""
    total = 0
    for m in (history or []):
        role = str(m.get("role", "")) or ""
        content = str(m.get("content", "")) or ""
        total += len(content) + len(role) + _MESSAGE_OVERHEAD_CHARS
    return total


# ---------------------------------------------------------------------------
# Per-turn analysis
# ---------------------------------------------------------------------------


def analyze_product_experience(
    *,
    turn_index: int = 1,
    session_id: str = "",
    conversation_history: list | None = None,
    timing: dict | None = None,
    diagnostics: dict | None = None,
    first_turn: bool = False,
    history_count_at_start: int = 0,
    turn_metrics_at_start: int = 0,
    startup_timestamp: str = "",
) -> dict:
    """Deterministic per-turn product-experience assessment.

    Pure function of the pipeline's existing per-turn record (Developer
    Console diagnostics, timing, conversation history). Mutates nothing.
    Returns a JSON-serialisable record for the Developer Console and the
    session-level ``ProductExperienceSummary``.

    ``startup`` / ``session`` measurements are only emitted on the first
    observed turn of a session (``first_turn=True``).
    """
    timing = timing or {}
    diagnostics = diagnostics or {}
    history = list(conversation_history or [])

    extraction_breakdown = dict(timing.get("extraction_breakdown") or {})
    merge_ms = float(extraction_breakdown.get("merge_ms", 0.0) or 0.0)

    developer_console = {
        "section_count": len(diagnostics),
        "total_entries": _entry_count(diagnostics),
        "largest_section": _largest_section(diagnostics),
        "prompt_size_chars": _json_chars(diagnostics.get("Prompt")) if "Prompt" in diagnostics else 0,
        "max_nesting_depth": _max_depth(diagnostics) if diagnostics else 0,
        "estimated_rendered_lines": _estimate_rendered_lines(diagnostics),
    }

    export = {
        "total_messages": len(history),
        "estimated_export_chars": _estimate_export_chars(history),
        "diagnostics_size_chars": _json_chars(diagnostics),
        "total_export_chars": (
            _estimate_export_chars(history) + _json_chars(diagnostics)
        ),
    }

    performance = {
        "extraction_ms": round(float(timing.get("extraction_ms", 0.0) or 0.0)),
        "merge_ms": round(merge_ms),
        "objective_ms": round(float(timing.get("objective_ms", 0.0) or 0.0)),
        "prompt_build_ms": round(float(timing.get("prompt_ms", 0.0) or 0.0)),
        "llm_ms": round(float(timing.get("llm_ms", 0.0) or 0.0)),
        # "render" is not separately measurable from the backend; the
        # post-reply finalize work is the closest observational signal.
        "render_ms": round(float(timing.get("finalize_ms", 0.0) or 0.0)),
        "total_ms": round(float(timing.get("total_ms", 0.0) or 0.0)),
    }

    record: Dict[str, Any] = {
        "turn_index": turn_index,
        "developer_console": developer_console,
        "export": export,
        "performance": performance,
    }

    if first_turn:
        prepare_ms = round(float(timing.get("prepare_ms", 0.0) or 0.0))
        first_message_ms = round(float(timing.get("total_ms", 0.0) or 0.0))
        history_at_start = int(history_count_at_start or 0)
        metrics_at_start = int(turn_metrics_at_start or 0)
        restored = bool(history_at_start > 0 or metrics_at_start > 0)
        previous_present = bool(history_at_start > 0)

        record["startup"] = {
            # Earliest measurable "one page is live" signal: the wall-clock
            # time of the first message processed by the backend.
            "app_startup_iso": startup_timestamp,
            # Session + state preparation before extraction (mentor "creation").
            "mentor_creation_ms": prepare_ms,
            "first_page_render_ms": None,  # not measurable from the backend
            "first_message_ms": first_message_ms,
            "total_startup_duration_ms": first_message_ms,
        }
        record["session"] = {
            "session_id": session_id,
            "session_restored": restored,
            "message_count_at_start": history_at_start,
            "previous_conversation_present": previous_present,
            # Observation only: a restored/present prior conversation is the
            # situation where a browser refresh would discard in-memory UI.
            "refresh_required": previous_present,
        }

    return record


# ---------------------------------------------------------------------------
# Developer Console per-turn section
# ---------------------------------------------------------------------------


def product_experience_diagnostics_section(record: dict | None) -> dict:
    """Developer Console projection of one turn's product-experience record.

    Returns an empty dict when there is no record this turn.
    """
    if not record:
        return {}
    dc = record.get("developer_console") or {}
    export = record.get("export") or {}
    perf = record.get("performance") or {}
    largest = dc.get("largest_section")
    largest_text = (
        f"{largest[0]} ({largest[1]} chars)" if largest else "—"
    )
    return {
        "Turn": record.get("turn_index"),
        "Developer Console → Sections": dc.get("section_count"),
        "Developer Console → Entries": dc.get("total_entries"),
        "Developer Console → Largest Section": largest_text,
        "Developer Console → Prompt Size": f"{dc.get('prompt_size_chars', 0)} chars",
        "Developer Console → Max Nesting Depth": dc.get("max_nesting_depth"),
        "Developer Console → Est. Rendered Lines": dc.get("estimated_rendered_lines"),
        "Export → Total Messages": export.get("total_messages"),
        "Export → Estimated Chars": export.get("estimated_export_chars"),
        "Export → Diagnostics Size": f"{export.get('diagnostics_size_chars', 0)} chars",
        "Export → Total Chars": export.get("total_export_chars"),
        "Performance → Extraction": f"{perf.get('extraction_ms', 0)} ms",
        "Performance → Merge": f"{perf.get('merge_ms', 0)} ms",
        "Performance → Objective": f"{perf.get('objective_ms', 0)} ms",
        "Performance → Prompt Build": f"{perf.get('prompt_build_ms', 0)} ms",
        "Performance → LLM": f"{perf.get('llm_ms', 0)} ms",
        "Performance → Render": f"{perf.get('render_ms', 0)} ms",
        "Performance → Total": f"{perf.get('total_ms', 0)} ms",
    }


# ---------------------------------------------------------------------------
# Session-level aggregate
# ---------------------------------------------------------------------------


@dataclass
class ProductExperienceSummary:
    """Append-only aggregate of per-turn product-experience records.

    Persists with the session (JSON-safe). Never read by any decision path —
    Developer Console + offline report only.
    """

    turn_count: int = 0
    startup: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)
    session_ids_seen: List[str] = field(default_factory=list)
    restored_count: int = 0
    fresh_count: int = 0
    turn_records: List[dict] = field(default_factory=list)
    # performance aggregates
    total_extraction_ms: float = 0.0
    total_merge_ms: float = 0.0
    total_objective_ms: float = 0.0
    total_prompt_build_ms: float = 0.0
    total_llm_ms: float = 0.0
    total_render_ms: float = 0.0
    total_turn_ms: float = 0.0
    max_turn_ms: float = 0.0
    # developer console aggregates
    total_sections: int = 0
    total_diag_entries: int = 0
    total_prompt_chars: int = 0
    max_nesting_depth: int = 0
    total_estimated_lines: int = 0
    # export aggregates
    total_messages: int = 0
    max_messages: int = 0
    total_diagnostics_chars: int = 0
    total_export_chars: int = 0

    # ---- aggregation -------------------------------------------------------

    def add_record(self, record: dict | None) -> None:
        if not record:
            return
        self.turn_count += 1

        # Startup / session are captured once (on the first observed turn).
        if record.get("startup") and not self.startup:
            self.startup = record["startup"]
        if record.get("session") and not self.session:
            self.session = record["session"]
            sid = record["session"].get("session_id")
            if sid and sid not in self.session_ids_seen:
                self.session_ids_seen.append(sid)
            if record["session"].get("session_restored"):
                self.restored_count += 1
            else:
                self.fresh_count += 1

        perf = record.get("performance") or {}
        self.total_extraction_ms += float(perf.get("extraction_ms", 0) or 0)
        self.total_merge_ms += float(perf.get("merge_ms", 0) or 0)
        self.total_objective_ms += float(perf.get("objective_ms", 0) or 0)
        self.total_prompt_build_ms += float(perf.get("prompt_build_ms", 0) or 0)
        self.total_llm_ms += float(perf.get("llm_ms", 0) or 0)
        self.total_render_ms += float(perf.get("render_ms", 0) or 0)
        total_ms = float(perf.get("total_ms", 0) or 0)
        self.total_turn_ms += total_ms
        self.max_turn_ms = max(self.max_turn_ms, total_ms)

        dc = record.get("developer_console") or {}
        self.total_sections += int(dc.get("section_count", 0) or 0)
        self.total_diag_entries += int(dc.get("total_entries", 0) or 0)
        self.total_prompt_chars += int(dc.get("prompt_size_chars", 0) or 0)
        self.max_nesting_depth = max(
            self.max_nesting_depth, int(dc.get("max_nesting_depth", 0) or 0)
        )
        self.total_estimated_lines += int(dc.get("estimated_rendered_lines", 0) or 0)

        export = record.get("export") or {}
        messages = int(export.get("total_messages", 0) or 0)
        self.total_messages += messages
        self.max_messages = max(self.max_messages, messages)
        self.total_diagnostics_chars += int(export.get("diagnostics_size_chars", 0) or 0)
        self.total_export_chars += int(export.get("total_export_chars", 0) or 0)

        self.turn_records.append(record)

    def merge(self, other: "ProductExperienceSummary") -> None:
        self.turn_count += other.turn_count
        if not self.startup and other.startup:
            self.startup = dict(other.startup)
        if not self.session and other.session:
            self.session = dict(other.session)
        for sid in other.session_ids_seen:
            if sid and sid not in self.session_ids_seen:
                self.session_ids_seen.append(sid)
        self.restored_count += other.restored_count
        self.fresh_count += other.fresh_count
        self.turn_records.extend(other.turn_records)
        self.total_extraction_ms += other.total_extraction_ms
        self.total_merge_ms += other.total_merge_ms
        self.total_objective_ms += other.total_objective_ms
        self.total_prompt_build_ms += other.total_prompt_build_ms
        self.total_llm_ms += other.total_llm_ms
        self.total_render_ms += other.total_render_ms
        self.total_turn_ms += other.total_turn_ms
        self.max_turn_ms = max(self.max_turn_ms, other.max_turn_ms)
        self.total_sections += other.total_sections
        self.total_diag_entries += other.total_diag_entries
        self.total_prompt_chars += other.total_prompt_chars
        self.max_nesting_depth = max(
            self.max_nesting_depth, other.max_nesting_depth
        )
        self.total_estimated_lines += other.total_estimated_lines
        self.total_messages += other.total_messages
        self.max_messages = max(self.max_messages, other.max_messages)
        self.total_diagnostics_chars += other.total_diagnostics_chars
        self.total_export_chars += other.total_export_chars

    # ---- metrics -----------------------------------------------------------

    def average_turn_time_ms(self) -> float:
        if not self.turn_count:
            return 0.0
        return round(self.total_turn_ms / self.turn_count, 1)

    def average_section_count(self) -> float:
        if not self.turn_count:
            return 0.0
        return round(self.total_sections / self.turn_count, 1)

    def average_diag_entries(self) -> float:
        if not self.turn_count:
            return 0.0
        return round(self.total_diag_entries / self.turn_count, 1)

    def average_prompt_chars(self) -> float:
        if not self.turn_count:
            return 0.0
        return round(self.total_prompt_chars / self.turn_count, 1)

    def average_export_chars(self) -> float:
        if not self.turn_count:
            return 0.0
        return round(self.total_export_chars / self.turn_count, 1)

    # ---- persistence / display ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "startup": self.startup,
            "session": self.session,
            "session_ids_seen": list(self.session_ids_seen),
            "restored_count": self.restored_count,
            "fresh_count": self.fresh_count,
            "turn_records": self.turn_records,
            "total_extraction_ms": self.total_extraction_ms,
            "total_merge_ms": self.total_merge_ms,
            "total_objective_ms": self.total_objective_ms,
            "total_prompt_build_ms": self.total_prompt_build_ms,
            "total_llm_ms": self.total_llm_ms,
            "total_render_ms": self.total_render_ms,
            "total_turn_ms": self.total_turn_ms,
            "max_turn_ms": self.max_turn_ms,
            "total_sections": self.total_sections,
            "total_diag_entries": self.total_diag_entries,
            "total_prompt_chars": self.total_prompt_chars,
            "max_nesting_depth": self.max_nesting_depth,
            "total_estimated_lines": self.total_estimated_lines,
            "total_messages": self.total_messages,
            "max_messages": self.max_messages,
            "total_diagnostics_chars": self.total_diagnostics_chars,
            "total_export_chars": self.total_export_chars,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "ProductExperienceSummary":
        d = d or {}
        return cls(
            turn_count=int(d.get("turn_count", 0)),
            startup=dict(d.get("startup", {}) or {}),
            session=dict(d.get("session", {}) or {}),
            session_ids_seen=list(d.get("session_ids_seen", []) or []),
            restored_count=int(d.get("restored_count", 0)),
            fresh_count=int(d.get("fresh_count", 0)),
            turn_records=list(d.get("turn_records", []) or []),
            total_extraction_ms=float(d.get("total_extraction_ms", 0.0)),
            total_merge_ms=float(d.get("total_merge_ms", 0.0)),
            total_objective_ms=float(d.get("total_objective_ms", 0.0)),
            total_prompt_build_ms=float(d.get("total_prompt_build_ms", 0.0)),
            total_llm_ms=float(d.get("total_llm_ms", 0.0)),
            total_render_ms=float(d.get("total_render_ms", 0.0)),
            total_turn_ms=float(d.get("total_turn_ms", 0.0)),
            max_turn_ms=float(d.get("max_turn_ms", 0.0)),
            total_sections=int(d.get("total_sections", 0)),
            total_diag_entries=int(d.get("total_diag_entries", 0)),
            total_prompt_chars=int(d.get("total_prompt_chars", 0)),
            max_nesting_depth=int(d.get("max_nesting_depth", 0)),
            total_estimated_lines=int(d.get("total_estimated_lines", 0)),
            total_messages=int(d.get("total_messages", 0)),
            max_messages=int(d.get("max_messages", 0)),
            total_diagnostics_chars=int(d.get("total_diagnostics_chars", 0)),
            total_export_chars=int(d.get("total_export_chars", 0)),
        )

    def to_display(self) -> dict:
        """Developer Console projection of the running aggregate."""
        return {
            "Turns Analyzed": self.turn_count,
            "Session Restored": "Yes" if self.session.get("session_restored") else "No",
            "Session Id": self.session.get("session_id", "—"),
            "Message Count at Start": self.session.get("message_count_at_start", 0),
            "Previous Conversation Present": "Yes" if self.session.get("previous_conversation_present") else "No",
            "Startup → First Message": f"{self.startup.get('first_message_ms', 0)} ms",
            "Startup → Mentor Creation": f"{self.startup.get('mentor_creation_ms', 0)} ms",
            "Startup → Total": f"{self.startup.get('total_startup_duration_ms', 0)} ms",
            "Avg Turn Time": f"{self.average_turn_time_ms()} ms",
            "Total Turn Time": f"{round(self.total_turn_ms)} ms",
            "Max Turn Time": f"{round(self.max_turn_ms)} ms",
            "Avg Extraction": f"{round(self.total_extraction_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg Merge": f"{round(self.total_merge_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg Objective": f"{round(self.total_objective_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg Prompt Build": f"{round(self.total_prompt_build_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg LLM": f"{round(self.total_llm_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg Render": f"{round(self.total_render_ms / self.turn_count) if self.turn_count else 0} ms",
            "Avg Developer Console Sections": self.average_section_count(),
            "Avg Developer Console Entries": self.average_diag_entries(),
            "Max Nesting Depth": self.max_nesting_depth,
            "Avg Prompt Size": f"{self.average_prompt_chars()} chars",
            "Avg Estimated Rendered Lines": (
                round(self.total_estimated_lines / self.turn_count) if self.turn_count else 0
            ),
            "Total Export Chars": self.total_export_chars,
            "Avg Export Chars": f"{self.average_export_chars()} chars",
        }