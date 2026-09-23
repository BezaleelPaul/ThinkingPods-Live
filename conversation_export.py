"""
conversation_export.py — deterministic conversation export utilities.

Pure, standard-library-only. Produces the four export artefacts the
Streamlit app exposes:

  * **Copy Conversation**            — markdown of the conversation only.
  * **Copy Conversation + Diagnostics** — markdown of the conversation with
                                     each assistant turn's Developer Console
                                     diagnostics (timing + sections).
  * **Export Markdown**              — full export: Conversation, Developer
                                     Diagnostics, Mission Memory, and
                                     Conversation Metrics.
  * **Export JSON**                  — the complete session as a canonical,
                                     byte-deterministic JSON document that can
                                     be reloaded.

Determinism guarantees (all unit-tested):

  * the same input messages always produce the same markdown text,
  * JSON serialisation is *canonical* (keys sorted recursively, stable
    indent / ensure_ascii) so ``export_session_json(reload_session_json(x))
    == x`` byte-for-byte,
  * nothing is mutated — diagnostics, timing, and message dicts are read by
    reference and never changed.

The module NEVER:

  * changes conversation behavior, extraction, objectives, prompts, or any
    pipeline logic,
  * imports streamlit — it stays a cycle-free leaf and is unit-tested.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

__all__ = [
    "build_conversation_copy",
    "build_conversation_diagnostics_copy",
    "build_full_markdown",
    "build_session_payload",
    "conversation_metrics_from_messages",
    "export_session_json",
    "reload_session_json",
    "serialize_session_json",
]

_FORMAT_VERSION = 1

_ROLE_LABELS = {
    "assistant": "Mentor",
    "user": "User",
    "system": "System",
}


def _role_label(role) -> str:
    return _ROLE_LABELS.get(role, role or "Unknown")


def _content(message: dict) -> str:
    return str(message.get("content") or "")


def _preview(text: str, limit: int = 48) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def build_conversation_section(messages: List[dict]) -> str:
    """Markdown of the raw conversation (roles + content)."""
    if not messages:
        return "_No messages yet._"
    parts = []
    for message in messages:
        role = _role_label(message.get("role"))
        content = _content(message)
        parts.append(f"**{role}:**\n\n{content}")
    return "\n\n---\n\n".join(parts)


def _render_timing_lines(timing: Optional[dict]) -> List[str]:
    if not timing:
        return []
    total = timing.get("total_ms", 0)
    lines = [f"  Total: {_format_ms(total)}"]
    for label, key in _TIMING_LABELS:
        val = timing.get(key)
        if val is not None:
            lines.append(f"  {label}: {_format_ms(val)}")
    return lines


_TIMING_LABELS = [
    ("LLM", "llm_ms"),
    ("Prompt", "prompt_ms"),
    ("Extraction", "extraction_ms"),
    ("Prepare", "prepare_ms"),
    ("Objective", "objective_ms"),
    ("Lifecycle", "lifecycle_ms"),
    ("Finalize", "finalize_ms"),
]


def _format_ms(ms) -> str:
    try:
        ms = float(ms or 0)
    except (TypeError, ValueError):
        ms = 0.0
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{round(ms)} ms"


def _render_diagnostics_block(
    diagnostics: Optional[dict], timing: Optional[dict]
) -> List[str]:
    """One turn's Developer Console as a text block (timing + sections)."""
    from diag_render import render_diag_items

    lines: List[str] = []
    timing_lines = _render_timing_lines(timing)
    if timing_lines:
        lines.append("**Performance:**")
        lines.extend(timing_lines)
        lines.append("")
    if not diagnostics:
        if not timing_lines:
            lines.append("_No diagnostics._")
        return lines
    for section_name, items in diagnostics.items():
        lines.append(f"**{section_name}:**")
        if not items:
            lines.append("  No state changes")
        else:
            lines.extend(f"  {line}" for line in render_diag_items(items))
        lines.append("")
    return lines


def build_diagnostics_section(messages: List[dict]) -> str:
    """Markdown Developer Diagnostics block: one subsection per assistant
    turn that carried timing/diagnostics."""
    if not messages:
        return "_No diagnostics._"
    parts: List[str] = []
    ordinal = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        diagnostics = message.get("diagnostics")
        timing = message.get("timing")
        if not diagnostics and not timing:
            continue
        ordinal += 1
        preview = _preview(_content(message))
        parts.append(f"### Turn {ordinal} — {preview}")
        block = "\n".join(_render_diagnostics_block(diagnostics, timing)).rstrip()
        parts.append(f"```text\n{block}\n```")
    if not parts:
        return "_No diagnostics._"
    return "\n\n".join(parts)


def build_mission_memory_section(
    project_name: str = "MyProject",
    saved_missions: Optional[List[str]] = None,
) -> str:
    """Markdown Mission Memory block: current project + persisted missions."""
    lines = [f"- **Project:** {project_name or 'MyProject'}"]
    missions = list(saved_missions or [])
    if missions:
        lines.append("- **Saved Missions:**")
        for name in missions:
            lines.append(f"  - {name}")
    else:
        lines.append("- **Saved Missions:** _(none)_")
    return "\n".join(lines)


def conversation_metrics_from_messages(messages: List[dict]) -> dict:
    """The Conversation Metrics dict from the latest turn that carried one
    (deterministic: last non-empty ``ConversationMetrics`` section wins)."""
    result: dict = {}
    for message in messages:
        diagnostics = message.get("diagnostics") or {}
        metrics = diagnostics.get("ConversationMetrics")
        if metrics:
            result = dict(metrics)
    return result


def build_metrics_section_text(metrics: Optional[dict]) -> str:
    """Markdown Conversation Metrics block from a metrics dict."""
    metrics = metrics or {}
    if not metrics:
        return "_No conversation metrics._"
    return "\n".join(f"- **{key}:** {value}" for key, value in metrics.items())


def build_conversation_copy(messages: List[dict]) -> str:
    """Copy Conversation: markdown of the conversation only."""
    return build_conversation_section(messages)


def build_conversation_diagnostics_copy(
    messages: List[dict], *, project_name: str = "MyProject"
) -> str:
    """Copy Conversation + Diagnostics: conversation followed by the
    Developer Diagnostics block."""
    header = f"# Conversation + Diagnostics\n\n**Project:** {project_name}\n\n"
    body = build_conversation_section(messages)
    diagnostics = build_diagnostics_section(messages)
    return (
        f"{header}"
        f"## Conversation\n\n{body}\n\n"
        f"## Developer Diagnostics\n\n{diagnostics}\n"
    )


def build_full_markdown(
    messages: List[dict],
    *,
    project_name: str = "MyProject",
    timestamp: str = "",
    saved_missions: Optional[List[str]] = None,
    metrics: Optional[dict] = None,
) -> str:
    """Full Export Markdown: Conversation, Developer Diagnostics, Mission
    Memory, and Conversation Metrics."""
    metrics = metrics if metrics is not None else conversation_metrics_from_messages(messages)
    ts = timestamp or "(not recorded)"
    return (
        f"# Conversation Export\n\n"
        f"**Project:** {project_name}\n"
        f"**Exported:** {ts}\n\n"
        f"## Conversation\n\n{build_conversation_section(messages)}\n\n"
        f"## Developer Diagnostics\n\n{build_diagnostics_section(messages)}\n\n"
        f"## Mission Memory\n\n{build_mission_memory_section(project_name, saved_missions)}\n\n"
        f"## Conversation Metrics\n\n{build_metrics_section_text(metrics)}\n"
    )


# ---------------------------------------------------------------------------
# JSON export (canonical, byte-deterministic)
# ---------------------------------------------------------------------------


def build_session_payload(
    messages: List[dict],
    *,
    project_name: str = "MyProject",
    dt_phase: str = "Empathize",
    session_id: str = "",
    timestamp: str = "",
    saved_missions: Optional[List[str]] = None,
    conversation_metrics: Optional[dict] = None,
) -> dict:
    """The complete-session payload. Messages keep their full dicts
    (role, content, audio, timing, diagnostics) — nothing is stripped or
    mutated. Conversation metrics come from the messages when not supplied."""
    return {
        "format_version": _FORMAT_VERSION,
        "exported_at": timestamp,
        "project_name": project_name,
        "dt_phase": dt_phase,
        "session_id": session_id,
        "messages": [dict(m) for m in messages],
        "mission_memory": {
            "saved_missions": list(saved_missions or []),
        },
        "conversation_metrics": dict(
            conversation_metrics
            if conversation_metrics is not None
            else conversation_metrics_from_messages(messages)
        ),
    }


def _canonical(obj: Any) -> Any:
    """Recursively sort dict keys so the same logical document always
    serialises to identical bytes."""
    if isinstance(obj, dict):
        return {key: _canonical(obj[key]) for key in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(item) for item in obj]
    return obj


def export_session_json(payload: dict) -> str:
    """Canonical, byte-deterministic JSON for a session payload."""
    return json.dumps(
        _canonical(payload), indent=2, ensure_ascii=False, sort_keys=True
    )


def serialize_session_json(
    messages: List[dict],
    *,
    project_name: str = "MyProject",
    dt_phase: str = "Empathize",
    session_id: str = "",
    timestamp: str = "",
    saved_missions: Optional[List[str]] = None,
    conversation_metrics: Optional[dict] = None,
) -> str:
    """One-call convenience: build the payload and return its JSON."""
    payload = build_session_payload(
        messages,
        project_name=project_name,
        dt_phase=dt_phase,
        session_id=session_id,
        timestamp=timestamp,
        saved_missions=saved_missions,
        conversation_metrics=conversation_metrics,
    )
    return export_session_json(payload)


def reload_session_json(text: str) -> dict:
    """Parse exported JSON back into the session payload dict.

    Raises ``json.JSONDecodeError`` on malformed input; the caller decides
    how to surface that. Round-trips byte-identically:
    ``export_session_json(reload_session_json(text)) == text``.
    """
    return json.loads(text)
