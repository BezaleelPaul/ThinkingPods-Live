"""
relevant_context.py — deterministic Relevant Context Builder (Phase 1 dynamic coach).

Architecture role
-----------------
The response model previously received the *entire* ProjectState plus a
one-turn conversation window plus a stack of imperative bullets. That is
both too much (full state dump, much of it irrelevant this turn) and too
little (no salience: new vs old, corrected vs confirmed, volunteered vs
elicited are undistinguished; user wording beyond one turn is absent).

This module selects **3–8 highly relevant snippets** for THIS turn from
structures the pipeline already owns:

* validated state diff (new facts, with values),
* correction pairs (old → new),
* open threads / deferred topics (memory keys + reasons),
* the current objective's remaining need,
* the immediately preceding exchange (verbatim, raw user wording kept).

Priority order (salience, not recency alone):

1. new information, 2. corrections, 3. unresolved/open threads,
4. current objective need, 5. immediately preceding conversation,
6. older volunteered context.

Deliberately NOT included: full history dumps, resolved-thread replays,
user dossiers, vector retrieval. Strict budget: at most 8 snippets and
2400 characters of rendered text (~600 tokens).

Pure, deterministic, no LLM, no I/O. The selector never invents content:
every snippet is quoted or directly derived from pipeline state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from conversation_brief import field_label

__all__ = [
    "MAX_SNIPPETS",
    "MAX_CHARS",
    "build_relevant_context",
    "render_relevant_context",
]

MAX_SNIPPETS = 8
MAX_CHARS = 2400


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _history_turns(conversation_history: Optional[list], limit: int = 4) -> list:
    """Most recent ``{role, content}`` turns, oldest-first, content-clipped."""
    history = [m for m in (conversation_history or []) if isinstance(m, dict)]
    return history[-limit:] if limit else []


def build_relevant_context(
    *,
    brief=None,
    state_after: Optional[dict] = None,
    memory=None,
    conversation_history: Optional[list] = None,
    last_assistant_message: Optional[str] = None,
    current_target: Optional[str] = None,
) -> List[dict]:
    """Select up to 8 salient snippets for this turn.

    Each snippet is ``{"tag": ..., "text": ...}`` where tag is one of
    CORRECTED / NEW / NEED / OPEN / VOLUNTEERED / YOU SAID / I ASKED.
    ``brief`` (a ``ConversationBrief``) supplies new facts, correction,
    uncertainty, and the current objective; ``memory`` is the session's
    ``ConversationMemory``; ``state_after`` is the post-extraction snapshot
    used only to attach current values to memory-keyed threads.
    """
    snippets: List[dict] = []
    seen_fields: set = set()

    new_facts = list(getattr(brief, "new_facts", []) or [])
    correction = getattr(brief, "correction", None) or None
    uncertainty = bool(getattr(brief, "uncertainty", False))
    current_objective = getattr(brief, "current_objective", "") or ""
    why_now = getattr(brief, "why_now", "") or ""

    # 1. Correction first — the model must anchor on the new value.
    if correction:
        fields = list(correction.get("fields") or [])
        note = (correction.get("note") or "").strip()
        if fields:
            seen_fields.update(fields)
            label = ", ".join(field_label(f) for f in fields)
            text = f"User corrected {label}."
            if note:
                text += f" Note: {_clip(note, 140)}"
            # Attach the now-current value(s) so the model anchors correctly.
            current_vals = []
            for f in fields:
                v = (state_after or {}).get(f)
                if isinstance(v, list):
                    current_vals.extend([i for i in v if i][:2])
                elif v:
                    current_vals.append(str(v))
            if current_vals:
                text += " Current: " + "; ".join(
                    _clip(str(v), 80) for v in current_vals[:2]
                )
            snippets.append({"tag": "CORRECTED", "text": text})
        elif note:
            snippets.append({"tag": "CORRECTED", "text": f"User made a correction. Note: {_clip(note, 160)}"})

    # 2. New facts with values — what the model should acknowledge precisely.
    for fact in new_facts:
        f = fact.get("field", "")
        v = fact.get("value", "")
        if not v or f in seen_fields:
            continue
        seen_fields.add(f)
        snippets.append(
            {"tag": "NEW", "text": f"{field_label(f)}: {_clip(str(v), 140)}"}
        )
        if len(snippets) >= MAX_SNIPPETS:
            return snippets

    # 3. Uncertainty on the just-shared answer.
    if uncertainty:
        snippets.append(
            {
                "tag": "OPEN",
                "text": "The user's latest answer was partial or uncertain — "
                "treat it gently, do not assume details they did not give.",
            }
        )
        if len(snippets) >= MAX_SNIPPETS:
            return snippets

    # 4/5. Open threads + deferred topics (excluding the current target and
    # fields already covered above). Memory holds keys+reasons only; attach
    # the live value from state so the model sees user wording, not labels.
    if memory is not None:
        for rec in list(getattr(memory, "open_threads", []) or []):
            f = rec.get("field", "")
            if not f or f == current_target or f in seen_fields:
                continue
            seen_fields.add(f)
            snippets.append(
                {
                    "tag": "OPEN",
                    "text": f"Unfinished topic — {field_label(f)}: "
                    f"{_clip(rec.get('reason', ''), 100)}",
                }
            )
            if len(snippets) >= MAX_SNIPPETS:
                return snippets
        for rec in list(getattr(memory, "deferred_topics", []) or []):
            f = rec.get("field", "")
            if not f or f == current_target or f in seen_fields:
                continue
            seen_fields.add(f)
            val = (state_after or {}).get(f)
            if isinstance(val, list):
                val = "; ".join(_clip(str(i), 60) for i in val[:2])
            extra = f" (they mentioned: {_clip(str(val), 120)})" if val else ""
            snippets.append(
                {
                    "tag": "VOLUNTEERED",
                    "text": f"User earlier volunteered {field_label(f)}{extra} — "
                    "return to it naturally once the current point is addressed.",
                }
            )
            if len(snippets) >= MAX_SNIPPETS:
                return snippets

    # 6. Current objective need — what we are trying to learn and why.
    if current_objective:
        need = f"Still working to understand: {current_objective}."
        if why_now:
            need += f" {_clip(why_now, 140)}"
        snippets.append({"tag": "NEED", "text": need})
        if len(snippets) >= MAX_SNIPPETS:
            return snippets

    # 7. Immediately preceding exchange, verbatim (raw wording preserved).
    if last_assistant_message and last_assistant_message.strip():
        snippets.append(
            {"tag": "I ASKED", "text": _clip(last_assistant_message, 200)}
        )
        if len(snippets) >= MAX_SNIPPETS:
            return snippets
    recent_user = None
    for m in reversed(_history_turns(conversation_history, limit=4)):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            recent_user = m["content"]
            break
    if recent_user:
        snippets.append({"tag": "YOU SAID", "text": _clip(recent_user, 280)})
        if len(snippets) >= MAX_SNIPPETS:
            return snippets

    return snippets[:MAX_SNIPPETS]


def render_relevant_context(snippets: List[dict]) -> str:
    """Render snippets as short ``- [TAG] text`` lines under a char budget."""
    lines: List[str] = []
    used = 0
    for s in snippets or []:
        line = f"- [{s.get('tag', 'CTX')}] {(s.get('text') or '').strip()}"
        if used + len(line) > MAX_CHARS:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)
