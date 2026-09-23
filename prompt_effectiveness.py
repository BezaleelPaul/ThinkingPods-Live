"""
prompt_effectiveness.py — Prompt Effectiveness Audit.

Observation-only developer tool.  Replays the golden conversation fixtures
through the live ``mentor.process_mentor_turn`` pipeline with a fully
mocked (deterministic) LLM, then selectively disables one prompt-guidance
section at a time and measures which sections actually influence the
mentor's replies.

The pipeline itself is NEVER modified.  This module:

  * does not change ``mentor.py`` behaviour,
  * does not change ``PromptBuilder``,
  * does not change extraction, objectives, or lifecycle,
  * performs no LLM calls of its own.

How it works
------------
1. Every fixture is replayed turn-by-turn with a prompt-reflective stub
   LLM.  The stub deterministically encodes which guidance signals were
   present in the prompt into a short reply (which still passes the
   mentor's reply-enforcement rules).
2. The same fixtures are replayed once per prompt section, with that
   section's instruction bullets stripped from the prompt *inside the
   stub* before the reply is produced.
3. For each section the per-turn reply and each encoded signal (question
   family, coaching strategy, insight, conversation move, memory,
   transition, objective, instructions) are compared against the full
   prompt baseline and change rates are aggregated.

Because the stub is the only prompt consumer, the change rates report how
sensitive the mentor's (simulated) replies are to each guidance section.
Pipeline decisions (objective, transition type, question family, …) are
computed before the prompt is built and are therefore provably unchanged
— the audit surfaces that stability explicitly.

Sections that can be disabled
-----------------------------
``base_instructions``
    The per-strategy mentoring instruction bullets under ``Instructions``.
``conversation_memory``
    The ``resume_hint`` bullets from ``conversation_memory``.
``question_family``
    The family-guidance bullets from ``PromptBuilder``.
``coaching_strategy``
    The ``Current coaching strategy: …`` bullet.
``insight_guidance``
    The ``Insight detected: …`` bullet.
``conversation_move``
    The ``Conversation move: …`` bullet.

Report columns
--------------
Prompt Section | Changed Reply (%) | Changed Question (%) |
Changed Transition (%) | Changed Objective (%) |
Average Prompt Tokens | Average Completion Tokens | Estimated Latency

Token accounting uses a word proxy (``len(text.split())``).  Estimated
latency is a linear model ``250 + 18*prompt_tokens + 15*completion_tokens``
milliseconds — an order-of-magnitude estimate for a small local model,
documented for ranking, not for benchmarking.

Usage
-----
Run the full audit and write a report::

    python prompt_effectiveness.py run -o effectiveness.md
    python prompt_effectiveness.py run -o effectiveness.html

Emit a Replay-Lab-compatible capture (full or with one section removed)::

    python prompt_effectiveness.py capture --full -o full.json
    python prompt_effectiveness.py capture --ablate coaching_strategy -o no_coaching.json

Compare a full capture against an ablated capture::

    python prompt_effectiveness.py compare --full full.json \\
        --ablated no_coaching.json -o diff.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("COMPLEXITY_GATE", "true")

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import _build_extraction, _seed_session  # noqa: E402
from replay_lab import _capture_turn, compare_captures  # noqa: E402

# module4 label table -> reverse map to decode the objective signal from the
# "Current Objective" section of a prompt.
from module4.prompt_builder import _OBJECTIVE_LABELS  # noqa: E402

_OBJECTIVE_BY_LABEL: dict[str, str] = {
    label.strip(): obj.value for obj, label in _OBJECTIVE_LABELS.items()
}

# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

_SIGNALS = (
    "objective", "family", "coaching", "insight", "move",
    "transition", "memory", "instructions",
)

# The order in which sections appear in reports (spec order).
_SECTION_ORDER = (
    "base_instructions",
    "conversation_memory",
    "question_family",
    "coaching_strategy",
    "insight_guidance",
    "conversation_move",
)

_SECTION_LABELS = {
    "base_instructions": "Base mentoring instructions",
    "conversation_memory": "Conversation memory",
    "question_family": "Question family guidance",
    "coaching_strategy": "Coaching strategy",
    "insight_guidance": "Insight guidance",
    "conversation_move": "Conversation move",
}

# Bullet-line prefixes used to identify each guidance section inside the
# prompt's Instructions block — plus the Phase 1 dynamic prompt's
# ALLOWED CONVERSATIONAL MOVES block (the move/family guidance now lives
# there as an allowed set instead of single-bullet commands).
_DYNAMIC_MOVE_PREFIXES = (
    "ELICIT_INFORMATION:",
    "EXPAND_IDEA:",
    "CONNECT_INFORMATION:",
    "VALIDATE_DISCOVERY:",
    "CHALLENGE_ASSUMPTION:",
    "SUMMARIZE_PROGRESS:",
    "TRANSITION_TOPIC:",
    "CLOSE_TOPIC:",
)
_GUIDANCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "conversation_memory": (
        "The user has an unfinished topic on",
        "The user previously mentioned",
        "The user already acknowledged the empathy summary",
    ),
    "question_family": (
        "Avoid repeating questions from these already-asked families",
        "Ask a NEW question angle from the family",
        # Phase 1 dynamic carriers (ALLOWED CONVERSATIONAL MOVES block).
        "A fresh question angle such as",
        "Already-asked angles to avoid repeating",
    ),
    "coaching_strategy": (
        "Current coaching strategy:",
    ),
    "insight_guidance": (
        "Insight detected:",
    ),
    "conversation_move": (
        "Conversation move:",
    ) + _DYNAMIC_MOVE_PREFIXES + (
        "Choose the move that fits best",
    ),
}

# All guidance prefixes, for detecting "non-guidance" (base) bullets.
_ALL_GUIDANCE_PREFIXES: tuple[str, ...] = tuple(
    p for prefixes in _GUIDANCE_PREFIXES.values() for p in prefixes
)

_SECTION_META = [
    {
        "id": sid,
        "label": _SECTION_LABELS[sid],
        "prefixes": _GUIDANCE_PREFIXES.get(sid, ()),
        "is_guidance": sid != "base_instructions",
    }
    for sid in _SECTION_ORDER
]


def section_label(section_id: str) -> str:
    return _SECTION_LABELS.get(section_id, section_id)


def _bullet_is_guidance(line: str) -> bool:
    body = line[2:] if line.startswith("- ") else line
    body = body.strip()
    return any(body.startswith(p) for p in _ALL_GUIDANCE_PREFIXES)


def _bullet_is_section(line: str, prefixes: tuple[str, ...]) -> bool:
    body = line[2:] if line.startswith("- ") else line
    body = body.strip()
    return any(body.startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# Prompt surgery — remove one guidance section's bullets (observation-only)
# ---------------------------------------------------------------------------


def _remove_section(prompt: str, section_id: str) -> str:
    """Return a copy of ``prompt`` with the named section's instruction
    bullets removed from the Instructions block.

    Multi-line bullets (e.g. the coaching/insight/move bullets with their
    ``Guidance:`` continuation line) are removed as a unit.  The five
    structural sections (Role / Current Objective / Known Project State /
    Latest Conversation / Instructions) are left intact.
    """
    meta = next((m for m in _SECTION_META if m["id"] == section_id), None)
    if meta is None:
        return prompt

    def _should_remove(line: str) -> bool:
        if meta["is_guidance"]:
            return _bullet_is_section(line, meta["prefixes"])
        # Base instructions = every non-guidance bullet.
        return line.startswith("- ") and not _bullet_is_guidance(line)

    lines = prompt.split("\n")
    out: list[str] = []
    in_instructions = False
    i = 0
    # Phase 1: guidance bullets also live under ALLOWED CONVERSATIONAL MOVES.
    section_headers = ("Instructions", "ALLOWED CONVERSATIONAL MOVES")
    while i < len(lines):
        line = lines[i]
        if line.strip() in section_headers:
            in_instructions = True
            out.append(line)
            i += 1
            continue
        if in_instructions and line.startswith("- ") and _should_remove(line):
            # Skip this bullet plus its continuation lines: indented
            # follow-ups and legacy "Guidance:" lines rendered without the
            # "- " prefix. Blank lines and section headers terminate the
            # bullet (never swallow structural headers).
            i += 1
            while i < len(lines) and (
                (lines[i][:1].isspace() and lines[i].strip())
                or lines[i].startswith("Guidance:")
            ):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Prompt parsing — deterministic signal extraction (what a guidance-obeying
# model would "see" and reflect).
# ---------------------------------------------------------------------------


def _bullet_bodies(prompt: str) -> list[str]:
    """Return the text bodies of every bullet line under Instructions
    (legacy prompt) or ALLOWED CONVERSATIONAL MOVES (Phase 1 prompt)."""
    lines = prompt.split("\n")
    bodies: list[str] = []
    in_instructions = False
    for line in lines:
        stripped = line.strip()
        if stripped in ("Instructions", "ALLOWED CONVERSATIONAL MOVES"):
            in_instructions = True
            continue
        if stripped in ("HARD CONSTRAINTS", "USER MESSAGE"):
            in_instructions = False
            continue
        if in_instructions and line.startswith("- "):
            bodies.append(line[2:].strip())
    return bodies


def _parse_prompt(prompt: str) -> dict[str, Any]:
    """Extract the guidance signals present in a prompt string."""
    sig: dict[str, Any] = {}

    lines = prompt.split("\n")
    # Objective from the "Current Objective" section (skip the rule and
    # blank lines that precede the label).
    for idx, line in enumerate(lines):
        if line.strip() == "Current Objective":
            for nxt in lines[idx + 1:]:
                stripped = nxt.strip()
                if stripped and not stripped.startswith("-"):
                    sig["objective"] = _OBJECTIVE_BY_LABEL.get(
                        stripped, stripped
                    )
                    break
            break
    # Phase 1 dynamic prompt: objective lives under CONVERSATION GOAL
    # (same human-readable labels, so the same reverse map applies).
    if "objective" not in sig:
        for idx, line in enumerate(lines):
            if line.strip() == "CONVERSATION GOAL":
                for nxt in lines[idx + 1:]:
                    stripped = nxt.strip()
                    if stripped and not stripped.startswith("-"):
                        sig["objective"] = _OBJECTIVE_BY_LABEL.get(
                            stripped, stripped
                        )
                        break
                break

    bodies = _bullet_bodies(prompt)
    family_label: Optional[str] = None
    for b in bodies:
        if b.startswith("Ask a NEW question angle from the family: "):
            family_label = b.split(":", 1)[1].strip().rstrip(".")
        elif b.startswith("A fresh question angle such as"):
            # Dynamic carrier: 'A fresh question angle such as 'X' ...'.
            quoted = b.split("'", 2)
            family_label = quoted[1].strip() if len(quoted) > 1 else b
        elif b.startswith("Avoid repeating questions from these already-asked families"):
            family_label = family_label or "avoided"
        elif b.startswith("Already-asked angles to avoid repeating"):
            family_label = family_label or "avoided"
        if b.startswith("Current coaching strategy: "):
            sig["coaching"] = b.split(":", 1)[1].strip().split()[0].upper()
        elif b.startswith("Insight detected: "):
            head = b.split(":", 1)[1].strip()
            sig["insight"] = head.split()[0].upper()
        elif b.startswith("Conversation move: "):
            sig["move"] = b.split(":", 1)[1].strip().split()[0].upper()
        else:
            # Dynamic carrier: "- MOVE_NAME: guidance" lines.
            for prefix in _DYNAMIC_MOVE_PREFIXES:
                if b.startswith(prefix):
                    sig.setdefault(
                        "move", prefix.rstrip(":").upper())
                    break
    if family_label:
        sig["family"] = family_label

    sig["memory"] = sum(
        1
        for b in bodies
        if any(b.startswith(p) for p in _GUIDANCE_PREFIXES["conversation_memory"])
    )
    # Dynamic carrier: "- [TAG] ..." relevant-context lines.
    sig["memory"] += sum(
        1 for line in lines if line.startswith("- [")
    )
    sig["instructions"] = sum(
        1 for b in bodies if not _bullet_is_guidance("- " + b)
    )
    sig["transition"] = "yes" if (
        sig.get("move") == "TRANSITION_TOPIC"
        or sig.get("coaching") == "TRANSITION"
    ) else "no"
    return sig


def _encode_reply(sig: dict[str, Any]) -> str:
    """Deterministic prompt-reflective reply that passes the mentor's
    reply-enforcement rules (one question, no advice markers, <55 words)."""
    parts: list[str] = []
    for key in _SIGNALS:
        val = sig.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            val = "yes" if val else "no"
        parts.append(f"{key}={val}")
    body = "; ".join(parts) if parts else "no guidance signals"
    return f"Signals: {body}. What would you like to explore?"


def _decode_reply(reply: str) -> dict[str, Any]:
    """Inverse of ``_encode_reply`` — parse ``key=value`` signal tokens."""
    import re
    out: dict[str, Any] = {}
    if not reply:
        return out
    for key, val in re.findall(r"(\w+)=([^;\s.]+)", reply):
        out[key] = val
    return out


# ---------------------------------------------------------------------------
# Token / latency estimates (word-proxy tokens; linear latency model)
# ---------------------------------------------------------------------------


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _estimate_latency_ms(prompt_tokens: float, completion_tokens: float) -> float:
    return round(250.0 + prompt_tokens * 18.0 + completion_tokens * 15.0, 1)


# ---------------------------------------------------------------------------
# Prompt-reflective stub LLM
# ---------------------------------------------------------------------------


class PromptReflectiveOllama:
    """Deterministic stand-in for the ``ollama`` module.

    Reads the prompt, (optionally) strips one guidance section, parses the
    remaining signals, and emits a reply that reflects them.  Records the
    prompt it actually processed so the audit can measure token counts.
    """

    def __init__(self, ablate: Optional[str] = None):
        self._ablate = ablate
        self.last_processed_prompt: str = ""

    def chat(self, **kwargs) -> dict:
        messages = kwargs.get("messages", [])
        prompt = messages[0]["content"] if messages else ""
        if self._ablate:
            prompt = _remove_section(prompt, self._ablate)
        self.last_processed_prompt = prompt
        sig = _parse_prompt(prompt)
        return {"message": {"content": _encode_reply(sig)}}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _replay_fixtures(fixtures, ablate: Optional[str]) -> list[dict]:
    """Replay every fixture with a prompt-reflective stub, optionally
    ablating one section.  Returns Replay-Lab-compatible fixture dicts
    (turn records from ``replay_lab._capture_turn``) augmented with the
    internal ``_`` prefixed audit fields."""
    import mentor
    from session_manager import get_session_manager

    out: list[dict] = []
    for fixture in fixtures:
        _seed_session(fixture)
        turns = []
        for index, turn in enumerate(fixture.turns):
            stub = PromptReflectiveOllama(ablate=ablate)
            extraction = _build_extraction(turn)
            mgr = get_session_manager()
            state_before = (
                mgr.get_active_session_data().project_state.to_state_dict()
            )
            with mock.patch.dict("sys.modules", {"ollama": stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    reply, _session, timing, diagnostics = (
                        mentor.process_mentor_turn(
                            turn.user,
                            username=fixture.username,
                            project_name=fixture.project_name,
                        )
                    )
            state_after = (
                get_session_manager()
                .get_active_session_data()
                .project_state.to_state_dict()
            )
            record = _capture_turn(
                index, turn, reply, diagnostics, timing, state_before, state_after
            )
            record["_signals"] = _decode_reply(reply)
            record["_reply"] = reply
            record["_prompt_tokens"] = _token_count(stub.last_processed_prompt)
            record["_completion_tokens"] = _token_count(reply)
            turns.append(record)

        session_summary = (
            get_session_manager()
            .get_active_session_data()
            .mentor_decision_summary
        )
        out.append({
            "fixture_name": fixture.name,
            "description": fixture.description,
            "username": fixture.username,
            "turns": turns,
            "summary": {
                "turn_count": session_summary.turn_count,
                "objective_counts": dict(session_summary.objective_counts),
                "continuity_counts": dict(session_summary.continuity_counts),
                "quality_counts": dict(session_summary.quality_counts),
                "transition_counts": dict(session_summary.transition_counts),
                "continuity_score": session_summary.continuity_score(),
                "quality_score": session_summary.quality_score(),
                "transition_score": session_summary.transition_score(),
                "appropriate_count": session_summary.appropriate_count,
                "acknowledged_count": session_summary.acknowledged_count,
                "restarted_count": session_summary.restarted_count,
            },
        })
    return out


def _strip_internal(record: dict) -> dict:
    return {k: v for k, v in record.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Capture API (Replay-Lab compatible)
# ---------------------------------------------------------------------------


def capture_full(fixtures=None, label: str = "full") -> dict:
    """Replay-Lab-compatible capture of the full prompt run."""
    fixtures = fixtures if fixtures is not None else load_all_fixtures()
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixtures": [
            {
                "fixture_name": f["fixture_name"],
                "description": f["description"],
                "username": f["username"],
                "turns": [_strip_internal(t) for t in f["turns"]],
                "summary": f["summary"],
            }
            for f in _replay_fixtures(fixtures, ablate=None)
        ],
    }


def capture_ablation(
    fixtures=None, section: str = "base_instructions", label: Optional[str] = None
) -> dict:
    """Replay-Lab-compatible capture with one prompt section disabled."""
    fixtures = fixtures if fixtures is not None else load_all_fixtures()
    label = label or f"without_{section}"
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixtures": [
            {
                "fixture_name": f["fixture_name"],
                "description": f["description"],
                "username": f["username"],
                "turns": [_strip_internal(t) for t in f["turns"]],
                "summary": f["summary"],
            }
            for f in _replay_fixtures(fixtures, ablate=section)
        ],
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _signal_change_rate(changes: dict, total: int, signal: str) -> float:
    return round(100.0 * changes.get(signal, 0) / total, 1) if total else 0.0


def audit_prompt_effectiveness(fixtures=None, sections=None) -> dict:
    """Run the full audit.

    Returns a structured dict with a baseline row and one row per section:
    change rates for every signal, average prompt / completion tokens, and
    estimated latency, plus a ranking ordered by reply influence.
    """
    fixtures = fixtures if fixtures is not None else load_all_fixtures()
    baseline = _replay_fixtures(fixtures, ablate=None)
    section_ids = [m["id"] for m in _SECTION_META] if sections is None else sections

    total = sum(len(f["turns"]) for f in baseline)
    base_prompt = sum(
        t["_prompt_tokens"] for f in baseline for t in f["turns"]
    ) / max(total, 1)
    base_completion = sum(
        t["_completion_tokens"] for f in baseline for t in f["turns"]
    ) / max(total, 1)

    rows = []
    for sid in section_ids:
        ablated = _replay_fixtures(fixtures, ablate=sid)
        changes = {s: 0 for s in _SIGNALS}
        changed_reply = 0
        prompt_sum = 0.0
        completion_sum = 0.0
        for fb, fa in zip(baseline, ablated):
            for tb, ta in zip(fb["turns"], fa["turns"]):
                if tb["_reply"] != ta["_reply"]:
                    changed_reply += 1
                sb, sa = tb["_signals"], ta["_signals"]
                # Per-signal comparison only applies to genuine stub replies.
                # A fallback reply (e.g. the family-aware template) carries no
                # encoded signals — skip it so it does not skew per-signal
                # rates; it still counts toward Changed Reply above.
                if sb and sa:
                    for s in _SIGNALS:
                        if sb.get(s) != sa.get(s):
                            changes[s] += 1
                prompt_sum += ta["_prompt_tokens"]
                completion_sum += ta["_completion_tokens"]
        avg_prompt = prompt_sum / max(total, 1)
        avg_completion = completion_sum / max(total, 1)
        rows.append({
            "id": sid,
            "label": section_label(sid),
            "changed_reply_pct": round(100.0 * changed_reply / total, 1) if total else 0.0,
            "changed_question_pct": _signal_change_rate(changes, total, "family"),
            "changed_coaching_pct": _signal_change_rate(changes, total, "coaching"),
            "changed_insight_pct": _signal_change_rate(changes, total, "insight"),
            "changed_move_pct": _signal_change_rate(changes, total, "move"),
            "changed_memory_pct": _signal_change_rate(changes, total, "memory"),
            "changed_transition_pct": _signal_change_rate(changes, total, "transition"),
            "changed_objective_pct": _signal_change_rate(changes, total, "objective"),
            "changed_instructions_pct": _signal_change_rate(changes, total, "instructions"),
            "avg_prompt_tokens": round(avg_prompt, 1),
            "avg_completion_tokens": round(avg_completion, 1),
            "estimated_latency_ms": _estimate_latency_ms(avg_prompt, avg_completion),
            "turns_total": total,
            "turns_changed": changed_reply,
        })

    baseline_row = {
        "id": "full",
        "label": "Full prompt (baseline)",
        "changed_reply_pct": 0.0,
        "changed_question_pct": 0.0,
        "changed_coaching_pct": 0.0,
        "changed_insight_pct": 0.0,
        "changed_move_pct": 0.0,
        "changed_memory_pct": 0.0,
        "changed_transition_pct": 0.0,
        "changed_objective_pct": 0.0,
        "changed_instructions_pct": 0.0,
        "avg_prompt_tokens": round(base_prompt, 1),
        "avg_completion_tokens": round(base_completion, 1),
        "estimated_latency_ms": _estimate_latency_ms(base_prompt, base_completion),
        "turns_total": total,
        "turns_changed": 0,
    }

    ranking = sorted(rows, key=lambda r: r["changed_reply_pct"], reverse=True)
    return {
        "baseline": baseline_row,
        "sections": rows,
        "ranking": [r["label"] for r in ranking],
        "turn_count": total,
        "fixture_count": len(fixtures),
    }


# ---------------------------------------------------------------------------
# Developer Console section builder
# ---------------------------------------------------------------------------


def prompt_effectiveness_section(results: dict) -> dict:
    """Developer-Console-compatible projection of an audit result.

    Produces a section with ``Ranking`` and one sub-block per section so
    the front-end's generic diagnostics renderer can display it as-is.
    """
    section: dict[str, Any] = {"Ranking": list(results.get("ranking", []))}
    for row in [results.get("baseline")] + list(results.get("sections", [])):
        section[row["label"]] = {
            "Changed Reply": f"{row['changed_reply_pct']}%",
            "Changed Question": f"{row['changed_question_pct']}%",
            "Changed Transition": f"{row['changed_transition_pct']}%",
            "Changed Objective": f"{row['changed_objective_pct']}%",
            "Avg Prompt Tokens": row["avg_prompt_tokens"],
            "Avg Completion Tokens": row["avg_completion_tokens"],
            "Estimated Latency (ms)": row["estimated_latency_ms"],
        }
    return section


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


_MAIN_COLUMNS = (
    ("label", "Prompt Section"),
    ("changed_reply_pct", "Changed Reply (%)"),
    ("changed_question_pct", "Changed Question (%)"),
    ("changed_transition_pct", "Changed Transition (%)"),
    ("changed_objective_pct", "Changed Objective (%)"),
    ("avg_prompt_tokens", "Average Prompt Tokens"),
    ("avg_completion_tokens", "Average Completion Tokens"),
    ("estimated_latency_ms", "Estimated Latency (ms)"),
)

_SIGNAL_COLUMNS = (
    ("changed_coaching_pct", "Coaching (%)"),
    ("changed_insight_pct", "Insight (%)"),
    ("changed_move_pct", "Move (%)"),
    ("changed_memory_pct", "Memory (%)"),
    ("changed_instructions_pct", "Instructions (%)"),
)


def _rows_for_report(results: dict) -> list[dict]:
    return [results["baseline"]] + list(results["sections"])


def generate_markdown_report(results: dict) -> str:
    lines = [
        "# Prompt Effectiveness Audit",
        "",
        f"- **Fixtures:** {results['fixture_count']}",
        f"- **Turns:** {results['turn_count']}",
        "",
        "## Sections ranked by influence",
        "",
    ]
    for rank, label in enumerate(results["ranking"], start=1):
        lines.append(f"{rank}. {label}")
    lines += ["", "## Influence table", "", "| " + " | ".join(c[1] for c in _MAIN_COLUMNS) + " |"]
    lines.append("|" + "---|" * len(_MAIN_COLUMNS))
    for row in _rows_for_report(results):
        lines.append(
            "| " + " | ".join(str(row[k]).replace("|", "\\|") for k, _ in _MAIN_COLUMNS) + " |"
        )
    lines += ["", "## Per-signal change rates", "", "| " + " | ".join(["Prompt Section"] + [c[1] for c in _SIGNAL_COLUMNS]) + " |"]
    lines.append("|" + "---|" * (len(_SIGNAL_COLUMNS) + 1))
    for row in _rows_for_report(results):
        vals = [str(row[k]) for _, k in _SIGNAL_COLUMNS] if False else [
            str(row[c[0]]).replace("|", "\\|") for c in _SIGNAL_COLUMNS
        ]
        lines.append("| " + row["label"] + " | " + " | ".join(vals) + " |")
    lines += ["", "> Note: `Changed Objective (%)` is structurally 0% — the "
                  "objective lives in a non-ablated section, proving prompt "
                  "guidance never alters objective decisions."]
    return "\n".join(lines) + "\n"


def generate_html_report(results: dict) -> str:
    def _table(columns):
        head = "".join(f"<th>{c[1]}</th>" for c in columns)
        body_rows = []
        for row in _rows_for_report(results):
            tds = "".join(f"<td>{row[c[0]]}</td>" for c in columns)
            body_rows.append(f"<tr>{tds}</tr>")
        return (
            "<table><thead><tr>" + head + "</tr></thead><tbody>"
            + "".join(body_rows) + "</tbody></table>"
        )

    ranking = "<ol>" + "".join(f"<li>{l}</li>" for l in results["ranking"]) + "</ol>"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prompt Effectiveness Audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; }}
th, td {{ border: 1px solid #ccc; padding: .4rem .7rem; text-align: right; }}
th {{ background: #f4f4f4; }}
td:first-child {{ text-align: left; }}
.note {{ color: #555; font-size: .9rem; }}
</style></head><body>
<h1>Prompt Effectiveness Audit</h1>
<p>Fixtures: {results['fixture_count']} &middot; Turns: {results['turn_count']}</p>
<h2>Sections ranked by influence</h2>
{ranking}
<h2>Influence table</h2>
{_table(_MAIN_COLUMNS)}
<h2>Per-signal change rates</h2>
{_table(_SIGNAL_COLUMNS)}
<p class="note">Note: <code>Changed Objective (%)</code> is structurally 0&percnt; —
the objective lives in a non-ablated section, proving prompt guidance never
alters objective decisions.</p>
</body></html>
"""


def compare_full_vs_ablation(full: dict, ablated: dict) -> dict:
    """Replay-Lab comparison between a full-prompt capture and a capture
    with one section removed, plus the effectiveness row for that section."""
    comparison = compare_captures(full, ablated)
    return {
        "comparison": comparison,
        "ablated_section": ablated.get("label", "unknown"),
        "effectiveness_row": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt_effectiveness.py",
        description="Prompt Effectiveness Audit (observation-only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full audit and write a report")
    p_run.add_argument("-o", "--output", default="effectiveness.md")
    p_run.add_argument("--sections", default=None,
                       help="comma-separated section ids to audit")

    p_cap = sub.add_parser("capture", help="emit a Replay-Lab-compatible capture")
    p_cap.add_argument("--full", action="store_true", help="capture the full prompt run")
    p_cap.add_argument("--ablate", default=None, help="section id to disable")
    p_cap.add_argument("-o", "--output", required=True)

    p_cmp = sub.add_parser("compare", help="compare full vs ablated captures")
    p_cmp.add_argument("--full", required=True)
    p_cmp.add_argument("--ablated", required=True)
    p_cmp.add_argument("-o", "--output", default=None)

    args = parser.parse_args(argv)

    if args.command == "run":
        sections = args.sections.split(",") if args.sections else None
        results = audit_prompt_effectiveness(sections=sections)
        text = (
            generate_html_report(results)
            if args.output.lower().endswith(".html")
            else generate_markdown_report(results)
        )
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.output}")
        for label in results["ranking"]:
            print(f"  {label}")
        return 0

    if args.command == "capture":
        fixtures = load_all_fixtures()
        if args.full:
            data = capture_full(fixtures, label="full")
        else:
            data = capture_ablation(fixtures, section=args.ablate or "base_instructions")
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"wrote {args.output} ({data['label']})")
        return 0

    if args.command == "compare":
        with open(args.full, encoding="utf-8") as fh:
            full = json.load(fh)
        with open(args.ablated, encoding="utf-8") as fh:
            ablated = json.load(fh)
        comparison = compare_full_vs_ablation(full, ablated)["comparison"]
        text = [
            f"# Replay Lab: {full.get('label')} vs {ablated.get('label')}",
            "",
            f"- Turns (baseline): {comparison['total_turns_baseline']}",
            f"- Turns (candidate): {comparison['total_turns_candidate']}",
            "",
            "## Fixture verdicts",
            "",
            "| Fixture | Verdict | Changed | Improved | Regressed |",
            "|---------|---------|---------|----------|-----------|",
        ]
        for fd in comparison["fixture_diffs"]:
            text.append(
                f"| {fd['fixture_name']} | {fd.get('verdict','—')} | "
                f"{fd.get('turns_changed',0)} | {fd.get('turns_improved',0)} | "
                f"{fd.get('turns_regressed',0)} |"
            )
        text.append("")
        text.append("> Compare: disabling this prompt section changes the captured "
                    "signals above. See `prompt_effectiveness.py run` for the ranked "
                    "influence table.")
        report = "\n".join(text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(report)
            print(f"wrote {args.output}")
        else:
            print(report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
