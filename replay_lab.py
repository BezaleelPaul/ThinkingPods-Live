"""
replay_lab.py — Conversation Replay Lab.

A developer-productivity tool for comparing two runs of the same golden
conversation fixtures after code changes.  It replays every fixture
through the live ``mentor.process_mentor_turn`` pipeline (fully mocked,
deterministic — no real LLM) and captures per-turn:

  * ProjectState evolution (before / after)
  * Objective
  * Question family
  * Transition type
  * Insight detection (type / confidence / signals)
  * Conversation move (move / reason / signals)
  * Conversation memory (open / deferred / resolved threads)
  * Full LLM prompt
  * Mentor reply
  * Diagnostics (continuity, quality, hybrid usage, extraction latency)

A capture is a plain JSON-serialisable dict saved to disk.  Two captures
can be compared to produce an HTML or Markdown report that highlights,
per fixture and per turn:

  * Objective changes
  * Different questions
  * Different transitions
  * Different extractions (state mutations)
  * Different prompts
  * Conversation metrics: continuity, question quality, transition
    score, hybrid usage, extraction latency

Each metric is classified as **Improved**, **Unchanged**, or
**Regressed** relative to the baseline run.

No runtime behaviour changes.  No architecture changes.  No extraction
changes.  This module is purely observational.

Usage
-----

Replay and save a capture::

    python replay_lab.py capture --label before -o captures/before.json

(…make code changes…)

    python replay_lab.py capture --label after -o captures/after.json

Compare and emit a report::

    python replay_lab.py compare --baseline captures/before.json \\
        --candidate captures/after.json --format html -o report.html

Or generate a single capture with embedded report::

    python replay_lab.py capture --label demo -o demo.json --report demo_report.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("COMPLEXITY_GATE", "true")

from tests.goldens.fixtures import load_all_fixtures, load_fixture  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)

_STATE_KEYS = (
    "personas", "problems", "current_solutions", "pain_points",
    "evidence", "impacts", "frequency",
)

_METRIC_HIGHER_IS_BETTER = (
    "continuity_score", "quality_score", "transition_score",
)
_METRIC_LOWER_IS_BETTER = (
    "extraction_ms",
)


# ---------------------------------------------------------------------------
# 1. Capture
# ---------------------------------------------------------------------------


def _safe_get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def _capture_turn(
    index: int,
    turn,
    reply: str,
    diagnostics: dict,
    timing,
    state_before: dict,
    state_after: dict,
) -> dict:
    """Extract every signal-of-interest from one turn's diagnostics."""
    pipeline = diagnostics.get("Pipeline", {})
    mentor_dec = diagnostics.get("MentorDecision", {})
    qf = diagnostics.get("QuestionFamilies", {})
    memory = diagnostics.get("Memory", {})
    extraction = diagnostics.get("Extraction", {})
    hybrid = diagnostics.get("HybridExtraction", {})
    md_summary = diagnostics.get("MentorDecisionSummary", {})
    insight = diagnostics.get("InsightDetection", {})
    move = diagnostics.get("ConversationMove", {})
    prompt = diagnostics.get("Prompt", "")
    conversation_failure = diagnostics.get("ConversationFailure", {})
    cf_summary = diagnostics.get("ConversationFailureSummary", {})

    return {
        "turn_index": index,
        "user_message": turn.user,
        "reply": reply,
        "objective": pipeline.get("Objective"),
        "response_strategy": pipeline.get("Response Strategy"),
        "lifecycle_decision": pipeline.get("Lifecycle Decision"),
        "stage": pipeline.get("Stage"),
        "question_family": qf.get("Question Family"),
        "planned_family": qf.get("Planned Family"),
        "reask": qf.get("Re-ask Sanctioned", False),
        "asked_families": qf.get("Previously Asked Families", []),
        "transition_type": mentor_dec.get("Transition Type"),
        "transition_reason": mentor_dec.get("Transition Reason"),
        "continuity": mentor_dec.get("Conversation Continuity"),
        "continuity_reason": mentor_dec.get("Continuity Reason"),
        "question_quality": mentor_dec.get("Question Quality"),
        "quality_reason": mentor_dec.get("Quality Reason"),
        "objective_appropriate": mentor_dec.get("Objective Appropriate"),
        "acknowledged": mentor_dec.get("Acknowledged Information"),
        "restarted": mentor_dec.get("Restarted Topic"),
        "better_alternative": mentor_dec.get("Better Alternative"),
        "state_before": state_before,
        "state_after": state_after,
        "state_changes": diagnostics.get("StateChanges", {}),
        "memory_open_threads": memory.get("Open Threads", []),
        "memory_deferred_topics": memory.get("Deferred Topics", []),
        "memory_resolved_threads": memory.get("Resolved Threads", []),
        "extraction_message_type": extraction.get("message_type"),
        "extraction_updates": _safe_get(extraction, "rule_based_extraction", default=extraction.get("updates", [])),
        "hybrid_llm_invoked": _safe_get(hybrid, "LLM Invoked"),
        "hybrid_objective_satisfied_by_rules": _safe_get(hybrid, "Objective Satisfied By Rules"),
        "insight_type": insight.get("Insight Type"),
        "insight_confidence": insight.get("Confidence"),
        "insight_signals": insight.get("Signals", []),
        "insight_explanation": insight.get("Explanation", ""),
        "conversation_move": move.get("Move"),
        "conversation_move_reason": move.get("Reason", ""),
        "conversation_move_signals": move.get("Signals", []),
        "prompt": prompt,
        "failure_primary": conversation_failure.get("Primary Failure"),
        "failure_secondary": conversation_failure.get("Secondary Failures"),
        "failure_confidence": conversation_failure.get("Confidence"),
        "failure_reason": conversation_failure.get("Reason", ""),
        "failure_summary_turns": cf_summary.get("Turns Analyzed"),
        "failure_summary_rate": cf_summary.get("Turn Failure Rate"),
        "continuity_score": _safe_get(md_summary, "Continuity Score"),
        "quality_score": _safe_get(md_summary, "Question Quality Score"),
        "transition_score": _safe_get(md_summary, "Transition Score"),
        "extraction_ms": timing.extraction_ms if timing else None,
        "llm_ms": timing.llm_ms if timing else None,
        "total_ms": timing.total_ms if timing else None,
    }


def capture_conversations(label: str, fixtures=None) -> dict:
    """Replay every fixture and return a JSON-serialisable capture dict."""
    fixtures = fixtures if fixtures is not None else load_all_fixtures()
    import mentor
    from session_manager import get_session_manager

    results = []
    for fixture in fixtures:
        _seed_session(fixture)
        turns = []
        for index, turn in enumerate(fixture.turns):
            ollama_stub = (
                _RaisingOllama()
                if turn.ollama_reply is None
                else _StubOllama([turn.ollama_reply])
            )
            extraction = _build_extraction(turn)

            mgr = get_session_manager()
            state_before = mgr.get_active_session_data().project_state.to_state_dict()

            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
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

            turns.append(
                _capture_turn(index, turn, reply, diagnostics, timing, state_before, state_after)
            )

        session_summary = (
            get_session_manager()
            .get_active_session_data()
            .mentor_decision_summary
        )
        results.append({
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

    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixture_count": len(results),
        "fixtures": results,
    }


def save_capture(capture: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(capture, f, indent=2, ensure_ascii=False, default=str)


def load_capture(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Compare
# ---------------------------------------------------------------------------


@dataclass
class TurnDiff:
    turn_index: int
    user_message: str
    changes: list[dict] = field(default_factory=list)
    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)


@dataclass
class FixtureDiff:
    fixture_name: str
    turns: list[TurnDiff] = field(default_factory=list)
    summary_diff: dict = field(default_factory=dict)
    verdict: str = "unchanged"


def _diff_value(name: str, a, b) -> dict | None:
    if a == b:
        return None
    return {"field": name, "baseline": a, "candidate": b}


def _diff_state(a: dict, b: dict) -> list[dict]:
    out = []
    for key in _STATE_KEYS:
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            out.append({"field": key, "baseline": va, "candidate": vb})
    return out


def _classify_metric(name: str, a, b) -> str | None:
    """Return 'improved', 'regressed', 'unchanged', or None (incomparable)."""
    if a is None or b is None:
        return None
    try:
        fa = float(str(a).rstrip("%"))
        fb = float(str(b).rstrip("%"))
    except (ValueError, TypeError):
        return None
    if fa == fb:
        return "unchanged"
    if name in _METRIC_HIGHER_IS_BETTER:
        return "improved" if fb > fa else "regressed"
    if name in _METRIC_LOWER_IS_BETTER:
        return "improved" if fb < fa else "regressed"
    return None


def compare_turns(a: dict, b: dict) -> TurnDiff:
    """Compare two captured turns and classify changes."""
    diff = TurnDiff(
        turn_index=a.get("turn_index", 0),
        user_message=a.get("user_message", ""),
    )

    simple_fields = [
        "objective", "response_strategy", "lifecycle_decision",
        "question_family", "planned_family", "reask",
        "transition_type", "continuity", "question_quality",
        "objective_appropriate", "acknowledged", "restarted",
        "insight_type", "insight_confidence",
        "conversation_move",
        "reply",
    ]
    for f in simple_fields:
        change = _diff_value(f, a.get(f), b.get(f))
        if change:
            diff.changes.append(change)

    prompt_change = _diff_value("prompt", a.get("prompt"), b.get("prompt"))
    if prompt_change:
        diff.changes.append(prompt_change)

    state_diffs = _diff_state(a.get("state_after", {}), b.get("state_after", {}))
    for sd in state_diffs:
        diff.changes.append({"field": f"state.{sd['field']}", "baseline": sd["baseline"], "candidate": sd["candidate"]})

    mem_a = {
        "open": a.get("memory_open_threads", []),
        "deferred": a.get("memory_deferred_topics", []),
        "resolved": a.get("memory_resolved_threads", []),
    }
    mem_b = {
        "open": b.get("memory_open_threads", []),
        "deferred": b.get("memory_deferred_topics", []),
        "resolved": b.get("memory_resolved_threads", []),
    }
    if mem_a != mem_b:
        diff.changes.append({"field": "conversation_memory", "baseline": mem_a, "candidate": mem_b})

    hybrid_change = _diff_value("hybrid_llm_invoked", a.get("hybrid_llm_invoked"), b.get("hybrid_llm_invoked"))
    if hybrid_change:
        diff.changes.append(hybrid_change)

    metrics = [
        ("continuity_score", a.get("continuity_score"), b.get("continuity_score")),
        ("quality_score", a.get("quality_score"), b.get("quality_score")),
        ("transition_score", a.get("transition_score"), b.get("transition_score")),
        ("extraction_ms", a.get("extraction_ms"), b.get("extraction_ms")),
    ]
    for name, va, vb in metrics:
        verdict = _classify_metric(name, va, vb)
        if verdict == "improved":
            diff.improvements.append({"field": name, "baseline": va, "candidate": vb})
        elif verdict == "regressed":
            diff.regressions.append({"field": name, "baseline": va, "candidate": vb})

    return diff


def compare_fixtures(fix_a: dict, fix_b: dict) -> FixtureDiff:
    fd = FixtureDiff(fixture_name=fix_a["fixture_name"])
    turns_a = {t["turn_index"]: t for t in fix_a.get("turns", [])}
    turns_b = {t["turn_index"]: t for t in fix_b.get("turns", [])}

    for idx in sorted(set(turns_a) | set(turns_b)):
        ta = turns_a.get(idx)
        tb = turns_b.get(idx)
        if ta is None:
            fd.turns.append(TurnDiff(turn_index=idx, user_message="<added>",
                                     changes=[{"field": "_turn_added", "candidate": tb}]))
            continue
        if tb is None:
            fd.turns.append(TurnDiff(turn_index=idx, user_message="<removed>",
                                     changes=[{"field": "_turn_removed", "baseline": ta}]))
            continue
        fd.turns.append(compare_turns(ta, tb))

    sa = fix_a.get("summary", {})
    sb = fix_b.get("summary", {})
    for key in ("continuity_score", "quality_score", "transition_score"):
        verdict = _classify_metric(key, sa.get(key), sb.get(key))
        if verdict and verdict != "unchanged":
            fd.summary_diff[key] = {
                "baseline": sa.get(key), "candidate": sb.get(key), "verdict": verdict,
            }

    has_regressions = any(t.regressions for t in fd.turns) or any(
        v["verdict"] == "regressed" for v in fd.summary_diff.values()
    )
    has_improvements = any(t.improvements for t in fd.turns) or any(
        v["verdict"] == "improved" for v in fd.summary_diff.values()
    )
    has_changes = any(t.changes for t in fd.turns)
    if has_regressions:
        fd.verdict = "regressed"
    elif has_improvements and not has_regressions:
        fd.verdict = "improved"
    elif has_changes:
        fd.verdict = "changed"
    else:
        fd.verdict = "unchanged"

    return fd


def compare_captures(baseline: dict, candidate: dict) -> dict:
    fix_a = {f["fixture_name"]: f for f in baseline.get("fixtures", [])}
    fix_b = {f["fixture_name"]: f for f in candidate.get("fixtures", [])}

    fixture_diffs = []
    for name in sorted(set(fix_a) | set(fix_b)):
        fa = fix_a.get(name)
        fb = fix_b.get(name)
        if fa is None:
            fixture_diffs.append({"fixture_name": name, "verdict": "added"})
            continue
        if fb is None:
            fixture_diffs.append({"fixture_name": name, "verdict": "removed"})
            continue
        fd = compare_fixtures(fa, fb)
        fixture_diffs.append({
            "fixture_name": fd.fixture_name,
            "verdict": fd.verdict,
            "turn_count": len(fd.turns),
            "turns_changed": sum(1 for t in fd.turns if t.changes),
            "turns_improved": sum(1 for t in fd.turns if t.improvements),
            "turns_regressed": sum(1 for t in fd.turns if t.regressions),
            "summary_diff": fd.summary_diff,
            "turn_diffs": [
                {
                    "turn_index": t.turn_index,
                    "user_message": t.user_message,
                    "changes": t.changes,
                    "improvements": t.improvements,
                    "regressions": t.regressions,
                }
                for t in fd.turns
                if t.changes or t.improvements or t.regressions
            ],
        })

    sa = baseline.get("fixtures", [{}])
    sb = candidate.get("fixtures", [{}])

    total_turns_a = sum(f.get("summary", {}).get("turn_count", 0) for f in baseline.get("fixtures", []))
    total_turns_b = sum(f.get("summary", {}).get("turn_count", 0) for f in candidate.get("fixtures", []))

    return {
        "baseline_label": baseline.get("label", "baseline"),
        "candidate_label": candidate.get("label", "candidate"),
        "baseline_timestamp": baseline.get("timestamp"),
        "candidate_timestamp": candidate.get("timestamp"),
        "total_turns_baseline": total_turns_a,
        "total_turns_candidate": total_turns_b,
        "fixture_diffs": fixture_diffs,
    }


# ---------------------------------------------------------------------------
# 3. Report — Markdown
# ---------------------------------------------------------------------------


def _fmt_val(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)


def generate_markdown_report(comparison: dict) -> str:
    lines = []
    bl = comparison["baseline_label"]
    cl = comparison["candidate_label"]
    lines.append(f"# Conversation Replay Lab Report")
    lines.append("")
    lines.append(f"- **Baseline:** `{bl}` ({comparison.get('baseline_timestamp', '—')})")
    lines.append(f"- **Candidate:** `{cl}` ({comparison.get('candidate_timestamp', '—')})")
    lines.append(f"- **Turns replayed (baseline):** {comparison['total_turns_baseline']}")
    lines.append(f"- **Turns replayed (candidate):** {comparison['total_turns_candidate']}")
    lines.append("")

    verdicts = {"improved": 0, "regressed": 0, "changed": 0, "unchanged": 0, "added": 0, "removed": 0}
    for fd in comparison["fixture_diffs"]:
        v = fd["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    lines.append("## Summary")
    lines.append("")
    lines.append("| Verdict | Fixtures |")
    lines.append("|---------|----------|")
    for v in ("improved", "unchanged", "changed", "regressed", "added", "removed"):
        if verdicts.get(v):
            lines.append(f"| {v.title()} | {verdicts[v]} |")
    lines.append("")

    for fd in comparison["fixture_diffs"]:
        icon = {"improved": "✅", "regressed": "❌", "changed": "⚠️",
                "unchanged": "➖", "added": "➕", "removed": "➖"}.get(fd["verdict"], "❓")
        lines.append(f"## {icon} {fd['fixture_name']} — *{fd['verdict']}*")
        lines.append("")
        lines.append(f"- Turns: {fd['turn_count']}, Changed: {fd['turns_changed']}, "
                     f"Improved: {fd['turns_improved']}, Regressed: {fd['turns_regressed']}")

        if fd.get("summary_diff"):
            lines.append("")
            lines.append("### Session Metrics")
            lines.append("| Metric | Baseline | Candidate | Verdict |")
            lines.append("|--------|----------|-----------|---------|")
            for key, val in fd["summary_diff"].items():
                lines.append(f"| {key} | {val['baseline']} | {val['candidate']} | {val['verdict']} |")
            lines.append("")

        changed_turns = fd.get("turn_diffs", [])
        if changed_turns:
            lines.append("### Turn-level Diff")
            lines.append("")
            for td in changed_turns:
                lines.append(f"#### Turn {td['turn_index']}: `{td['user_message'][:80]}`")
                lines.append("")
                if td.get("improvements"):
                    lines.append("**Improvements:**")
                    for imp in td["improvements"]:
                        lines.append(f"- ✅ `{imp['field']}`: {imp['baseline']} → {imp['candidate']}")
                    lines.append("")
                if td.get("regressions"):
                    lines.append("**Regressions:**")
                    for reg in td["regressions"]:
                        lines.append(f"- ❌ `{reg['field']}`: {reg['baseline']} → {reg['candidate']}")
                    lines.append("")
                if td.get("changes"):
                    lines.append("**Changes:**")
                    lines.append("| Field | Baseline | Candidate |")
                    lines.append("|-------|----------|-----------|")
                    for c in td["changes"]:
                        field_name = c["field"]
                        if field_name == "prompt":
                            lines.append(f"| prompt | (see diff) | (see diff) |")
                        else:
                            lines.append(f"| {field_name} | {_fmt_val(c.get('baseline'))} | {_fmt_val(c.get('candidate'))} |")
                    lines.append("")
        else:
            lines.append("*No per-turn changes.*")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3b. Report — HTML
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conversation Replay Lab Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; max-width: 1200px; }}
  h1, h2, h3, h4 {{ color: #1a1a2e; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f4f4f4; font-weight: 600; }}
  .verdict-improved {{ color: #16a34a; font-weight: 600; }}
  .verdict-regressed {{ color: #dc2626; font-weight: 600; }}
  .verdict-changed {{ color: #d97706; font-weight: 600; }}
  .verdict-unchanged {{ color: #6b7280; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
  .badge-improved {{ background: #dcfce7; color: #166534; }}
  .badge-regressed {{ background: #fee2e2; color: #991b1b; }}
  .badge-changed {{ background: #fef3c7; color: #92400e; }}
  .badge-unchanged {{ background: #f3f4f6; color: #4b5563; }}
  pre {{ background: #f8f8f8; padding: 8px; overflow-x: auto; border-radius: 4px; font-size: 12px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .summary-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; }}
  .summary-card h3 {{ margin-top: 0; }}
  .diff-val {{ max-width: 400px; word-break: break-word; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _html_escape(s) -> str:
    if s is None:
        return "—"
    import html as _html
    return _html.escape(_fmt_val(s))


def _verdict_badge(verdict: str) -> str:
    return f'<span class="badge badge-{verdict}">{verdict}</span>'


def generate_html_report(comparison: dict) -> str:
    bl = comparison["baseline_label"]
    cl = comparison["candidate_label"]
    parts = []

    parts.append("<h1>Conversation Replay Lab Report</h1>")
    parts.append(f"<p><strong>Baseline:</strong> <code>{_html_escape(bl)}</code> "
                 f"({comparison.get('baseline_timestamp', '—')})<br>"
                 f"<strong>Candidate:</strong> <code>{_html_escape(cl)}</code> "
                 f"({comparison.get('candidate_timestamp', '—')})</p>")

    verdicts = {"improved": 0, "regressed": 0, "changed": 0, "unchanged": 0, "added": 0, "removed": 0}
    for fd in comparison["fixture_diffs"]:
        v = fd["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    parts.append('<div class="summary-grid">')
    for v in ("improved", "unchanged", "changed", "regressed", "added", "removed"):
        if verdicts.get(v):
            parts.append(
                f'<div class="summary-card"><h3 class="verdict-{v}">{v.title()}</h3>'
                f'<p style="font-size:2rem;margin:0;">{verdicts[v]}</p></div>'
            )
    parts.append('</div>')

    for fd in comparison["fixture_diffs"]:
        parts.append(f'<h2>{_verdict_badge(fd["verdict"])} {fd["fixture_name"]}</h2>')
        parts.append(
            f'<p>Turns: {fd["turn_count"]} | Changed: {fd["turns_changed"]} | '
            f'Improved: {fd["turns_improved"]} | Regressed: {fd["turns_regressed"]}</p>'
        )

        if fd.get("summary_diff"):
            parts.append("<h3>Session Metrics</h3>")
            parts.append('<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Verdict</th></tr>')
            for key, val in fd["summary_diff"].items():
                parts.append(
                    f'<tr><td>{_html_escape(key)}</td><td>{_html_escape(val["baseline"])}</td>'
                    f'<td>{_html_escape(val["candidate"])}</td>'
                    f'<td class="verdict-{val["verdict"]}">{_verdict_badge(val["verdict"])}</td></tr>'
                )
            parts.append('</table>')

        changed_turns = fd.get("turn_diffs", [])
        if changed_turns:
            parts.append("<h3>Turn-level Diff</h3>")
            for td in changed_turns:
                parts.append(f'<h4>Turn {td["turn_index"]}: <code>{_html_escape(td["user_message"][:100])}</code></h4>')

                if td.get("improvements"):
                    parts.append('<table><tr><th>Improvements</th><th>Baseline</th><th>Candidate</th></tr>')
                    for imp in td["improvements"]:
                        parts.append(
                            f'<tr><td class="verdict-improved">✅ {_html_escape(imp["field"])}</td>'
                            f'<td>{_html_escape(imp["baseline"])}</td>'
                            f'<td>{_html_escape(imp["candidate"])}</td></tr>'
                        )
                    parts.append('</table>')

                if td.get("regressions"):
                    parts.append('<table><tr><th>Regressions</th><th>Baseline</th><th>Candidate</th></tr>')
                    for reg in td["regressions"]:
                        parts.append(
                            f'<tr><td class="verdict-regressed">❌ {_html_escape(reg["field"])}</td>'
                            f'<td>{_html_escape(reg["baseline"])}</td>'
                            f'<td>{_html_escape(reg["candidate"])}</td></tr>'
                        )
                    parts.append('</table>')

                if td.get("changes"):
                    parts.append('<table><tr><th>Field</th><th>Baseline</th><th>Candidate</th></tr>')
                    for c in td["changes"]:
                        field_name = c["field"]
                        if field_name == "prompt":
                            parts.append(
                                f'<tr><td><strong>prompt</strong></td>'
                                f'<td colspan="2"><details><summary>Click to expand prompt diff</summary>'
                                f'<pre>{_html_escape(c.get("baseline", ""))}</pre>'
                                f'<pre>{_html_escape(c.get("candidate", ""))}</pre>'
                                f'</details></td></tr>'
                            )
                        else:
                            parts.append(
                                f'<tr><td><strong>{_html_escape(field_name)}</strong></td>'
                                f'<td class="diff-val">{_html_escape(c.get("baseline"))}</td>'
                                f'<td class="diff-val">{_html_escape(c.get("candidate"))}</td></tr>'
                            )
                    parts.append('</table>')
        else:
            parts.append('<p>No per-turn changes.</p>')

    return _HTML_TEMPLATE.format(body="\n".join(parts))


def generate_report(comparison: dict, fmt: str = "html") -> str:
    if fmt == "markdown":
        return generate_markdown_report(comparison)
    return generate_html_report(comparison)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------


def _cli_capture(args):
    print(f"Replaying all fixtures for capture '{args.label}'...")
    capture = capture_conversations(args.label)
    save_capture(capture, args.output)
    total_turns = sum(f.get("summary", {}).get("turn_count", 0) for f in capture.get("fixtures", []))
    print(f"  Captured {len(capture.get('fixtures', []))} fixtures ({total_turns} turns) -> {args.output}")

    if args.report:
        if args.baseline:
            print(f"  Loading baseline '{args.baseline}' for comparison...")
            baseline = load_capture(args.baseline)
            comparison = compare_captures(baseline, capture)
            report = generate_report(comparison, fmt=args.format)
        else:
            print("  No baseline provided; generating standalone summary report.")
            comparison = compare_captures(capture, capture)
            report = generate_report(comparison, fmt=args.format)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report written to {args.report}")


def _cli_compare(args):
    baseline = load_capture(args.baseline)
    candidate = load_capture(args.candidate)
    comparison = compare_captures(baseline, candidate)
    report = generate_report(comparison, fmt=args.format)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    verdicts = {}
    for fd in comparison["fixture_diffs"]:
        v = fd["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    print(f"Comparison report written to {args.output}")
    print(f"  Verdicts: {verdicts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Conversation Replay Lab — capture, compare, and report."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Replay fixtures and save a capture file.")
    cap.add_argument("--label", required=True, help="Label for this capture (e.g. 'before').")
    cap.add_argument("-o", "--output", required=True, help="Output JSON file path.")
    cap.add_argument("--report", default=None, help="Optional: also write a report file.")
    cap.add_argument("--format", default="html", choices=["html", "markdown"], help="Report format.")
    cap.add_argument("--baseline", default=None, help="Optional baseline capture to compare against in the report.")
    cap.set_defaults(func=_cli_capture)

    cmp = sub.add_parser("compare", help="Compare two capture files and generate a report.")
    cmp.add_argument("--baseline", required=True, help="Baseline capture JSON file.")
    cmp.add_argument("--candidate", required=True, help="Candidate capture JSON file.")
    cmp.add_argument("-o", "--output", required=True, help="Output report file path.")
    cmp.add_argument("--format", default="html", choices=["html", "markdown"], help="Report format.")
    cmp.set_defaults(func=_cli_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
