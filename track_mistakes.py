"""track_mistakes.py — Unified Mistake & Failure Tracking Engine for ReqGPT / ThinkingPods.

Provides end-to-end mistake tracking across:
  1. Golden Scenarios: Replays verified conversation fixtures to audit mentor failures.
  2. Saved Sessions: Scans user sessions in `sessions/` to detect real-world conversational errors.
  3. Turn-by-Turn Diagnostics: Explains the root cause of every mistake with actionable recommendations.

Usage:
  python track_mistakes.py                     # Run complete audit (goldens + saved sessions)
  python track_mistakes.py --goldens           # Audit golden test fixtures
  python track_mistakes.py --sessions          # Audit all saved user sessions in sessions/
  python track_mistakes.py --session <file>    # Audit a specific session file
  python track_mistakes.py --verbose           # Detailed turn-by-turn mistake log
  python track_mistakes.py --export            # Export report to docs/reports/Mistakes_Audit_Report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversation_failure_audit import (
    ConversationFailure,
    ConversationFailureRecord,
    ConversationFailureSummary,
    analyze_conversation_failure,
    failure_label,
)

# Terminal color codes (safe fallback on Windows)
try:
    import colorama
    colorama.init()
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
except Exception:
    RED = GREEN = YELLOW = BLUE = CYAN = BOLD = RESET = ""


# ---------------------------------------------------------------------------
# Data Models for Mistake Tracking
# ---------------------------------------------------------------------------

@dataclass
class MistakeOccurrence:
    """Represents a single mistake detected in a conversation turn."""
    source: str                 # "golden:<fixture_id>" or "session:<filename>"
    turn_index: int
    primary_failure: str
    secondary_failures: List[str]
    user_input: str
    assistant_reply: str
    confidence: float
    explanation: str
    recommendation: str

    @property
    def severity(self) -> str:
        critical = {
            "MISUNDERSTOOD_RESPONSE",
            "ABRUPT_TRANSITION",
            "MEMORY_FAILURE",
            "MOVE_SELECTION",
        }
        warning = {
            "REPEATED_INFORMATION",
            "OVER_EXPLORATION",
            "UNSUPPORTED_INFERENCE",
            "MISSED_INSIGHT",
            "SOCIAL_FAILURE",
        }
        if self.primary_failure in critical:
            return "CRITICAL"
        if self.primary_failure in warning:
            return "WARNING"
        return "INFO"


@dataclass
class MistakeReport:
    """Aggregated results across all audited conversations."""
    total_turns: int = 0
    clean_turns: int = 0
    flawed_turns: int = 0
    mistakes: List[MistakeOccurrence] = field(default_factory=list)
    category_counts: Counter[str] = field(default_factory=Counter)

    @property
    def failure_rate(self) -> float:
        return (self.flawed_turns / self.total_turns * 100.0) if self.total_turns else 0.0


# ---------------------------------------------------------------------------
# Explanation & Fix Recommendations
# ---------------------------------------------------------------------------

RECOMMENDATIONS: Dict[str, Tuple[str, str]] = {
    "MISUNDERSTOOD_RESPONSE": (
        "Mentor re-asked a question for a field that already has a usable value in project state.",
        "Check question family rotation and ensure the objective engine recognizes the field as completed.",
    ),
    "ABRUPT_TRANSITION": (
        "Mentor transitioned to a new topic or summary without acknowledging the user's input.",
        "Ensure conversational memory marks threads as acknowledged before triggering lifecycle transitions.",
    ),
    "REPEATED_INFORMATION": (
        "User re-offered information because previous mentor reply did not confirm or store it.",
        "Review extractor field mappings and ensure prompt reflects acknowledged facts.",
    ),
    "OVER_EXPLORATION": (
        "Mentor continued interrogating a topic that was already fully resolved.",
        "Tune objective engine sufficiency threshold to advance earlier once required fields are populated.",
    ),
    "UNSUPPORTED_INFERENCE": (
        "Mentor response implied a magnitude or frequency without supporting evidence in state.",
        "Check prompt builder constraints and enforce that assumptions are stated as hypotheses.",
    ),
    "MISSED_INSIGHT": (
        "Project state holds extracted data that conversational memory never opened a thread for.",
        "Update conversation memory to track newly extracted scalar and list fields automatically.",
    ),
    "MEMORY_FAILURE": (
        "Conversational memory contains contradictory statuses (e.g. field open and acknowledged simultaneously).",
        "Inspect update_conversation_memory transitions and thread resolution hooks.",
    ),
    "SOCIAL_FAILURE": (
        "Mentor response reprimanded or scolded the user outside an intentional challenge move.",
        "Enforce polite, constructive mentor phrasing in response strategy prompt.",
    ),
    "MOVE_SELECTION": (
        "Selected conversational move does not match the actual reply structure (e.g. elicit without question).",
        "Verify ConversationMove classifier and prompt instructions for the chosen move.",
    ),
    "META_INTENT": (
        "User turn was non-substantive (greeting, validation, trust check) rather than design content.",
        "Ensure meta-intent handler acknowledges the user smoothly and bridges back to design thinking.",
    ),
}


# ---------------------------------------------------------------------------
# Golden Conversation Auditor
# ---------------------------------------------------------------------------

def audit_goldens() -> MistakeReport:
    """Audits all golden test fixtures against the live mentor pipeline."""
    report = MistakeReport()
    try:
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import _RaisingOllama, _StubOllama, _seed_session
        import mentor
    except ImportError as err:
        print(f"[!] Warning: Could not import golden fixtures: {err}")
        return report

    os.environ["COMPLEXITY_GATE"] = "true"
    from session_manager import get_session_manager

    fixtures = load_all_fixtures()
    for fixture in fixtures:
        _seed_session(fixture)
        for idx, turn in enumerate(fixture.turns, start=1):
            report.total_turns += 1

            ollama_stub = (
                _RaisingOllama()
                if turn.ollama_reply is None
                else _StubOllama([turn.ollama_reply])
            )

            from tests.goldens.runner import _build_extraction
            canned = _build_extraction(turn)

            from unittest import mock
            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                with mock.patch("mentor.MemoryExtractor.extract", return_value=canned):
                    reply = mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )

            session_data = get_session_manager().get_active_session_data()
            summary = session_data.conversation_failure_summary
            rec_dict = summary.turn_records[-1] if summary.turn_records else {}
            rec = ConversationFailureRecord.from_dict(rec_dict)
            prim = rec.primary_failure.value if isinstance(rec.primary_failure, ConversationFailure) else str(rec.primary_failure)

            if rec.primary_failure != ConversationFailure.NONE and prim != "NONE":
                report.flawed_turns += 1
                report.category_counts[prim] += 1

                expl, reco = RECOMMENDATIONS.get(
                    prim,
                    ("Detected conversational inconsistency.", "Review pipeline turn diagnostics."),
                )
                if rec.reason:
                    expl = f"{expl} (Details: {rec.reason})"

                secs = [
                    s.value if isinstance(s, ConversationFailure) else str(s)
                    for s in rec.secondary_failures
                ]

                fix_id = getattr(fixture, "name", fixture.project_name)
                report.mistakes.append(MistakeOccurrence(
                    source=f"golden:{fix_id}",
                    turn_index=idx,
                    primary_failure=prim,
                    secondary_failures=secs,
                    user_input=turn.user,
                    assistant_reply=reply if isinstance(reply, str) else "",
                    confidence=rec.confidence,
                    explanation=expl,
                    recommendation=reco,
                ))
            else:
                report.clean_turns += 1

    return report


# ---------------------------------------------------------------------------
# Saved Session Auditor
# ---------------------------------------------------------------------------

def audit_session_file(filepath: str) -> List[MistakeOccurrence]:
    """Audits a single saved session JSON file for mistakes."""
    mistakes: List[MistakeOccurrence] = []
    if not os.path.exists(filepath):
        return mistakes

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        return mistakes

    # Session files might contain turns, diagnostics, or history
    turns = data.get("turns") or data.get("history") or []
    for idx, turn_data in enumerate(turns, start=1):
        if not isinstance(turn_data, dict):
            continue

        diag = turn_data.get("diagnostics") or {}
        fail_dict = diag.get("ConversationFailure") or {}
        rec = ConversationFailureRecord.from_dict(fail_dict)
        prim = rec.primary_failure.value if isinstance(rec.primary_failure, ConversationFailure) else str(rec.primary_failure)

        user_msg = turn_data.get("user") or turn_data.get("user_message") or ""
        asst_msg = turn_data.get("assistant") or turn_data.get("reply") or ""

        if rec.primary_failure != ConversationFailure.NONE and prim != "NONE":
            expl, reco = RECOMMENDATIONS.get(
                prim,
                ("Inconsistency recorded in session diagnostics.", "Review turn details."),
            )
            if rec.reason:
                expl = f"{expl} [{rec.reason}]"

            secs = [
                s.value if isinstance(s, ConversationFailure) else str(s)
                for s in rec.secondary_failures
            ]
            mistakes.append(MistakeOccurrence(
                source=f"session:{os.path.basename(filepath)}",
                turn_index=idx,
                primary_failure=prim,
                secondary_failures=secs,
                user_input=user_msg,
                assistant_reply=asst_msg,
                confidence=rec.confidence,
                explanation=expl,
                recommendation=reco,
            ))

    return mistakes


def audit_all_sessions(sessions_dir: str = "sessions") -> MistakeReport:
    """Audits all JSON session files in the given directory."""
    report = MistakeReport()
    session_files = glob.glob(os.path.join(sessions_dir, "*.json"))

    for sfile in session_files:
        mistakes = audit_session_file(sfile)
        # Approximate turn count from file
        try:
            with open(sfile, "r", encoding="utf-8") as f:
                d = json.load(f)
                t_count = len(d.get("turns") or d.get("history") or [])
        except Exception:
            t_count = len(mistakes)

        report.total_turns += t_count
        report.flawed_turns += len(mistakes)
        report.clean_turns += max(0, t_count - len(mistakes))
        for m in mistakes:
            report.category_counts[m.primary_failure] += 1
            report.mistakes.append(m)

    return report


# ---------------------------------------------------------------------------
# Output & Reporting
# ---------------------------------------------------------------------------

def print_mistake_summary(report: MistakeReport, title: str = "Mistake Tracking Summary") -> None:
    """Prints a styled summary dashboard of detected mistakes."""
    width = 68
    print()
    print("=" * width)
    print(f"  {BOLD}{title}{RESET}")
    print("=" * width)
    print(f"  Total Turns Analyzed:  {report.total_turns}")
    print(f"  Clean Turns:           {GREEN}{report.clean_turns}{RESET}")
    print(f"  Turns with Mistakes:   {RED if report.flawed_turns else GREEN}{report.flawed_turns}{RESET}")
    print(f"  Mistake Rate:          {RED if report.failure_rate > 15 else YELLOW}{report.failure_rate:.1f}%{RESET}")
    print("-" * width)

    if not report.mistakes:
        print(f"  {GREEN}[OK] No conversational mistakes detected! System is operating cleanly.{RESET}")
        print("=" * width)
        print()
        return

    print(f"  {BOLD}Mistakes by Category:{RESET}")
    for cat, count in report.category_counts.most_common():
        label = failure_label(cat)
        pct = (count / report.total_turns * 100.0) if report.total_turns else 0.0
        bar = "#" * max(1, int(pct / 2))
        print(f"    {label:<28} {count:>3}  ({pct:>4.1f}%)  {YELLOW}{bar}{RESET}")

    print("-" * width)
    print(f"  {BOLD}Top Actionable Recommendations:{RESET}")
    seen_cats = set()
    for m in report.mistakes:
        if m.primary_failure not in seen_cats:
            seen_cats.add(m.primary_failure)
            _, reco = RECOMMENDATIONS.get(m.primary_failure, ("", "Review diagnostics."))
            print(f"    * [{failure_label(m.primary_failure)}]: {reco}")
    print("=" * width)
    print()


def print_turn_details(mistakes: List[MistakeOccurrence]) -> None:
    """Prints detailed turn-by-turn breakdown of each mistake."""
    print(f"\n{BOLD}Turn-by-Turn Mistake Log ({len(mistakes)} items):{RESET}")
    for i, m in enumerate(mistakes, start=1):
        color = RED if m.severity == "CRITICAL" else (YELLOW if m.severity == "WARNING" else BLUE)
        print(f"\n[{i}] {color}{BOLD}[{m.severity}] {failure_label(m.primary_failure)}{RESET}")
        print(f"    Source: {m.source} (Turn {m.turn_index})")
        print(f"    User:   \"{m.user_input[:90]}\"")
        if m.assistant_reply:
            print(f"    Mentor: \"{m.assistant_reply[:90]}\"")
        print(f"    Cause:  {m.explanation}")
        print(f"    Fix:    {m.recommendation}")


def export_markdown_report(report: MistakeReport, output_path: str = "docs/reports/Mistakes_Audit_Report.md") -> None:
    """Exports a comprehensive markdown report for the team."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [
        "# Mistake & Failure Tracking Report",
        "",
        f"**Date:** System Audit",
        f"**Total Turns Audited:** {report.total_turns}",
        f"**Clean Turns:** {report.clean_turns}",
        f"**Turns with Mistakes:** {report.flawed_turns} ({report.failure_rate:.1f}%)",
        "",
        "## 1. Executive Summary",
        "",
        "| Category | Occurrences | Percentage | Severity |",
        "|---|---|---|---|",
    ]

    for cat, count in report.category_counts.most_common():
        pct = (count / report.total_turns * 100.0) if report.total_turns else 0.0
        label = failure_label(cat)
        expl, _ = RECOMMENDATIONS.get(cat, ("", ""))
        sev = "Critical" if cat in {"MISUNDERSTOOD_RESPONSE", "ABRUPT_TRANSITION", "MEMORY_FAILURE"} else "Warning"
        lines.append(f"| **{label}** | {count} | {pct:.1f}% | {sev} |")

    lines.extend([
        "",
        "## 2. Actionable Fix Recommendations",
        "",
    ])

    seen = set()
    for m in report.mistakes:
        if m.primary_failure not in seen:
            seen.add(m.primary_failure)
            _, reco = RECOMMENDATIONS.get(m.primary_failure, ("", ""))
            lines.append(f"### {failure_label(m.primary_failure)}")
            lines.append(f"- **Explanation:** {m.explanation}")
            lines.append(f"- **Recommended Fix:** {reco}")
            lines.append("")

    lines.extend([
        "## 3. Sample Turn-by-Turn Mistake Log",
        "",
    ])

    for i, m in enumerate(report.mistakes[:20], start=1):
        lines.append(f"#### Mistake #{i}: {failure_label(m.primary_failure)} (`{m.source}` T{m.turn_index})")
        lines.append(f"- **User:** \"{m.user_input}\"")
        if m.assistant_reply:
            lines.append(f"- **Mentor:** \"{m.assistant_reply}\"")
        lines.append(f"- **Root Cause:** {m.explanation}")
        lines.append(f"- **Action:** {m.recommendation}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Full Mistake Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ReqGPT Unified Mistake & Failure Tracker")
    parser.add_argument("--goldens", action="store_true", help="Audit golden conversation fixtures")
    parser.add_argument("--sessions", action="store_true", help="Audit saved user sessions in sessions/")
    parser.add_argument("--session", type=str, default=None, help="Audit a specific session file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show turn-by-turn detailed mistake logs")
    parser.add_argument("--export", action="store_true", help="Export report to docs/reports/Mistakes_Audit_Report.md")
    args = parser.parse_args()

    combined_report = MistakeReport()

    # Determine what to run
    run_all = not (args.goldens or args.sessions or args.session)

    if args.session:
        print(f"Auditing specific session file: {args.session}...")
        mistakes = audit_session_file(args.session)
        combined_report.total_turns = len(mistakes)
        combined_report.flawed_turns = len(mistakes)
        for m in mistakes:
            combined_report.category_counts[m.primary_failure] += 1
            combined_report.mistakes.append(m)
        print_mistake_summary(combined_report, f"Session Audit: {os.path.basename(args.session)}")

    if run_all or args.goldens:
        print("Auditing Golden Conversation Fixtures...")
        golden_report = audit_goldens()
        combined_report.total_turns += golden_report.total_turns
        combined_report.clean_turns += golden_report.clean_turns
        combined_report.flawed_turns += golden_report.flawed_turns
        combined_report.category_counts.update(golden_report.category_counts)
        combined_report.mistakes.extend(golden_report.mistakes)
        if not run_all:
            print_mistake_summary(golden_report, "Golden Conversations Mistake Audit")

    if run_all or args.sessions:
        print("Auditing Saved User Sessions (sessions/)...")
        session_report = audit_all_sessions("sessions")
        combined_report.total_turns += session_report.total_turns
        combined_report.clean_turns += session_report.clean_turns
        combined_report.flawed_turns += session_report.flawed_turns
        combined_report.category_counts.update(session_report.category_counts)
        combined_report.mistakes.extend(session_report.mistakes)
        if not run_all:
            print_mistake_summary(session_report, "User Sessions Mistake Audit")

    if run_all:
        print_mistake_summary(combined_report, "ReqGPT Global Mistake Tracking Audit")

    if args.verbose and combined_report.mistakes:
        print_turn_details(combined_report.mistakes)

    if args.export or run_all:
        export_markdown_report(combined_report)


if __name__ == "__main__":
    main()
