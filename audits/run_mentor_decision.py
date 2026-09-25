"""
run_mentor_decision.py — deterministic Mentor Decision Audit.

Measurement ONLY. This script replays every golden conversation
fixture against the live ``mentor.process_mentor_turn`` pipeline
(Semantic Complexity Gate ON, the production default) and aggregates
the per-turn ``MentorDecision`` diagnostics produced by
``mentor_decision_audit`` into an end-of-run report:

  * Objective Distribution — how often each objective was pursued.
  * Repeated Objectives — objectives the mentor returned to.
  * Conversation Continuity Score — GOOD/PARTIAL/POOR weighted.
  * Question Quality Score — Natural / Slightly Repetitive / Repeated /
    Questionnaire-like weighted.
  * Most Common Weak Transitions — (prev → current objective) pairs
    where continuity was POOR or quality was Repeated/Questionnaire-like.
  * Better-Alternative Turns — turns where another objective would
    likely have produced a better conversation.

Nothing about objectives, lifecycle, prompts, extraction, or replies is
changed — this script only reports measurements the pipeline already
records on every turn.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's
    canned extraction — the same source the audit classifies.
  * No real LLM is called.

Usage:
    python run_mentor_decision.py
"""

import os
import sys
from unittest import mock

os.environ["COMPLEXITY_GATE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)
from mentor_decision_audit import MentorDecisionSummary  # noqa: E402


def _collect() -> tuple[MentorDecisionSummary, int, int]:
    summary = MentorDecisionSummary()
    turns_with_decision = 0
    turns_total = 0

    for fixture in load_all_fixtures():
        import mentor  # noqa: F402

        _seed_session(fixture)
        for _index, turn in enumerate(fixture.turns):
            ollama_stub = (
                _RaisingOllama()
                if turn.ollama_reply is None
                else _StubOllama([turn.ollama_reply])
            )
            extraction = _build_extraction(turn)
            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    _reply, _session, _timing, diagnostics = (
                        mentor.process_mentor_turn(
                            turn.user,
                            username=fixture.username,
                            project_name=fixture.project_name,
                        )
                    )

            turns_total += 1
            if diagnostics.get("MentorDecision"):
                turns_with_decision += 1

        # mentor already aggregated this fixture's raw per-turn records
        # into the session summary; fold it into the global report.
        from session_manager import get_session_manager  # noqa: F402

        session_summary = (
            get_session_manager()
            .get_active_session_data()
            .mentor_decision_summary
        )
        summary.merge(session_summary)

    return summary, turns_total, turns_with_decision


def _print_report(
    summary: MentorDecisionSummary,
    turns_total: int,
    turns_with_decision: int,
) -> None:
    print("=" * 68)
    print("Mentor Decision Audit")
    print("=" * 68)
    print(f"  Turns replayed:          {turns_total}")
    print(f"  Turns with decision audit: {turns_with_decision}")
    print(f"  Total turns aggregated:  {summary.turn_count}")
    print()

    print("--- Objective Distribution ---")
    for obj, count in sorted(
        summary.objective_counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {obj:<20} {count}")
    print()

    repeated = summary.repeated_objectives()
    print("--- Repeated Objectives ---")
    if not repeated:
        print("  (none)")
    for obj, count in sorted(repeated.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {obj:<20} {count} turns")
    print()

    print("--- Conversation Continuity ---")
    for k, v in sorted(summary.continuity_counts.items()):
        print(f"  {k:<10} {v}")
    print(f"  Continuity Score: {summary.continuity_score():.1f}%")
    print()

    print("--- Question Quality ---")
    for k, v in sorted(summary.quality_counts.items()):
        print(f"  {k:<24} {v}")
    print(f"  Question Quality Score: {summary.quality_score():.1f}%")
    print()

    print("--- Transition Type ---")
    for k, v in sorted(summary.transition_counts.items()):
        print(f"  {k:<20} {v}")
    print(f"  Transition Score: {summary.transition_score():.1f}%")
    print()

    print("--- Most Common Weak Transitions ---")
    weak = summary.most_common_weak_transitions()
    if not weak:
        print("  (none)")
    for fp, count in sorted(weak.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:>3}  {fp}")
    print()

    better = summary.better_alternative_turns
    print("--- Better-Alternative Turns ---")
    if not better:
        print("  (none)")
    for e in better:
        print(
            f"  Turn {e.get('turn')}: {e.get('objective')} -> "
            f"better = {e.get('better_alternative')} -- {e.get('reason')}"
        )
        print(f"    User: {e.get('message')!r}")
        print(f"    Q:    {e.get('question')!r}")
    print("=" * 68)


def main():
    summary, turns_total, turns_with_decision = _collect()
    _print_report(summary, turns_total, turns_with_decision)
    return summary


if __name__ == "__main__":
    main()