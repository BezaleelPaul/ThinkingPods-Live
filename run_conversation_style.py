"""
run_conversation_style.py — deterministic Conversation Style Audit.

Measurement ONLY. Replays every golden conversation fixture against the live
``mentor.process_mentor_turn`` pipeline (Semantic Complexity Gate ON, the
production default) and aggregates the per-turn ``ConversationStyle``
diagnostics produced by ``conversation_style_audit`` into an end-of-run
report:

  * Most Common Openings
  * Opening Diversity Score
  * Average Reply Length (words / sentences)
  * Reflection Rate
  * Summary Rate
  * Bridge Rate
  * Questionnaire Rate (immediately asks a question)
  * Topic Restart Rate
  * Multiple-Question Rate
  * Begins-with-Acknowledgment Rate

Nothing about objectives, lifecycle, prompts, extraction, or replies is
changed — this script only reports measurements the pipeline already records
on every turn.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's
    canned extraction — the same source the audit classifies.
  * No real LLM is called.

Usage:
    python run_conversation_style.py
"""

import os
import sys
from unittest import mock

os.environ["COMPLEXITY_GATE"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)
from conversation_style_audit import ConversationStyleSummary  # noqa: E402


def _collect() -> tuple[ConversationStyleSummary, int]:
    summary = ConversationStyleSummary()
    turns_total = 0

    for fixture in load_all_fixtures():
        import mentor  # noqa: F402

        _seed_session(fixture)
        for turn in fixture.turns:
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

        from session_manager import get_session_manager  # noqa: F402

        session_summary = (
            get_session_manager()
            .get_active_session_data()
            .conversation_style_summary
        )
        summary.merge(session_summary)

    return summary, turns_total


def _print_report(
    summary: ConversationStyleSummary,
    turns_total: int,
) -> None:
    print("=" * 68)
    print("Conversation Style Audit")
    print("=" * 68)
    print(f"  Turns replayed:   {turns_total}")
    print(f"  Turns aggregated: {summary.turn_count}")
    print()

    print("--- Most Common Openings ---")
    openings = summary.most_common_openings(limit=10)
    if not openings:
        print("  (none)")
    for opening, count in openings:
        print(f"  {count:>3}  {opening!r}")
    print()

    print(f"--- Opening Diversity Score: {summary.opening_diversity_score():.2f} / 1.00 ---")
    print()

    print("--- Average Reply Length ---")
    print(
        f"  {summary.average_reply_length():.1f} words/reply "
        f"({summary.total_sentences} sentences total)"
    )
    print()

    print("--- Per-Style Rates ---")
    for label, num in (
        ("Reflection",             summary.reflection_count),
        ("Summary",                summary.summary_count),
        ("Bridge to Prior Info",   summary.bridge_count),
        ("Immediate Question",     summary.questionnaire_count),
        ("Topic Restart",          summary.restart_count),
        ("Multiple Unrelated Qs",  summary.multiple_question_count),
        ("Begins with Acknowledgment", summary.acknowledgment_count),
    ):
        rate = (num / summary.turn_count * 100) if summary.turn_count else 0.0
        print(f"  {label:<28} {num:>3}/{summary.turn_count:<3} ({rate:5.1f}%)")
    print("=" * 68)


def main():
    summary, turns_total = _collect()
    _print_report(summary, turns_total)
    return summary


if __name__ == "__main__":
    main()