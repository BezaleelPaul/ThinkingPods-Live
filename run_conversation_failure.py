"""
run_conversation_failure.py — deterministic Conversation Failure Report.

Replays every golden conversation fixture against the live
``mentor.process_mentor_turn`` pipeline (Semantic Complexity Gate ON) and
aggregates the per-turn ``ConversationFailure`` diagnostics produced by
``conversation_failure_audit`` into an end-of-run report: turns analysed,
failure rate, per-category counts/percentages, most common failure, and
actionable recommendations.

Nothing about objectives, extraction, lifecycle, prompts, or replies is
changed — this script only reports measurements the pipeline already records
on every turn.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's canned
    extraction — the same source the audit observes.
  * No real LLM is called.

Usage:
    python run_conversation_failure.py
"""

import os
import time
from unittest import mock

os.environ["COMPLEXITY_GATE"] = "true"

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)
from conversation_failure_audit import (  # noqa: E402
    ConversationFailure,
    ConversationFailureSummary,
    failure_label,
)


def _collect() -> tuple[ConversationFailureSummary, int]:
    summary = ConversationFailureSummary()
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
                    mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )
            turns_total += 1

        from session_manager import get_session_manager  # noqa: F402

        session_summary = (
            get_session_manager()
            .get_active_session_data()
            .conversation_failure_summary
        )
        summary.merge(session_summary)

    return summary, turns_total


def _print_report(summary: ConversationFailureSummary, turns_total: int) -> None:
    print("=" * 68)
    print("  Conversation Failure Report")
    print("=" * 68)
    print(f"  Turns replayed:   {turns_total}")
    print(f"  Turns aggregated: {summary.total_turns()}")
    print(f"  Turns with failures: {summary.total_failures()}")
    print(f"  Turn failure rate:   {summary.failure_rate():.0%}")
    print()

    counts = summary.failure_counts()
    percents = summary.per_failure_percentages()

    if counts:
        print("  Failure Categories")
        width = max(len(failure_label(ConversationFailure(c))) for c in counts) + 2
        for category, count in counts.items():
            label = failure_label(ConversationFailure(category))
            pct = percents.get(category, 0.0)
            bar = "#" * int(round(pct / 100.0 * 40))
            print(f"    {label:<{width}} {count:>4}  {pct:5.1f}%  {bar}")
        print()
        most = summary.most_common_failure()
        if most:
            print(f"  Most common failure: {most}")
        print(f"  Categories fired:    {len(counts)}")
        print()

    print("  Recommendations")
    lines = []
    if summary.total_turns() and summary.failure_rate() >= 0.5:
        lines.append(
            "- More than half of turns carry a failure; review the mentor's "
            "handling of already-answered fields and topic transitions."
        )
    if counts.get(ConversationFailure.MISUNDERSTOOD_RESPONSE.value):
        lines.append(
            "- Mentor re-asked fields already holding a value; consider "
            "cross-checking project state before re-asking."
        )
    if counts.get(ConversationFailure.OVER_EXPLORATION.value):
        lines.append(
            "- Mentor over-explored resolved fields; consider broader topic "
            "rotation after a field reaches sufficiency."
        )
    if counts.get(ConversationFailure.MEMORY_FAILURE.value):
        lines.append(
            "- Conversational memory self-contradicts (acknowledged yet open); "
            "review how acknowledgment/transition updates threads."
        )
    if counts.get(ConversationFailure.MOVE_SELECTION.value):
        lines.append(
            "- Some turns selected a move that did not match the generated "
            "reply (e.g. ELICIT_INFORMATION without a question)."
        )
    if counts.get(ConversationFailure.REPEATED_INFORMATION.value):
        lines.append(
            "- User re-offered content for already-acknowledged fields; the "
            "mentor may transition too early or over-close topics."
        )
    if not lines:
        lines.append(
            "- No meaningful failure patterns observed across golden fixtures. "
            "Keep auditing as fixtures grow."
        )
    for line in lines:
        print(f"    {line}")
    print("=" * 68)


def main():
    start = time.perf_counter()
    summary, turns_total = _collect()
    print(f"  (replay took {round(time.perf_counter() - start, 2)} s)")
    _print_report(summary, turns_total)
    return summary


if __name__ == "__main__":
    main()