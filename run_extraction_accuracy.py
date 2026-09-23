"""
run_extraction_accuracy.py — deterministic LLM Extraction Accuracy Audit.

Measurement ONLY. This script replays every golden conversation fixture
against the live ``mentor.process_mentor_turn`` pipeline (Semantic
Complexity Gate ON, the production default) and aggregates the per-turn
``ExtractionAccuracy`` diagnostics produced by ``extraction_accuracy`` into
an end-of-run report:

  * Extraction Accuracy Summary — per-update label counts
    (Correct / Partially Correct / Incorrect / Missed Opportunity).
  * Per-field precision — Audience, Problems, Pain, Motivation, Evidence,
    Frequency, Current Solution, Impact.
  * Most common mistakes — reason-key frequency.

Nothing about extraction, the hybrid decision, objectives, lifecycle,
prompts, or replies is changed — this script only reports measurements that
the pipeline already records on every LLM-invoked turn.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's canned
    extraction — the same source the audit classifies.
  * No real LLM is called.

Usage:
    python run_extraction_accuracy.py
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
from extraction_accuracy import ExtractionAccuracySummary  # noqa: E402


def _collect() -> tuple[ExtractionAccuracySummary, int, int]:
    summary = ExtractionAccuracySummary()
    turns_with_llm = 0
    turns_with_audit = 0

    for fixture in load_all_fixtures():
        import mentor

        _seed_session(fixture)
        for index, turn in enumerate(fixture.turns):
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

            hybrid = diagnostics.get("HybridExtraction") or {}
            if hybrid.get("LLM Invoked") == "Yes":
                turns_with_llm += 1
            if diagnostics.get("ExtractionAccuracy"):
                turns_with_audit += 1

        # mentor already aggregated this fixture's raw per-turn accuracy
        # records into the session summary; fold it into the global report.
        from session_manager import get_session_manager

        session_summary = get_session_manager().get_active_session_data().extraction_accuracy_summary
        summary.merge(session_summary)

    return summary, turns_with_llm, turns_with_audit


def _print_report(summary: ExtractionAccuracySummary, turns_with_llm: int, turns_with_audit: int) -> None:
    total = summary.total()
    print("=" * 68)
    print("LLM Extraction Accuracy Audit")
    print("=" * 68)
    print(
        f"  Turns with LLM invoked:  {turns_with_llm}"
    )
    print(f"  Turns with accuracy audit: {turns_with_audit}")
    print(f"  Updates classified:       {total}")
    print(f"  Turn-level missed facts:  {summary.unattached_misses}")
    print()

    def _pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "0.0%"

    print("--- Extraction Accuracy Summary ---")
    print(f"  Correct:             {summary.correct}  ({_pct(summary.correct)})")
    print(
        f"  Partially Correct:   {summary.partially_correct}  "
        f"({_pct(summary.partially_correct)})"
    )
    print(
        f"  Incorrect:           {summary.incorrect}  ({_pct(summary.incorrect)})"
    )
    print(
        f"  Missed Opportunity:  {summary.missed}  ({_pct(summary.missed)})"
    )

    print()
    print("--- Per-field precision ---")
    for label, info in summary.per_field_precision().items():
        if info["classified"]:
            print(
                f"  {label:<16} {info['correct']}/{info['classified']} "
                f"({info['pct']}%)"
            )
        else:
            print(f"  {label:<16} (no updates)")

    print()
    print("--- Most common mistakes ---")
    if not summary.reason_counts:
        print("  (none)")
    for reason, count in sorted(
        summary.reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        print(f"  {count:>3}  {reason}")

    if summary.examples:
        print()
        print("--- Example missed opportunities ---")
        for ex in summary.examples:
            print(f"  Message: {ex['message']!r}")
            for m in ex["missed"]:
                print(f"      missed {m['field']}: {m['fact']} ({m['kind']})")
    print("=" * 68)


def main():
    summary, turns_with_llm, turns_with_audit = _collect()
    _print_report(summary, turns_with_llm, turns_with_audit)
    return summary


if __name__ == "__main__":
    main()
