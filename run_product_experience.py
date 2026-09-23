"""
run_product_experience.py — deterministic Product Experience Report.

Replays every golden conversation fixture against the live
``mentor.process_mentor_turn`` pipeline (Semantic Complexity Gate ON) and
aggregates the per-turn ``ProductExperience`` diagnostics produced by
``product_experience_audit`` into an end-of-run report covering Startup,
Session, Performance, Developer Console, Export, and Recommendations.

Nothing about objectives, extraction, lifecycle, prompts, or replies is
changed — this script only reports measurements the pipeline already records
on every turn.

The run is fully deterministic:

  * ``ollama`` is stubbed (deterministic fallback replies).
  * ``mentor.MemoryExtractor.extract`` is patched with the fixture's
    canned extraction — the same source the audit observes.
  * No real LLM is called.

Usage:
    python run_product_experience.py
"""

import os
import time
from unittest import mock
from datetime import datetime, timezone

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
from product_experience_audit import ProductExperienceSummary  # noqa: E402


def _collect() -> tuple[ProductExperienceSummary, int]:
    summary = ProductExperienceSummary()
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
            .product_experience_summary
        )
        summary.merge(session_summary)

    return summary, turns_total


def _format_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{round(ms)} ms"


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _print_report(
    summary: ProductExperienceSummary,
    turns_total: int,
) -> None:
    print("=" * 68)
    print("  Product Experience Report")
    print("=" * 68)
    print(f"  Turns replayed:   {turns_total}")
    print(f"  Turns aggregated:  {summary.turn_count}")
    print()

    # ---- Startup ----
    print("  Startup")
    startup = summary.startup
    if startup:
        print(f"    First message:     {_format_ms(startup.get('first_message_ms', 0))}")
        print(f"    Mentor creation:   {_format_ms(startup.get('mentor_creation_ms', 0))}")
        print(f"    Total startup:     {_format_ms(startup.get('total_startup_duration_ms', 0))}")
    else:
        print("    (no startup data recorded)")
    print()

    # ---- Session ----
    print("  Session")
    s = summary.session
    if s:
        print(f"    Session id:        {s.get('session_id', '—')}")
        print(f"    Restored:          {'Yes' if s.get('session_restored') else 'No'}")
        print(f"    Messages at start: {s.get('message_count_at_start', 0)}")
        print(f"    Prior conversation: {'Yes' if s.get('previous_conversation_present') else 'No'}")
        print(f"    Refresh required:  {'Yes' if s.get('refresh_required') else 'No'}")
    else:
        print("    (no session data recorded)")
    print()

    # ---- Performance ----
    print("  Performance")
    tc = summary.turn_count or 1
    print(f"    Avg turn time:       {_format_ms(summary.average_turn_time_ms())}")
    print(f"    Max turn time:       {_format_ms(summary.max_turn_ms)}")
    print(f"    Avg extraction:      {_format_ms(summary.total_extraction_ms / tc)}")
    print(f"    Avg merge:           {_format_ms(summary.total_merge_ms / tc)}")
    print(f"    Avg objective:       {_format_ms(summary.total_objective_ms / tc)}")
    print(f"    Avg prompt build:    {_format_ms(summary.total_prompt_build_ms / tc)}")
    print(f"    Avg LLM:             {_format_ms(summary.total_llm_ms / tc)}")
    print(f"    Avg render:          {_format_ms(summary.total_render_ms / tc)}")
    exc_pct = (
        (summary.total_extraction_ms / summary.total_turn_ms * 100)
        if summary.total_turn_ms else 0.0
    )
    print(f"    Extraction time (%): {exc_pct:.1f}%")
    print()

    # ---- Developer Console ----
    print("  Developer Console")
    print(f"    Avg sections:        {summary.average_section_count()}")
    print(f"    Avg entries per turn:{summary.average_diag_entries()}")
    print(f"    Max nesting depth:   {summary.max_nesting_depth}")
    print(f"    Avg prompt size:     {summary.average_prompt_chars()} chars")
    print(f"    Avg rendered lines:  {round(summary.total_estimated_lines / tc)}")
    print()

    # ---- Export ----
    print("  Export")
    print(f"    Total messages:      {summary.total_messages}")
    print(f"    Max per turn:        {summary.max_messages}")
    print(f"    Total diag size:     {summary.total_diagnostics_chars} chars")
    print(f"    Total export chars:  {summary.total_export_chars}")
    print()

    # ---- Recommendations ----
    print("  Recommendations")
    lines = []
    if summary.max_turn_ms > 1000:
        lines.append("- Slowest turn exceeds 1 s; consider reducing extraction/LLM overhead.")
    if summary.max_nesting_depth > 5:
        lines.append(f"- Diagnostics nesting depth is {summary.max_nesting_depth}; review if any section can be flattened.")
    if summary.average_prompt_chars() > 4000:
        lines.append(f"- Avg prompt size is large ({summary.average_prompt_chars()} chars); consider trimming instructions.")
    exc_pct2 = (
        (summary.total_extraction_ms / summary.total_turn_ms * 100)
        if summary.total_turn_ms else 0.0
    )
    if exc_pct2 > 60:
        lines.append(f"- Extraction dominates at {exc_pct2:.1f}% of turn time; model or caching candidates.")
    if not lines:
        lines.append("- All metrics are within typical bounds. No actionable recommendations.")
    for l_ in lines:
        print(f"    {l_}")
    print("=" * 68)


def main():
    start = time.perf_counter()
    summary, turns_total = _collect()
    elapsed_s = round(time.perf_counter() - start, 2)
    _print_report(summary, turns_total)
    return summary


if __name__ == "__main__":
    main()