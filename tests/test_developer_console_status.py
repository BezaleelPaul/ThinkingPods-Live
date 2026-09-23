"""
tests/test_developer_console_status.py — regression tests for the visual
status indicators added to the Developer Console.

Covers:

  * ``section_status`` — the deterministic per-key classifier mapping every
    known diagnostics section to ``SectionStatus`` (healthy/warning/problem)
    plus a short human reason, WITHOUT ever mutating or dropping diagnostics,
  * ``status_icon`` / ``status_label`` — the cosmetic presentation helpers,
  * tree integration — ``build_tree_plan`` attaches ``status`` /
    ``status_reason`` to every ``SectionNode`` (including the synthetic
    ``Performance`` block) and leaves the underlying diagnostics untouched.

These are pure-model tests (no Streamlit) and never touch conversation
behavior, extraction, objectives, or the live pipeline.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from developer_console import (  # noqa: E402
    SectionStatus,
    build_tree_plan,
    build_turn,
    section_status,
    status_icon,
    status_label,
)

HEALTHY = SectionStatus.HEALTHY
WARNING = SectionStatus.WARNING
PROBLEM = SectionStatus.PROBLEM


class TestStatusPresentation(unittest.TestCase):

    def test_icons_and_labels_are_stable(self):
        self.assertEqual(status_icon(HEALTHY), "🟢")
        self.assertEqual(status_icon(WARNING), "🟡")
        self.assertEqual(status_icon(PROBLEM), "🔴")

    def test_labels_are_human(self):
        self.assertEqual(status_label(HEALTHY), "Healthy")
        self.assertEqual(status_label(WARNING), "Warning")
        self.assertEqual(status_label(PROBLEM), "Problem")

    def test_default_healthy_for_unknown_key(self):
        status, reason = section_status("TotallyNewSection", {"a": 1})
        self.assertIs(status, HEALTHY)
        self.assertEqual(reason, "")


class TestPerformanceStatus(unittest.TestCase):

    def test_fast_turn_is_healthy(self):
        status, reason = section_status("Performance", timing={"total_ms": 366})
        self.assertIs(status, HEALTHY)
        self.assertEqual(reason, "")

    def test_slow_turn_is_warning(self):
        status, reason = section_status("Performance", timing={"total_ms": 4000})
        self.assertIs(status, WARNING)
        self.assertIn("Slow", reason)

    def test_very_slow_turn_is_problem(self):
        status, reason = section_status("Performance", timing={"total_ms": 84561})
        self.assertIs(status, PROBLEM)
        self.assertIn("Slow", reason)

    def test_slow_llm_is_problem(self):
        status, reason = section_status(
            "Performance", timing={"total_ms": 2000, "llm_ms": 9000})
        self.assertIs(status, PROBLEM)
        self.assertIn("LLM", reason)

    def test_missing_timing_is_healthy(self):
        status, _ = section_status("Performance", None, timing=None)
        self.assertIs(status, HEALTHY)


class TestConversationFailureStatus(unittest.TestCase):

    def test_unsupported_inference_is_problem(self):
        status, reason = section_status("ConversationFailure", {
            "Primary Failure": "Unsupported inference",
            "Secondary Failures": "None",
            "Confidence": "0.98",
        })
        self.assertIs(status, PROBLEM)
        self.assertEqual(reason, "Unsupported inference detected")

    def test_none_primary_is_healthy(self):
        status, _ = section_status("ConversationFailure", {
            "Primary Failure": "None", "Secondary Failures": "None"})
        self.assertIs(status, HEALTHY)

    def test_secondary_only_is_warning(self):
        status, _ = section_status("ConversationFailure", {
            "Primary Failure": "None", "Secondary Failures": "Repeated information"})
        self.assertIs(status, WARNING)

    def test_summary_with_failures_is_problem(self):
        status, reason = section_status("ConversationFailureSummary", {
            "Turns with Failures": 3, "Turn Failure Rate": "30%"})
        self.assertIs(status, PROBLEM)
        self.assertIn("3", reason)


class TestExtractionStatus(unittest.TestCase):

    def test_rule_llm_disagreement_is_warning(self):
        status, reason = section_status("ExtractionComparison", {
            "Decision": {"Rule sufficient": "No", "LLM Applied": "Yes"},
            "Fields Only Found by LLM": ["frequency"],
        })
        self.assertIs(status, WARNING)
        self.assertEqual(reason, "Rule/LLM disagreement")

    def test_llm_discarded_is_warning(self):
        status, _ = section_status("ExtractionComparison", {
            "Decision": {"LLM Applied": "No"},
            "LLM Extractor": ["frequency"],
        })
        self.assertIs(status, WARNING)

    def test_agreeing_extraction_is_healthy(self):
        status, _ = section_status("ExtractionComparison", {
            "Decision": {"Rule sufficient": "Yes", "LLM Applied": "Yes"},
            "Fields That Match": ["frequency"],
        })
        self.assertIs(status, HEALTHY)

    def test_incorrect_extraction_is_problem(self):
        status, _ = section_status("ExtractionAccuracy", {
            "Extracted Updates": [{"label": "Incorrect", "field": "frequency"}],
        })
        self.assertIs(status, PROBLEM)

    def test_dropped_missed_opportunity_is_problem(self):
        status, _ = section_status("ExtractionAccuracy", {
            "Missed Opportunities": [{"field": "evidence", "attached": "No"}],
        })
        self.assertIs(status, PROBLEM)

    def test_attached_missed_opportunity_is_warning(self):
        status, _ = section_status("ExtractionAccuracy", {
            "Missed Opportunities": [{"field": "evidence", "attached": "Yes"}],
        })
        self.assertIs(status, WARNING)

    def test_summary_is_problem_when_counts_exist(self):
        status, _ = section_status("ExtractionAccuracySummary", {
            "Incorrect": 1, "Missed Opportunity": 2})
        self.assertIs(status, PROBLEM)


class TestRecoveryStatus(unittest.TestCase):

    def test_contradiction_is_problem(self):
        status, reason = section_status("Recovery", {
            "Category": "CONTRADICTION", "Detected Inconsistency": "Frequency changed"})
        self.assertIs(status, PROBLEM)
        self.assertEqual(reason, "Frequency changed")

    def test_none_category_is_healthy(self):
        status, _ = section_status("Recovery", {"Category": "NONE"})
        self.assertIs(status, HEALTHY)

    def test_uncertain_answer_is_warning(self):
        status, _ = section_status("Recovery", {"Category": "UNCERTAIN_ANSWER"})
        self.assertIs(status, WARNING)


class TestOtherSections(unittest.TestCase):

    def test_prompt_is_healthy_by_default(self):
        status, _ = section_status("Prompt", "You are a mentor.")
        self.assertIs(status, HEALTHY)

    def test_metrics_healthy_without_problems(self):
        status, _ = section_status("ConversationMetrics", {
            "Repeated Question Attempts": 0})
        self.assertIs(status, HEALTHY)

    def test_metrics_repeated_attempts_is_warning(self):
        status, _ = section_status("ConversationMetrics", {
            "Repeated Question Attempts": 2})
        self.assertIs(status, WARNING)

    def test_memory_open_threads_is_warning(self):
        status, reason = section_status("Memory", {
            "Open Threads": [{"field": "frequency", "reason": "asked", "turn": 1}]})
        self.assertIs(status, WARNING)
        self.assertIn("1", reason)

    def test_project_state_empty_is_warning(self):
        status, _ = section_status("ProjectState", {})
        self.assertIs(status, WARNING)

    def test_objective_trace_missing_fields_is_warning(self):
        status, _ = section_status("ObjectiveTrace", {
            "Missing Fields": ["evidence", "impacts"]})
        self.assertIs(status, WARNING)

    def test_hybrid_extraction_double_no_is_problem(self):
        status, _ = section_status("HybridExtraction", {
            "Objective Satisfied By Rules": "No", "LLM Invoked": "No"})
        self.assertIs(status, PROBLEM)


class TestTreeIntegration(unittest.TestCase):

    def _turn(self):
        return build_turn(
            turn_index=0,
            ordinal=1,
            role="assistant",
            content="Hi",
            timing={"total_ms": 900.0, "llm_ms": 500.0},
            diagnostics={
                "ConversationFailure": {
                    "Primary Failure": "Unsupported inference",
                    "Secondary Failures": "None",
                },
                "Prompt": "You are a mentor.",
            },
        )

    def _sections(self, turn):
        node = build_tree_plan([turn], selected_index=0).turns[0]
        out = {}
        for category in node.categories:
            for section in category.sections:
                out[section.key] = section
        return out

    def test_section_status_attached_to_nodes(self):
        sections = self._sections(self._turn())
        self.assertEqual(sections["ConversationFailure"].status, PROBLEM.value)
        self.assertNotEqual(sections["ConversationFailure"].status_reason, "")
        self.assertEqual(sections["Prompt"].status, HEALTHY.value)
        self.assertEqual(sections["Performance"].status, HEALTHY.value)

    def test_performance_status_reflects_timing(self):
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics={}, timing={"total_ms": 84561})
        sections = self._sections(turn)
        self.assertEqual(sections["Performance"].status, PROBLEM.value)
        self.assertIn("Slow", sections["Performance"].status_reason)

    def test_status_never_mutates_diagnostics(self):
        diag = {
            "ConversationFailure": {"Primary Failure": "Unsupported inference"},
        }
        snapshot = copy.deepcopy(diag)
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics=diag, timing={"total_ms": 10})
        self._sections(turn)
        self.assertEqual(diag, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)