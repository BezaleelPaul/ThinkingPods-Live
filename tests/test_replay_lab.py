"""
tests/test_replay_lab.py — regression tests for the Conversation Replay Lab.

Tests are deterministic (mocked ollama + mocked MemoryExtractor, reusing
the golden-fixture infrastructure).  No real LLM is called.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import replay_lab  # noqa: E402
from tests.goldens.fixtures import (  # noqa: E402
    GoldenConversationFixture,
    GoldenTurn,
)
from tests.goldens.runner import (  # noqa: E402
    _RaisingOllama,
    _build_extraction,
    _seed_session,
)


def _make_fixture(name="lab_test", turns=2):
    """Build a minimal fixture for deterministic capture tests."""
    turn_list = []
    for i in range(turns):
        turn_list.append(
            GoldenTurn(
                user=f"user message {i}",
                extraction={
                    "message_type": "MEANINGFUL",
                    "updates": [
                        {"operation": "ADD", "field": "personas", "value": f"persona {i}"}
                    ],
                },
                ollama_reply=None,
            )
        )
    return GoldenConversationFixture(
        name=name,
        scenario=0,
        description="lab test fixture",
        username="lab_user",
        project_name="LabProject",
        turns=turn_list,
    )


# ---------------------------------------------------------------------------
# Capture tests
# ---------------------------------------------------------------------------


class TestCapture(unittest.TestCase):
    """capture_conversations produces a JSON-serialisable dict with all
    per-turn signals of interest."""

    def test_capture_returns_well_formed_dict(self):
        fixture = _make_fixture()
        capture = replay_lab.capture_conversations("test", fixtures=[fixture])
        self.assertEqual(capture["label"], "test")
        self.assertEqual(capture["fixture_count"], 1)
        self.assertEqual(len(capture["fixtures"]), 1)
        self.assertEqual(capture["fixtures"][0]["fixture_name"], "lab_test")
        self.assertEqual(len(capture["fixtures"][0]["turns"]), 2)

    def test_captured_turn_has_all_required_fields(self):
        fixture = _make_fixture()
        capture = replay_lab.capture_conversations("test", fixtures=[fixture])
        turn = capture["fixtures"][0]["turns"][0]
        required_fields = [
            "turn_index", "user_message", "reply", "objective",
            "question_family", "transition_type", "continuity",
            "question_quality", "state_before", "state_after", "prompt",
            "memory_open_threads", "memory_deferred_topics",
            "memory_resolved_threads", "extraction_ms",
            "continuity_score", "quality_score", "transition_score",
            "hybrid_llm_invoked",
        ]
        for f in required_fields:
            self.assertIn(f, turn, f"captured turn missing field: {f}")

    def test_capture_is_json_serialisable(self):
        fixture = _make_fixture()
        capture = replay_lab.capture_conversations("test", fixtures=[fixture])
        # Should not raise
        s = json.dumps(capture, ensure_ascii=False, default=str)
        self.assertTrue(s)

    def test_save_and_load_capture_round_trip(self):
        fixture = _make_fixture()
        capture = replay_lab.capture_conversations("round_trip", fixtures=[fixture])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            path = f.name
        try:
            replay_lab.save_capture(capture, path)
            loaded = replay_lab.load_capture(path)
            self.assertEqual(loaded["label"], "round_trip")
            self.assertEqual(
                loaded["fixtures"][0]["fixture_name"], "lab_test"
            )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Compare tests
# ---------------------------------------------------------------------------


class TestCompare(unittest.TestCase):
    """The comparison engine detects changes, improvements, regressions."""

    def _make_turn(
        self, index=0, user="hi", reply="q", objective="PERSONAS",
        question_family="PERSONAS_WHO", transition_type="Direct",
        continuity="GOOD", question_quality="Natural",
        continuity_score="50.0%", extraction_ms=1.0,
        state_after=None, prompt="prompt",
    ):
        return {
            "turn_index": index,
            "user_message": user,
            "reply": reply,
            "objective": objective,
            "response_strategy": "ASK_QUESTION",
            "lifecycle_decision": "CONTINUE",
            "stage": "Empathize",
            "question_family": question_family,
            "planned_family": question_family,
            "reask": False,
            "asked_families": [],
            "transition_type": transition_type,
            "transition_reason": "r",
            "continuity": continuity,
            "continuity_reason": "r",
            "question_quality": question_quality,
            "quality_reason": "r",
            "objective_appropriate": "Yes",
            "acknowledged": "Yes",
            "restarted": "No",
            "better_alternative": None,
            "state_before": {},
            "state_after": state_after or {},
            "state_changes": {},
            "memory_open_threads": [],
            "memory_deferred_topics": [],
            "memory_resolved_threads": [],
            "extraction_message_type": "MEANINGFUL",
            "extraction_updates": [],
            "hybrid_llm_invoked": "No",
            "hybrid_objective_satisfied_by_rules": None,
            "prompt": prompt,
            "continuity_score": continuity_score,
            "quality_score": "50.0%",
            "transition_score": "50.0%",
            "extraction_ms": extraction_ms,
            "llm_ms": 0.0,
            "total_ms": 0.0,
        }

    def _make_capture(self, label, turns):
        return {
            "label": label,
            "timestamp": "2026-01-01",
            "fixture_count": 1,
            "fixtures": [{
                "fixture_name": "f1",
                "description": "",
                "username": "u",
                "turns": turns,
                "summary": {
                    "turn_count": len(turns),
                    "objective_counts": {},
                    "continuity_counts": {},
                    "quality_counts": {},
                    "transition_counts": {},
                    "continuity_score": 50.0,
                    "quality_score": 50.0,
                    "transition_score": 50.0,
                    "appropriate_count": 0,
                    "acknowledged_count": 0,
                    "restarted_count": 0,
                },
            }],
        }

    def test_identical_captures_unchanged(self):
        turn = self._make_turn()
        a = self._make_capture("a", [turn])
        b = self._make_capture("b", [turn])
        diff = replay_lab.compare_captures(a, b)
        self.assertEqual(diff["fixture_diffs"][0]["verdict"], "unchanged")
        self.assertEqual(diff["fixture_diffs"][0]["turns_changed"], 0)

    def test_objective_change_detected(self):
        ta = self._make_turn(objective="PERSONAS")
        tb = self._make_turn(objective="PROBLEMS")
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        fd = diff["fixture_diffs"][0]
        self.assertEqual(fd["verdict"], "changed")
        self.assertEqual(fd["turns_changed"], 1)
        change_fields = [c["field"] for c in fd["turn_diffs"][0]["changes"]]
        self.assertIn("objective", change_fields)

    def test_continuity_score_improvement_detected(self):
        ta = self._make_turn(continuity_score="50.0%")
        tb = self._make_turn(continuity_score="90.0%")
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        td = diff["fixture_diffs"][0]["turn_diffs"][0]
        self.assertTrue(td["improvements"])
        self.assertEqual(td["improvements"][0]["field"], "continuity_score")

    def test_extraction_latency_regression_detected(self):
        ta = self._make_turn(extraction_ms=1.0)
        tb = self._make_turn(extraction_ms=999.0)
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        fd = diff["fixture_diffs"][0]
        self.assertEqual(fd["verdict"], "regressed")
        td = fd["turn_diffs"][0]
        self.assertTrue(td["regressions"])
        self.assertEqual(td["regressions"][0]["field"], "extraction_ms")

    def test_extraction_improvement_detected(self):
        # Lower extraction latency = improvement
        ta = self._make_turn(extraction_ms=999.0)
        tb = self._make_turn(extraction_ms=1.0)
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        td = diff["fixture_diffs"][0]["turn_diffs"][0]
        self.assertTrue(td["improvements"])
        self.assertEqual(td["improvements"][0]["field"], "extraction_ms")

    def test_state_change_detected(self):
        ta = self._make_turn(state_after={"personas": ["a"]})
        tb = self._make_turn(state_after={"personas": ["b"]})
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        change_fields = [
            c["field"]
            for c in diff["fixture_diffs"][0]["turn_diffs"][0]["changes"]
        ]
        self.assertIn("state.personas", change_fields)

    def test_prompt_change_detected(self):
        ta = self._make_turn(prompt="prompt A")
        tb = self._make_turn(prompt="prompt B")
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        change_fields = [
            c["field"]
            for c in diff["fixture_diffs"][0]["turn_diffs"][0]["changes"]
        ]
        self.assertIn("prompt", change_fields)

    def test_transition_type_change_detected(self):
        ta = self._make_turn(transition_type="Topic Restart")
        tb = self._make_turn(transition_type="Direct")
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        change_fields = [
            c["field"]
            for c in diff["fixture_diffs"][0]["turn_diffs"][0]["changes"]
        ]
        self.assertIn("transition_type", change_fields)

    def test_question_family_change_detected(self):
        ta = self._make_turn(question_family="PERSONAS_WHO")
        tb = self._make_turn(question_family="PROBLEMS_CORE")
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        change_fields = [
            c["field"]
            for c in diff["fixture_diffs"][0]["turn_diffs"][0]["changes"]
        ]
        self.assertIn("question_family", change_fields)

    def test_memory_change_detected(self):
        ta = self._make_turn()
        ta["memory_open_threads"] = [{"field": "problems", "reason": "partial"}]
        tb = self._make_turn()
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        change_fields = [
            c["field"]
            for c in diff["fixture_diffs"][0]["turn_diffs"][0]["changes"]
        ]
        self.assertIn("conversation_memory", change_fields)

    def test_summary_metric_improvement_in_fixture_diff(self):
        a = self._make_capture("a", [self._make_turn()])
        a["fixtures"][0]["summary"]["continuity_score"] = 40.0
        b = self._make_capture("b", [self._make_turn()])
        b["fixtures"][0]["summary"]["continuity_score"] = 90.0
        diff = replay_lab.compare_captures(a, b)
        sd = diff["fixture_diffs"][0]["summary_diff"]
        self.assertIn("continuity_score", sd)
        self.assertEqual(sd["continuity_score"]["verdict"], "improved")

    def test_no_regressions_when_only_improvements(self):
        ta = self._make_turn(continuity_score="50.0%", extraction_ms=10.0)
        tb = self._make_turn(continuity_score="90.0%", extraction_ms=1.0)
        a = self._make_capture("a", [ta])
        b = self._make_capture("b", [tb])
        diff = replay_lab.compare_captures(a, b)
        self.assertEqual(diff["fixture_diffs"][0]["verdict"], "improved")

    def test_missing_fixture_in_candidate_marked_removed(self):
        turn = self._make_turn()
        a = self._make_capture("a", [turn])
        b = {"label": "b", "timestamp": "x", "fixture_count": 0, "fixtures": []}
        diff = replay_lab.compare_captures(a, b)
        self.assertEqual(diff["fixture_diffs"][0]["verdict"], "removed")

    def test_added_fixture_in_candidate_marked_added(self):
        turn = self._make_turn()
        a = {"label": "a", "timestamp": "x", "fixture_count": 0, "fixtures": []}
        b = self._make_capture("b", [turn])
        diff = replay_lab.compare_captures(a, b)
        self.assertEqual(diff["fixture_diffs"][0]["verdict"], "added")


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


class TestReportGeneration(unittest.TestCase):
    """HTML and Markdown reports render the comparison cleanly."""

    def _make_simple_capture(self, label, objective="PERSONAS", score=50.0):
        return {
            "label": label,
            "timestamp": "2026-01-01",
            "fixture_count": 1,
            "fixtures": [{
                "fixture_name": "f1",
                "description": "",
                "username": "u",
                "turns": [
                    {
                        "turn_index": 0,
                        "user_message": "hi",
                        "reply": "q",
                        "objective": objective,
                        "continuity_score": f"{score:.1f}%",
                        "extraction_ms": 1.0,
                        "state_after": {"personas": ["a"]},
                        "prompt": "p",
                        "memory_open_threads": [],
                        "memory_deferred_topics": [],
                        "memory_resolved_threads": [],
                        "hybrid_llm_invoked": "No",
                        "question_family": "PERSONAS_WHO",
                        "transition_type": "Direct",
                        "continuity": "GOOD",
                        "question_quality": "Natural",
                    }
                ],
                "summary": {
                    "turn_count": 1,
                    "continuity_score": score,
                    "quality_score": 50.0,
                    "transition_score": 50.0,
                },
            }],
        }

    def test_markdown_report_contains_baseline_and_candidate_labels(self):
        a = self._make_simple_capture("before")
        b = self._make_simple_capture("after")
        comparison = replay_lab.compare_captures(a, b)
        report = replay_lab.generate_markdown_report(comparison)
        self.assertIn("Conversation Replay Lab Report", report)
        self.assertIn("`before`", report)
        self.assertIn("`after`", report)

    def test_markdown_report_shows_unchanged_when_identical(self):
        a = self._make_simple_capture("a")
        b = self._make_simple_capture("b")
        comparison = replay_lab.compare_captures(a, b)
        report = replay_lab.generate_markdown_report(comparison)
        self.assertIn("unchanged", report)

    def test_markdown_report_shows_improvement(self):
        a = self._make_simple_capture("a", score=40.0)
        b = self._make_simple_capture("b", score=90.0)
        comparison = replay_lab.compare_captures(a, b)
        report = replay_lab.generate_markdown_report(comparison)
        self.assertIn("Improvements", report)
        self.assertIn("continuity_score", report)

    def test_html_report_contains_required_html_structure(self):
        a = self._make_simple_capture("a")
        b = self._make_simple_capture("b")
        comparison = replay_lab.compare_captures(a, b)
        report = replay_lab.generate_html_report(comparison)
        self.assertIn("<!DOCTYPE html>", report)
        self.assertIn("<html", report)
        self.assertIn("</html>", report)
        self.assertIn("Conversation Replay Lab", report)

    def test_html_report_contains_verdict_badge_for_change(self):
        a = self._make_simple_capture("a", objective="PERSONAS")
        b = self._make_simple_capture("b", objective="PROBLEMS")
        comparison = replay_lab.compare_captures(a, b)
        report = replay_lab.generate_html_report(comparison)
        self.assertIn("badge", report)

    def test_generate_report_dispatches_on_format(self):
        a = self._make_simple_capture("a")
        b = self._make_simple_capture("b")
        comparison = replay_lab.compare_captures(a, b)
        md = replay_lab.generate_report(comparison, fmt="markdown")
        html = replay_lab.generate_report(comparison, fmt="html")
        self.assertIn("# Conversation Replay Lab Report", md)
        self.assertIn("<!DOCTYPE html>", html)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """The argparse CLI dispatches capture and compare sub-commands."""

    def test_capture_subcommand_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.json")
            with mock.patch.object(
                replay_lab, "capture_conversations",
                return_value={"label": "x", "timestamp": "t",
                              "fixture_count": 0, "fixtures": []},
            ):
                replay_lab.main(["capture", "--label", "x", "-o", path])
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["label"], "x")

    def test_compare_subcommand_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.json")
            cand_path = os.path.join(tmp, "cand.json")
            rep_path = os.path.join(tmp, "rep.html")
            base = {"label": "b", "timestamp": "t", "fixture_count": 0, "fixtures": []}
            cand = {"label": "c", "timestamp": "t", "fixture_count": 0, "fixtures": []}
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(base, f)
            with open(cand_path, "w", encoding="utf-8") as f:
                json.dump(cand, f)
            replay_lab.main([
                "compare", "--baseline", base_path,
                "--candidate", cand_path,
                "-o", rep_path, "--format", "html",
            ])
            self.assertTrue(os.path.exists(rep_path))
            with open(rep_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<!DOCTYPE html>", content)

    def test_compare_subcommand_markdown_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, "base.json")
            cand_path = os.path.join(tmp, "cand.json")
            rep_path = os.path.join(tmp, "rep.md")
            base = {"label": "b", "timestamp": "t", "fixture_count": 0, "fixtures": []}
            cand = {"label": "c", "timestamp": "t", "fixture_count": 0, "fixtures": []}
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(base, f)
            with open(cand_path, "w", encoding="utf-8") as f:
                json.dump(cand, f)
            replay_lab.main([
                "compare", "--baseline", base_path,
                "--candidate", cand_path,
                "-o", rep_path, "--format", "markdown",
            ])
            self.assertTrue(os.path.exists(rep_path))
            with open(rep_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# Conversation Replay Lab Report", content)


# ---------------------------------------------------------------------------
# Regression: replay lab is observation-only
# ---------------------------------------------------------------------------


class TestReplayLabIsObservationOnly(unittest.TestCase):
    """Replay Lab must not change mentor behavior — running it leaves
    the pipeline in the same state the golden runner would."""

    def test_capture_does_not_alter_objective_sequence(self):
        fixture = _make_fixture()
        capture = replay_lab.capture_conversations("obs", fixtures=[fixture])
        first_turn = capture["fixtures"][0]["turns"][0]
        # First turn after seeding personas on an empty state -> objective
        # should be PROBLEMS (the next-unmet field), exactly as the
        # golden runner produces.
        self.assertEqual(first_turn["objective"], "PROBLEMS")

    def test_capture_records_final_state_evolution(self):
        fixture = _make_fixture(turns=3)
        capture = replay_lab.capture_conversations("obs", fixtures=[fixture])
        last_turn = capture["fixtures"][0]["turns"][-1]
        self.assertIn("persona 2", last_turn["state_after"].get("personas", []))


if __name__ == "__main__":
    unittest.main()
