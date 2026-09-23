"""
tests/test_product_experience.py — regression tests for the observation-only
Product Experience Audit.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import product_experience_audit as audit  # noqa: E402
from product_experience_audit import (  # noqa: E402
    ProductExperienceSummary,
    analyze_product_experience,
    product_experience_diagnostics_section,
)


# ---------------------------------------------------------------------------
# Unit tests for analyze_product_experience
# ---------------------------------------------------------------------------


class TestAnalyzeProductExperience(unittest.TestCase):

    def test_empty_inputs_all_keys_present(self):
        r = analyze_product_experience()
        self.assertIn("turn_index", r)
        self.assertIn("developer_console", r)
        self.assertIn("export", r)
        self.assertIn("performance", r)
        self.assertNotIn("startup", r)
        self.assertNotIn("session", r)

    def test_startup_and_session_emitted_on_first_turn(self):
        r = analyze_product_experience(
            turn_index=1,
            first_turn=True,
            startup_timestamp="2026-08-04T12:00:00Z",
            history_count_at_start=0,
            turn_metrics_at_start=0,
            session_id="abc123",
            timing={"prepare_ms": 42.0, "total_ms": 100.0},
        )
        self.assertIn("startup", r)
        self.assertIn("session", r)
        self.assertEqual(r["startup"]["app_startup_iso"], "2026-08-04T12:00:00Z")
        self.assertEqual(r["startup"]["mentor_creation_ms"], 42)
        self.assertIsNone(r["startup"]["first_page_render_ms"])
        self.assertEqual(r["startup"]["first_message_ms"], 100)
        self.assertFalse(r["session"]["session_restored"])
        self.assertEqual(r["session"]["session_id"], "abc123")

    def test_restored_session_detected(self):
        r = analyze_product_experience(
            first_turn=True,
            history_count_at_start=4,
            startup_timestamp="2026-08-04T12:00:00Z",
        )
        self.assertIn("session", r)
        self.assertTrue(r["session"]["session_restored"])
        self.assertTrue(r["session"]["previous_conversation_present"])

    def test_performance_fields_rounded(self):
        r = analyze_product_experience(
            turn_index=3,
            timing={
                "extraction_ms": 50.0,
                "objective_ms": 20.0,
                "prompt_ms": 10.0,
                "llm_ms": 200.0,
                "finalize_ms": 5.0,
                "total_ms": 320.0,
                "extraction_breakdown": {"merge_ms": 5},
            },
            diagnostics={
                "Pipeline": {"Stage": "Empathize", "Objective": "PROBLEMS"},
                "ProjectState": {"Personas": ["students"]},
            },
            conversation_history=[{"role": "assistant", "content": "Hi there"}],
        )
        self.assertEqual(r["developer_console"]["section_count"], 2)
        self.assertEqual(r["performance"]["extraction_ms"], 50)
        self.assertEqual(r["performance"]["merge_ms"], 5)
        self.assertEqual(r["performance"]["total_ms"], 320)
        self.assertEqual(r["export"]["total_messages"], 1)

    def test_developer_console_detects_largest_section(self):
        r = analyze_product_experience(
            diagnostics={"A": [1], "B": [1, 2, 3], "C": [1, 2]},
        )
        largest = r["developer_console"]["largest_section"]
        self.assertIsNotNone(largest)
        self.assertEqual(largest[0], "B")  # B has more entries

    def test_export_estimate_near_markdown_size(self):
        hist = [{"role": "assistant", "content": "Hello user"}]
        r = analyze_product_experience(conversation_history=hist)
        # 12 chars overhead + "Hello user"(11) + "assistant"(9) = ~32
        self.assertGreater(r["export"]["estimated_export_chars"], 20)
        self.assertLess(r["export"]["estimated_export_chars"], 50)


# ---------------------------------------------------------------------------
# Diagnostics section tests
# ---------------------------------------------------------------------------


class TestDiagnosticsSection(unittest.TestCase):

    def test_empty_without_record(self):
        self.assertEqual(product_experience_diagnostics_section(None), {})
        self.assertEqual(product_experience_diagnostics_section({}), {})

    def test_projection_copies_expected_fields(self):
        r = analyze_product_experience(
            turn_index=7,
            timing={"extraction_ms": 55, "total_ms": 110},
            diagnostics={"A": 1},
        )
        proj = product_experience_diagnostics_section(r)
        self.assertEqual(proj["Turn"], 7)
        self.assertEqual(proj["Performance → Extraction"], "55 ms")

    def test_section_is_json_serialisable(self):
        r = analyze_product_experience(turn_index=1, timing={"total_ms": 100})
        proj = product_experience_diagnostics_section(r)
        round_tripped = json.loads(json.dumps(proj, default=str))
        self.assertEqual(round_tripped["Turn"], 1)


# ---------------------------------------------------------------------------
# Summary aggregation tests
# ---------------------------------------------------------------------------


def _make_record(
    turn_index=1,
    first_turn=False,
    startup_timestamp="2026-08-04T12:00:00Z",
    session_id="Test123",
    history_at_start=0,
    timing_ms=None,
    export_messages=0,
    sections=0,
):
    timing_ms = timing_ms or {}
    return analyze_product_experience(
        turn_index=turn_index,
        session_id=session_id,
        conversation_history=(
            [{"role": "user", "content": "x"}] * export_messages
        ),
        timing={
            "extraction_ms": timing_ms.get("extraction", 50),
            "objective_ms": timing_ms.get("objective", 20),
            "prompt_ms": timing_ms.get("prompt", 10),
            "llm_ms": timing_ms.get("llm", 100),
            "finalize_ms": timing_ms.get("finalize", 5),
            "total_ms": timing_ms.get("total", 200),
            "extraction_breakdown": {"merge_ms": timing_ms.get("merge", 3)},
        },
        diagnostics={"Test": True} if sections else {},
        first_turn=first_turn,
        history_count_at_start=history_at_start,
        turn_metrics_at_start=1 if history_at_start else 0,
        startup_timestamp=startup_timestamp if first_turn else "",
    )


class TestSummaryAggregation(unittest.TestCase):

    def test_add_record_stores_startup_once(self):
        s = ProductExperienceSummary()
        s.add_record(_make_record(first_turn=True, session_id="s1"))
        self.assertEqual(s.turn_count, 1)
        self.assertIn("session_id", s.session)
        self.assertEqual(s.session["session_id"], "s1")
        self.assertEqual(s.fresh_count, 1)

    def test_multiple_turn_counts_and_no_overwrite(self):
        s = ProductExperienceSummary()
        s.add_record(_make_record(1, first_turn=True, session_id="s1",
                                  startup_timestamp="time1"))
        s.add_record(_make_record(2, first_turn=False))
        s.add_record(_make_record(3, first_turn=False))
        self.assertEqual(s.turn_count, 3)
        self.assertEqual(s.startup["app_startup_iso"], "time1")
        self.assertEqual(s.session["session_id"], "s1")

    def test_json_roundtrip(self):
        s = ProductExperienceSummary()
        s.add_record(_make_record(1, first_turn=True))
        s.add_record(_make_record(2))
        d = s.to_dict()
        restored = ProductExperienceSummary.from_dict(d)
        self.assertEqual(s.turn_count, restored.turn_count)
        self.assertEqual(s.startup, restored.startup)
        self.assertEqual(s.session, restored.session)
        self.assertEqual(s.total_extraction_ms, restored.total_extraction_ms)

    def test_merge_combines_shape(self):
        a = ProductExperienceSummary()
        a.add_record(_make_record(1, first_turn=True, session_id="s1",
                                  timing_ms={"total": 100}))
        a.add_record(_make_record(2, timing_ms={"total": 200}))
        b = ProductExperienceSummary()
        b.add_record(_make_record(1, first_turn=True, session_id="s2",
                                  timing_ms={"total": 150}))
        a.merge(b)
        self.assertEqual(a.turn_count, 3)
        self.assertEqual(a.fresh_count, 2)
        self.assertIn("s1", a.session_ids_seen)
        self.assertIn("s2", a.session_ids_seen)
        self.assertEqual(a.total_turn_ms, 100 + 200 + 150)

    def test_zero_rates_for_empty_summary(self):
        s = ProductExperienceSummary()
        self.assertEqual(s.average_turn_time_ms(), 0.0)
        self.assertEqual(s.average_diag_entries(), 0.0)
        self.assertEqual(s.average_prompt_chars(), 0.0)
        self.assertEqual(s.average_export_chars(), 0.0)

    def test_to_display(self):
        s = ProductExperienceSummary()
        s.add_record(_make_record(1, first_turn=True))
        d = s.to_display()
        self.assertIn("Turns Analyzed", d)
        self.assertIn("Avg Turn Time", d)
        self.assertIn("Session Id", d)

    def test_from_empty_dict_produces_empty_summary(self):
        s = ProductExperienceSummary.from_dict(None)
        self.assertEqual(s.turn_count, 0)
        s = ProductExperienceSummary.from_dict({})
        self.assertEqual(s.turn_count, 0)


# ---------------------------------------------------------------------------
# Live pipeline wiring tests
# ---------------------------------------------------------------------------


class TestWiringLivePipeline(unittest.TestCase):

    def _run_one_turn(self, fixture):
        import mentor
        from tests.goldens.runner import (
            _RaisingOllama,
            _StubOllama,
            _build_extraction,
            _seed_session,
        )
        from session_manager import get_session_manager

        _seed_session(fixture)
        mgr = get_session_manager()
        turn = fixture.turns[0]
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
                reply, _session, _timing, diag = mentor.process_mentor_turn(
                    turn.user,
                    username=fixture.username,
                    project_name=fixture.project_name,
                )
        return reply, diag, mgr

    def test_product_experience_sections_appear_in_diagnostics(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        _reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertIn("ProductExperience", diagnostics)
        self.assertIn("ProductExperienceSummary", diagnostics)

    def test_reply_unchanged_and_audit_present(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertIsNotNone(reply)
        self.assertTrue(reply.strip())
        self.assertIn("ProductExperience", diagnostics)

    def test_session_summary_persists_across_turns(self):
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import (
            _RaisingOllama,
            _build_extraction,
            _seed_session,
        )
        from session_manager import get_session_manager
        import mentor

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)

        for turn in fixture.turns:
            ollama_stub = _RaisingOllama()
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

        sd = get_session_manager().get_active_session_data()
        pexp = sd.product_experience_summary
        self.assertEqual(pexp.turn_count, len(fixture.turns))
        self.assertIn("session_id", pexp.session)
        self.assertEqual(len(pexp.turn_records), len(fixture.turns))

    def test_session_persistence_roundtrip_survives_summary(self):
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import (
            _RaisingOllama,
            _build_extraction,
            _seed_session,
        )
        from session_manager import get_session_manager
        import mentor

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)

        for turn in fixture.turns:
            ollama_stub = _RaisingOllama()
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

        sd = get_session_manager().get_active_session_data()
        pexp = sd.product_experience_summary
        self.assertEqual(pexp.turn_count, len(fixture.turns))
        # Roundtrip via JSON
        d = pexp.to_dict()
        restored = ProductExperienceSummary.from_dict(d)
        self.assertEqual(restored.turn_count, pexp.turn_count)
        self.assertEqual(restored.startup, pexp.startup)
        self.assertEqual(restored.session, pexp.session)

    def test_observation_only_objectives_unchanged(self):
        """Adding the audit must never alter objectives, stage, lifecycle,
        or families — the Pipeline section stays intact."""
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import (
            _RaisingOllama,
            _build_extraction,
            _seed_session,
        )
        import mentor

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)

        turn = fixture.turns[-1]
        ollama_stub = _RaisingOllama()
        extraction = _build_extraction(turn)
        with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", return_value=extraction
            ):
                reply, _s, _t, diag = mentor.process_mentor_turn(
                    turn.user,
                    username=fixture.username,
                    project_name=fixture.project_name,
                )

        self.assertTrue(reply and reply.strip())
        self.assertIn("ProductExperience", diag)
        self.assertIn("Pipeline", diag)
        self.assertIn("Objective", diag["Pipeline"])
        self.assertIn("Stage", diag["Pipeline"])
        self.assertIn("Lifecycle Decision", diag["Pipeline"])

    def test_audit_failure_does_not_break_conversation(self):
        """Prove the audit is wrapped in try/except by patching
        analyze_product_experience to throw; the conversation must still
        produce a valid reply and diagnostics."""
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import (
            _RaisingOllama,
            _build_extraction,
            _seed_session,
        )
        import mentor
        from product_experience_audit import analyze_product_experience as _orig_analyze

        def _failing_analyze(**kw):
            raise RuntimeError("simulated product-experience audit failure")

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)
        turn = fixture.turns[0]
        ollama_stub = _RaisingOllama()
        extraction = _build_extraction(turn)

        with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
            with mock.patch("mentor.MemoryExtractor.extract", return_value=extraction):
                with mock.patch(
                    "product_experience_audit.analyze_product_experience",
                    side_effect=RuntimeError("boom"),
                ):
                    reply, _session, _timing, diag = mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )

        self.assertTrue(reply and reply.strip())
        # Pipeline still records decisions even though audit threw
        self.assertIn("Pipeline", diag)
        self.assertIn("Objective", diag["Pipeline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)