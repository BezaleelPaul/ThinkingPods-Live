"""
tests/test_prompt_effectiveness.py — unit + integration tests for the
Prompt Effectiveness Audit (observation-only developer tool).
"""

import json
import os
import sys
import tempfile
import unittest
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prompt_effectiveness as pe  # noqa: E402
from tests.goldens.fixtures import load_all_fixtures  # noqa: E402


def _small_fixtures():
    return load_all_fixtures()[:1]


def _build_sample_prompt(**extra):
    """A real five-section prompt with all guidance sections present."""
    from module4.prompt_builder import build_prompt
    from module3 import ObjectiveEngine
    from module4 import ResponseStrategy
    from memory_extractor import ProjectState

    state = ProjectState()
    obj = ObjectiveEngine().determine_next(state)
    kwargs = {
        "project_state": state,
        "conversation_objective": obj,
        "response_strategy": ResponseStrategy.ASK_QUESTION,
        "coaching_bullet": [
            "Current coaching strategy: EXPLORE\nGuidance: begin openly"
        ],
        "insight_bullet": [
            "Insight detected: STRONG_EVIDENCE [HIGH confidence]\n"
            "Guidance: validate"
        ],
        "move_bullet": [
            "Conversation move: VALIDATE_DISCOVERY\nGuidance: acknowledge"
        ],
        "resume_hint": [
            "The user has an unfinished topic on Problems: x. "
            "Do not restart it from scratch - build on what they already shared."
        ],
    }
    kwargs.update(extra)
    return build_prompt(**kwargs)


@lru_cache(maxsize=1)
def _audit_results():
    return pe.audit_prompt_effectiveness(fixtures=_small_fixtures())


# ---------------------------------------------------------------------------
# Unit tests — prompt parsing
# ---------------------------------------------------------------------------


class TestParsePrompt(unittest.TestCase):
    def test_extracts_all_guidance_signals(self):
        sig = pe._parse_prompt(_build_sample_prompt())
        self.assertEqual(sig["objective"], "PERSONAS")
        self.assertEqual(sig["coaching"], "EXPLORE")
        self.assertEqual(sig["insight"], "STRONG_EVIDENCE")
        self.assertEqual(sig["move"], "VALIDATE_DISCOVERY")
        self.assertEqual(sig["memory"], 1)
        self.assertGreaterEqual(sig["instructions"], 1)
        self.assertEqual(sig["transition"], "no")

    def test_extracts_question_family(self):
        sig = pe._parse_prompt(
            _build_sample_prompt(
                question_family="PROBLEMS_CORE",
                previously_asked_families=["FREQUENCY_ESTIMATE"],
            )
        )
        self.assertIn("family", sig)

    def test_transition_signal_from_transition_move(self):
        sig = pe._parse_prompt(
            _build_sample_prompt(
                move_bullet=[
                    "Conversation move: TRANSITION_TOPIC\nGuidance: bridge"
                ]
            )
        )
        self.assertEqual(sig["transition"], "yes")

    def test_transition_signal_from_transition_coaching(self):
        sig = pe._parse_prompt(
            _build_sample_prompt(
                coaching_bullet=[
                    "Current coaching strategy: TRANSITION\nGuidance: bridge"
                ]
            )
        )
        self.assertEqual(sig["transition"], "yes")


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip(self):
        sig = pe._parse_prompt(_build_sample_prompt())
        reply = pe._encode_reply(sig)
        decoded = pe._decode_reply(reply)
        for key in pe._SIGNALS:
            expected = str(sig.get(key)) if sig.get(key) is not None else None
            self.assertEqual(decoded.get(key), expected, key)

    def test_empty_signal_dict(self):
        reply = pe._encode_reply({})
        self.assertIn("no guidance signals", reply)
        self.assertEqual(pe._decode_reply(reply), {})

    def test_reply_passes_enforcement_shape(self):
        # One question, no advice markers, short — shape that survives the
        # mentor's reply-enforcement rules.
        reply = pe._encode_reply(pe._parse_prompt(_build_sample_prompt()))
        self.assertEqual(reply.count("?"), 1)
        self.assertLess(len(reply.split()), 55)
        for marker in ("you should", "i suggest", "the solution is"):
            self.assertNotIn(marker, reply.lower())


# ---------------------------------------------------------------------------
# Unit tests — prompt surgery
# ---------------------------------------------------------------------------


class TestRemoveSection(unittest.TestCase):
    def setUp(self):
        self.prompt = _build_sample_prompt()

    def test_coaching_removal_only_removes_coaching(self):
        removed = pe._remove_section(self.prompt, "coaching_strategy")
        self.assertNotIn("Current coaching strategy:", removed)
        self.assertIn("Conversation move:", removed)
        self.assertIn("Insight detected:", removed)
        self.assertIn("unfinished topic", removed)

    def test_move_removal_only_removes_move(self):
        removed = pe._remove_section(self.prompt, "conversation_move")
        self.assertNotIn("Conversation move:", removed)
        self.assertIn("Current coaching strategy:", removed)
        self.assertIn("Insight detected:", removed)

    def test_insight_removal_only_removes_insight(self):
        removed = pe._remove_section(self.prompt, "insight_guidance")
        self.assertNotIn("Insight detected:", removed)
        self.assertIn("Current coaching strategy:", removed)
        self.assertIn("Conversation move:", removed)

    def test_memory_removal_only_removes_resume_hints(self):
        removed = pe._remove_section(self.prompt, "conversation_memory")
        self.assertNotIn("unfinished topic", removed)
        self.assertIn("Current coaching strategy:", removed)

    def test_base_removal_keeps_guidance_but_drops_strategy_bullets(self):
        removed = pe._remove_section(self.prompt, "base_instructions")
        self.assertNotIn(
            "Acknowledge what the user just shared", removed
        )
        self.assertIn("Current coaching strategy:", removed)
        self.assertIn("Conversation move:", removed)

    def test_structural_sections_survive(self):
        for sid in pe._SECTION_ORDER:
            removed = pe._remove_section(self.prompt, sid)
            for header in ("Role", "Current Objective", "Known Project State",
                           "Latest Conversation", "Instructions"):
                self.assertIn(header, removed, sid)


# ---------------------------------------------------------------------------
# Unit tests — token / latency estimates
# ---------------------------------------------------------------------------


class TestEstimates(unittest.TestCase):
    def test_token_count_word_proxy(self):
        self.assertEqual(pe._token_count("a b c"), 3)
        self.assertEqual(pe._token_count(""), 1)

    def test_latency_increases_with_tokens(self):
        low = pe._estimate_latency_ms(100, 10)
        high = pe._estimate_latency_ms(500, 50)
        self.assertLess(low, high)


# ---------------------------------------------------------------------------
# Unit tests — section registry
# ---------------------------------------------------------------------------


class TestSectionRegistry(unittest.TestCase):
    def test_all_spec_sections_present_in_order(self):
        ids = [m["id"] for m in pe._SECTION_META]
        self.assertEqual(
            ids,
            [
                "base_instructions",
                "conversation_memory",
                "question_family",
                "coaching_strategy",
                "insight_guidance",
                "conversation_move",
            ],
        )

    def test_section_label_maps(self):
        self.assertEqual(pe.section_label("coaching_strategy"),
                         "Coaching strategy")


# ---------------------------------------------------------------------------
# Integration — audit
# ---------------------------------------------------------------------------


class TestAuditIntegration(unittest.TestCase):
    def test_audit_shape(self):
        res = _audit_results()
        self.assertEqual(res["turn_count"], len(_small_fixtures()[0].turns))
        self.assertEqual(len(res["sections"]), 6)
        self.assertEqual(len(res["ranking"]), 6)

    def test_baseline_zero_changes(self):
        base = _audit_results()["baseline"]
        self.assertEqual(base["changed_reply_pct"], 0.0)
        self.assertEqual(base["changed_objective_pct"], 0.0)

    def test_ranking_sorted_by_reply_influence(self):
        res = _audit_results()
        pcts = [r["changed_reply_pct"] for r in res["sections"]]
        by_label = {r["label"]: r["changed_reply_pct"] for r in res["sections"]}
        ranked = [by_label[l] for l in res["ranking"]]
        self.assertEqual(ranked, sorted(ranked, reverse=True))

    def test_move_section_changes_move_signal(self):
        # Phase 1 intentional change: coaching strategy no longer exists as
        # a separable prompt bullet (it shapes the allowed-move set), so
        # the ablation-sensitive signal is now the conversational move.
        res = _audit_results()
        row = next(r for r in res["sections"] if r["id"] == "conversation_move")
        self.assertGreater(row["changed_reply_pct"], 0.0)
        self.assertGreater(row["changed_move_pct"], 0.0)

    def test_objective_never_changes(self):
        res = _audit_results()
        for row in res["sections"]:
            self.assertEqual(row["changed_objective_pct"], 0.0, row["id"])

    def test_ablating_sections_does_not_alter_pipeline_decisions(self):
        # The audit is observation-only: removing a guidance section never
        # changes the deterministic pre-prompt decisions.  Compare per-turn
        # objective / transition / question-family across full and ablated
        # captures.
        fixtures = _small_fixtures()
        full = pe.capture_full(fixtures, label="full")
        abl = pe.capture_ablation(fixtures, section="coaching_strategy")
        for fa, fb in zip(full["fixtures"], abl["fixtures"]):
            for ta, tb in zip(fa["turns"], fb["turns"]):
                self.assertEqual(ta["objective"], tb["objective"])
                self.assertEqual(ta["transition_type"], tb["transition_type"])
                self.assertEqual(ta["question_family"], tb["question_family"])

    def test_prompt_tokens_reduce_when_section_ablated(self):
        res = _audit_results()
        base_tokens = res["baseline"]["avg_prompt_tokens"]
        for row in res["sections"]:
            self.assertLessEqual(row["avg_prompt_tokens"], base_tokens)


# ---------------------------------------------------------------------------
# Integration — reports & developer console section
# ---------------------------------------------------------------------------


class TestReports(unittest.TestCase):
    def test_markdown_report_contains_columns_and_sections(self):
        text = pe.generate_markdown_report(_audit_results())
        for col in ("Changed Reply (%)", "Changed Question (%)",
                    "Changed Transition (%)", "Changed Objective (%)",
                    "Average Prompt Tokens", "Estimated Latency (ms)"):
            self.assertIn(col, text)
        for row in _audit_results()["sections"]:
            self.assertIn(row["label"], text)

    def test_html_report_contains_table_and_ranking(self):
        text = pe.generate_html_report(_audit_results())
        self.assertIn("<table>", text)
        self.assertIn("<ol>", text)
        self.assertIn("Prompt Effectiveness Audit", text)

    def test_developer_console_section(self):
        res = _audit_results()
        section = pe.prompt_effectiveness_section(res)
        self.assertIn("Ranking", section)
        self.assertIn(res["baseline"]["label"], section)
        self.assertIn(res["sections"][0]["label"], section)
        blk = section[res["sections"][0]["label"]]
        self.assertIn("Changed Reply", blk)
        self.assertIn("Estimated Latency (ms)", blk)


# ---------------------------------------------------------------------------
# Integration — replay-lab compatibility & CLI
# ---------------------------------------------------------------------------


class TestCaptureCompatibility(unittest.TestCase):
    def test_capture_full_shape(self):
        full = pe.capture_full(_small_fixtures(), label="full")
        self.assertEqual(full["label"], "full")
        fixture = full["fixtures"][0]
        self.assertIn("turns", fixture)
        self.assertIn("summary", fixture)
        self.assertIn("turn_count", fixture["summary"])
        self.assertNotIn("_signals", fixture["turns"][0])

    def test_capture_ablation_shape(self):
        abl = pe.capture_ablation(_small_fixtures(), section="conversation_move")
        self.assertEqual(abl["label"], "without_conversation_move")

    def test_compare_full_vs_ablation(self):
        fixtures = _small_fixtures()
        full = pe.capture_full(fixtures, label="full")
        abl = pe.capture_ablation(fixtures, section="coaching_strategy")
        result = pe.compare_full_vs_ablation(full, abl)
        self.assertIn("comparison", result)
        self.assertEqual(result["ablated_section"], "without_coaching_strategy")
        comp = result["comparison"]
        self.assertEqual(comp["total_turns_baseline"],
                         comp["total_turns_candidate"])

    def test_cli_run_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "effectiveness.md")
            from prompt_effectiveness import main
            rc = main(["run", "--sections", "coaching_strategy",
                       "-o", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Coaching strategy", text)


if __name__ == "__main__":
    unittest.main()
