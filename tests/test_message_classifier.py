"""
tests/test_message_classifier.py — regression tests for the observation-only
Message-Type Shadow Classifier (``message_classifier``) and its Developer
Console wiring.

The shadow vote is INDEPENDENT of — and never overrides — Module 1's rule
extractor. Covers:
  * disabled-by-default behavior (``Status="disabled"``);
  * the OK path against a mocked zero-shot pipeline (no network, no model);
  * phrase -> MessageType-label mapping and rules-agreement computation;
  * the never-raises guarantee;
  * Developer Console projection, category, and status traffic lights;
  * live-pipeline wiring: the section appears, decision sections stay
    untouched, and an exploding classifier cannot break a turn.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import message_classifier as mc  # noqa: E402
from message_classifier import (  # noqa: E402
    CANDIDATE_LABELS,
    ClassificationResult,
    _HYPOTHESIS_STATEMENTS,
    classify_message,
    message_classification_diagnostics_section,
)


def _env(enabled_value):
    return {"CLASSIFIER_ENABLED": enabled_value}


def _fake_raw(best_label, best_score=0.91):
    """Zero-shot pipeline output: hypothesis statements ordered best-first."""
    others = [s for l, s in _HYPOTHESIS_STATEMENTS.items() if l != best_label]
    return {
        "sequence": "some text",
        "labels": [_HYPOTHESIS_STATEMENTS[best_label]] + others,
        "scores": [best_score, 0.03, 0.03, 0.03],
    }


class TestDisabledByDefault(unittest.TestCase):
    def setUp(self):
        mc._reset_pipeline_for_tests()

    def test_disabled_returns_disabled_status(self):
        with mock.patch.dict(os.environ, _env("false")):
            result = classify_message("we need this for our project")
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.label, "")
        self.assertIsNone(result.agrees_with_rules)

    def test_disabled_never_touches_the_pipeline(self):
        with mock.patch.dict(os.environ, _env("false")):
            with mock.patch(
                "message_classifier._get_pipeline",
                side_effect=AssertionError("pipeline must not load"),
            ):
                result = classify_message("hello")
        self.assertEqual(result.status, "disabled")

    def test_default_env_is_disabled(self):
        env = dict(os.environ)
        env.pop("CLASSIFIER_ENABLED", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(mc.enabled())

    def test_candidate_labels_match_module1(self):
        from memory_extractor import MessageType

        self.assertEqual(
            set(CANDIDATE_LABELS),
            {m.value for m in MessageType},
        )


class TestOkPath(unittest.TestCase):
    def setUp(self):
        mc._reset_pipeline_for_tests()

    def _run(
        self,
        phrase,
        score=0.91,
        rule_label="",
        text="Students skip their medication every day",
    ):
        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch(
                "message_classifier._get_pipeline",
                return_value=lambda _t, **_k: _fake_raw(phrase, score),
            ):
                return classify_message(text, rule_label=rule_label)

    def test_meaningful_vote(self):
        result = self._run("MEANINGFUL", rule_label="MEANINGFUL")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.label, "MEANINGFUL")
        self.assertEqual(result.rule_label, "MEANINGFUL")
        self.assertTrue(result.agrees_with_rules)
        self.assertAlmostEqual(result.score, 0.91, places=4)

    def test_no_update_phrase_maps_correctly(self):
        result = self._run("NO_UPDATE", rule_label="MEANINGFUL")
        self.assertEqual(result.label, "NO_UPDATE")
        self.assertFalse(result.agrees_with_rules)

    def test_ambiguous_phrase_maps_correctly(self):
        result = self._run("AMBIGUOUS", rule_label="AMBIGUOUS")
        self.assertEqual(result.label, "AMBIGUOUS")
        self.assertTrue(result.agrees_with_rules)

    def test_end_phrase_maps_correctly(self):
        result = self._run("END", rule_label="END")
        self.assertEqual(result.label, "END")
        self.assertTrue(result.agrees_with_rules)

    def test_no_rule_label_means_no_agreement_value(self):
        result = self._run("MEANINGFUL", rule_label="")
        self.assertIsNone(result.agrees_with_rules)
        self.assertEqual(result.to_dict()["Agrees With Rules"], "")

    def test_record_is_json_safe(self):
        result = self._run("MEANINGFUL", rule_label="MEANINGFUL")
        json.dumps(result.to_dict())

    def test_hypotheses_cover_exactly_the_four_labels(self):
        self.assertEqual(set(_HYPOTHESIS_STATEMENTS), set(CANDIDATE_LABELS))
        self.assertEqual(
            len(set(_HYPOTHESIS_STATEMENTS.values())),
            len(_HYPOTHESIS_STATEMENTS),
        )


class TestNeverRaises(unittest.TestCase):
    def setUp(self):
        mc._reset_pipeline_for_tests()

    def test_missing_pipeline_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch("message_classifier._get_pipeline", return_value=None):
                result = classify_message("some text", rule_label="MEANINGFUL")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.rule_label, "MEANINGFUL")

    def test_exploding_pipeline_is_unavailable(self):
        def _boom(*_a, **_k):
            raise RuntimeError("oom")

        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch("message_classifier._get_pipeline", return_value=_boom):
                result = classify_message("some text")
        self.assertEqual(result.status, "unavailable")
        self.assertIn("RuntimeError", result.error)

    def test_empty_input_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            result = classify_message("")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error, "empty input")

    def test_none_input_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            result = classify_message(None)
        self.assertEqual(result.status, "unavailable")


class TestDiagnosticsProjection(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(message_classification_diagnostics_section(None), {})
        self.assertEqual(message_classification_diagnostics_section({}), {})

    def test_passthrough_is_a_copy(self):
        record = ClassificationResult(
            label="MEANINGFUL",
            score=0.8,
            status="ok",
            rule_label="MEANINGFUL",
            agrees_with_rules=True,
        ).to_dict()
        section = message_classification_diagnostics_section(record)
        self.assertEqual(section, record)
        self.assertIsNot(section, record)


class TestDeveloperConsole(unittest.TestCase):
    def test_display_label(self):
        from developer_console import DIAG_SECTION_DISPLAY

        self.assertEqual(
            DIAG_SECTION_DISPLAY["MessageClassification"],
            "Message Classification",
        )

    def test_category_is_quality_audits(self):
        from developer_console import section_category

        self.assertEqual(section_category("MessageClassification"), "Quality Audits")

    def test_status_ok_agreeing_is_healthy(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status(
            "MessageClassification",
            {"Status": "ok", "Agrees With Rules": "Yes"},
        )
        self.assertEqual(status, SectionStatus.HEALTHY)
        self.assertEqual(reason, "")

    def test_status_disagreement_is_warning(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status(
            "MessageClassification",
            {"Status": "ok", "Agrees With Rules": "No"},
        )
        self.assertEqual(status, SectionStatus.WARNING)
        self.assertIn("disagrees", reason)

    def test_status_disabled_is_warning(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status("MessageClassification", {"Status": "disabled"})
        self.assertEqual(status, SectionStatus.WARNING)
        self.assertEqual(reason, "Classifier disabled")


# ---------------------------------------------------------------------------
# Live-pipeline wiring (golden stubs; no Ollama, no model download)
# ---------------------------------------------------------------------------


class TestWiringLivePipeline(unittest.TestCase):
    @staticmethod
    def _run_one_turn(fixture):
        import mentor
        from tests.goldens.runner import (
            _RaisingOllama,
            _StubOllama,
            _build_extraction,
            _seed_session,
        )

        _seed_session(fixture)
        turn = fixture.turns[0]
        ollama_stub = (
            _RaisingOllama()
            if turn.ollama_reply is None
            else _StubOllama([turn.ollama_reply])
        )
        extraction = _build_extraction(turn)
        env = {"SENTIMENT_ENABLED": "false", "CLASSIFIER_ENABLED": "false"}
        with mock.patch.dict(os.environ, env):
            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    reply, _session, _timing, diag = mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )
        return reply, diag

    def test_section_appears_when_disabled(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        reply, diagnostics = self._run_one_turn(fixture)
        self.assertTrue(reply.strip())
        self.assertEqual(diagnostics["MessageClassification"].get("Status"), "disabled")

    def test_patched_vote_reaches_diagnostics_with_rule_comparison(self):
        from tests.goldens.fixtures import load_all_fixtures
        from message_classifier import ClassificationResult

        fixture = load_all_fixtures()[0]
        canned = ClassificationResult(
            label="NO_UPDATE",
            score=0.62,
            model="mnli",
            status="ok",
            latency_ms=44.0,
            rule_label="MEANINGFUL",
            agrees_with_rules=False,
        )
        import mentor

        with mock.patch("mentor.classify_message", return_value=canned):
            reply, diagnostics = self._run_one_turn(fixture)
        self.assertTrue(reply.strip())
        section = diagnostics["MessageClassification"]
        self.assertEqual(section["Shadow Label"], "NO_UPDATE")
        self.assertEqual(section["Rules Label"], "MEANINGFUL")
        self.assertEqual(section["Agrees With Rules"], "No")

    def test_rule_label_comes_from_extraction_capture(self):
        """The live call must pass Module 1's MessageType as the rule label."""
        from tests.goldens.fixtures import load_all_fixtures
        from message_classifier import ClassificationResult
        import mentor

        fixture = load_all_fixtures()[0]
        seen = {}

        def _spy(text, rule_label=""):
            seen["rule_label"] = rule_label
            return ClassificationResult(status="ok", rule_label=rule_label)

        with mock.patch("mentor.classify_message", side_effect=_spy):
            _reply, _diag = self._run_one_turn(fixture)
        self.assertIn("rule_label", seen)
        self.assertIn(
            seen["rule_label"],
            {"MEANINGFUL", "NO_UPDATE", "AMBIGUOUS", "END"},
        )

    def test_exploding_classifier_cannot_break_the_turn(self):
        from tests.goldens.fixtures import load_all_fixtures
        import mentor

        fixture = load_all_fixtures()[0]
        with mock.patch("mentor.classify_message", side_effect=RuntimeError("boom")):
            reply, diagnostics = self._run_one_turn(fixture)
        self.assertIsNotNone(reply)
        self.assertTrue(reply.strip())
        self.assertEqual(diagnostics["MessageClassification"], {})

    def test_decision_sections_untouched(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        _reply, diagnostics = self._run_one_turn(fixture)
        self.assertEqual(diagnostics["Pipeline"]["Objective"], "PROBLEMS")
        self.assertIn("Extraction", diagnostics)


if __name__ == "__main__":
    unittest.main()
