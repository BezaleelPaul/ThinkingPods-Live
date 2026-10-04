"""
tests/test_sentiment.py — regression tests for the observation-only
Sentiment Analysis (``sentiment``) and its Developer Console wiring.

Covers:
  * disabled-by-default behavior (no model, ``Status="disabled"``) — the
    default path every existing test / golden conversation exercises;
  * the OK path against a mocked pipeline (no network, no model download);
  * the never-raises guarantee (missing pipeline, exploding pipeline,
    empty input);
  * the Developer Console projection (``sentiment_diagnostics_section``),
    display/category mapping, and section status traffic lights;
  * live-pipeline wiring: the section appears in diagnostics, decision
    sections stay untouched, and an exploding analyzer cannot break a turn.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentiment  # noqa: E402
from sentiment import (  # noqa: E402
    SentimentResult,
    analyze_sentiment,
    sentiment_diagnostics_section,
)


def _env(enabled_value):
    """Environment with the sentiment flag pinned (others untouched)."""
    return {"SENTIMENT_ENABLED": enabled_value}


class TestDisabledByDefault(unittest.TestCase):
    def setUp(self):
        sentiment._reset_pipeline_for_tests()

    def test_disabled_returns_disabled_status(self):
        with mock.patch.dict(os.environ, _env("false")):
            result = analyze_sentiment("I hate this app")
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.label, "")
        self.assertEqual(result.to_dict()["Status"], "disabled")

    def test_disabled_never_touches_the_pipeline(self):
        with mock.patch.dict(os.environ, _env("false")):
            with mock.patch(
                "sentiment._get_pipeline",
                side_effect=AssertionError("pipeline must not load"),
            ):
                result = analyze_sentiment("hello")
        self.assertEqual(result.status, "disabled")

    def test_default_env_is_disabled(self):
        env = dict(os.environ)
        env.pop("SENTIMENT_ENABLED", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(sentiment.enabled())


class TestOkPath(unittest.TestCase):
    def setUp(self):
        sentiment._reset_pipeline_for_tests()

    def _run(self, raw, text="My patients keep missing doses and it frustrates me"):
        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch("sentiment._get_pipeline", return_value=lambda _t: raw):
                return analyze_sentiment(text)

    def test_negative_sentiment(self):
        result = self._run([[{"label": "negative", "score": 0.9734}]])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.label, "NEGATIVE")
        self.assertAlmostEqual(result.score, 0.9734, places=4)
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertEqual(result.model, sentiment.DEFAULT_MODEL)

    def test_positive_sentiment(self):
        result = self._run([[{"label": "positive", "score": 0.88}]])
        self.assertEqual(result.label, "POSITIVE")

    def test_unknown_label_becomes_neutral(self):
        result = self._run([[{"label": "LABEL_1", "score": 0.5}]])
        self.assertEqual(result.label, "NEUTRAL")

    def test_record_is_json_safe(self):
        result = self._run([[{"label": "negative", "score": 0.9}]])
        json.dumps(result.to_dict())  # must not raise


class TestNeverRaises(unittest.TestCase):
    def setUp(self):
        sentiment._reset_pipeline_for_tests()

    def test_missing_pipeline_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch("sentiment._get_pipeline", return_value=None):
                result = analyze_sentiment("some text")
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(result.error)

    def test_exploding_pipeline_is_unavailable(self):
        def _boom(_text):
            raise RuntimeError("cuda exploded")

        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch("sentiment._get_pipeline", return_value=_boom):
                result = analyze_sentiment("some text")
        self.assertEqual(result.status, "unavailable")
        self.assertIn("RuntimeError", result.error)

    def test_empty_input_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            result = analyze_sentiment("   ")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error, "empty input")

    def test_none_input_is_unavailable(self):
        with mock.patch.dict(os.environ, _env("true")):
            result = analyze_sentiment(None)
        self.assertEqual(result.status, "unavailable")

    def test_analyze_never_raises_with_broken_get_pipeline(self):
        # _get_pipeline itself swallows import errors and returns None — but
        # if the loader is broken anyway, analyze_sentiment must still catch it.
        with mock.patch.dict(os.environ, _env("true")):
            with mock.patch(
                "sentiment._get_pipeline",
                side_effect=RuntimeError("loader broke"),
            ):
                result = analyze_sentiment("text")
        self.assertEqual(result.status, "unavailable")


class TestDiagnosticsProjection(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(sentiment_diagnostics_section(None), {})
        self.assertEqual(sentiment_diagnostics_section({}), {})

    def test_passthrough_is_a_copy(self):
        record = SentimentResult(
            label="NEGATIVE", score=0.9, model="m", status="ok"
        ).to_dict()
        section = sentiment_diagnostics_section(record)
        self.assertEqual(section, record)
        self.assertIsNot(section, record)


class TestDeveloperConsole(unittest.TestCase):
    def test_display_label(self):
        from developer_console import DIAG_SECTION_DISPLAY

        self.assertEqual(
            DIAG_SECTION_DISPLAY["SentimentAnalysis"], "Sentiment Analysis"
        )

    def test_category_is_quality_audits(self):
        from developer_console import section_category

        self.assertEqual(section_category("SentimentAnalysis"), "Quality Audits")

    def test_status_ok_is_healthy(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status(
            "SentimentAnalysis", {"Status": "ok", "Label": "POSITIVE"}
        )
        self.assertEqual(status, SectionStatus.HEALTHY)
        self.assertEqual(reason, "")

    def test_status_disabled_is_warning(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status("SentimentAnalysis", {"Status": "disabled"})
        self.assertEqual(status, SectionStatus.WARNING)
        self.assertEqual(reason, "Classifier disabled")

    def test_status_unavailable_is_warning(self):
        from developer_console import SectionStatus, section_status

        status, reason = section_status(
            "SentimentAnalysis", {"Status": "unavailable", "Error": "x"}
        )
        self.assertEqual(status, SectionStatus.WARNING)
        self.assertEqual(reason, "Classifier unavailable")


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

    def test_sections_appear_when_disabled(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        reply, diagnostics = self._run_one_turn(fixture)
        self.assertTrue(reply.strip())
        self.assertEqual(diagnostics["SentimentAnalysis"].get("Status"), "disabled")
        self.assertEqual(diagnostics["MessageClassification"].get("Status"), "disabled")

    def test_patched_result_reaches_diagnostics(self):
        from tests.goldens.fixtures import load_all_fixtures
        from sentiment import SentimentResult

        fixture = load_all_fixtures()[0]
        canned = SentimentResult(
            label="POSITIVE", score=0.91, model="sst2", status="ok", latency_ms=12.0
        )
        import mentor

        with mock.patch("mentor.analyze_sentiment", return_value=canned):
            reply, diagnostics = self._run_one_turn(fixture)
        self.assertTrue(reply.strip())
        self.assertEqual(diagnostics["SentimentAnalysis"]["Label"], "POSITIVE")
        self.assertEqual(diagnostics["SentimentAnalysis"]["Status"], "ok")

    def test_exploding_analyzer_cannot_break_the_turn(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        import mentor

        with mock.patch("mentor.analyze_sentiment", side_effect=RuntimeError("boom")):
            reply, diagnostics = self._run_one_turn(fixture)
        self.assertIsNotNone(reply)
        self.assertTrue(reply.strip())
        # capture never got the record -> the section degrades to {}
        self.assertEqual(diagnostics["SentimentAnalysis"], {})

    def test_decision_sections_untouched(self):
        from tests.goldens.fixtures import load_all_fixtures

        fixture = load_all_fixtures()[0]
        _reply, diagnostics = self._run_one_turn(fixture)
        # The golden's first-turn objective must be exactly what the golden
        # runner asserts — the classifier never feeds a decision path.
        self.assertEqual(diagnostics["Pipeline"]["Objective"], "PROBLEMS")
        self.assertIn("ProjectState", diagnostics)
        self.assertIn("Extraction", diagnostics)


if __name__ == "__main__":
    unittest.main()
