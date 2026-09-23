"""
tests/test_turn_profiler.py — unit tests for the Phase 1.5 latency audit
instrumentation (``timing.TurnProfiler`` + ``timed_ollama_chat``).

All tests are deterministic (stubbed ollama, no network). The
instrumentation is observation-only by contract: identical calls with and
without a profiler must behave identically.
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timing import TurnProfiler, add_stage_ms, timed_ollama_chat  # noqa: E402


class _DictOllama:
    """Stub returning a dict-style response with server metadata."""

    def __init__(self, calls=None, fail=False):
        self.calls = calls if calls is not None else []
        self.fail = fail

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("ollama down")
        return {
            "message": {"content": "hello there"},
            "prompt_eval_count": 35,
            "eval_count": 3,
            "total_duration": 13_000_000_000,
            "load_duration": 11_000_000_000,
            "prompt_eval_duration": 1_200_000_000,
            "eval_duration": 200_000_000,
        }


class TestTurnProfiler(unittest.TestCase):
    def test_empty_profile(self):
        d = TurnProfiler().to_dict()
        self.assertEqual(d["stages"], {})
        self.assertEqual(d["stage_order"], [])
        self.assertEqual(d["llm_calls"], [])

    def test_spans_ordered_and_summed(self):
        p = TurnProfiler()
        with p.span("prepare"):
            time.sleep(0.01)
        with p.span("objective"):
            pass
        with p.span("prepare"):
            pass
        d = p.to_dict()
        self.assertEqual(d["stage_order"],
                         ["prepare", "objective", "prepare"])
        self.assertGreaterEqual(d["stages"]["prepare"], 5.0)
        self.assertGreaterEqual(p.span_ms("objective"), 0.0)
        self.assertEqual(p.span_ms("missing"), 0.0)

    def test_span_records_on_exception(self):
        p = TurnProfiler()
        with self.assertRaises(ValueError):
            with p.span("risky"):
                raise ValueError("boom")
        self.assertEqual(p.to_dict()["stage_order"], ["risky"])

    def test_record_llm_call(self):
        p = TurnProfiler()
        rec = p.record_llm_call(
            purpose="response_generation", model="m", prompt_chars=100,
            duration_ms=12.5, prompt_tokens=30, eval_count=5,
            server_ms={"eval_duration": 3.0}, gated=False, failed=False,
        )
        self.assertEqual(rec, p.to_dict()["llm_calls"][0])
        self.assertFalse(rec["failed"])


class TestTimedOllamaChat(unittest.TestCase):
    def test_records_metadata_and_returns_response(self):
        stub = _DictOllama()
        p = TurnProfiler()
        with mock.patch.dict("sys.modules", {"ollama": stub}):
            resp = timed_ollama_chat(
                p, purpose="response_generation", model="qwen2.5:3b",
                messages=[{"role": "user", "content": "hi"}],
                options={"temperature": 0.4},
            )
        self.assertEqual(resp["message"]["content"], "hello there")
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["model"], "qwen2.5:3b")
        self.assertEqual(stub.calls[0]["options"], {"temperature": 0.4})
        (rec,) = p.to_dict()["llm_calls"]
        self.assertEqual(rec["purpose"], "response_generation")
        self.assertEqual(rec["prompt_chars"], 2)
        self.assertEqual(rec["prompt_tokens"], 35)
        self.assertEqual(rec["eval_count"], 3)
        self.assertEqual(rec["server_ms"]["load_duration"], 11000.0)
        self.assertEqual(rec["server_ms"]["eval_duration"], 200.0)
        self.assertFalse(rec["gated"])
        self.assertFalse(rec["failed"])

    def test_none_profiler_is_passthrough(self):
        stub = _DictOllama()
        with mock.patch.dict("sys.modules", {"ollama": stub}):
            resp = timed_ollama_chat(
                None, purpose="x", model="m",
                messages=[{"role": "user", "content": "hi"}],
                options=None,
            )
        self.assertEqual(resp["message"]["content"], "hello there")
        self.assertEqual(len(stub.calls), 1)
        self.assertNotIn("options", stub.calls[0])

    def test_failure_recorded_and_reraised(self):
        stub = _DictOllama(fail=True)
        p = TurnProfiler()
        with mock.patch.dict("sys.modules", {"ollama": stub}):
            with self.assertRaises(RuntimeError):
                timed_ollama_chat(
                    p, purpose="extraction", model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    options={}, gated=True,
                )
        (rec,) = p.to_dict()["llm_calls"]
        self.assertTrue(rec["failed"])
        self.assertTrue(rec["gated"])
        self.assertIsNone(rec["prompt_tokens"])

    def test_object_style_response(self):
        class ObjResp:
            prompt_eval_count = 10
            eval_count = 2

            def __getitem__(self, key):
                if key == "message":
                    return {"content": "ok"}
                raise KeyError(key)

        class ObjOllama:
            def chat(self, **kwargs):
                return ObjResp()

        p = TurnProfiler()
        with mock.patch.dict("sys.modules", {"ollama": ObjOllama()}):
            resp = timed_ollama_chat(
                p, purpose="x", model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertEqual(resp["message"]["content"], "ok")
        (rec,) = p.to_dict()["llm_calls"]
        self.assertEqual(rec["prompt_tokens"], 10)
        self.assertEqual(rec["eval_count"], 2)


class TestTurnProfileExposure(unittest.TestCase):
    def test_process_mentor_turn_exposes_profile(self):
        import mentor
        from session_manager import get_session_manager
        from tests.goldens.fixtures import GoldenConversationFixture, GoldenTurn
        from tests.goldens.runner import _RaisingOllama, _seed_session

        _seed_session(GoldenConversationFixture(
            name="prof", scenario=0, description="prof",
            username="prof_user", project_name="ProfProject",
            turns=[GoldenTurn(user="seed")],
        ))
        get_session_manager().get_active_session_data().conversation_history = []
        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            _reply, _session, timing, _diag = mentor.process_mentor_turn(
                "College students miss deadlines.",
                username="prof_user", project_name="ProfProject")
        profile = getattr(timing, "profile", None)
        self.assertIsNotNone(profile)
        stages = profile["stages"]
        for expected in ("prepare", "context_gate", "extraction", "objective",
                         "recovery", "family", "lifecycle", "memory",
                         "generate_reply", "finalize", "record_turn"):
            self.assertIn(expected, stages, expected)
        # to_dict shape unchanged (profile rides outside it).
        d = timing.to_dict()
        self.assertNotIn("profile", d)
        self.assertIn("total_ms", d)

    def test_accumulator_compat(self):
        # TurnProfiler is NOT the _timing accumulator; plain dicts still work.
        acc: dict = {}
        add_stage_ms(acc, "rules", 0.0, 0.01)
        self.assertGreater(acc["rules"], 0)


if __name__ == "__main__":
    unittest.main()
