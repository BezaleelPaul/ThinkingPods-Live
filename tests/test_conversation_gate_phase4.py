"""
tests/test_conversation_gate_phase4.py — Phase 4 replay-evaluation baseline.

FREEZES the evaluated behaviour of the Conversational Context Gate so any
future change surfaces against this documented baseline. The numbers below
are produced by ``python phase4_replay.py`` (evaluation-only harness).

PHASE 5 BASELINE UPDATE (intentional changes, explained):
  * NORMAL_DT CONTROL detection 6/6 — the third-person "don't know"
    false positive ("people don't know where to start") is fixed by
    person-scoping the DONT_KNOW lexicon.
  * CORRECTION 6/6 pauses and TOPIC_SHIFT 7/7 pauses — the Phase 4
    false negatives (meta-correction frames, changed-the-project,
    let's-switch-topics, talk-about-something-else) are detected and
    paused at HIGH confidence.
  * CONFUSED pause total 8/8 — "I don't get what you mean" and the
    context-dependent "Can you explain that?" now pause.
  * Pause-turn pivot-language ingestion 0 — paused HYPOTHETICAL /
    TOPIC_SHIFT / CORRECTION turns strip clause-initial pivot heads and
    suppress extraction ONLY when no substantive DT content remains.
  * Hard false positives shrink to two zero-information cases (messages
    with an explicit hypothetical marker but no extractor-reachable
    content, so suppression loses nothing). The substantive
    "Teachers suppose that students review notes nightly." moved to the
    intentional paused-content-preserved bucket.

If one of these assertions fails after an intentional rule change, update
the baseline here AND the replay report together.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phase4_replay  # noqa: E402

# Evaluate once for the whole class (deterministic, stubbed pipeline).
_METRICS = phase4_replay.evaluate(report_path=None)

_KNOWN_HARD_FPS = {
    # Explicit hypothetical marker + NO extractor-reachable content:
    # pausing loses nothing; label-level limitation documented in report.
    ("Imagine badges as rewards in the classroom economy.",
     "HYPOTHETICAL", "HIGH"),
    ("Suppose it rains - attendance drops, that's all.",
     "HYPOTHETICAL", "HIGH"),
}
_KNOWN_PRESERVED = {
    # Substantive hypotheticals: paused conversationally, extraction kept.
    ("Teachers suppose that students review notes nightly.",
     "HYPOTHETICAL", "HIGH"),
}
_KNOWN_SOFT_FLAGS: set = set()

_EXPECTED_CATEGORY_TABLE = {
    # category: (det_ok, det_total, pause_ok, pause_total, contam, pollute)
    "CORRECTION": (6, 6, 6, 6, 0, 0),
    "TOPIC_SHIFT": (7, 7, 7, 7, 0, 0),
    "DIRECT_QUESTION": (3, 3, 3, 3, 0, 0),
    "CONFUSED": (9, 9, 8, 8, 0, 0),
    "HYPOTHETICAL": (5, 5, 5, 5, 0, 0),
    "AMBIGUOUS": (5, 5, 5, 5, 0, 0),
    "NORMAL_DT CONTROL": (6, 6, 6, 6, 0, 0),
}


class TestPhase4Baseline(unittest.TestCase):
    def test_category_table_unchanged(self):
        for cat, expected in _EXPECTED_CATEGORY_TABLE.items():
            agg = _METRICS["categories"][cat]
            actual = (
                agg["detection_ok"], agg["detection_total"],
                agg["pause_ok"], agg["pause_total"],
                agg["contamination"], agg["family_pollution"],
            )
            self.assertEqual(actual, expected, cat)

    def test_hard_false_positives_exactly_the_known_set(self):
        got = {(m, mode, conf) for m, mode, conf in _METRICS["hard_false_positives"]}
        self.assertEqual(got, _KNOWN_HARD_FPS)

    def test_substantive_hypotheticals_pause_without_information_loss(self):
        got = {(m, mode, conf)
               for m, mode, conf in _METRICS["paused_content_preserved"]}
        self.assertEqual(got, _KNOWN_PRESERVED)

    def test_soft_flags_exactly_the_known_set(self):
        got = {(m, mode, conf) for m, mode, conf in _METRICS["soft_flags"]}
        self.assertEqual(got, _KNOWN_SOFT_FLAGS)

    def test_cross_turn_resumption_all_pass(self):
        self.assertEqual(
            (_METRICS["resumption_pass"], _METRICS["resumption_total"]), (5, 5))
        for r in _METRICS["resumptions"]:
            self.assertTrue(r["ok"], r["name"])
            self.assertTrue(r["resumed_canonically"], r["name"])

    def test_no_pivot_language_ingestion_anywhere(self):
        self.assertEqual(_METRICS["pause_turn_ingestion"], 0)
        ingested = [r["name"] for r in _METRICS["resumptions"]
                    if r.get("ingested_at_pause")]
        self.assertEqual(ingested, [])

    def test_safety_matrix_perfect_and_traceless(self):
        self.assertEqual(_METRICS["safety_accuracy"], 1.0)
        self.assertTrue(_METRICS["safety_no_trace"])
        self.assertEqual(_METRICS["safety_misses"], [])
        self.assertFalse(any(r["intercepted"] and not r["should"]
                             for r in _METRICS["safety_rows"]))

    def test_zero_llm_overhead(self):
        self.assertEqual(_METRICS["extra_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
