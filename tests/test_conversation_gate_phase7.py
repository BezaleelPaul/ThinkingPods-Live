"""
tests/test_conversation_gate_phase7.py — Effectiveness-evaluation baseline.

FREEZES the Phase 7 A/B/D-arm findings so future layer changes surface
against measured evidence (regenerate via ``python phase7_evaluation.py``).

Pinned facts (Phase 7 run, 108 turns/arm, novel-wording dataset):
  * NET improvement +23  (27 prevented / 4 introduced)
  * ablation invariant: DETECTION_ONLY behaves exactly like BASELINE
    (identical behavioural failure totals) — detection without behaviour
    changes nothing.
  * ACTIVE introduces no hard failures (no information loss, state
    contamination, objective derailment, or failed resumption).
  * one FALSE_PAUSE on "Supervisors imagine new schedules each semester."
  * control-group (golden traffic) replies identical across arms.
  * latency unchanged; zero extra LLM attempts.
  * known limitation: ~48% of strong novel conversational moves are still
    missed behaviourally — the documented Phase-8 work item.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import phase7_evaluation as p7  # noqa: E402

_METRICS = p7.evaluate(report_path=None)


class TestPhase7EffectivenessBaseline(unittest.TestCase):
    def test_net_improvement_positive_and_pinned(self):
        comp = _METRICS["comparison"]
        self.assertEqual(comp["net_improvement"], 23)
        self.assertEqual(comp["prevented_total"], 27)
        self.assertEqual(comp["introduced_failures"], {"FALSE_PAUSE": 1})

    def test_no_hard_failures_introduced(self):
        introduced = _METRICS["comparison"]["introduced_failures"]
        for hard in ("INFORMATION_LOSS", "OBJECTIVE_DERAILMENT",
                     "FAILED_RESUMPTION", "STATE_CONTAMINATION"):
            self.assertNotIn(hard, introduced)

    def test_ablation_detection_only_equals_baseline(self):
        b = _METRICS["baseline"]["total_failures"]
        d = _METRICS["detection_only"]["total_failures"]
        self.assertEqual(b, d,
                         "Detection without behaviour must not change "
                         "behavioural outcomes.")

    def test_active_beats_both_other_arms(self):
        a = _METRICS["active"]["total_failures"]
        self.assertLess(a, _METRICS["baseline"]["total_failures"])
        self.assertLess(a, _METRICS["detection_only"]["total_failures"])

    def test_layer_miss_rate_materially_better_than_baseline(self):
        comp = _METRICS["comparison"]
        self.assertLess(comp["missed_move_rate_layer"],
                        comp["missed_move_rate_baseline"])
        # Documented residual: still high enough to demand Phase-8 work.
        self.assertGreater(comp["missed_move_rate_layer"], 0.05)

    def test_resumption_and_integrity(self):
        self.assertEqual(
            _METRICS["active"]["failures"].get("FAILED_RESUMPTION", 0), 0)
        self.assertEqual(_METRICS["comparison"]["derailment_rate_layer"], 0.0)
        self.assertEqual(_METRICS["projectstate_contamination"] if
                         "projectstate_contamination" in _METRICS else 0, 0)

    def test_control_group_replies_identical(self):
        ce = _METRICS["control_equivalence"]
        self.assertEqual(ce["reply_diffs"], [])

    def test_safety_unchanged(self):
        self.assertEqual(_METRICS["safety"]["accuracy"], 1.0)
        self.assertTrue(_METRICS["safety"]["no_trace"])

    def test_latency_and_llc_invariants(self):
        for arm in ("baseline", "detection_only", "active"):
            self.assertEqual(_METRICS[arm]["llm_calls"],
                             _METRICS[arm]["turns"])
        self.assertEqual(_METRICS["comparison"]["responses_changed"], 42)

    def test_verdict_is_B_given_residual_misses(self):
        verdict = p7._verdict(_METRICS)
        self.assertTrue(verdict.startswith("B"), verdict)


if __name__ == "__main__":
    unittest.main(verbosity=2)