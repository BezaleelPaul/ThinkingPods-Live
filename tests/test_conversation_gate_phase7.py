"""
tests/test_conversation_gate_phase7.py — Effectiveness-evaluation baseline.

FREEZES the Phase 7 A/B/D-arm findings so future layer changes surface
against measured evidence (regenerate via
``python audits/phase7_evaluation.py --report docs/reports/Phase7_Effectiveness_Report.md``).

Pinned facts (Phase 7 run, 108 turns/arm, novel-wording dataset):
  * NET improvement +24  (25 prevented / 1 introduced)
  * ablation invariant: DETECTION_ONLY behaves exactly like BASELINE
    (identical behavioural failure totals) — detection without behaviour
    changes nothing.
  * ACTIVE introduces no hard failures (no information loss, state
    contamination, objective derailment, or failed resumption).
  * FAILED_RESUMPTION (layer) = 0. Resumption is measured as REGRESSION —
    fields lost since the pre-pause state, or missing what the non-paused
    BASELINE arm already holds (audits/phase7_evaluation.py::
    _apply_resumption). It is deliberately NOT "the objective label must
    change each turn": that label stays on the same objective across turns
    in every arm, and the old rule flagged 13 turns where nothing was lost.
  * one FALSE_PAUSE on "Supervisors imagine new schedules each semester."
  * control-group (golden traffic) replies identical across arms.
  * latency unchanged; zero extra LLM attempts.
  * known limitation: 5 of 33 strong novel conversational moves are still
    missed behaviourally (missed_move_rate_layer 0.1515 vs 0.9091
    baseline) — the documented Phase-8 work item.
  * verdict "A" (previously pinned "B" purely because of the broken
    resumption metric — behaviour did not change, only the measurement).
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
        self.assertEqual(comp["net_improvement"], 24)
        self.assertEqual(comp["prevented_total"], 25)
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

    def test_verdict_is_A_with_low_residual_miss_rate(self):
        verdict = p7._verdict(_METRICS)
        self.assertTrue(verdict.startswith("A"), verdict)


class _Anno:
    """Minimal stand-in for the turn annotation _rubric_scores reads."""

    move = "normal"
    strength = "strong"


def _make_record(turn, state, objective="PROBLEMS", check=False):
    return {
        "turn": turn,
        "objective": objective,
        "state_after": state,
        "mode": "NORMAL_DT",
        "pause": False,
        "ack": False,
        "reply_family": None,
        "reply": "ok",
        "_anno": _Anno(),
        "_failures": set(),
        "_scores": {},
        "_resumption_check": check,
    }


class TestResumptionMetricSemantics(unittest.TestCase):
    """FAILED_RESUMPTION means the pause lost ground, not label churn."""

    def _apply(self, records, baseline=None):
        p7._apply_resumption(records, baseline=baseline)
        return {r["turn"]: r["_failures"] for r in records}

    def test_unchanged_objective_label_is_not_a_failure(self):
        state = {"problems": ["late refills"]}
        records = [
            _make_record(1, state),
            _make_record(2, dict(state)),
            _make_record(3, dict(state), check=True),
        ]
        self.assertNotIn("FAILED_RESUMPTION", self._apply(records)[3])

    def test_fields_lost_since_pre_pause_are_flagged(self):
        records = [
            _make_record(1, {"problems": ["late refills"]}),
            _make_record(2, {"problems": ["late refills"]}),
            _make_record(3, {}, check=True),
        ]
        self.assertIn("FAILED_RESUMPTION", self._apply(records)[3])

    def test_falling_behind_non_paused_baseline_is_flagged(self):
        records = [
            _make_record(1, {"problems": ["late refills"]}),
            _make_record(2, {"problems": ["late refills"]}),
            _make_record(3, {"problems": ["late refills"]}, check=True),
        ]
        baseline = [
            _make_record(1, {"problems": ["late refills"]}),
            _make_record(2, {"problems": ["late refills"]}),
            _make_record(3, {"problems": ["late refills"],
                             "frequency": ["weekly"]}),
        ]
        self.assertIn("FAILED_RESUMPTION", self._apply(records, baseline)[3])

    def test_content_dropped_by_pause_but_recovered_on_resume_is_ok(self):
        records = [
            _make_record(1, {"problems": ["late refills"]}),
            _make_record(2, {"problems": ["late refills"]}),
            _make_record(3, {"problems": ["late refills"],
                             "frequency": ["weekly"]}, check=True),
        ]
        baseline = [
            _make_record(1, {"problems": ["late refills"]}),
            _make_record(2, {"problems": ["late refills"],
                             "frequency": ["weekly"]}),
            _make_record(3, {"problems": ["late refills"],
                             "frequency": ["weekly"]}),
        ]
        self.assertNotIn("FAILED_RESUMPTION", self._apply(records, baseline)[3])

if __name__ == "__main__":
    unittest.main(verbosity=2)