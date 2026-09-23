"""
tests/test_priority_engine.py

Unit tests for module3.priority_engine.PriorityEngine.

Covers (per the Module 3 spec test categories):
  - personas missing
  - problems missing
  - pain points missing
  - wrap-up objective (all fields complete -> WRAP_UP)
  - priority ordering (higher priority wins over lower)
  - ties (multiple rules share the winning priority)
  - Confidence bands (1.0 unique, 0.8 tiebreak)

Also covers:
  - read-only contract on CoverageReport (no mutation)
  - the strict CoverageReport-only input boundary
  - custom rule registries (extensibility)
  - the misconfigured empty-registry failure mode
  - the implicit assumption that WRAP_UP must NOT be a rule entry
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import REQUIRED_FIELDS, StateField

from module3 import DEFAULT_RULES, CoverageReport, Objective, PriorityEngine, Rule
from module3.priority_engine import PriorityDecision
from module3.rules import ordered_candidates, validate_rules


# ---------------------------------------------------------------------------
# Small helpers — build CoverageReports declaratively and concisely.
# ---------------------------------------------------------------------------

def _report(**overrides):
    """Build a CoverageReport whose defaults are all-false; pass True to fill."""
    kwargs = {sf.value: False for sf in REQUIRED_FIELDS}
    kwargs.update(overrides)
    return CoverageReport(**kwargs)


def _all_complete_report():
    """Convenience: every required field covered."""
    return _report(**{sf.value: True for sf in REQUIRED_FIELDS})


# ---------------------------------------------------------------------------
# Default-rule behaviour — these match the Empathize order in DEFAULT_RULES
# ---------------------------------------------------------------------------


class TestPriorityEngine_DefaultRules(unittest.TestCase):

    def setUp(self):
        # Default engine uses DEFAULT_RULES (the canonical Empathize order).
        self.engine = PriorityEngine()

    def test_personas_missing_returns_personas(self):
        report = _report()  # nothing covered, highest-priority gap is personas
        decision = self.engine.decide(report)
        self.assertEqual(decision.objective, Objective.PERSONAS)
        self.assertEqual(decision.confidence, PriorityEngine.CONFIDENCE_UNIQUE)
        self.assertIsNotNone(decision.winning_rule)
        self.assertEqual(decision.winning_rule.field, StateField.PERSONAS)
        self.assertEqual(decision.tied_rules, [])

    def test_problems_missing_with_personas_present(self):
        # Personas covered -> the next-highest missing field is problems (90).
        report = _report(personas=True)
        decision = self.engine.decide(report)
        self.assertEqual(decision.objective, Objective.PROBLEMS)

    def test_pain_points_missing_only(self):
        # Everything except pain_points covered -> pain_points objective.
        report = _report(
            personas=True,
            problems=True,
            current_solutions=True,
            pain_points=False,
            evidence=True,
            frequency=True,
        )
        decision = self.engine.decide(report)
        self.assertEqual(decision.objective, Objective.PAIN_POINTS)
        self.assertEqual(decision.confidence, 1.0)

    def test_evidence_missing_only(self):
        # All covered except evidence (priority 50).
        report = _report(
            personas=True,
            problems=True,
            current_solutions=True,
            pain_points=True,
            evidence=False,
            frequency=True,
        )
        decision = self.engine.decide(report)
        self.assertEqual(decision.objective, Objective.EVIDENCE)

    def test_frequency_missing_among_others(self):
        # frequency has priority 80, so among the missing fields it should
        # win out over current_solutions (70), pain_points (60), evidence (50).
        report = _report(
            personas=True,
            problems=True,
            current_solutions=False,
            pain_points=False,
            evidence=False,
            frequency=False,
        )
        decision = self.engine.decide(report)
        self.assertEqual(decision.objective, Objective.FREQUENCY)


class TestPriorityEngine_WrapUp(unittest.TestCase):
    """When the report is all-complete, the engine returns WRAP_UP."""

    def test_all_complete_returns_wrap_up(self):
        engine = PriorityEngine()
        decision = engine.decide(_all_complete_report())
        self.assertEqual(decision.objective, Objective.WRAP_UP)
        self.assertEqual(decision.confidence, PriorityEngine.CONFIDENCE_UNIQUE)
        self.assertIsNone(decision.winning_rule)
        self.assertEqual(decision.tied_rules, [])

    def test_wrap_up_reasoning_records_completion(self):
        engine = PriorityEngine()
        decision = engine.decide(_all_complete_report())
        joined = " ".join(decision.reasoning)
        self.assertIn("All required", joined)
        self.assertIn("WRAP_UP", joined)


class TestPriorityEngine_PriorityOrdering(unittest.TestCase):
    """Higher priority rules must beat lower priority rules."""

    def test_priority_100_beats_priority_50(self):
        # Nothing covered — the rule with priority 100 (personas) must win
        # even though evidence (priority 50) is also missing.
        engine = PriorityEngine()
        decision = engine.decide(_report())
        self.assertEqual(decision.winning_rule.priority, 100)
        self.assertEqual(decision.objective, Objective.PERSONAS)

    def test_priority_lower_wins_when_higher_already_met(self):
        # Personas already filled — engine skips its rule and goes to
        # problems (priority 90) even though other rules still match.
        engine = PriorityEngine()
        decision = engine.decide(_report(personas=True))
        self.assertEqual(decision.winning_rule.priority, 90)
        self.assertEqual(decision.objective, Objective.PROBLEMS)

    def test_priority_descending_sequence(self):
        """Walking completion forward yields objectives in declared priority order."""
        engine = PriorityEngine()
        state = _report()  # all missing
        expected_order = [  # descending priority per DEFAULT_RULES
            (Objective.PERSONAS,        100),
            (Objective.PROBLEMS,        90),
            (Objective.FREQUENCY,       80),
            (Objective.CURRENT_SOLUTIONS, 70),
            (Objective.PAIN_POINTS,     60),
            (Objective.EVIDENCE,        50),
        ]
        for objective, priority in expected_order:
            decision = engine.decide(state)
            self.assertEqual(decision.objective, objective,
                             f"expected {objective.name}, got {decision.objective.name}")
            self.assertEqual(decision.winning_rule.priority, priority)
            # Mark this objective's field as complete in the synthetic report.
            state = _report(**{**state.as_dict(), decision.winning_rule.field.value: True})
        # Now everything is covered -> WRAP_UP.
        decision = engine.decide(state)
        self.assertEqual(decision.objective, Objective.WRAP_UP)


class TestPriorityEngine_Ties(unittest.TestCase):
    """When two candidate rules share the highest priority -> deterministic tiebreak."""

    def _tie_rules(self):
        """Two rules with equal priority for two distinct StateFields."""
        return [
            Rule(field=StateField.PERSONAS, priority=50, objective=Objective.PERSONAS),
            Rule(field=StateField.EVIDENCE, priority=50, objective=Objective.EVIDENCE),
        ]

    def test_tie_breaks_by_declared_enum_order(self):
        engine = PriorityEngine(rules=self._tie_rules())
        # Both rules match (no field covered) -> PERSONAS is declared first
        # in the StateField enum, so it must win.
        decision = engine.decide(_report())
        self.assertEqual(decision.objective, Objective.PERSONAS)
        self.assertEqual(decision.winning_rule.field, StateField.PERSONAS)

    def test_tie_lowers_confidence(self):
        engine = PriorityEngine(rules=self._tie_rules())
        decision = engine.decide(_report())
        self.assertEqual(decision.confidence, PriorityEngine.CONFIDENCE_TIEBREAK)
        self.assertEqual(decision.confidence, 0.8)

    def test_tie_rules_all_appear_in_tied_rules(self):
        engine = PriorityEngine(rules=self._tie_rules())
        decision = engine.decide(_report())
        self.assertEqual(len(decision.tied_rules), 2)
        tied_fields = {r.field for r in decision.tied_rules}
        self.assertEqual(tied_fields, {StateField.PERSONAS, StateField.EVIDENCE})

    def test_tie_reasoning_mentions_tie_and_tiebreak(self):
        engine = PriorityEngine(rules=self._tie_rules())
        decision = engine.decide(_report())
        joined = " ".join(decision.reasoning)
        self.assertIn("Priority tie among 2 candidate rules", joined)
        self.assertIn("Tiebreak", joined)

    def test_tie_picks_first_declared_field_when_differs_from_rule_list_order(self):
        """Tiebreak is by StateField enum order, NOT by declared rule-order."""
        # Reverse the rule list so evidence appears first — tiebreak must
        # still pick PERSONAS (first in StateField's enum).
        rules = [
            Rule(field=StateField.EVIDENCE, priority=50, objective=Objective.EVIDENCE),
            Rule(field=StateField.PERSONAS, priority=50, objective=Objective.PERSONAS),
        ]
        engine = PriorityEngine(rules=rules)
        decision = engine.decide(_report())
        self.assertEqual(decision.objective, Objective.PERSONAS)


class TestPriorityEngine_Contract(unittest.TestCase):

    def test_does_not_mutate_input_report(self):
        """The report is a frozen dataclass, but assert the engine didn't try."""
        report = _report(personas=True, problems=False)
        # Capture pre-state by re-creating the same report; we cannot
        # mutate it (frozen), but we can confirm its public view is intact.
        before = report.as_dict()
        PriorityEngine().decide(report)
        self.assertEqual(report.as_dict(), before)

    def test_rejects_non_coverage_report(self):
        with self.assertRaises(TypeError):
            PriorityEngine().decide({"personas": True})  # type: ignore[arg-type]

    def test_rejects_none(self):
        with self.assertRaises(TypeError):
            PriorityEngine().decide(None)  # type: ignore[arg-type]

    def test_default_rules_round_trip_via_engine_property(self):
        engine = PriorityEngine()
        # engine.rules returns a defensive copy; should equal DEFAULT_RULES.
        self.assertEqual(engine.rules, DEFAULT_RULES)
        # And mutating the returned copy MUST NOT affect the engine.
        copy_rules = engine.rules
        if copy_rules:
            copy_rules.pop()
        self.assertEqual(len(engine.rules), len(DEFAULT_RULES))

    def test_returns_priority_decision_namedtuple(self):
        decision = PriorityEngine().decide(_report())
        self.assertIsInstance(decision, PriorityDecision)

    def test_decision_has_required_fields(self):
        decision = PriorityEngine().decide(_report())
        # All five namedtuple fields present and well-typed.
        self.assertTrue(hasattr(decision, "objective"))
        self.assertTrue(hasattr(decision, "winning_rule"))
        self.assertTrue(hasattr(decision, "tied_rules"))
        self.assertTrue(hasattr(decision, "confidence"))
        self.assertTrue(hasattr(decision, "reasoning"))
        self.assertIsInstance(decision.objective, Objective)
        self.assertIsInstance(decision.reasoning, list)


class TestPriorityEngine_Misconfiguration(unittest.TestCase):

    def test_empty_registry_with_missing_fields_raises(self):
        # No rules -> a non-complete report has no candidates -> RuntimeError
        # (deterministic failure rather than silent WRAP_UP).
        engine = PriorityEngine(rules=[])
        with self.assertRaises(RuntimeError):
            engine.decide(_report())

    def test_empty_registry_with_complete_report_still_returns_wrap_up(self):
        # With an empty rule registry AND all-complete report, the algorithm
        # short-circuits to WRAP_UP before even consulting the rules.
        engine = PriorityEngine(rules=[])
        decision = engine.decide(_all_complete_report())
        self.assertEqual(decision.objective, Objective.WRAP_UP)

    def test_validate_rules_rejects_wrapup_rule_entry(self):
        """WRAP_UP must not be a Rule field-gap entry; it is derived."""
        bad_rules = [
            Rule(field=StateField.PERSONAS, priority=100, objective=Objective.PERSONAS),
            Rule(field=StateField.PROBLEMS, priority=90, objective=Objective.WRAP_UP),  # invalid
        ]
        with self.assertRaises(ValueError):
            validate_rules(bad_rules)

    def test_validate_rules_rejects_duplicate_field(self):
        bad_rules = [
            Rule(field=StateField.PERSONAS, priority=100, objective=Objective.PERSONAS),
            Rule(field=StateField.PERSONAS, priority=50, objective=Objective.PERSONAS),
        ]
        with self.assertRaises(ValueError):
            validate_rules(bad_rules)

    def test_validate_rules_rejects_objective_field_mismatch(self):
        """A Rule whose objective does not match its field is rejected.

        Note: this invariant is enforced by ``Rule.__post_init__`` itself
        ( Rule construction raises ValueError), so the check fires at the
        rule level — before ``validate_rules`` ever sees the entry. Both
        layers guard this invariant; here we assert the immediate raising
        layer (Rule construction), which is the one a misconfigured
        contributor actually hits first.
        """
        with self.assertRaises(ValueError):
            Rule(
                field=StateField.PERSONAS,
                priority=100,
                objective=Objective.EVIDENCE,  # mismatch
            )


class TestOrderedCandidatesHelper(unittest.TestCase):
    """Sanity-check the ordering helper the engine leans on."""

    def test_orders_by_priority_descending(self):
        rules = [
            Rule(field=StateField.EVIDENCE, priority=50, objective=Objective.EVIDENCE),
            Rule(field=StateField.PERSONAS, priority=100, objective=Objective.PERSONAS),
            Rule(field=StateField.PAIN_POINTS, priority=60, objective=Objective.PAIN_POINTS),
        ]
        # All fields missing -> all candidates.
        report = _report()
        ordered = ordered_candidates(rules, report)
        priorities = [r.priority for (r, _i) in ordered]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        self.assertEqual(priorities, [100, 60, 50])

    def test_filters_out_completed_fields(self):
        rules = [
            Rule(field=StateField.PERSONAS, priority=100, objective=Objective.PERSONAS),
            Rule(field=StateField.PROBLEMS, priority=90, objective=Objective.PROBLEMS),
        ]
        # Personas already covered -> only the problems rule should be a candidate.
        report = _report(personas=True)
        ordered = ordered_candidates(rules, report)
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0][0].field, StateField.PROBLEMS)

    def test_tiebreak_is_enum_order(self):
        # Same priority for both -> StateField enum order breaks the tie;
        # PERSONAS (enum index 0) comes before EVIDENCE (enum index 4).
        rules = [
            Rule(field=StateField.EVIDENCE, priority=50, objective=Objective.EVIDENCE),
            Rule(field=StateField.PERSONAS, priority=50, objective=Objective.PERSONAS),
        ]
        report = _report()
        ordered = ordered_candidates(rules, report)
        fields = [r.field for (r, _i) in ordered]
        self.assertEqual(fields, [StateField.PERSONAS, StateField.EVIDENCE])


if __name__ == "__main__":
    unittest.main(verbosity=2)
