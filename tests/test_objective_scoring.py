"""
tests/test_objective_scoring.py

Information-gain objective scoring — the ObjectiveEngine improvement that
replaces pure checklist-priority selection ("what item is missing and declared
most important?") with information-gain selection ("which single question
would reduce uncertainty the most?").

Scope:
  * Regression — on canonical Empathize states the new scorer reproduces the
    legacy PriorityEngine selection exactly when there is no conversation
    context (and with a *barely* touched context it still agrees).
  * Determinism — identical (state, context) pairs always produce identical
    scores, ordering, reasoning, and selection traces.
  * Context sensitivity — user-discussed topics earn a bonus, already-asked
    question families earn a small penalty, and neither alone may overturn
    the canonical ordering.
  * Pivot — when the user has already discussed a lower-priority field AND
    the higher-priority field's question family was already asked, the engine
    pivots to the discussed field. This is the behaviour the legacy checklist
    engine could not express.
  * Developer Console — the Selection trace (candidates + component scores +
    selected + rejected reasons) is attached to every non-WRAP_UP objective
    and is JSON-serialisable (the X-Diagnostics wire contract).
  * WRAP_UP / lifecycle unchanged.
"""

import json
import os
import sys
import unittest
from unittest import mock

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField  # noqa: E402

from module3 import (  # noqa: E402
    CandidateScore,
    CompletenessChecker,
    ConversationObjective,
    Objective,
    ObjectiveContext,
    ObjectiveEngine,
    PriorityEngine,
    build_candidate_scores,
)


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _legacy_objective(state):
    """The pre-improvement selection: highest-priority missing field."""
    coverage = CompletenessChecker().evaluate(state)
    return PriorityEngine().decide(coverage).objective


def _state(**kwargs):
    return ProjectState(**kwargs)


# ---------------------------------------------------------------------------
# Regression: new scorer == legacy PriorityEngine on canonical states
# ---------------------------------------------------------------------------


class TestScoringMatchesLegacyBaseline(unittest.TestCase):
    """No conversation context -> the information-gain scorer reproduces the
    legacy checklist selection exactly, so the golden paths stay pinned."""

    _STATES = [
        _state(),
        _state(personas=["students"]),
        _state(personas=["students"], problems=["forgetting deadlines"]),
        _state(
            personas=["students"],
            problems=["forgetting deadlines"],
            frequency="weekly",
        ),
        _state(
            personas=["students"],
            problems=["forgetting deadlines"],
            current_solutions=["pill box"],
            pain_points=["stress"],
            frequency="daily",
        ),
        _state(
            personas=["students"],
            problems=["forgetting deadlines"],
            current_solutions=["pill box"],
            evidence=["surveyed 20"],
            frequency="daily",
        ),
        _state(
            personas=["students"],
            problems=["forgetting deadlines"],
            current_solutions=["pill box"],
            pain_points=["stress"],
            evidence=["surveyed 20"],
            frequency="daily",
        ),
    ]

    def test_new_engine_agrees_with_legacy_on_every_canonical_state(self):
        engine = ObjectiveEngine()
        for i, state in enumerate(self._STATES):
            new = engine.determine_next(state).objective
            legacy = _legacy_objective(state)
            self.assertEqual(
                new, legacy,
                f"state[{i}]: new scorer selected {new.name}, "
                f"legacy selected {legacy.name}",
            )

    def test_full_state_still_wraps_up(self):
        obj = ObjectiveEngine().determine_next(self._STATES[-1])
        self.assertTrue(obj.is_wrap_up)
        self.assertIsNone(obj.selection_trace)

    def test_partial_state_has_no_legacy_reordering(self):
        # Only pain_points + evidence missing: legacy picks pain_points (60)
        # over evidence (50); the scorer must agree.
        state = _state(
            personas=["x"], problems=["x"], current_solutions=["x"],
            pain_points=[], evidence=[], frequency="weekly",
        )
        self.assertEqual(ObjectiveEngine().determine_next(state).objective,
                         _legacy_objective(state))

    def test_execution_order_is_canonical_without_context(self):
        engine = ObjectiveEngine()
        state = _state()
        visited = []
        while True:
            obj = engine.determine_next(state)
            visited.append(obj.objective)
            if obj.is_wrap_up:
                break
            field = obj.targeted_field()
            if field in (StateField.FREQUENCY,):
                setattr(state, field.value, "x")
            else:
                setattr(state, field.value, ["x"])
        self.assertEqual(visited, [
            Objective.PERSONAS,
            Objective.PROBLEMS,
            Objective.FREQUENCY,
            Objective.CURRENT_SOLUTIONS,
            Objective.PAIN_POINTS,
            Objective.EVIDENCE,
            Objective.WRAP_UP,
        ])


# ---------------------------------------------------------------------------
# Determinism of the scorer
# ---------------------------------------------------------------------------


class TestScoringDeterministic(unittest.TestCase):

    def test_identical_state_and_context_produce_identical_objectives(self):
        engine = ObjectiveEngine()
        state = _state(personas=["students"], frequency="weekly")
        ctx = ObjectiveContext(
            user_messages=("they currently use sticky notes",),
            asked_families=frozenset({"FREQUENCY_ESTIMATE"}),
        )
        a = engine.determine_next(state, context=ctx)
        b = engine.determine_next(state, context=ctx)
        self.assertEqual(a, b)
        self.assertEqual(a.selection_trace, b.selection_trace)
        self.assertEqual(a.reasoning, b.reasoning)

    def test_context_order_independence(self):
        # discussion is an "any message" signal, so message ordering cannot
        # change the result.
        engine = ObjectiveEngine()
        state = _state(personas=["students"])
        base = engine.determine_next(
            state, context=ObjectiveContext(user_messages=("a", "b"))
        )
        swapped = engine.determine_next(
            state, context=ObjectiveContext(user_messages=("b", "a"))
        )
        self.assertEqual(base, swapped)

    def test_scores_are_stable_plain_numbers(self):
        scores = build_candidate_scores(
            ObjectiveEngine().rules,
            CompletenessChecker().evaluate(_state()),
            ObjectiveContext(),
        )
        for s in scores:
            self.assertIsInstance(s, CandidateScore)
            self.assertIsInstance(s.score, float)
            self.assertEqual(s.score,
                             s.info_value + s.unlock_bonus
                             + s.discussion_bonus - s.repeat_penalty)


# ---------------------------------------------------------------------------
# Context sensitivity
# ---------------------------------------------------------------------------


class TestScoringContextSensitivity(unittest.TestCase):

    def test_repeat_penalty_alone_does_not_unseat_the_leader(self):
        """Golden-11-style: every frequency family already asked, frequency
        still missing. The field remains the top priority — repeat penalty is
        deliberately too small to overturn a one-band priority gap."""
        state = _state(personas=["students"], problems=["forgetting"])
        ctx = ObjectiveContext(
            asked_families=frozenset({
                "FREQUENCY_ESTIMATE", "FREQUENCY_PERCENTAGE",
                "FREQUENCY_PREVALENCE", "FREQUENCY_EXAMPLES",
            })
        )
        obj = ObjectiveEngine().determine_next(state, context=ctx)
        self.assertEqual(obj.objective, Objective.FREQUENCY)

    def test_discussion_bonus_alone_does_not_reorder_canonical_winners(self):
        """A discussed lower-priority field does not beat an untouched,
        clearly-more-important field (personas > problems by far more than
        the discussion bonus)."""
        state = _state()
        ctx = ObjectiveContext(user_messages=("the problem is getting worse",))
        obj = ObjectiveEngine().determine_next(state, context=ctx)
        self.assertEqual(obj.objective, Objective.PERSONAS)

    def test_discussion_plus_repeat_pivots_to_the_discussed_field(self):
        """The improvement: when the user has ALREADY discussed current
        solutions AND a frequency family was already asked, the mentor pivots
        to the discussed field instead of re-asking frequency. The legacy
        engine always picked FREQUENCY here."""
        state = _state(personas=["students"], problems=["forgetting"])
        no_context = ObjectiveEngine().determine_next(state)
        self.assertEqual(no_context.objective, Objective.FREQUENCY)

        ctx = ObjectiveContext(
            user_messages=("they currently use sticky notes and a planner",),
            asked_families=frozenset({"FREQUENCY_ESTIMATE"}),
        )
        pivoted = ObjectiveEngine().determine_next(state, context=ctx)
        self.assertEqual(pivoted.objective, Objective.CURRENT_SOLUTIONS)
        # The trace explains the pivot deterministically.
        self.assertEqual(pivoted.selection_trace["selected"],
                         "CURRENT_SOLUTIONS")
        freq = next(
            c for c in pivoted.selection_trace["candidates"]
            if c["objective"] == "FREQUENCY"
        )
        self.assertTrue(freq["repeated"])
        cs = next(
            c for c in pivoted.selection_trace["candidates"]
            if c["objective"] == "CURRENT_SOLUTIONS"
        )
        self.assertTrue(cs["discussed"])
        self.assertGreater(cs["score"], freq["score"])


# ---------------------------------------------------------------------------
# Selection trace — Developer Console payload
# ---------------------------------------------------------------------------


class TestSelectionTrace(unittest.TestCase):

    def setUp(self):
        self.state = _state(personas=["students"], frequency="weekly")
        self.ctx = ObjectiveContext(
            user_messages=("they use a planner today",),
            asked_families=frozenset({"FREQUENCY_ESTIMATE"}),
        )
        self.obj = ObjectiveEngine().determine_next(self.state, context=self.ctx)

    def test_trace_is_attached_for_non_wrap_up(self):
        self.assertIsNotNone(self.obj.selection_trace)
        self.assertEqual(self.obj.selection_trace["mode"], "information_gain")

    def test_candidates_sorted_by_score_desc(self):
        cands = self.obj.selection_trace["candidates"]
        scores = [c["score"] for c in cands]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_candidate_carries_component_scores(self):
        for c in self.obj.selection_trace["candidates"]:
            for key in ("info_value", "unlock_bonus",
                        "discussion_bonus", "repeat_penalty"):
                self.assertIn(key, c["components"])
                self.assertIsInstance(c["components"][key], float)
            self.assertIn("discussed", c)
            self.assertIn("repeated", c)

    def test_selected_matches_objective(self):
        self.assertEqual(self.obj.selection_trace["selected"],
                         self.obj.objective.value)
        self.assertTrue(self.obj.selection_trace["reason_selected"])

    def test_rejected_lists_every_loser_with_reason(self):
        trace = self.obj.selection_trace
        cands = trace["candidates"]
        expected_losers = {c["objective"] for c in cands} - {trace["selected"]}
        self.assertEqual({r["objective"] for r in trace["rejected"]},
                         expected_losers)
        self.assertTrue(all(r["reason"] for r in trace["rejected"]))

    def test_trace_is_json_serialisable(self):
        round_tripped = json.loads(json.dumps(self.obj.selection_trace))
        self.assertEqual(round_tripped["selected"],
                         self.obj.selection_trace["selected"])
        self.assertEqual(round_tripped["candidates"],
                         self.obj.selection_trace["candidates"])

    def test_reasoning_explains_the_decision(self):
        joined = "\n".join(self.obj.reasoning)
        self.assertIn("Information-gain scoring", joined)
        self.assertIn("Selected", joined)
        self.assertIn("Rejected", joined)


# ---------------------------------------------------------------------------
# Live pipeline — Selection section in the Developer Console
# ---------------------------------------------------------------------------


class TestScoringPipelineDiagnostics(unittest.TestCase):

    AUDIT_USER = "audit_scoring"
    AUDIT_PROJ = "audit_scoring_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_live_turn_exposes_selection_section(self):
        import mentor
        from memory_extractor import ExtractionResult, MessageType

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                _reply, _sess, _timing, diagnostics = mentor.process_mentor_turn(
                    "hello",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
        trace = diagnostics["ObjectiveTrace"]
        self.assertIn("Selection", trace)
        selection = trace["Selection"]
        self.assertEqual(selection["selected"], "PERSONAS")
        self.assertEqual(
            [c["objective"] for c in selection["candidates"]][0], "PERSONAS"
        )
        self.assertTrue(selection["rejected"])
        # Survives the wire contract.
        round_tripped = json.loads(json.dumps(diagnostics))
        self.assertEqual(round_tripped["ObjectiveTrace"]["Selection"],
                         selection)


if __name__ == "__main__":
    unittest.main(verbosity=2)
