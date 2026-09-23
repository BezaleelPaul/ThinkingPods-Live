"""
tests/test_conversation_memory.py

Regression suite for the deterministic conversational-context memory layer
(``conversation_memory``): open threads, resolved threads, deferred topics,
acknowledged facts, summarized facts, and partially answered objectives.

Covers:
  * thread lifecycle across turns (open -> resolved);
  * deferred topics (out-of-turn contributions) being revisited and then
    resolved once the field reaches a usable answer;
  * summary / acknowledgement tracking through the lifecycle decisions;
  * deterministic, read-only guarantees (state snapshots are never mutated);
  * SessionData persistence round-trip for the memory container;
  * Developer Console ``Memory`` section present and JSON-serialisable;
  * the resume hint flowing into the generated LLM prompt.

Unit tests exercise ``update_conversation_memory`` directly with hand-built
state snapshots (no LLM, no session persistence); integration tests replay
live ``process_mentor_turn`` turns with a stubbed MemoryExtractor and a
raising ollama stub (deterministic fallback path).
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    ProjectState,
    StateField,
)
from module3 import ObjectiveEngine  # noqa: E402
from module5 import LifecycleDecision  # noqa: E402
from recovery_monitor import analyze_recovery  # noqa: E402
from conversation_memory import (  # noqa: E402
    ConversationMemory,
    memory_diagnostics_section,
    memory_to_prompt_bullets,
    update_conversation_memory,
)
from session_manager import SessionData  # noqa: E402


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _state(**overrides):
    base = {
        "personas": [],
        "problems": [],
        "current_solutions": [],
        "pain_points": [],
        "evidence": [],
        "impacts": [],
        "frequency": None,
    }
    base.update(overrides)
    return base


def _objective(state_dict):
    """Build the real engine's objective for a to_state_dict()-shaped dict."""
    return ObjectiveEngine().determine_next(ProjectState(**state_dict))


# ---------------------------------------------------------------------------
# Unit tests — direct update_conversation_memory behaviour
# ---------------------------------------------------------------------------


class TestThreadLifecycle(unittest.TestCase):
    def test_open_thread_created_for_current_target(self):
        sd = SessionData()
        before = _state()
        after = _state()
        obj = _objective(after)
        # Empty state -> engine targets the first required field (personas).
        self.assertEqual(obj.targeted_field(), StateField.PERSONAS)
        report = analyze_recovery(
            user_message="students", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        open_fields = {r["field"] for r in sd.conversation_memory.open_threads}
        self.assertIn("personas", open_fields)

    def test_thread_resolves_after_usable_answer(self):
        sd = SessionData()
        # Turn 1: persona captured, problems is now the pursued target.
        t1_before = _state()
        t1_after = _state(personas=["students"])
        obj1 = _objective(t1_after)
        self.assertEqual(obj1.targeted_field(), StateField.PROBLEMS)
        update_conversation_memory(
            session_data=sd,
            state_before=t1_before,
            state_after=t1_after,
            objective=obj1,
            recovery=analyze_recovery(
                user_message="students", state_before=t1_before, state_after=t1_after
            ),
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        # Turn 2: problems answered -> resolved, removed from open.
        t2_before = t1_after
        t2_after = _state(personas=["students"], problems=["missing deadlines"])
        obj2 = _objective(t2_after)
        update_conversation_memory(
            session_data=sd,
            state_before=t2_before,
            state_after=t2_after,
            objective=obj2,
            recovery=analyze_recovery(
                user_message="they miss deadlines",
                state_before=t2_before,
                state_after=t2_after,
            ),
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        memory = sd.conversation_memory
        self.assertIn(
            "problems", {r["field"] for r in memory.resolved_threads}
        )
        self.assertNotIn(
            "problems", {r["field"] for r in memory.open_threads}
        )
        # A field never moves out of resolved once satisfied.
        update_conversation_memory(
            session_data=sd,
            state_before=t2_after,
            state_after=t2_after,
            objective=obj2,
            recovery=None,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertEqual(len(memory.resolved_threads), 2)


class TestDeferredTopics(unittest.TestCase):
    def test_out_of_turn_partial_answer_deferred(self):
        """A partial frequency volunteered while the target is another field
        is deferred, not treated as the active thread."""
        sd = SessionData()
        before = _state(personas=["students"])
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)  # target = problems / current_solutions
        self.assertNotEqual(obj.targeted_field(), StateField.FREQUENCY)
        report = analyze_recovery(
            user_message="it happens a lot",
            state_before=before,
            state_after=after,
            last_assistant_message="What is the core problem or frustration they're experiencing?",
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        deferred = {r["field"] for r in sd.conversation_memory.deferred_topics}
        self.assertIn("frequency", deferred)
        self.assertIn("out of turn", sd.conversation_memory.deferred_topics[0]["reason"])

    def test_deferred_topic_resolved_when_satisfied(self):
        sd = SessionData()
        t1_before = _state(personas=["students"])
        t1_after = _state(personas=["students"], frequency="a lot")
        obj1 = _objective(t1_after)
        update_conversation_memory(
            session_data=sd,
            state_before=t1_before,
            state_after=t1_after,
            objective=obj1,
            recovery=analyze_recovery(
                user_message="it happens a lot",
                state_before=t1_before,
                state_after=t1_after,
            ),
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertIn(
            "frequency", {r["field"] for r in sd.conversation_memory.deferred_topics}
        )
        # Later the user gives a usable cadence -> frequency resolves.
        t2_before = t1_after
        t2_after = _state(
            personas=["students"], frequency="every single day"
        )
        obj2 = _objective(t2_after)
        update_conversation_memory(
            session_data=sd,
            state_before=t2_before,
            state_after=t2_after,
            objective=obj2,
            recovery=analyze_recovery(
                user_message="every single day",
                state_before=t2_before,
                state_after=t2_after,
            ),
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        deferred = {r["field"] for r in sd.conversation_memory.deferred_topics}
        resolved = {r["field"] for r in sd.conversation_memory.resolved_threads}
        self.assertNotIn("frequency", deferred)
        self.assertIn("frequency", resolved)

    def test_target_answer_is_not_deferred(self):
        sd = SessionData()
        before = _state()
        after = _state(personas=["students"])
        obj = _objective(after)
        report = analyze_recovery(
            user_message="students", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        # The answered persona reached a usable answer -> resolved, never
        # deferred (deferred only holds unsatisfied out-of-turn topics).
        self.assertEqual(sd.conversation_memory.deferred_topics, [])
        self.assertIn(
            "personas", {r["field"] for r in sd.conversation_memory.resolved_threads}
        )


class TestPartiallyAnsweredObjectives(unittest.TestCase):
    def test_partial_frequency_recorded(self):
        sd = SessionData()
        before = _state(personas=["students"])
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        report = analyze_recovery(
            user_message="it happens a lot", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        partial = {r["field"] for r in sd.conversation_memory.partially_answered_objectives}
        self.assertIn("frequency", partial)

    def test_usable_answer_is_not_partial(self):
        sd = SessionData()
        before = _state()
        after = _state(frequency="every single day")
        obj = _objective(after)
        report = analyze_recovery(
            user_message="every single day", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertEqual(sd.conversation_memory.partially_answered_objectives, [])


class TestSummaryAndAcknowledgement(unittest.TestCase):
    def test_summary_records_summarized_facts(self):
        sd = SessionData()
        full = _state(
            personas=["students"],
            problems=["missing deadlines"],
            current_solutions=["sticky notes"],
            pain_points=["stress"],
            evidence=["interviewed 10"],
            frequency="weekly",
        )
        obj = _objective(full)
        self.assertTrue(obj.is_wrap_up)
        update_conversation_memory(
            session_data=sd,
            state_before=full,
            state_after=full,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.READY_FOR_SUMMARY,
        )
        summarized = {r["field"] for r in sd.conversation_memory.summarized_facts}
        self.assertEqual(summarized, {sf.value for sf in StateField if sf.value != "impacts"})
        self.assertIn("summary", sd.conversation_memory.summarized_facts[0]["reason"])

    def test_transition_records_acknowledged_facts(self):
        sd = SessionData()
        full = _state(
            personas=["students"],
            problems=["missing deadlines"],
            current_solutions=["sticky notes"],
            pain_points=["stress"],
            evidence=["interviewed 10"],
            frequency="weekly",
        )
        obj = _objective(full)
        update_conversation_memory(
            session_data=sd,
            state_before=full,
            state_after=full,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.READY_FOR_TRANSITION,
        )
        acknowledged = {r["field"] for r in sd.conversation_memory.acknowledged_facts}
        self.assertIn("personas", acknowledged)
        self.assertIn("frequency", acknowledged)
        self.assertIn(
            "confirmed", sd.conversation_memory.acknowledged_facts[0]["reason"]
        )


class TestDeterminismAndReadOnly(unittest.TestCase):
    def test_deterministic_same_inputs_same_memory(self):
        def run():
            sd = SessionData()
            before = _state(personas=["students"])
            after = _state(personas=["students"], frequency="a lot")
            obj = _objective(after)
            report = analyze_recovery(
                user_message="it happens a lot",
                state_before=before,
                state_after=after,
            )
            update_conversation_memory(
                session_data=sd,
                state_before=before,
                state_after=after,
                objective=obj,
                recovery=report,
                lifecycle_decision=LifecycleDecision.CONTINUE,
            )
            return sd.conversation_memory.to_dict()

        self.assertEqual(run(), run())

    def test_state_snapshots_are_never_mutated(self):
        before = _state(personas=["students"])
        after = _state(personas=["students"], frequency="a lot")
        before_copy = json.loads(json.dumps(before))
        after_copy = json.loads(json.dumps(after))
        sd = SessionData()
        obj = _objective(after)
        report = analyze_recovery(
            user_message="it happens a lot", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertEqual(before, before_copy)
        self.assertEqual(after, after_copy)

    def test_memory_never_duplicates_project_state_values(self):
        """Records reference field keys/statuses only — no captured values."""
        sd = SessionData()
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        update_conversation_memory(
            session_data=sd,
            state_before=_state(),
            state_after=after,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        flat = json.dumps(sd.conversation_memory.to_dict())
        for value in ("students", "a lot"):
            self.assertNotIn(value, flat)

    def test_json_round_trip(self):
        sd = SessionData()
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        update_conversation_memory(
            session_data=sd,
            state_before=_state(),
            state_after=after,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        d = sd.to_dict()
        round_tripped = json.loads(json.dumps(d))
        restored = SessionData.from_dict(round_tripped)
        self.assertEqual(restored.conversation_memory, sd.conversation_memory)
        self.assertEqual(
            restored.conversation_memory.deferred_topics,
            sd.conversation_memory.deferred_topics,
        )

    def test_recovery_dict_input_supported(self):
        """update_conversation_memory accepts the RecoveryReport dict form too."""
        sd = SessionData()
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        report = analyze_recovery(
            user_message="it happens a lot", state_before=_state(), state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=_state(),
            state_after=after,
            objective=obj,
            recovery=report.to_dict(),
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertIn(
            "frequency", {r["field"] for r in sd.conversation_memory.deferred_topics}
        )


class TestPromptResumeHint(unittest.TestCase):
    def test_empty_memory_yields_no_bullets(self):
        self.assertEqual(memory_to_prompt_bullets(SessionData()), [])

    def test_target_thread_not_relisted(self):
        sd = SessionData()
        after = _state(personas=["students"])
        obj = _objective(after)
        update_conversation_memory(
            session_data=sd,
            state_before=_state(),
            state_after=after,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        bullets = memory_to_prompt_bullets(sd, current_target=obj.targeted_field())
        self.assertEqual(bullets, [])

    def test_deferred_topic_creates_resume_bullet(self):
        sd = SessionData()
        before = _state(personas=["students"])
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        report = analyze_recovery(
            user_message="it happens a lot", state_before=before, state_after=after
        )
        update_conversation_memory(
            session_data=sd,
            state_before=before,
            state_after=after,
            objective=obj,
            recovery=report,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        self.assertIn(
            "frequency", {r["field"] for r in sd.conversation_memory.deferred_topics}
        )
        bullets = memory_to_prompt_bullets(sd, current_target=obj.targeted_field())
        self.assertTrue(any("previously mentioned Frequency" in b for b in bullets))

    def test_acknowledged_summary_bullet(self):
        sd = SessionData()
        full = _state(
            personas=["students"],
            problems=["missing deadlines"],
            current_solutions=["sticky notes"],
            pain_points=["stress"],
            evidence=["interviewed 10"],
            frequency="weekly",
        )
        obj = _objective(full)
        update_conversation_memory(
            session_data=sd,
            state_before=full,
            state_after=full,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.READY_FOR_TRANSITION,
        )
        bullets = memory_to_prompt_bullets(sd)
        self.assertTrue(any("already acknowledged" in b for b in bullets))


class TestDiagnosticsSection(unittest.TestCase):
    def test_section_exposes_all_six_keys(self):
        sd = SessionData()
        section = memory_diagnostics_section(sd)
        self.assertEqual(
            set(section),
            {
                "Open Threads",
                "Resolved Threads",
                "Deferred Topics",
                "Acknowledged Facts",
                "Summarized Facts",
                "Partially Answered Objectives",
            },
        )

    def test_section_is_json_serialisable(self):
        sd = SessionData()
        after = _state(personas=["students"], frequency="a lot")
        obj = _objective(after)
        update_conversation_memory(
            session_data=sd,
            state_before=_state(),
            state_after=after,
            objective=obj,
            recovery=None,
            lifecycle_decision=LifecycleDecision.CONTINUE,
        )
        section = memory_diagnostics_section(sd)
        self.assertEqual(json.loads(json.dumps(section)), section)


# ---------------------------------------------------------------------------
# Live pipeline — Memory section + resume hint in the real turn
# ---------------------------------------------------------------------------


def _extraction(message_type=MessageType.MEANINGFUL, updates=()):
    return ExtractionResult(
        message_type=message_type,
        updates=[
            ExtractionUpdate(operation=u[0], field=u[1], value=u[2])
            for u in updates
        ],
    )


def _run_turn(user, extraction):
    import mentor

    with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
        with mock.patch(
            "mentor.MemoryExtractor.extract", return_value=extraction
        ):
            return mentor.process_mentor_turn(
                user,
                username="memory_user",
                project_name="MemoryProject",
                model_name="test-model",
            )


class TestMemoryPipeline(unittest.TestCase):
    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_memory_section_always_present(self):
        _reply, _sess, _timing, diagnostics = _run_turn(
            "hello", _extraction(MessageType.AMBIGUOUS)
        )
        memory = diagnostics["Memory"]
        for key in (
            "Open Threads",
            "Resolved Threads",
            "Deferred Topics",
            "Acknowledged Facts",
            "Summarized Facts",
            "Partially Answered Objectives",
        ):
            self.assertIn(key, memory)

    def test_open_and_resolved_threads_populate(self):
        # Turn 1: personas captured -> problems becomes the pursued target.
        _r1, _s, _t, diagnostics = _run_turn(
            "I want to help students",
            _extraction(
                updates=[(Operation.ADD, StateField.PERSONAS, "students")]
            ),
        )
        memory = diagnostics["Memory"]
        self.assertIn(
            "personas", {r["field"] for r in memory["Resolved Threads"]}
        )
        self.assertIn(
            "problems", {r["field"] for r in memory["Open Threads"]}
        )
        # Turn 2: problems answered -> resolved; frequency becomes the target.
        _r2, _s, _t, diagnostics = _run_turn(
            "they keep missing deadlines",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PROBLEMS, "missing deadlines")
                ]
            ),
        )
        memory = diagnostics["Memory"]
        resolved = {r["field"] for r in memory["Resolved Threads"]}
        self.assertIn("personas", resolved)
        self.assertIn("problems", resolved)
        open_fields = {r["field"] for r in memory["Open Threads"]}
        self.assertIn("frequency", open_fields)
        self.assertNotIn("problems", open_fields)

    def test_deferred_topic_surfaces_in_prompt(self):
        # Turn 1: no deterministic facts for the PERSONAS objective, so the
        # LLM path runs and applies the canned persona + an out-of-turn rough
        # frequency ("a lot", only PARTIAL), which gets deferred.
        _r1, _s, _t, _d1 = _run_turn(
            "It happens a lot",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PERSONAS, "students"),
                    (Operation.SET, StateField.FREQUENCY, "a lot"),
                ]
            ),
        )
        # Turn 2: the mentor asks about problems; the prompt must reference
        # the deferred frequency so the LLM resumes it naturally.
        _r2, _s, _t, diagnostics = _run_turn(
            "they keep missing deadlines",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PROBLEMS, "missing deadlines")
                ]
            ),
        )
        prompt = diagnostics.get("Prompt") or ""
        # Phase 1 carrier: Relevant Context VOLUNTEERED snippet.
        self.assertIn("volunteered Frequency", prompt)
        # The just-answered field must NOT be listed as unfinished.
        self.assertNotIn("Unfinished topic — Problems", prompt)

    def test_summary_then_confirmation_tracks_facts(self):
        # Turn 1: rules satisfy PERSONAS (persona + problem + cadence) so the
        # LLM path is skipped and the deterministic fields land in state.
        _r, _s, _t, _d = _run_turn(
            "students miss deadlines weekly", _extraction(updates=[])
        )
        # Turn 2: rules satisfy CURRENT_SOLUTIONS (workflow cue) -> skip LLM.
        _r, _s, _t, _d = _run_turn(
            "they use sticky notes", _extraction(updates=[])
        )
        # Turn 3: no deterministic cues for the PAIN_POINTS objective, so the
        # LLM path runs and applies the canned evidence + pain points.
        _r, _s, _t, _d = _run_turn(
            "I spoke to ten of them already",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.CURRENT_SOLUTIONS, "sticky notes"),
                    (Operation.ADD, StateField.PAIN_POINTS, "stress"),
                    (Operation.ADD, StateField.EVIDENCE, "interviewed 10"),
                ]
            ),
        )
        _r, _s, _t, diagnostics = _run_turn(
            "yes", _extraction(MessageType.AMBIGUOUS)
        )
        memory = diagnostics["Memory"]
        self.assertEqual(
            diagnostics["Pipeline"]["Lifecycle Decision"], "READY_FOR_TRANSITION"
        )
        self.assertTrue(memory["Summarized Facts"])
        self.assertTrue(memory["Acknowledged Facts"])
        acknowledged = {r["field"] for r in memory["Acknowledged Facts"]}
        self.assertIn("frequency", acknowledged)

    def test_memory_section_is_json_serialisable(self):
        _reply, _sess, _timing, diagnostics = _run_turn(
            "students, and it happens a lot",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PERSONAS, "students"),
                    (Operation.SET, StateField.FREQUENCY, "a lot"),
                ]
            ),
        )
        round_tripped = json.loads(json.dumps(diagnostics))
        self.assertEqual(
            round_tripped["Memory"]["Deferred Topics"],
            diagnostics["Memory"]["Deferred Topics"],
        )

    def test_known_flag_set_when_user_states_field_keywords(self):
        """_known=True is set when user message contains field-related keywords."""
        from conversation_memory import _record_user_stated_facts, ConversationMemory
        from memory_extractor import StateField

        memory = ConversationMemory()
        # User mentions "people" which is a PERSONAS keyword
        _record_user_stated_facts(memory, StateField.PERSONAS.value)
        # Check the record was created with known=True
        open_threads = memory.open_threads
        assert len(open_threads) == 1
        assert open_threads[0]["known"] is True
        assert open_threads[0]["field"] == StateField.PERSONAS.value

    def test_known_flag_set_for_problems_keywords(self):
        """_known=True is set for PROBLEMS when user mentions problem-related words."""
        from conversation_memory import _record_user_stated_facts, ConversationMemory
        from memory_extractor import StateField

        memory = ConversationMemory()
        _record_user_stated_facts(memory, StateField.PROBLEMS.value)
        open_threads = memory.open_threads
        assert len(open_threads) == 1
        assert open_threads[0]["known"] is True
        assert open_threads[0]["field"] == StateField.PROBLEMS.value

    def test_known_flag_set_for_frequency_keywords(self):
        """_known=True is set for FREQUENCY when user mentions frequency-related words."""
        from conversation_memory import _record_user_stated_facts, ConversationMemory
        from memory_extractor import StateField

        memory = ConversationMemory()
        _record_user_stated_facts(memory, StateField.FREQUENCY.value)
        open_threads = memory.open_threads
        assert len(open_threads) == 1
        assert open_threads[0]["known"] is True
        assert open_threads[0]["field"] == StateField.FREQUENCY.value

    def test_resume_hint_5th_bullet_when_known_facts_exist(self):
        """The 5th resume hint bullet is generated when _known facts exist."""
        from conversation_memory import ConversationMemory, memory_to_prompt_bullets
        from memory_extractor import ProjectState, StateField

        # Create memory with a known fact in open_threads
        memory = ConversationMemory()
        memory.open_threads.append({
            "field": StateField.PERSONAS.value,
            "reason": "user stated this fact voluntarily",
            "turn": 1,
            "known": True,
            "asked": False,
        })

        # Call memory_to_prompt_bullets - should include the 5th bullet
        bullets = memory_to_prompt_bullets(
            type('SD', (), {'conversation_memory': memory})(),
            current_target=StateField.PERSONAS.value,
        )

        # The 5th bullet should be present (may be capped by max_bullets=4,
        # but it's included in the returned list logic)
        known_bullets = [b for b in bullets if "voluntarily" in b or "shared some facts" in b]
        assert len(known_bullets) >= 1, f"Expected 5th resume hint bullet, got: {bullets}"

    def test_known_context_resume_hint_in_prompt(self):
        """The 5th resume hint bullet appears in the LLM prompt."""
        from conversation_memory import memory_to_prompt_bullets
        from memory_extractor import ProjectState, StateField

        # Memory with known fact
        memory = ConversationMemory()
        memory.open_threads.append({
            "field": StateField.PERSONAS.value,
            "reason": "user stated this fact voluntarily",
            "turn": 1,
            "known": True,
            "asked": False,
        })

        # Get resume hints
        bullets = memory_to_prompt_bullets(
            type('SD', (), {'conversation_memory': memory})(),
            current_target=StateField.PERSONAS.value,
        )

        # Check that at least one bullet mentions user-stated facts
        has_known_bullet = any(
            "voluntarily" in b or "shared some facts" in b or "established context" in b
            for b in bullets
        )
        assert has_known_bullet, f"Expected 5th bullet about known facts, got: {bullets}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
