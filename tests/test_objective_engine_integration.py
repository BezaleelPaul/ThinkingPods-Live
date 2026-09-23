"""
tests/test_objective_engine_integration.py

Integration tests for Module 4 — wiring the Objective Engine (Module 3) +
Response Strategy Engine + Prompt Builder (Module 4) into
mentor.process_mentor_turn, the live conversation pipeline.

These tests verify the Empathize v2 architecture requirement that the
application now uses Module 3 + Module 4 as the SINGLE authoritative planning
and response-shaping pipeline. They are NOT unit tests for Module 3 / 4 —
those modules' own behaviour is covered by tests/test_objective_engine.py /
test_priority_engine.py / test_completeness_checker.py /
test_response_strategy.py / test_prompt_builder.py. Here we assert the
integration contract:

  1. ObjectiveEngine.determine_next() is invoked during EVERY mentor turn
     (verified by observing the canonical [ObjectiveEngine] log line and by
     the resulting planning decision).
  2. ResponseStrategyEngine.determine_strategy() is invoked during EVERY
     mentor turn (verified by the canonical [ResponseStrategyEngine] log
     line and the resulting ResponseStrategy).
  3. QuestionPlanner is RETIRED. Its class is no longer importable from
     `mentor`, and no legacy prompt-builder path remains in the codebase.
  4. The chosen objective comes from Module 3 — empty state -> PERSONAS,
     one field missing -> that objective, all fields gathered -> WRAP_UP.
  5. WRAP_UP produces ResponseStrategy.GENERATE_SUMMARY, transitions the
     conversation OUT of the Empathize stage via the existing
     StageController mechanism (-> "Empathize_Complete"), and the
     deterministic fallback reply is a summary + confirmation question.
  6. Existing behaviour remains functional — the deterministic fallback
     reply path (used when the LLM is unreachable, e.g. in CI) still
     produces a well-formed, non-empty, single-question reply.

Differences from the existing unit-test suite:
  * Imports mentor.process_mentor_turn (the live pipeline) — not a stub.
  * Uses a dedicated audit user so the real `sessions/` directory's other
    JSON files are untouched. Temp projects are torn down per-test.
  * No Ollama required: the pipeline's existing try/except swallows LLM
    failures and falls back to deterministic templates built from Module 3 +
    Module 4 structured inputs (build_deterministic_fallback).
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState  # noqa: E402

# Importing mentor triggers Module 1/2/3/4 + the module3 / module4 packages.
# With the lazy `import ollama` change, no Ollama client installation is
# required.
import mentor  # noqa: E402
from mentor import (  # noqa: E402
    ChecklistManager,
    MentorSession,
    STAGE_EMPATHIZE,
    STAGE_EMPATHIZE_COMPLETE,
    _OBJECTIVE_ENGINE,
    _RESPONSE_STRATEGY_ENGINE,
    build_deterministic_fallback,
    process_mentor_turn,
)
from module3 import Objective, ObjectiveEngine  # noqa: E402
from module4 import (  # noqa: E402
    PromptBuilder as Module4PromptBuilder,
    ResponseStrategy,
    ResponseStrategyEngine,
    build_prompt as build_module4_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_full_session(username: str, project: str) -> None:
    """Pre-populate a session with all fields via the new SessionManager."""
    from session_manager import get_session_manager
    mgr = get_session_manager()
    mgr.reset_runtime_state(username=username, project_title=project)
    sd = mgr.get_active_session_data()
    sd.project_state = ProjectState(
        personas=["students"],
        problems=["buried messages"],
        current_solutions=["whatsapp groups"],
        pain_points=["personal connection"],
        evidence=["user interviews"],
        frequency="daily",
    )
    mgr.save_session_data(mgr.get_active_session_id(), sd)


def _clear_session(username: str, project: str) -> None:
    """Archive and clear the active session."""
    from session_manager import get_session_manager
    try:
        mgr = get_session_manager()
        mgr.reset_runtime_state()
    except Exception:
        pass


def _capture_log(stdout_lines: list, marker: str) -> list:
    """Return log lines starting with the given marker."""
    return [line for line in stdout_lines if line.startswith(marker)]


def _run_turn(message, username, project):
    """Run a mentor turn and capture stdout log lines."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        reply, session, _, _ = process_mentor_turn(
            message, username=username, project_name=project
        )
    return reply, session, buf.getvalue().splitlines()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestModule4Integration(unittest.TestCase):
    """Module 3 + Module 4 drive every mentor turn."""

    AUDIT_USER = "audit_integration"
    AUDIT_PROJ = "audit_project"

    def setUp(self):
        # Isolate each test from the real sessions/ dir by removing this
        # test's expected file path before each run.
        _clear_session(self.AUDIT_USER, self.AUDIT_PROJ)

    def tearDown(self):
        _clear_session(self.AUDIT_USER, self.AUDIT_PROJ)

    # ---- 1. Module 3 is invoked on every turn -----------------------------

    def test_objective_engine_is_a_singleton_reused_across_turns(self):
        """The pipeline-level ObjectiveEngine is process-wide and stable."""
        self.assertIsInstance(_OBJECTIVE_ENGINE, ObjectiveEngine)
        self.assertIs(mentor._OBJECTIVE_ENGINE, _OBJECTIVE_ENGINE)

    def test_response_strategy_engine_is_a_singleton_reused_across_turns(self):
        """The pipeline-level ResponseStrategyEngine is process-wide and stable."""
        self.assertIsInstance(_RESPONSE_STRATEGY_ENGINE, ResponseStrategyEngine)
        self.assertIs(mentor._RESPONSE_STRATEGY_ENGINE, _RESPONSE_STRATEGY_ENGINE)

    def test_empty_state_yields_personas_objective_and_ask_question_strategy(self):
        """Empty state -> Module 3 returns PERSONAS; Module 4 returns ASK_QUESTION."""
        reply, session, log = _run_turn(
            "hello", self.AUDIT_USER, self.AUDIT_PROJ
        )
        # Stage is still Empathize (nothing gathered yet).
        self.assertEqual(session.current_stage,
                         STAGE_EMPATHIZE)
        # Module 3 invoked.
        obj_logs = _capture_log(log, "[ObjectiveEngine]")
        self.assertTrue(obj_logs)
        self.assertIn("objective=PERSONAS", obj_logs[-1])
        # Module 4 invoked.
        strat_logs = _capture_log(log, "[ResponseStrategyEngine]")
        self.assertTrue(strat_logs)
        self.assertIn("strategy=ASK_QUESTION", strat_logs[-1])
        # Fallback reply (LLM unreachable in CI) is the PERSONAS question.
        self.assertTrue(reply and len(reply.strip()) > 5)
        self.assertIn("benefit", reply.lower())

    def test_engine_invoked_every_turn_with_continuity(self):
        """Running multiple turns keeps Module 3 + Module 4 active throughout."""
        for _ in range(3):
            reply, _, log = _run_turn(
                "not sure", self.AUDIT_USER, self.AUDIT_PROJ
            )
            self.assertTrue(reply and reply.strip())
            self.assertTrue(_capture_log(log, "[ObjectiveEngine]"))
            self.assertTrue(_capture_log(log, "[ResponseStrategyEngine]"))

    # ---- 2. QuestionPlanner is retired -----------------------------------

    def test_question_planner_class_is_not_importable_from_mentor(self):
        """Module 4 has removed QuestionPlanner from the codebase."""
        self.assertFalse(hasattr(mentor, "QuestionPlanner"))
        self.assertFalse(hasattr(mentor, "FALLBACK_QUESTIONS"))
        self.assertFalse(hasattr(mentor, "objective_to_legacy_planning"))
        self.assertFalse(hasattr(mentor, "_STATEFIELD_TO_LEGACY_KEY"))
        self.assertFalse(hasattr(mentor, "_STATEFIELD_ACTION_DESC"))

    def test_no_legacy_prompt_builder_class_in_mentor(self):
        """The legacy `mentor.PromptBuilder` class is gone; only the module4
        one survives. The PromptBuilder contract lives in module4."""
        # mentor.PromptBuilder is no longer defined in mentor.py.
        pb = getattr(mentor, "PromptBuilder", None)
        self.assertIsNone(pb)
        # The Module 4 PromptBuilder is the only one available.
        self.assertTrue(hasattr(Module4PromptBuilder, "build_prompt"))

    # ---- 3. Chosen objective comes from Module 3 -------------------------

    def test_objective_is_module3_output_for_partial_state(self):
        """Partial state -> Module 3 chooses the highest-priority missing
        field; Module 4 returns ASK_QUESTION for it."""
        from session_manager import get_session_manager
        mgr = get_session_manager()
        mgr.reset_runtime_state(username=self.AUDIT_USER, project_title=self.AUDIT_PROJ)
        sd = mgr.get_active_session_data()
        sd.project_state = ProjectState(personas=["students"])
        mgr.save_session_data(mgr.get_active_session_id(), sd)

        reply, sess, log = _run_turn(
            "what else", self.AUDIT_USER, self.AUDIT_PROJ
        )
        # The pipeline populated project_state.personas = ['students'],
        # so Module 3 must target PROBLEMS next (priority 90, next in order).
        from session_manager import get_session_manager
        ps = get_session_manager().get_active_session_data().project_state
        obj = _OBJECTIVE_ENGINE.determine_next(ps)
        self.assertEqual(obj.objective, Objective.PROBLEMS)
        self.assertIn("problems", [f.value for f in obj.missing_fields])
        # Module 4 maps PROBLEMS to ASK_QUESTION deterministically.
        strat = _RESPONSE_STRATEGY_ENGINE.determine_strategy(obj, ps)
        self.assertEqual(strat, ResponseStrategy.ASK_QUESTION)
        # And the live pipeline emitted the matching log lines.
        self.assertIn("objective=PROBLEMS",
                      " ".join(_capture_log(log, "[ObjectiveEngine]")))

    # ---- 4. WRAP_UP transitions the stage AND maps to GENERATE_SUMMARY ----

    def test_wrap_up_yields_generate_summary_strategy(self):
        """WRAP_UP Objective maps to GENERATE_SUMMARY ResponseStrategy."""
        full_ps = ProjectState(
            personas=["x"], problems=["y"], current_solutions=["z"],
            pain_points=["q"], evidence=["w"], frequency="daily",
        )
        obj = _OBJECTIVE_ENGINE.determine_next(full_ps)
        self.assertTrue(obj.is_wrap_up)
        strat = _RESPONSE_STRATEGY_ENGINE.determine_strategy(obj, full_ps)
        self.assertEqual(strat, ResponseStrategy.GENERATE_SUMMARY)

    def test_wrap_up_transitions_out_of_empathize_with_summary_strategy(self):
        """A full-state WRAP_UP turn yields READY_FOR_SUMMARY, NOT immediately
        EMPATHIZE_COMPLETE. The Module 5 lifecycle requires explicit user
        confirmation: the first WRAP_UP turn produces a SUMMARY + stays in
        EMPATHIZE; only READY_FOR_TRANSITION (user says yes) flips the stage."""
        _seed_full_session(self.AUDIT_USER, self.AUDIT_PROJ)

        from session_manager import get_session_manager
        pre_stage = get_session_manager().get_active_session_data().current_stage
        self.assertEqual(pre_stage, STAGE_EMPATHIZE)

        reply, session, log = _run_turn(
            "let us wrap up", self.AUDIT_USER, self.AUDIT_PROJ
        )

        # Module 5: first WRAP_UP → READY_FOR_SUMMARY; stage stays EMPATHIZE
        # until the user explicitly confirms. StageController only flips on
        # READY_FOR_TRANSITION.
        lifecycle_log = " ".join(log)
        self.assertIn("decision=READY_FOR_SUMMARY", lifecycle_log)
        self.assertEqual(session.current_stage,
                         STAGE_EMPATHIZE)
        # The strategy log for this turn still reads GENERATE_SUMMARY
        # (the integrator overrode the module4 candidate for the prompt).
        self.assertIn("strategy=GENERATE_SUMMARY",
                      " ".join(_capture_log(log, "[ResponseStrategyEngine]")))
        # Reply is the deterministic empathy-summary fallback (non-empty,
        # single-question tail).
        self.assertTrue(reply and reply.strip().endswith("?"))

        # --- Second turn: user explicitly confirms the summary → flips stage -
        reply2, session2, log2 = _run_turn(
            "yes", self.AUDIT_USER, self.AUDIT_PROJ
        )
        self.assertIn("decision=READY_FOR_TRANSITION",
                      " ".join(_capture_log(log2, "[LifecycleManager]")))
        self.assertEqual(session2.current_stage,
                         STAGE_EMPATHIZE_COMPLETE)
        self.assertNotEqual(session2.current_stage, STAGE_EMPATHIZE)

    def test_wrap_up_does_not_reopen_empathize_on_followup(self):
        """After the user confirms the summary (READY_FOR_TRANSITION),
        the stage stays at EMPATHIZE_COMPLETE — it does NOT reopen."""
        _seed_full_session(self.AUDIT_USER, self.AUDIT_PROJ)

        # First turn: full state, READY_FOR_SUMMARY, summary reply emitted.
        _run_turn("wrap up", self.AUDIT_USER, self.AUDIT_PROJ)

        # Second turn: user explicitly confirms.
        _, session_confirmed, _ = _run_turn(
            "yes", self.AUDIT_USER, self.AUDIT_PROJ
        )
        self.assertEqual(session_confirmed.current_stage,
                         STAGE_EMPATHIZE_COMPLETE)

        # Third turn: the stage stays closed — no reopening Empathize.
        _, session_after, _ = _run_turn(
            "anything else", self.AUDIT_USER, self.AUDIT_PROJ
        )
        self.assertEqual(session_after.current_stage,
                         STAGE_EMPATHIZE_COMPLETE)

    # ---- 5. Deterministic fallback still functional ----------------------

    def test_deterministic_fallback_ask_question_for_empty_state(self):
        """build_deterministic_fallback returns a single question for an
        ASK_QUESTION objective on an empty ProjectState."""
        ps = ProjectState()
        obj = _OBJECTIVE_ENGINE.determine_next(ps)
        strat = _RESPONSE_STRATEGY_ENGINE.determine_strategy(obj, ps)
        self.assertEqual(strat, ResponseStrategy.ASK_QUESTION)
        reply = build_deterministic_fallback(obj, strat, ps)
        self.assertTrue(reply and "?" in reply)

    def test_deterministic_fallback_summary_for_full_state(self):
        """build_deterministic_fallback returns a summary + confirmation
        question for a GENERATE_SUMMARY objective."""
        ps = ProjectState(
            personas=["students"], problems=["buried chats"],
            pain_points=["personal connection"],
        )
        obj = _OBJECTIVE_ENGINE.determine_next(ps)
        # Cannot use this state: it's incomplete, so override for the test
        # by building a complete state.
        ps_full = ProjectState(
            personas=["students"], problems=["buried chats"],
            current_solutions=["whatsapp"], pain_points=["personal connection"],
            evidence=["interviews"], frequency="daily",
        )
        obj = _OBJECTIVE_ENGINE.determine_next(ps_full)
        strat = _RESPONSE_STRATEGY_ENGINE.determine_strategy(obj, ps_full)
        self.assertEqual(strat, ResponseStrategy.GENERATE_SUMMARY)
        reply = build_deterministic_fallback(obj, strat, ps_full)
        self.assertTrue(reply and "?" in reply)
        self.assertIn("students", reply.lower())

    def test_fallback_reply_is_non_empty_without_ollama(self):
        """With Ollama unreachable (CI / offline), the deterministic
        fallback still produces a non-empty, one-question reply — same
        contract as the legacy pipeline but built from Module 3 + 4 inputs."""
        reply, _sess, _log = _run_turn(
            "i don't know", "audit_fallback", "audit_fb"
        )
        _clear_session("audit_fallback", "audit_fb")
        self.assertTrue(reply and len(reply.strip()) > 4)
        self.assertIn("?", reply)

    # ---- 6. Module 4 is the only prompt-construction path ----------------

    def test_module4_build_prompt_is_the_single_prompt_path(self):
        """The mentor module publishes build_module4_prompt (an alias of
        module4.PromptBuilder.build_prompt) — there should be no other
        build_prompt-style callable that produces a mentor LLM prompt."""
        self.assertTrue(callable(mentor.build_module4_prompt))
        self.assertIs(mentor.build_module4_prompt, build_module4_prompt)
        # Sanity: build_prompt produces a structured, five-section prompt.
        ps = ProjectState()
        obj = _OBJECTIVE_ENGINE.determine_next(ps)
        strat = _RESPONSE_STRATEGY_ENGINE.determine_strategy(obj, ps)
        prompt = build_module4_prompt(
            project_state=ps,
            conversation_objective=obj,
            response_strategy=strat,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        for section in ("Role", "Current Objective", "Known Project State",
                        "Latest Conversation", "Instructions"):
            self.assertIn(section + "\n--------------------------------", prompt)

    # ---- 7. ChecklistManager retained for dashboard (out-of-pipeline) ----

    def test_checklist_manager_still_importable_for_dashboard(self):
        """ChecklistManager is retained as a read-only Empathize-coverage
        view for the dashboard / server status endpoint; it must NOT be in
        the mentor decision pipe."""
        self.assertTrue(hasattr(ChecklistManager, "get_status_dict"))
        ps = ProjectState()
        status = ChecklistManager.get_status_dict(ps)
        self.assertEqual(len(status), len(ChecklistManager._FIELD_MAP))


# ---------------------------------------------------------------------------
# Module 3.5 regression guard — prompts decoded through Module 4 alone
# ---------------------------------------------------------------------------


class TestPipelineDeterministicOffline(unittest.TestCase):
    """Module 3 + 4 + deterministic fallback form a fully deterministic
    pipeline when the LLM is unreachable. The same inputs produce the
    same reply across multiple cold process pipelines."""

    AUDIT_USER = "audit_determinism"
    AUDIT_PROJ = "audit_determinism"

    def setUp(self):
        _clear_session(self.AUDIT_USER, self.AUDIT_PROJ)

    def tearDown(self):
        _clear_session(self.AUDIT_USER, self.AUDIT_PROJ)

    def test_same_empty_state_first_two_turns_produce_stable_replies(self):
        """Driving two separate fresh sessions through the same 'hello'
        first turn gives identical deterministic fallback replies."""
        user, proj = self.AUDIT_USER + "_a", self.AUDIT_PROJ + "_a"
        try:
            _clear_session(user, proj)
            reply_a, _sa, _, _ = process_mentor_turn(
                "hello", username=user, project_name=proj
            )
            _clear_session(user, proj)
            reply_b, _sb, _, _ = process_mentor_turn(
                "hello", username=user, project_name=proj
            )
            self.assertEqual(reply_a, reply_b)
            self.assertTrue(reply_a.strip())
        finally:
            _clear_session(user, proj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
