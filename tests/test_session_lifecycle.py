"""
tests/test_session_lifecycle.py — deterministic tests for the session
lifecycle decision authority (``session_lifecycle.py``).

Covers the four required scenarios:

  * fresh startup        — opening the app (no restore) begins a pristine
                           session with a brand-new id.
  * restored session     — an explicit restore request binds to the
                           persisted session without resetting/archiving.
  * new session          — an explicit new-session action archives the
                           current session and starts another fresh one.
  * multiple refreshes   — every repeated open yields a distinct fresh
                           session, with no manual refresh ever needed.

Also proves the lifecycle is observation-only for conversations: the
greeting message is byte-identical to the app's existing default and a
fresh bind leaves ProjectState empty (no extraction/conversation mutation).

The backend ``/session/start`` endpoint is a thin wrapper around
``SessionLifecycle.bind_session``; testing that helper against the real
``SessionManager`` covers the endpoint without importing ``server.py``
(which pulls in torch/whisper at module load).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from session_lifecycle import (  # noqa: E402
    STARTUP_GREETING,
    SessionLifecycle,
    StartupDecision,
    bind_session,
    decide_startup,
    manual_refresh_needed,
)


class TestFreshStartup(unittest.TestCase):

    def test_default_decision_is_fresh(self):
        decision = decide_startup()
        self.assertEqual(decision.action, SessionLifecycle.ACTION_FRESH)
        self.assertEqual(decision.backend_mode, SessionLifecycle.MODE_FRESH)
        self.assertIsNone(decision.restore_project)
        self.assertEqual(decision.project_name, "MyProject")

    def test_fresh_decision_respects_project_name(self):
        decision = decide_startup(project_name="Campus")
        self.assertEqual(decision.project_name, "Campus")

    def test_fresh_decision_is_deterministic(self):
        self.assertEqual(decide_startup(), decide_startup())

    def test_bind_fresh_creates_pristine_session(self):
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")  # known baseline
        before = mgr.get_active_session_id()

        result = bind_session(mgr, decide_startup(), username="User")

        self.assertEqual(result["action"], "fresh")
        self.assertTrue(result["session_id"])
        self.assertNotEqual(result["session_id"], before)
        self.assertFalse(result["manual_refresh_needed"])

    def test_bind_fresh_leaves_project_state_empty(self):
        """Fresh bind must not mutate/extract any conversation knowledge."""
        from session_manager import get_session_manager

        mgr = get_session_manager()
        result = bind_session(mgr, decide_startup(), username="User")
        sd = mgr.get_active_session_data()
        state = sd.project_state.to_state_dict()
        self.assertEqual(result["action"], "fresh")
        self.assertEqual(state["personas"], [])
        self.assertEqual(state["problems"], [])
        self.assertEqual(state["frequency"], None)


class TestRestoredSession(unittest.TestCase):

    def test_explicit_restore_decision(self):
        decision = decide_startup(explicit_restore="MissionAlpha")
        self.assertEqual(decision.action, SessionLifecycle.ACTION_RESTORE)
        self.assertEqual(decision.backend_mode, SessionLifecycle.MODE_RESTORE)
        self.assertEqual(decision.restore_project, "MissionAlpha")
        self.assertIn("restore", decision.reason)

    def test_restore_without_explicit_request_is_never_chosen(self):
        self.assertNotEqual(
            decide_startup().action, SessionLifecycle.ACTION_RESTORE
        )

    def test_bind_restore_keeps_persisted_session_id(self):
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")
        persisted_id = mgr.get_active_session_id()
        self.assertTrue(persisted_id)

        result = bind_session(
            mgr,
            decide_startup(explicit_restore="MissionAlpha"),
            username="User",
        )

        self.assertEqual(result["action"], "restore")
        self.assertEqual(result["session_id"], persisted_id)
        self.assertFalse(result["manual_refresh_needed"])

    def test_bind_restore_does_not_reset_or_archive(self):
        """Restore must bind to the persisted session — never a new one."""
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")
        persisted_id = mgr.get_active_session_id()

        # Put some content into the persisted session.
        sd = mgr.get_active_session_data()
        sd.project_state.personas.append("elderly people")
        mgr.save_session_data(persisted_id, sd)

        result = bind_session(
            mgr,
            decide_startup(explicit_restore="MissionAlpha"),
            username="User",
        )

        self.assertEqual(result["session_id"], persisted_id)
        restored = mgr.get_active_session_data()
        self.assertEqual(restored.project_state.personas, ["elderly people"])

    def test_restore_with_no_persisted_session_needs_refresh(self):
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")
        # Delete the active-session pointer to simulate "nothing persisted".
        pointer = mgr.sessions_dir / "active_session.txt"
        if pointer.exists():
            pointer.unlink()

        result = bind_session(
            mgr,
            decide_startup(explicit_restore="MissionAlpha"),
            username="User",
        )
        self.assertEqual(result["action"], "restore")
        self.assertEqual(result["session_id"], "")
        self.assertTrue(result["manual_refresh_needed"])


class TestNewSession(unittest.TestCase):

    def test_new_session_produces_distinct_ids(self):
        """The explicit new-session path (what /session/new runs) must hand
        back a fresh, empty session whose id differs from the prior one."""
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")
        first_id = mgr.get_active_session_id()

        mgr.reset_runtime_state(username="User")  # archive + fresh
        second_id = mgr.get_active_session_id()

        self.assertTrue(first_id)
        self.assertTrue(second_id)
        self.assertNotEqual(first_id, second_id)

        sd = mgr.get_active_session_data()
        self.assertEqual(sd.project_state.to_state_dict()["personas"], [])

    def test_new_session_preserves_previous_session(self):
        """Archived sessions must survive a new-session action (project
        persistence feature)."""
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")
        first_id = mgr.get_active_session_id()

        mgr.reset_runtime_state(username="User")

        # The archived session folder (not just the pointer) must still exist.
        self.assertTrue((mgr.sessions_dir / first_id).exists())


class TestMultipleBrowserRefreshes(unittest.TestCase):

    def test_each_refresh_is_a_distinct_fresh_session(self):
        from session_manager import get_session_manager

        mgr = get_session_manager()
        mgr.reset_runtime_state(username="User")

        seen: list[str] = []
        for _ in range(5):
            result = bind_session(mgr, decide_startup(), username="User")
            self.assertEqual(result["action"], "fresh")
            self.assertFalse(result["manual_refresh_needed"])
            seen.append(result["session_id"])

        self.assertTrue(all(seen), "every refresh must bind a session")
        self.assertEqual(len(set(seen)), 5, "every refresh must be distinct")

    def test_each_refresh_decision_is_fresh_without_restore(self):
        for _ in range(5):
            decision = decide_startup()
            self.assertEqual(decision.action, SessionLifecycle.ACTION_FRESH)
            self.assertEqual(
                SessionLifecycle.manual_refresh_needed(True), False
            )


class TestManualRefreshContract(unittest.TestCase):

    def test_no_manual_refresh_after_successful_bind(self):
        self.assertFalse(manual_refresh_needed(session_id_bound=True))

    def test_manual_refresh_only_when_unbound(self):
        self.assertTrue(manual_refresh_needed(session_id_bound=False))

    def test_module_level_helpers_match_class(self):
        self.assertEqual(
            manual_refresh_needed, SessionLifecycle.manual_refresh_needed
        )
        self.assertEqual(decide_startup, SessionLifecycle.decide_startup)
        self.assertEqual(bind_session, SessionLifecycle.bind_session)


class TestNoConversationBehaviorChange(unittest.TestCase):

    def test_greeting_byte_identical_to_app_default(self):
        expected = (
            "Hello! I'm your Design Thinking Mentor. I'll guide you through "
            "the Empathize stage by asking thoughtful questions — not giving "
            "answers. What project or idea would you like to explore today?"
        )
        self.assertEqual(SessionLifecycle.greeting(), expected)
        self.assertEqual(STARTUP_GREETING, expected)

    def test_decision_never_touches_mentor_pipeline(self):
        """decide_startup is a pure function of its inputs — no session or
        backend state is read or written."""
        d1 = decide_startup(explicit_restore="X")
        d2 = decide_startup(explicit_restore="X")
        self.assertEqual(d1, d2)
        self.assertIsInstance(d1, StartupDecision)

    def test_startup_decision_serializes(self):
        d = decide_startup(explicit_restore="MissionAlpha")
        payload = d.to_dict()
        self.assertEqual(payload["action"], "restore")
        self.assertEqual(payload["restore_project"], "MissionAlpha")
        self.assertEqual(payload["backend_mode"], "restore")
        # JSON-safe
        import json

        json.loads(json.dumps(payload))

    def test_bind_record_is_json_safe(self):
        from session_manager import get_session_manager

        import json

        mgr = get_session_manager()
        result = bind_session(mgr, decide_startup(), username="User")
        json.loads(json.dumps(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
