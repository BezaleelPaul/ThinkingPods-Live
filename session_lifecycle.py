"""
session_lifecycle.py — deterministic session-lifecycle decisions for the
Streamlit application startup.

Product rule (single source of truth, unit-tested here because the Streamlit
frontend itself is not covered by the suite):

    Opening the application creates a FRESH session unless the user
    explicitly requested to RESTORE a saved project.

The module is STRICTLY a decision + bind helper for the startup path:

  * ``SessionLifecycle.decide_startup`` decides *fresh* vs *restore* from a
    pure description of the page load (no Streamlit, no backend state).
  * ``SessionLifecycle.bind_session`` applies that decision to a
    ``SessionManager`` and returns the bound session record — so the same
    logic powers both the FastAPI ``/session/start`` endpoint and the
    deterministic backend tests, without ever importing ``server.py``.
  * ``SessionLifecycle.manual_refresh_needed`` encodes "No manual browser
    refresh should be required": a refresh is only needed when a session
    could not be bound (backend unreachable / no persisted session).

It NEVER:

  * changes conversation behavior, messages, ProjectState, objectives,
    extraction, prompts, lifecycle, or the Streamlit layout,
  * archives or deletes persisted sessions on the *restore* path (it binds
    to the persisted active session instead),
  * imports anything but the standard library, staying a cycle-free leaf.

Decisions returned by ``decide_startup``:

  * ``fresh``   — begin a pristine backend session (archive the current one,
                  start empty). Matches ``/session/start`` mode ``fresh``.
  * ``restore`` — bind to the persisted active session without resetting or
                  archiving. Matches ``/session/start`` mode ``restore``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "SessionLifecycle",
    "StartupDecision",
    "STARTUP_GREETING",
    "decide_startup",
    "bind_session",
    "manual_refresh_needed",
]

# The exact first message a fresh session shows. Kept byte-identical to the
# application's existing default greeting so conversation behavior is
# unchanged; this module only *documents* it for deterministic testing.
STARTUP_GREETING = (
    "Hello! I'm your Design Thinking Mentor. I'll guide you through the "
    "Empathize stage by asking thoughtful questions — not giving answers. "
    "What project or idea would you like to explore today?"
)


@dataclass(frozen=True)
class StartupDecision:
    """Immutable decision for one application startup.

    ``action`` is the human/UI-facing label; ``backend_mode`` is what the
    frontend sends to ``/session/start``. ``restore_project`` names the saved
    project a restore was explicitly requested for (``None`` on fresh).
    """

    action: str
    backend_mode: str
    project_name: str = "MyProject"
    restore_project: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "backend_mode": self.backend_mode,
            "project_name": self.project_name,
            "restore_project": self.restore_project,
            "reason": self.reason,
        }


class SessionLifecycle:
    """Deterministic session-lifecycle decision authority for the app."""

    ACTION_FRESH = "fresh"
    ACTION_RESTORE = "restore"

    MODE_FRESH = "fresh"
    MODE_RESTORE = "restore"

    @classmethod
    def decide_startup(
        cls,
        explicit_restore: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> StartupDecision:
        """Decide *fresh* vs *restore* for one page load.

        ``explicit_restore`` — the saved project the user asked to load.
        When provided the decision is ``restore`` (bind to the persisted
        session, never archive/reset). Otherwise it is ALWAYS ``fresh``:
        every application open without an explicit restore request begins a
        clean mentor session.
        """
        project_name = project_name or "MyProject"
        if explicit_restore:
            return StartupDecision(
                action=cls.ACTION_RESTORE,
                backend_mode=cls.MODE_RESTORE,
                project_name=project_name,
                restore_project=explicit_restore,
                reason="explicit restore requested — bind to persisted session",
            )
        return StartupDecision(
            action=cls.ACTION_FRESH,
            backend_mode=cls.MODE_FRESH,
            project_name=project_name,
            reason="fresh application open",
        )

    @staticmethod
    def manual_refresh_needed(session_id_bound: bool) -> bool:
        """Whether the frontend must ask the user to refresh the browser.

        A browser refresh is only ever required when a session could not be
        bound (backend unreachable). A successful fresh/restore bind needs
        none — the page reconciles itself.
        """
        return not session_id_bound

    @staticmethod
    def greeting() -> str:
        """The deterministic first assistant message for a fresh session.

        Identical to the application's existing default greeting.
        """
        return STARTUP_GREETING

    @staticmethod
    def bind_session(manager, decision: StartupDecision, username: str = "User") -> dict:
        """Apply a ``StartupDecision`` to a ``SessionManager``.

        * ``fresh``   — archive the current session (if any) and create a
                        pristine empty one; returns the new session id.
        * ``restore`` — bind to the persisted active session WITHOUT
                        resetting or archiving; returns its current id
                        (``""`` when no persisted session exists yet).

        Never raises for missing sessions. Returns a JSON-safe record:
        ``action``, ``backend_mode``, ``session_id``,
        ``manual_refresh_needed``.
        """
        if decision.backend_mode == SessionLifecycle.MODE_RESTORE:
            session_id = manager.get_active_session_id() or ""
        else:
            manager.reset_runtime_state(username=username)
            session_id = manager.get_active_session_id() or ""
        return {
            "action": decision.action,
            "backend_mode": decision.backend_mode,
            "session_id": session_id,
            "manual_refresh_needed": SessionLifecycle.manual_refresh_needed(
                bool(session_id)
            ),
        }


# Backwards-compatible module-level helpers (mirrors the audits' pattern of
# exposing both the class and a plain function).
decide_startup = SessionLifecycle.decide_startup
bind_session = SessionLifecycle.bind_session
manual_refresh_needed = SessionLifecycle.manual_refresh_needed
