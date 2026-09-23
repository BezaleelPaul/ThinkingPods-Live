"""
session_manager.py — Session Management for Design Thinking Mentor

Architecture role
-----------------
Frontend → POST /session/new → SessionManager → New Session Folder
                              → Initialize Empty ProjectState / Conversation History
                              → Return session_id

The SessionManager is OUTSIDE the mentoring pipeline. It manages session lifecycle
and provides the active ProjectState to the pipeline.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from memory_extractor import ProjectState
from conversation_memory import ConversationMemory
from hybrid_audit import HybridAuditSummary
from extraction_accuracy import ExtractionAccuracySummary
from mentor_decision_audit import MentorDecisionSummary
from conversation_style_audit import ConversationStyleSummary
from product_experience_audit import ProductExperienceSummary
from conversation_failure_audit import ConversationFailureSummary


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionMetadata:
    """Metadata for a single mentoring session."""
    session_id: str
    created_at: str  # ISO 8601 UTC
    status: str      # "active" | "archived" | "completed"
    phase: str       # "Empathize" | "Define" | "Ideate" | ...
    project_title: Optional[str] = None
    username: str = "User"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "status": self.status,
            "phase": self.phase,
            "project_title": self.project_title,
            "username": self.username,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionMetadata":
        return cls(
            session_id=d["session_id"],
            created_at=d["created_at"],
            status=d["status"],
            phase=d["phase"],
            project_title=d.get("project_title"),
            username=d.get("username", "User"),
        )


@dataclass
class SessionData:
    """All persisted data for a session."""
    project_state: ProjectState = field(default_factory=ProjectState)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    inference_history: list[dict[str, Any]] = field(default_factory=list)
    extraction_history: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_history: list[dict[str, Any]] = field(default_factory=list)
    # Migrated from MentorSession (PR14, PR16)
    project_name: str = "MyProject"
    current_stage: str = "Empathize"
    previous_questions: list[str] = field(default_factory=list)
    # Semantic families of questions the mentor already asked, in the order
    # they were asked (values of module3.question_families.QuestionFamily).
    # Consumed by the QuestionFamilyPlanner so the mentor avoids re-asking
    # from an already-covered family unless the previous answers were
    # insufficient.
    asked_question_families: list[str] = field(default_factory=list)
    unknown_facts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    empathize_summary_presented: bool = False
    previous_lifecycle_decision: Optional[str] = None
    # Developer-only observation log: one per-turn metrics record (see
    # conversation_metrics.record_turn). Append-only; NEVER read by any
    # decision path — purely for Developer Console quality metrics.
    turn_metrics: list[dict[str, Any]] = field(default_factory=list)
    # Conversational-context memory: open/resolved threads, deferred topics,
    # acknowledged & summarized facts, partially answered objectives. The
    # single memory home beyond the raw conversation history; persists with
    # the session. ProjectState remains the canonical project-knowledge store.
    conversation_memory: ConversationMemory = field(default_factory=ConversationMemory)
    # Measurement-only shadow-audit aggregate for the hybrid extraction
    # decision (see hybrid_audit.py). NEVER read by any decision path — it
    # exists solely for the Developer Console "Hybrid Summary" block.
    hybrid_audit_summary: "HybridAuditSummary" = field(default_factory=HybridAuditSummary)
    # Measurement-only accuracy audit aggregate for LLM-extracted updates
    # (see extraction_accuracy.py). NEVER read by any decision path — it
    # exists solely for the Developer Console "Extraction Accuracy Summary".
    extraction_accuracy_summary: "ExtractionAccuracySummary" = field(default_factory=ExtractionAccuracySummary)
    # Measurement-only audit aggregate for the mentor's conversational
    # decisions (objective appropriateness, continuity, question quality).
    # NEVER read by any decision path — it exists solely for the Developer
    # Console "Mentor Decision" block and the end-of-run report.
    mentor_decision_summary: "MentorDecisionSummary" = field(default_factory=MentorDecisionSummary)
    # Measurement-only audit aggregate for the mentor's reply *style*
    # (opening, length, acknowledgment, observation, bridge, question
    # count, summary markers, references to prior info, topic restart).
    # NEVER read by any decision path — it exists solely for the Developer
    # Console "Conversation Style" block and the end-of-run report.
    conversation_style_summary: "ConversationStyleSummary" = field(default_factory=ConversationStyleSummary)
    # Measurement-only audit aggregate for the end-to-end product experience
    # (startup timing, session restoration, Developer Console shape/cost,
    # export size, performance timeline). NEVER read by any decision path —
    # exists solely for the Developer Console "Product Experience" block.
    product_experience_summary: "ProductExperienceSummary" = field(default_factory=ProductExperienceSummary)
    # Measurement-only observation audit aggregate for *failed conversations*
    # (meta intents, re-asking resolved fields, abrupt transitions,
    # over-exploration, unsupported inferences, memory inconsistencies,
    # social slips, move/reply mismatches). NEVER read by any decision path —
    # exists solely for the Developer Console "Conversation Failure Summary"
    # block and the end-of-run report.
    conversation_failure_summary: "ConversationFailureSummary" = field(default_factory=ConversationFailureSummary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_state": self.project_state.__dict__,
            "conversation_history": self.conversation_history,
            "hypotheses": self.hypotheses,
            "inference_history": self.inference_history,
            "extraction_history": self.extraction_history,
            "lifecycle_history": self.lifecycle_history,
            "project_name": self.project_name,
            "current_stage": self.current_stage,
            "previous_questions": self.previous_questions,
            "asked_question_families": self.asked_question_families,
            "unknown_facts": self.unknown_facts,
            "assumptions": self.assumptions,
            "open_questions": self.open_questions,
            "empathize_summary_presented": self.empathize_summary_presented,
            "previous_lifecycle_decision": self.previous_lifecycle_decision,
            "turn_metrics": self.turn_metrics,
            "conversation_memory": self.conversation_memory.to_dict(),
            "hybrid_audit_summary": self.hybrid_audit_summary.to_dict(),
            "extraction_accuracy_summary": self.extraction_accuracy_summary.to_dict(),
            "mentor_decision_summary": self.mentor_decision_summary.to_dict(),
            "conversation_style_summary": self.conversation_style_summary.to_dict(),
            "product_experience_summary": self.product_experience_summary.to_dict(),
            "conversation_failure_summary": self.conversation_failure_summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionData":
        ps_data = d.get("project_state", {})
        project_state = ProjectState(
            personas=ps_data.get("personas", []),
            problems=ps_data.get("problems", []),
            current_solutions=ps_data.get("current_solutions", []),
            pain_points=ps_data.get("pain_points", []),
            evidence=ps_data.get("evidence", []),
            impacts=ps_data.get("impacts", []),
            frequency=ps_data.get("frequency"),
            previous_assistant_message=ps_data.get("previous_assistant_message"),
            previous_user_message=ps_data.get("previous_user_message"),
        )
        return cls(
            project_state=project_state,
            conversation_history=d.get("conversation_history", []),
            hypotheses=d.get("hypotheses", []),
            inference_history=d.get("inference_history", []),
            extraction_history=d.get("extraction_history", []),
            lifecycle_history=d.get("lifecycle_history", []),
            project_name=d.get("project_name", "MyProject"),
            current_stage=d.get("current_stage", "Empathize"),
            previous_questions=d.get("previous_questions", []),
            asked_question_families=d.get("asked_question_families", []),
            unknown_facts=d.get("unknown_facts", []),
            assumptions=d.get("assumptions", []),
            open_questions=d.get("open_questions", []),
            empathize_summary_presented=d.get("empathize_summary_presented", False),
            previous_lifecycle_decision=d.get("previous_lifecycle_decision"),
            turn_metrics=d.get("turn_metrics", []),
            conversation_memory=ConversationMemory.from_dict(
                d.get("conversation_memory", {})
            ),
            hybrid_audit_summary=HybridAuditSummary.from_dict(
                d.get("hybrid_audit_summary", {})
            ),
            extraction_accuracy_summary=ExtractionAccuracySummary.from_dict(
                d.get("extraction_accuracy_summary", {})
            ),
            mentor_decision_summary=MentorDecisionSummary.from_dict(
                d.get("mentor_decision_summary", {})
            ),
            conversation_style_summary=ConversationStyleSummary.from_dict(
                d.get("conversation_style_summary", {})
            ),
            product_experience_summary=ProductExperienceSummary.from_dict(
                d.get("product_experience_summary", {})
            ),
            conversation_failure_summary=ConversationFailureSummary.from_dict(
                d.get("conversation_failure_summary", {})
            ),
        )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _get_session_dir(session_id: str) -> Path:
    """Get the directory path for a session."""
    return SESSIONS_DIR / session_id


def _read_active_session_id() -> Optional[str]:
    """Read the active session ID from the pointer file."""
    pointer_file = SESSIONS_DIR / "active_session.txt"
    if pointer_file.exists():
        try:
            session_id = pointer_file.read_text().strip()
            if session_id and _get_session_dir(session_id).exists():
                return session_id
        except Exception:
            pass
    return None


def _write_active_session_id(session_id: str) -> None:
    """Write the active session ID to the pointer file."""
    pointer_file = SESSIONS_DIR / "active_session.txt"
    if session_id:
        pointer_file.write_text(session_id)
    elif pointer_file.exists():
        pointer_file.unlink()


def _generate_session_id() -> str:
    """Generate a unique session ID: 20260726T143215Z_8f3a7c"""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


# ---------------------------------------------------------------------------
# SessionManager Class
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages mentoring session lifecycle.

    Responsibilities:
    - Create new sessions with timestamped folders
    - Load/switch active sessions
    - Archive sessions (preserve for debugging)
    - Reset runtime state for new sessions
    - List available sessions

    Non-responsibilities (kept in mentoring pipeline):
    - Memory extraction
    - Inference
    - Objective selection
    - Response strategy
    - Prompt building
    """

    def __init__(self, sessions_dir: Path = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Session Creation
    # ---------------------------------------------------------------------

    def create_session(
        self,
        username: str = "User",
        project_title: Optional[str] = None,
        phase: str = "Empathize",
    ) -> SessionMetadata:
        """
        Create a completely new mentoring session.

        Steps:
        1. Generate unique session_id (timestamp + short UUID)
        2. Create session folder
        3. Initialize empty ProjectState, conversation history, etc.
        4. Write session.json metadata
        5. Set as active session

        Returns:
            SessionMetadata for the new session.
        """
        session_id = _generate_session_id()
        session_dir = _get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty session data
        session_data = SessionData()
        self._write_session_data(session_dir, session_data)

        # Write metadata
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            status="active",
            phase=phase,
            project_title=project_title,
            username=username,
        )
        self._write_metadata(session_dir, metadata)

        # Set as active
        _write_active_session_id(session_id)

        return metadata

    # ---------------------------------------------------------------------
    # Active Session Management
    # ---------------------------------------------------------------------

    def get_active_session_id(self) -> Optional[str]:
        """Return the currently active session ID, or None if none exists."""
        return _read_active_session_id()

    def get_active_project_state(self) -> Optional[ProjectState]:
        """Return the active session's ProjectState, or None if no session."""
        session_id = self.get_active_session_id()
        if session_id is None:
            return None
        session_dir = _get_session_dir(session_id)
        data = self._read_session_data(session_dir)
        return data.project_state

    def get_active_session_data(self) -> SessionData:
        """Load the active session's data, creating new if none exists."""
        session_id = self.get_active_session_id()
        if session_id is None:
            # No session exists, create one
            self.create_session()
            session_id = self.get_active_session_id()

        session_dir = _get_session_dir(session_id)
        return self._read_session_data(session_dir)

    def get_active_metadata(self) -> Optional[SessionMetadata]:
        """Load the active session's metadata."""
        session_id = self.get_active_session_id()
        if session_id is None:
            return None
        session_dir = _get_session_dir(session_id)
        return self._read_metadata(session_dir)

    def switch_session(self, session_id: str) -> bool:
        """
        Switch the active session to a different one.

        Archives the current active session (if any) and activates the new one.
        Returns True if successful.
        """
        target_dir = _get_session_dir(session_id)
        if not target_dir.exists():
            return False

        # Archive current active session
        current_id = self.get_active_session_id()
        if current_id and current_id != session_id:
            self._archive_session(current_id)

        # Activate target session
        self._set_session_status(session_id, "active")
        _write_active_session_id(session_id)
        return True

    def reset_runtime_state(
        self,
        username: str = "User",
        project_title: Optional[str] = None,
        phase: str = "Empathize",
    ) -> SessionMetadata:
        """
        Reset all runtime state for a completely fresh session.

        This is called when user clicks "New Session" in the frontend.
        Archives the current session and creates a brand new empty one.
        """
        current_id = self.get_active_session_id()
        if current_id:
            self._archive_session(current_id)

        # Create new session
        return self.create_session(username=username, project_title=project_title, phase=phase)

    # ---------------------------------------------------------------------
    # Session Persistence
    # ---------------------------------------------------------------------

    def save_session_data(self, session_id: str, data: SessionData) -> bool:
        """Persist session data to disk."""
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        self._write_session_data(session_dir, data)
        return True

    def save_project_state(self, session_id: str, project_state: ProjectState) -> bool:
        """Update just the ProjectState for a session."""
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        data = self._read_session_data(session_dir)
        data.project_state = project_state
        self._write_session_data(session_dir, data)
        return True

    def append_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> bool:
        """Append a turn to conversation history."""
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        data = self._read_session_data(session_dir)
        data.conversation_history.append({"role": role, "content": content})
        self._write_session_data(session_dir, data)
        return True

    def append_hypotheses(self, session_id: str, hypotheses: list[dict]) -> bool:
        """Append hypotheses to session."""
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        data = self._read_session_data(session_dir)
        data.hypotheses.extend(hypotheses)
        self._write_session_data(session_dir, data)
        return True

    # ---------------------------------------------------------------------
    # Session Listing & Archival
    # ---------------------------------------------------------------------

    def list_sessions(self) -> list[SessionMetadata]:
        """List all sessions, most recent first."""
        sessions = []
        for item in self.sessions_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                meta = self._read_metadata(item)
                if meta:
                    sessions.append(meta)
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def archive_current_session(self) -> bool:
        """Archive the currently active session."""
        session_id = self.get_active_session_id()
        if session_id:
            return self._archive_session(session_id)
        return False

    def _archive_session(self, session_id: str) -> bool:
        """Mark a session as archived."""
        return self._set_session_status(session_id, "archived")

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session folder (use with caution)."""
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        try:
            shutil.rmtree(session_dir)
            # If we deleted the active session, clear pointer
            if self.get_active_session_id() == session_id:
                _write_active_session_id("")
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Legacy Compatibility (for mentor.py SessionManager)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------

    def _write_metadata(self, session_dir: Path, metadata: SessionMetadata) -> None:
        meta_file = session_dir / "session.json"
        meta_file.write_text(json.dumps(metadata.to_dict(), indent=2))

    def _read_metadata(self, session_dir: Path) -> Optional[SessionMetadata]:
        meta_file = session_dir / "session.json"
        if not meta_file.exists():
            return None
        try:
            return SessionMetadata.from_dict(json.loads(meta_file.read_text()))
        except Exception:
            return None

    def _write_session_data(self, session_dir: Path, data: SessionData) -> None:
        data_file = session_dir / "session_data.json"
        data_file.write_text(json.dumps(data.to_dict(), indent=2))

    def _read_session_data(self, session_dir: Path) -> SessionData:
        data_file = session_dir / "session_data.json"
        if not data_file.exists():
            return SessionData()
        try:
            return SessionData.from_dict(json.loads(data_file.read_text()))
        except Exception:
            return SessionData()

    def _set_session_status(self, session_id: str, status: str) -> bool:
        session_dir = _get_session_dir(session_id)
        if not session_dir.exists():
            return False
        meta = self._read_metadata(session_dir)
        if not meta:
            return False
        meta = SessionMetadata(
            session_id=meta.session_id,
            created_at=meta.created_at,
            status=status,
            phase=meta.phase,
            project_title=meta.project_title,
            username=meta.username,
        )
        self._write_metadata(session_dir, meta)
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SESSION_MANAGER: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return the process-wide SessionManager instance."""
    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        _SESSION_MANAGER = SessionManager()
    return _SESSION_MANAGER