"""
Session pipeline — loading, MutatingSessionData, persistence, MentorSession bridge.

Responsibilities:
  * `MentorSession` — legacy runtime-metadata container (read-only dashboard view)
  * `_project_state_to_legacy_session`, `_build_legacy_session`, `build_mentor_session`
    — MentorSession construction from canonical ProjectState + SessionData
  * `_build_known_facts` — one-way projection of ProjectState into display strings
  * `_prepare_turn` — Stage-1 pipeline helper (load session, append user message)
  * `_finalize_session` — Stage-8 pipeline helper (persist, build return container)

Owns NO LLM interaction, NO extraction logic, NO decision-making.
"""

__all__ = [
    "MentorSession",
    "build_mentor_session",
    "_prepare_turn",
    "_finalize_session",
]

from memory_extractor import ProjectState
from session_manager import SessionData, get_session_manager
from module3 import classify_question


class MentorSession:
    """Runtime metadata only. No persistence mirrors."""

    def __init__(self, project_name="MyProject"):
        self.project_name = project_name
        self.current_stage = "Empathize"
        self.known_facts = []
        self.assumptions = []
        self.unknown_facts = []
        self.open_questions = []


def _build_known_facts(project_state: ProjectState, project_name: str) -> list[str]:
    facts = []
    mapping = [
        ("Project Name", project_name if project_name != "MyProject" else None),
        ("Target Audience", project_state.personas[0] if project_state.personas else None),
        ("Pain Point", project_state.problems[0] if project_state.problems else None),
        ("Motivation", project_state.pain_points[0] if project_state.pain_points else None),
        ("Existing Solution/Workflow", project_state.current_solutions[0] if project_state.current_solutions else None),
        ("Frequency", project_state.frequency),
        ("Evidence", project_state.evidence[0] if project_state.evidence else None),
        ("Impacts", project_state.impacts[0] if project_state.impacts else None),
    ]
    for label, val in mapping:
        if val:
            facts.append(f"{label}: {val}")
    return facts


def _project_state_to_legacy_session(project_state: ProjectState, project_name: str) -> "MentorSession":
    session = MentorSession(project_name)
    session.known_facts = _build_known_facts(project_state, project_name)
    return session


def _build_legacy_session(project_state: ProjectState, session_data: SessionData, project_name: str) -> MentorSession:
    session = _project_state_to_legacy_session(project_state, project_name)
    if session_data.project_name and session_data.project_name != "MyProject":
        session.project_name = session_data.project_name
    session.current_stage = session_data.current_stage
    session.assumptions = list(session_data.assumptions)
    session.unknown_facts = list(session_data.unknown_facts)
    session.open_questions = list(session_data.open_questions)
    return session


def build_mentor_session(project_name: str = "MyProject") -> MentorSession:
    mgr = get_session_manager()
    sd = mgr.get_active_session_data()
    return _build_legacy_session(sd.project_state, sd, project_name)


def _prepare_turn(user_message, storage_project_name):
    new_session_mgr = get_session_manager()
    session_data = new_session_mgr.get_active_session_data()
    project_state = session_data.project_state
    session_id = new_session_mgr.get_active_session_id()

    last_assistant_msg: str | None = None
    for turn in reversed(session_data.conversation_history):
        if turn.get("role") == "assistant":
            last_assistant_msg = turn.get("content")
            break

    session_data.conversation_history.append({"role": "user", "content": user_message})
    new_session_mgr.append_conversation(session_id, "user", user_message)
    return session_data, project_state, session_id, last_assistant_msg


def _finalize_session(reply, lifecycle_decision, session_data, project_state, storage_project_name, record_family=True):
    from module5 import LifecycleDecision

    if lifecycle_decision is LifecycleDecision.READY_FOR_SUMMARY:
        session_data.empathize_summary_presented = True

    session_data.conversation_history.append({"role": "assistant", "content": reply})
    if reply not in session_data.previous_questions:
        session_data.previous_questions.append(reply)

    # Record the semantic family of the question we just asked (if any) so
    # the QuestionFamilyPlanner can steer the next question away from it.
    # Paused conversational turns pass record_family=False: an acknowledgment
    # or clarification reply is not a Design Thinking question and must not
    # pollute asked_question_families.
    if record_family:
        family = classify_question(reply)
        if family is not None and family.value not in session_data.asked_question_families:
            session_data.asked_question_families.append(family.value)

    new_session_mgr = get_session_manager()
    session_id = new_session_mgr.get_active_session_id()
    if session_id is None:
        meta = new_session_mgr.create_session()
        session_id = meta.session_id
    new_session_mgr.append_conversation(session_id, "assistant", reply)
    session_data.project_state = project_state
    new_session_mgr.save_session_data(session_id, session_data)

    return _build_legacy_session(project_state, session_data, storage_project_name)
