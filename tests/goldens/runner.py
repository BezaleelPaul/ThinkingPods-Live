"""tests/goldens/runner.py — turn-by-turn golden conversation runner.

Replays a :class:`GoldenConversationFixture` against the live
``mentor.process_mentor_turn`` pipeline with a fully mocked runtime:

  * ``ollama`` is injected into ``sys.modules`` as a stub — a canned
    reply when the fixture supplies one, otherwise a stub that raises so
    the deterministic family-aware fallback path runs.
  * ``mentor.MemoryExtractor.extract`` is patched to return the fixture's
    ``ExtractionResult`` (or an AMBIGUOUS no-op, which lets the
    deterministic rule-based extractor run on the raw user text).

Every turn snapshots the post-turn state and runs the fixture's expected
checks. Failures are collected per turn with a ``Turn N: ...`` prefix so
a regression is immediately attributable to the exact failing turn.

The Semantic Complexity Gate (``COMPLEXITY_GATE``) is forced ON for the
whole replay: the fixtures now pin gate-on expectations (MEDIUM/HIGH
messages run the LLM even when the rules satisfy the current objective), so
the suite must not depend on the ambient environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from unittest import mock

from memory_extractor import (
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    StateField,
)

from .fixtures import GoldenConversationFixture, GoldenTurn

# ProjectState fields snapshot after each turn (mirrors to_state_dict).
_STATE_KEYS: tuple[str, ...] = (
    "personas",
    "problems",
    "current_solutions",
    "pain_points",
    "evidence",
    "impacts",
    "frequency",
)


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


class _StubOllama:
    """Stand-in that returns canned replies in order, then raises."""

    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, **kwargs):
        if not self._replies:
            raise RuntimeError("no more stub replies")
        return {"message": {"content": self._replies.pop(0)}}


@dataclass
class TurnResult:
    """Outcome of one replayed turn, including any check failures."""

    index: int
    user_message: str
    reply: str
    diagnostics: dict
    failures: list[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    """Outcome of a whole golden conversation."""

    fixture_name: str
    turn_results: list[TurnResult] = field(default_factory=list)

    @property
    def all_failures(self) -> list[str]:
        """Flattened failure list, each line naming the failing turn."""
        out: list[str] = []
        for tr in self.turn_results:
            for f in tr.failures:
                out.append(f"Turn {tr.index} [{tr.user_message!r}]: {f}")
        return out

    @property
    def passed(self) -> bool:
        return not self.all_failures


def _build_extraction(turn: GoldenTurn) -> ExtractionResult:
    """Turn the fixture's extraction dict into a typed ExtractionResult.

    When the fixture omits ``extraction`` an AMBIGUOUS result is returned so
    the pipeline falls through to the deterministic rule-based extractor.
    """
    if turn.extraction is None:
        return ExtractionResult(message_type=MessageType.AMBIGUOUS, updates=[])

    raw = turn.extraction
    updates = [
        ExtractionUpdate(
            operation=Operation(u["operation"]),
            field=StateField(u["field"]),
            value=u["value"],
        )
        for u in raw.get("updates", [])
    ]
    return ExtractionResult(
        message_type=MessageType(raw.get("message_type", "MEANINGFUL")),
        updates=updates,
    )


def _seed_session(fixture: GoldenConversationFixture) -> None:
    """Reset the session manager and apply the fixture's seed state."""
    from session_manager import get_session_manager

    mgr = get_session_manager()
    mgr.reset_runtime_state(
        username=fixture.username, project_title=fixture.project_name
    )
    sd = mgr.get_active_session_data()
    for key, value in fixture.seed_state.items():
        if key not in _STATE_KEYS:
            continue
        if isinstance(value, list):
            setattr(sd.project_state, key, list(value))
        else:
            setattr(sd.project_state, key, value)
    sd.asked_question_families = list(fixture.seed_asked_families)
    mgr.save_session_data(mgr.get_active_session_id(), sd)


def _post_turn_snapshot() -> dict:
    """Read the persisted post-turn session state (after _finalize_session)."""
    from session_manager import get_session_manager

    sd = get_session_manager().get_active_session_data()
    return {
        "state": sd.project_state.to_state_dict(),
        "asked_families": list(sd.asked_question_families),
        "unknown_facts": list(sd.unknown_facts),
        "stage": sd.current_stage,
    }


def _label_to_family(label: Optional[str]) -> Optional[str]:
    """Reverse of ``module3.question_families.family_label`` (label -> value)."""
    if not label:
        return None
    return label.upper().replace(" ", "_")


def _check_turn(
    turn: GoldenTurn,
    reply: str,
    diagnostics: dict,
    snapshot: dict,
    failures: list[str],
) -> None:
    """Run all of a turn's expected checks, appending failures."""
    prefix = ""

    if turn.expect_reply is not None and reply != turn.expect_reply:
        failures.append(
            f"reply mismatch — expected {turn.expect_reply!r}, got {reply!r}"
        )

    if turn.expect_state is not None:
        actual = snapshot["state"]
        for key, expected in turn.expect_state.items():
            got = actual.get(key)
            if isinstance(expected, list):
                if list(expected) != list(got or []):
                    failures.append(
                        f"state[{key}] mismatch — expected {list(expected)}, "
                        f"got {list(got or [])}"
                    )
            else:
                if expected != got:
                    failures.append(
                        f"state[{key}] mismatch — expected {expected!r}, "
                        f"got {got!r}"
                    )

    if turn.expect_objective is not None:
        actual_obj = diagnostics["Pipeline"]["Objective"]
        if actual_obj != turn.expect_objective:
            failures.append(
                f"objective mismatch — expected {turn.expect_objective}, "
                f"got {actual_obj}"
            )

    if turn.expect_checklist is not None:
        checklist = diagnostics["ChecklistReasoning"]
        for key, expected in turn.expect_checklist.items():
            if key not in checklist:
                failures.append(
                    f"checklist key {key!r} missing from diagnostics"
                )
                continue
            actual = checklist[key]["complete"]
            if actual != expected:
                failures.append(
                    f"checklist[{key}] complete mismatch — expected "
                    f"{expected}, got {actual}"
                )

    if turn.expect_family is not None:
        actual_label = diagnostics["QuestionFamilies"]["Question Family"]
        actual = _label_to_family(actual_label)
        if actual != turn.expect_family:
            failures.append(
                f"question family mismatch — expected {turn.expect_family}, "
                f"got {actual!r} (label {actual_label!r})"
            )

    if turn.expect_asked_families is not None:
        if list(snapshot["asked_families"]) != list(turn.expect_asked_families):
            failures.append(
                f"asked families mismatch — expected "
                f"{list(turn.expect_asked_families)}, "
                f"got {list(snapshot['asked_families'])}"
            )

    if turn.expect_unknown_facts is not None:
        if list(snapshot["unknown_facts"]) != list(turn.expect_unknown_facts):
            failures.append(
                f"unknown_facts mismatch — expected "
                f"{list(turn.expect_unknown_facts)}, "
                f"got {list(snapshot['unknown_facts'])}"
            )

    if turn.expect_stage is not None and snapshot["stage"] != turn.expect_stage:
        failures.append(
            f"stage mismatch — expected {turn.expect_stage}, "
            f"got {snapshot['stage']}"
        )

    if turn.expect_reask is not None:
        actual_reask = diagnostics["QuestionFamilies"]["Re-ask Sanctioned"]
        if actual_reask != turn.expect_reask:
            failures.append(
                f"reask mismatch — expected {turn.expect_reask}, "
                f"got {actual_reask}"
            )


def run_golden_conversation(
    fixture: GoldenConversationFixture,
) -> ConversationResult:
    """Replay a golden conversation turn-by-turn and return the result."""
    import mentor

    _seed_session(fixture)
    result = ConversationResult(fixture_name=fixture.name)

    # The fixtures pin Semantic-Complexity-Gate-ON expectations; force it so
    # the replay is deterministic regardless of the ambient environment.
    with mock.patch.dict(os.environ, {"COMPLEXITY_GATE": "true"}):
        for index, turn in enumerate(fixture.turns):
            ollama_stub = (
                _RaisingOllama()
                if turn.ollama_reply is None
                else _StubOllama([turn.ollama_reply])
            )
            extraction = _build_extraction(turn)

            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )

            snapshot = _post_turn_snapshot()
            failures: list[str] = []
            _check_turn(turn, reply, diagnostics, snapshot, failures)
            result.turn_results.append(
                TurnResult(
                    index=index,
                    user_message=turn.user,
                    reply=reply,
                    diagnostics=diagnostics,
                    failures=failures,
                )
            )

    if fixture.expect_final_objective is not None and result.turn_results:
        last = result.turn_results[-1]
        actual = last.diagnostics["Pipeline"]["Objective"]
        if actual != fixture.expect_final_objective:
            last.failures.append(
                f"final objective mismatch — expected "
                f"{fixture.expect_final_objective}, got {actual}"
            )

    if fixture.expect_final_stage is not None and result.turn_results:
        last = result.turn_results[-1]
        actual = last.diagnostics["Pipeline"]["Stage"]
        if actual != fixture.expect_final_stage:
            last.failures.append(
                f"final stage mismatch — expected {fixture.expect_final_stage}, "
                f"got {actual}"
            )

    return result
