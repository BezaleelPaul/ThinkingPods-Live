"""
tests/test_hybrid_audit.py

Regression tests for the measurement-only Hybrid Extraction Quality Audit
(``hybrid_audit`` + its wiring in ``mentor._extract_and_update_state``):

  * disabled (default) -> the pipeline is byte-identical to a non-audited
    run: the shadow extractor is NEVER called on a hybrid skip, and no
    ``HybridAudit`` / ``HybridSummary`` keys appear in diagnostics.
  * enabled (``HYBRID_AUDIT=true``) -> on a hybrid skip the shadow LLM runs
    a SECOND time WITHOUT applying its result; ``ProjectState``, objectives,
    lifecycle, prompts, and replies are untouched; the audit + summary appear
    ONLY in diagnostics.
  * a shadow batch that would fail ``StateBatchValidationError`` is reported
    as "no state change" (measurement cannot guess) and never crashes the
    pipeline.
  * ``HybridAuditSummary`` aggregation, risk classification, and
    ``to_display`` projection.
  * session persistence round-trip (``to_dict`` / ``from_dict``).

All tests are deterministic (mocked ollama + mocked MemoryExtractor).
"""

import copy
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hybrid_audit  # noqa: E402
from hybrid_audit import (  # noqa: E402
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RISK_NONE,
    HybridAuditSummary,
    build_audit_record,
)
from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    ProjectState,
    StateField,
)
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _meaningful(*updates) -> ExtractionResult:
    return ExtractionResult(
        message_type=MessageType.MEANINGFUL,
        updates=[
            ExtractionUpdate(
                operation=Operation(u[0]),
                field=StateField(u[1]),
                value=u[2],
            )
            for u in updates
        ],
    )


def _empty_ambiguous() -> ExtractionResult:
    return ExtractionResult(message_type=MessageType.AMBIGUOUS, updates=[])


class _EnvScope:
    """Context manager that pins HYBRID_AUDIT to a value for a block."""

    def __init__(self, value):
        self._value = value

    def __enter__(self):
        self._prev = os.environ.get("HYBRID_AUDIT")
        if self._value is None:
            os.environ.pop("HYBRID_AUDIT", None)
        else:
            os.environ["HYBRID_AUDIT"] = self._value
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("HYBRID_AUDIT", None)
        else:
            os.environ["HYBRID_AUDIT"] = self._prev


class TestBuildAuditRecord(unittest.TestCase):
    """Unit tests for ``build_audit_record`` risk/summary computation."""

    def test_none_when_shadow_empty(self):
        state = ProjectState()
        state.personas = ["students"]
        record = build_audit_record(
            {"target_audience": "students"},
            _empty_ambiguous(),
            copy.deepcopy(state),
            state,
        )
        self.assertEqual(record["hybrid_decision"], "skip")
        self.assertTrue(record["llm_skipped"])
        self.assertEqual(record["shadow_llm_fields"], [])
        self.assertEqual(record["additional_fields"], [])
        self.assertFalse(record["would_change_state"])
        self.assertFalse(record["would_change_objective"])
        self.assertEqual(record["risk_level"], RISK_NONE)

    def test_low_when_duplicate_only(self):
        state = ProjectState()
        state.personas = ["students"]
        after = copy.deepcopy(state)
        # Rules applied the same persona value; the shadow proposes the same.
        shadow = _meaningful(("ADD", "personas", "students"))
        record = build_audit_record(
            {"target_audience": "students"},
            shadow,
            copy.deepcopy(state),
            after,
        )
        self.assertEqual(record["additional_fields"], [])
        self.assertFalse(record["would_change_state"])
        self.assertEqual(record["risk_level"], RISK_LOW)

    def test_medium_when_new_fact_but_same_objective(self):
        state = ProjectState()
        state.personas = ["students"]
        after = copy.deepcopy(state)
        # Rules satisfied PERSONAS; the shadow would ALSO have pulled in a
        # fact for a field that is NOT the current objective (EVIDENCE comes
        # after PROBLEMS in priority, so the objective stays PROBLEMS).
        shadow = _meaningful(("ADD", "personas", "students"),
                             ("ADD", "evidence", "surveyed 10 students"))
        record = build_audit_record(
            {"target_audience": "students"},
            shadow,
            copy.deepcopy(state),
            after,
            session_data=SessionData(),
        )
        self.assertIn("evidence", record["additional_fields"])
        self.assertTrue(record["would_change_state"])
        self.assertFalse(record["would_change_objective"])
        self.assertEqual(record["risk_level"], RISK_MEDIUM)

    def test_high_when_objective_changes(self):
        state = ProjectState()
        after = copy.deepcopy(state)
        # Rules satisfied PERSONAS; the shadow's problem would move the
        # mentor's next objective from PROBLEMS to CURRENT_SOLUTIONS.
        shadow = _meaningful(("ADD", "personas", "students"),
                             ("ADD", "problems", "deadlines"))
        record = build_audit_record(
            {"target_audience": "students"},
            shadow,
            copy.deepcopy(state),
            after,
            session_data=SessionData(),
        )
        self.assertTrue(record["would_change_state"])
        self.assertTrue(record["would_change_objective"])
        self.assertEqual(record["risk_level"], RISK_HIGH)

    def test_invalid_shadow_batch_does_not_crash_and_reports_no_change(self):
        """A shadow batch that fails atomic validation must not crash the
        audit and is reported as 'no state change' (we cannot guess)."""
        state = ProjectState()
        after = copy.deepcopy(state)
        # SET on a list-typed field is invalid -> StateBatchValidationError.
        shadow = _meaningful(("SET", "personas", "students"))
        with mock.patch.object(hybrid_audit, "StateManager") as sm_cls:
            sm = sm_cls.return_value
            from state_manager import StateBatchValidationError

            sm.apply_extraction.side_effect = StateBatchValidationError(
                ["SET on list-typed field is invalid"]
            )
            record = build_audit_record(
                {"target_audience": "students"},
                shadow,
                copy.deepcopy(state),
                after,
            )
        self.assertFalse(record["would_change_state"])
        self.assertEqual(record["additional_fields"], [])
        self.assertEqual(record["risk_level"], RISK_LOW)


class TestHybridAuditSummary(unittest.TestCase):
    """Aggregation, risk classification, display projection, persistence."""

    def test_add_classifies_risk(self):
        summary = HybridAuditSummary()
        summary.add({"risk_level": RISK_NONE, "additional_fields": []})
        summary.add({"risk_level": RISK_LOW, "additional_fields": []})
        summary.add({"risk_level": RISK_MEDIUM, "additional_fields": ["problems"]})
        summary.add(
            {"risk_level": RISK_HIGH, "additional_fields": ["problems"],
             "would_change_objective": True}
        )
        summary.add(
            {"risk_level": RISK_HIGH, "additional_fields": ["evidence"],
             "would_change_objective": True}
        )
        self.assertEqual(summary.llm_skipped, 5)
        self.assertEqual(summary.no_difference, 1)
        self.assertEqual(summary.low_risk, 1)
        self.assertEqual(summary.medium_risk, 1)
        self.assertEqual(summary.high_risk, 2)
        self.assertEqual(summary.additional_field_counts["problems"], 2)
        self.assertEqual(summary.additional_field_counts["evidence"], 1)
        self.assertEqual(len(summary.high_risk_examples), 2)

    def test_high_risk_example_cap(self):
        summary = HybridAuditSummary()
        for i in range(10):
            summary.add(
                {"risk_level": RISK_HIGH,
                 "additional_fields": [f"field{i}"],
                 "would_change_objective": True}
            )
        self.assertEqual(summary.high_risk, 10)
        self.assertEqual(len(summary.high_risk_examples), 5)

    def test_to_display_labels(self):
        summary = HybridAuditSummary()
        summary.add({"risk_level": RISK_MEDIUM, "additional_fields": ["problems"]})
        display = summary.to_display()
        self.assertEqual(display["LLM skipped"], 1)
        self.assertEqual(display["Medium risk"], 1)
        self.assertEqual(display["Most commonly missed"], {"Problems": 1})
        self.assertEqual(display["High-risk examples"], [])

    def test_persistence_round_trip(self):
        summary = HybridAuditSummary()
        summary.add(
            {"risk_level": RISK_HIGH, "additional_fields": ["problems"],
             "would_change_objective": True}
        )
        restored = HybridAuditSummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())


class TestAuditWiringLivePipeline(unittest.TestCase):
    """Integration: the audit changes nothing when disabled and only adds
    diagnostics when enabled."""

    USER = "audit_user"
    PROJ = "AuditProject"

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _run_turn(self, user_message, patch_extract, audit_enabled=True):
        env_value = "true" if audit_enabled else None
        with _EnvScope(env_value):
            # COMPLEXITY_GATE=false keeps this suite pinned to the pure
            # hybrid skip behavior (rules satisfy => always skip). The gate's
            # own behavior is covered in test_complexity_gate.py.
            with mock.patch.dict(os.environ, {"COMPLEXITY_GATE": "false"}):
                with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
                    with mock.patch(
                        "mentor.MemoryExtractor.extract", side_effect=patch_extract
                    ):
                        return mentor.process_mentor_turn(
                            user_message,
                            username=self.USER,
                            project_name=self.PROJ,
                            model_name="test-model",
                        )

    def _active_state(self):
        from session_manager import get_session_manager

        return get_session_manager().get_active_session_data().project_state

    def test_disabled_never_runs_shadow_and_no_audit_sections(self):
        calls = []

        def _patched(*args, **kwargs):
            calls.append(True)
            raise AssertionError("LLM extractor must not run when disabled")

        reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
            audit_enabled=False,
        )
        self.assertEqual(calls, [])
        self.assertEqual(self._active_state().personas, ["elderly people"])
        self.assertNotIn("HybridAudit", diagnostics)
        self.assertNotIn("HybridSummary", diagnostics)
        self.assertIn("HybridExtraction", diagnostics)
        self.assertTrue(reply and reply.strip())

    def test_enabled_runs_shadow_exactly_once_without_applying(self):
        calls = []

        def _patched(*args, **kwargs):
            calls.append(True)
            return _meaningful(("ADD", "personas", "elderly people"),
                               ("ADD", "problems", "missed doses"))

        reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
            audit_enabled=True,
        )
        # Shadow extraction ran exactly once; the rules already satisfied
        # the objective so the real LLM never runs.
        self.assertEqual(len(calls), 1)
        # The shadow result was NEVER applied: only the rule value landed.
        self.assertEqual(self._active_state().personas, ["elderly people"])
        self.assertEqual(self._active_state().problems, [])

        audit = diagnostics["HybridAudit"]
        self.assertEqual(audit["Hybrid Decision"], "skip")
        self.assertEqual(audit["LLM Skipped"], "Yes")
        self.assertIn("Problems", audit["Additional Fields"])
        self.assertEqual(audit["Risk Level"], "HIGH")

        summary = diagnostics["HybridSummary"]
        self.assertEqual(summary["LLM skipped"], 1)
        self.assertEqual(summary["High risk"], 1)

    def test_enabled_state_and_objective_identical_to_disabled(self):
        """Same turn with audit on vs off must leave byte-identical state,
        objective, and non-audit diagnostics."""

        def _patched_off(*args, **kwargs):
            return _meaningful(("ADD", "personas", "elderly people"))

        def _patched_on(*args, **kwargs):
            # Shadow would extract an extra fact; still must not change state.
            return _meaningful(("ADD", "personas", "elderly people"),
                               ("ADD", "problems", "missed doses"))

        reply_off, _, _, diag_off = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched_off,
            audit_enabled=False,
        )
        state_off = self._active_state().to_state_dict()

        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state(
            username=self.USER, project_title=self.PROJ
        )

        reply_on, _, _, diag_on = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched_on,
            audit_enabled=True,
        )
        state_on = self._active_state().to_state_dict()

        self.assertEqual(state_on, state_off)
        self.assertEqual(reply_on, reply_off)
        self.assertEqual(
            diag_off["Pipeline"]["Objective"],
            diag_on["Pipeline"]["Objective"],
        )
        for key, value in diag_off.items():
            if key in (
                "HybridAudit",
                "HybridSummary",
                # Product Experience is measurement-only: it counts the
                # Developer Console sections present this turn, so it
                # legitimately observes the extra Hybrid* sections.
                "ProductExperience",
                "ProductExperienceSummary",
            ):
                continue
            self.assertEqual(
                diag_on.get(key), value, f"diagnostics[{key}] differs with audit on"
            )
        self.assertIn("HybridAudit", diag_on)
        self.assertIn("HybridSummary", diag_on)

    def test_session_persists_audit_summary_round_trip(self):
        from session_manager import get_session_manager

        sd = SessionData()
        sd.hybrid_audit_summary.add(
            {"risk_level": RISK_MEDIUM, "additional_fields": ["problems"]}
        )
        restored = SessionData.from_dict(sd.to_dict())
        self.assertEqual(
            restored.hybrid_audit_summary.to_dict(),
            sd.hybrid_audit_summary.to_dict(),
        )


if __name__ == "__main__":
    unittest.main()
