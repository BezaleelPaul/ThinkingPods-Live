"""Tests for the observation-only Developer Console audit.

Covers the per-turn ``DeveloperConsoleAudit`` measurements (navigation,
discoverability, interaction, density, layout), serialization, the Developer
Console diagnostics projection, the session-level
``DeveloperConsoleAuditSummary`` (aggregation, merge, persistence, display),
the observation-only guarantee (render plans are never mutated), determinism,
and an integration check that audits a real render plan produced by
``developer_console.build_render_plan``.

The audit NEVER touches the UI, never changes diagnostics or conversation
behavior, and never imports Streamlit / mentor — it only reads render plans.
"""

import json
import unittest

from developer_console import (
    CollapseState,
    build_render_plan,
    build_turn,
)
from developer_console_audit import (
    DeveloperConsoleAudit,
    DeveloperConsoleAuditSummary,
    analyze_developer_console,
    developer_console_diagnostics_section,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic diagnostics + render plans (built through the real
# developer_console renderer so the audited text is byte-identical to what
# the UI would render).
# ---------------------------------------------------------------------------


def _realistic_diagnostics():
    return {
        "Pipeline": {
            "Stage": "Empathize",
            "Objective": "PROBLEMS",
            "Response Strategy": "ASK_QUESTION",
            "Lifecycle Decision": "CONTINUE",
            "Summary Presented": False,
            "Model": "optimized-pods",
        },
        "ProjectState": {
            "Personas": ["Small business owners"],
            "Problems": ["Slow checkout"],
            "Frequency": "daily",
        },
        "Extraction": {
            "message_type": "MEANINGFUL",
            "updates": [
                {
                    "operation": "ADD",
                    "field": "problems",
                    "value": "Slow checkout",
                    "label": "Correct",
                    "reasons": ["matches text"],
                }
            ],
        },
        "Memory": {
            "Open Threads": ["problems"],
            "Resolved Threads": ["personas"],
            "Deferred Topics": [],
        },
        "ConversationFailure": {"Primary Failure": "None", "Confidence": 0.98},
        "Prompt": "You are a design thinking mentor.\nAsk one question.\nDo not advise.",
    }


def _timing():
    return {
        "total_ms": 1234.0,
        "extraction_ms": 500.0,
        "llm_ms": 600.0,
        "prompt_ms": 40.0,
    }


def _realistic_plan(collapse_state=None):
    turn = build_turn(
        turn_index=4,
        ordinal=2,
        role="assistant",
        content="Good - what core problem?",
        timing=_timing(),
        diagnostics=_realistic_diagnostics(),
    )
    return build_render_plan(turn, collapse_state)


def _all_expanded_plan():
    state = CollapseState(expanded=[
        "Performance", "Pipeline", "ProjectState", "Extraction", "Memory",
        "ConversationFailure", "Prompt",
    ])
    return _realistic_plan(state)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.audit = analyze_developer_console(
            render_plan=_realistic_plan(), turn_index=4, total_turns=12
        )

    def test_sections_per_turn(self):
        self.assertEqual(self.audit.sections_per_turn, 7)

    def test_expandable_count_equals_blocks(self):
        self.assertEqual(self.audit.expandable_count, 7)

    def test_collapsed_count_defaults(self):
        self.assertEqual(self.audit.collapsed_count, 5)

    def test_maximum_depth_from_rendered_text(self):
        self.assertEqual(self.audit.maximum_depth, 3)

    def test_prompt_location_depth(self):
        self.assertEqual(self.audit.prompt_location_depth, 7)

    def test_total_turns_from_context(self):
        self.assertEqual(self.audit.total_turns, 12)
        self.assertEqual(self.audit.turn_index, 4)

    def test_empty_plan_defaults(self):
        audit = analyze_developer_console(render_plan=[])
        self.assertEqual(audit.sections_per_turn, 0)
        self.assertEqual(audit.maximum_depth, 0)
        self.assertEqual(audit.expandable_count, 0)
        self.assertEqual(audit.collapsed_count, 0)
        self.assertIsNone(audit.prompt_location_depth)

    def test_garbage_plan_ignored(self):
        audit = analyze_developer_console(render_plan=12345)
        self.assertEqual(audit.sections_per_turn, 0)
        audit2 = analyze_developer_console(
            render_plan=["string", None]
        )
        self.assertEqual(audit2.sections_per_turn, 0)
        # dict entries are counted even when they carry no known keys
        audit3 = analyze_developer_console(render_plan=[{"no_key": 1}])
        self.assertEqual(audit3.sections_per_turn, 1)

    def test_prompt_absent_location_none(self):
        diagnostics = {
            "Pipeline": {"Stage": "Empathize"},
            "ProjectState": {"Personas": ["X"]},
        }
        turn = build_turn(
            turn_index=0, ordinal=1, role="assistant", content="c",
            timing={"total_ms": 10}, diagnostics=diagnostics,
        )
        audit = analyze_developer_console(render_plan=build_render_plan(turn))
        self.assertIsNone(audit.prompt_location_depth)
        self.assertIsNone(audit.clicks_to_prompt)


# ---------------------------------------------------------------------------
# Discoverability
# ---------------------------------------------------------------------------


class TestDiscoverability(unittest.TestCase):
    def test_defaults_performance_and_prompt_visible(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertTrue(audit.performance_visible)
        self.assertTrue(audit.prompt_visible_without_expanding)

    def test_project_state_hidden_by_default(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertFalse(audit.project_state_visible)

    def test_extraction_hidden_by_default(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertFalse(audit.extraction_visible)

    def test_all_visible_when_expanded(self):
        audit = analyze_developer_console(render_plan=_all_expanded_plan())
        self.assertTrue(audit.prompt_visible_without_expanding)
        self.assertTrue(audit.project_state_visible)
        self.assertTrue(audit.extraction_visible)
        self.assertTrue(audit.performance_visible)

    def test_missing_sections_not_visible(self):
        turn = build_turn(
            turn_index=0, ordinal=1, role="assistant", content="c",
            timing={"total_ms": 5}, diagnostics={"Pipeline": {"Stage": "X"}},
        )
        audit = analyze_developer_console(render_plan=build_render_plan(turn))
        self.assertFalse(audit.project_state_visible)
        self.assertFalse(audit.extraction_visible)
        # Performance is always emitted as a default-expanded block.
        self.assertTrue(audit.performance_visible)


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------


class TestInteraction(unittest.TestCase):
    def setUp(self):
        self.audit = analyze_developer_console(render_plan=_realistic_plan())

    def test_clicks_to_prompt_cumulative(self):
        self.assertEqual(self.audit.clicks_to_prompt, 5)

    def test_clicks_to_project_state(self):
        self.assertEqual(self.audit.clicks_to_project_state, 2)

    def test_clicks_to_memory(self):
        self.assertEqual(self.audit.clicks_to_memory, 4)

    def test_clicks_to_conversation_failure(self):
        self.assertEqual(self.audit.clicks_to_conversation_failure, 5)

    def test_clicks_zero_when_all_expanded(self):
        audit = analyze_developer_console(render_plan=_all_expanded_plan())
        self.assertEqual(audit.clicks_to_prompt, 0)
        self.assertEqual(audit.clicks_to_project_state, 0)
        self.assertEqual(audit.clicks_to_memory, 0)
        self.assertEqual(audit.clicks_to_conversation_failure, 0)

    def test_clicks_absent_target_none(self):
        turn = build_turn(
            turn_index=0, ordinal=1, role="assistant", content="c",
            timing={"total_ms": 5}, diagnostics={"Pipeline": {"Stage": "X"}},
        )
        audit = analyze_developer_console(render_plan=build_render_plan(turn))
        self.assertIsNone(audit.clicks_to_project_state)
        self.assertIsNone(audit.clicks_to_memory)
        self.assertIsNone(audit.clicks_to_conversation_failure)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


class TestDensity(unittest.TestCase):
    def setUp(self):
        self.audit = analyze_developer_console(render_plan=_realistic_plan())

    def test_average_characters_per_section(self):
        self.assertAlmostEqual(self.audit.average_characters_per_section, 91.6, places=1)

    def test_largest_section(self):
        largest = self.audit.largest_section
        self.assertEqual(largest["key"], "Pipeline")
        self.assertEqual(largest["characters"], 156)

    def test_smallest_section(self):
        smallest = self.audit.smallest_section
        self.assertEqual(smallest["key"], "ConversationFailure")
        self.assertEqual(smallest["characters"], 42)

    def test_density_empty_plan(self):
        audit = analyze_developer_console(render_plan=[])
        self.assertEqual(audit.average_characters_per_section, 0.0)
        self.assertIsNone(audit.largest_section)
        self.assertIsNone(audit.smallest_section)

    def test_section_profiles(self):
        profiles = self.audit.section_profiles
        self.assertEqual(len(profiles), 7)
        performance = profiles[0]
        self.assertEqual(performance["key"], "Performance")
        self.assertTrue(performance["visible"])
        prompt = profiles[-1]
        self.assertEqual(prompt["kind"], "prompt")
        extraction = [p for p in profiles if p["key"] == "Extraction"][0]
        self.assertEqual(extraction["depth"], 3)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout(unittest.TestCase):
    def test_panel_width_estimate(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertEqual(audit.panel_width_estimate, 46)

    def test_estimated_scrolling_distance(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertEqual(audit.estimated_scrolling_distance, 29)

    def test_visible_without_scrolling_default_height(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        self.assertEqual(audit.visible_sections_without_scrolling, 4)

    def test_visible_without_scrolling_custom_height(self):
        audit = analyze_developer_console(
            render_plan=_realistic_plan(),
            panel_config={"panel_height_lines": 100},
        )
        self.assertEqual(audit.visible_sections_without_scrolling, 7)
        self.assertEqual(audit.panel_height_lines, 100)

    def test_empty_plan_layout(self):
        audit = analyze_developer_console(render_plan=[])
        self.assertEqual(audit.panel_width_estimate, 0)
        self.assertEqual(audit.estimated_scrolling_distance, 0)
        self.assertEqual(audit.visible_sections_without_scrolling, 0)


# ---------------------------------------------------------------------------
# Record serialization + projection
# ---------------------------------------------------------------------------


class TestRecordSerialization(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self):
        audit = analyze_developer_console(
            render_plan=_realistic_plan(), turn_index=4, total_turns=12
        )
        restored = DeveloperConsoleAudit.from_dict(audit.to_dict())
        self.assertEqual(restored.to_dict(), audit.to_dict())

    def test_from_dict_defaults(self):
        audit = DeveloperConsoleAudit.from_dict({})
        self.assertEqual(audit.sections_per_turn, 0)
        self.assertIsNone(audit.prompt_location_depth)
        audit2 = DeveloperConsoleAudit.from_dict(None)
        self.assertEqual(audit2.turn_index, 0)

    def test_from_dict_tolerates_partial(self):
        audit = DeveloperConsoleAudit.from_dict(
            {"sections_per_turn": 3, "prompt_location_depth": 2}
        )
        self.assertEqual(audit.sections_per_turn, 3)
        self.assertEqual(audit.prompt_location_depth, 2)
        self.assertFalse(audit.project_state_visible)

    def test_json_serialisable(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        json.dumps(audit.to_dict())

    def test_to_display(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        display = audit.to_display()
        self.assertEqual(display["Navigation → Sections"], 7)
        self.assertIn("Interaction → Clicks to Prompt", display)


class TestDiagnosticsProjection(unittest.TestCase):
    def test_empty_without_audit(self):
        self.assertEqual(developer_console_diagnostics_section(None), {})

    def test_projection_fields(self):
        audit = analyze_developer_console(render_plan=_realistic_plan())
        proj = developer_console_diagnostics_section(audit)
        self.assertEqual(proj["Navigation → Sections"], 7)
        self.assertEqual(proj["Discoverability → Prompt"], "Yes")
        self.assertEqual(proj["Discoverability → Project State"], "No")
        self.assertEqual(proj["Interaction → Clicks to Prompt"], 5)
        self.assertEqual(proj["Layout → Visible Sections (no scroll)"], 4)

    def test_projection_absent_targets(self):
        turn = build_turn(
            turn_index=0, ordinal=1, role="assistant", content="c",
            timing={"total_ms": 5}, diagnostics={"Pipeline": {"Stage": "X"}},
        )
        audit = analyze_developer_console(render_plan=build_render_plan(turn))
        proj = developer_console_diagnostics_section(audit)
        self.assertEqual(proj["Interaction → Clicks to Prompt"], "—")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummaryMetrics(unittest.TestCase):
    def test_empty_summary(self):
        summary = DeveloperConsoleAuditSummary()
        self.assertEqual(summary.total_turns(), 0)
        self.assertEqual(summary.average_sections_per_turn(), 0.0)
        self.assertEqual(summary.maximum_depth(), 0)
        self.assertEqual(summary.average_expandable_count(), 0.0)
        self.assertEqual(summary.average_collapsed_count(), 0.0)
        self.assertIsNone(summary.average_prompt_location_depth())
        self.assertEqual(summary.prompt_discovery_rate(), 0.0)
        self.assertEqual(summary.project_state_discovery_rate(), 0.0)
        self.assertEqual(summary.extraction_discovery_rate(), 0.0)
        self.assertEqual(summary.performance_discovery_rate(), 0.0)
        self.assertIsNone(summary.average_clicks_to_prompt())
        self.assertIsNone(summary.average_clicks_to_project_state())
        self.assertEqual(summary.average_characters_per_section(), 0.0)
        self.assertIsNone(summary.largest_section())
        self.assertIsNone(summary.smallest_section())
        self.assertEqual(summary.average_panel_width_estimate(), 0.0)
        self.assertEqual(summary.total_scrolling_distance(), 0)
        self.assertEqual(summary.average_visible_sections_without_scrolling(), 0.0)

    def test_aggregation(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record(analyze_developer_console(
            render_plan=_realistic_plan(), turn_index=0, total_turns=2
        ))
        summary.add_record(analyze_developer_console(
            render_plan=_all_expanded_plan(), turn_index=1, total_turns=2
        ))
        self.assertEqual(summary.total_turns(), 2)
        self.assertEqual(summary.average_sections_per_turn(), 7.0)
        self.assertEqual(summary.maximum_depth(), 3)
        self.assertEqual(summary.average_expandable_count(), 7.0)
        self.assertEqual(summary.average_collapsed_count(), 2.5)
        self.assertAlmostEqual(summary.prompt_discovery_rate(), 1.0)
        self.assertAlmostEqual(summary.project_state_discovery_rate(), 0.5)
        self.assertAlmostEqual(summary.extraction_discovery_rate(), 0.5)
        self.assertAlmostEqual(summary.performance_discovery_rate(), 1.0)
        self.assertEqual(summary.average_clicks_to_prompt(), 2.5)
        self.assertEqual(summary.average_clicks_to_project_state(), 1.0)
        self.assertEqual(summary.total_scrolling_distance(), 58)

    def test_most_common_defaults_not_applicable(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record(analyze_developer_console(render_plan=_realistic_plan()))
        self.assertEqual(summary.largest_section()["key"], "Pipeline")
        self.assertEqual(summary.smallest_section()["key"], "ConversationFailure")


class TestSummaryPersistenceAndMerge(unittest.TestCase):
    def test_merge(self):
        a = DeveloperConsoleAuditSummary()
        a.add_record(analyze_developer_console(render_plan=_realistic_plan()))
        b = DeveloperConsoleAuditSummary()
        b.add_record(analyze_developer_console(render_plan=_all_expanded_plan()))
        a.merge(b)
        self.assertEqual(a.total_turns(), 2)
        self.assertAlmostEqual(a.project_state_discovery_rate(), 0.5)

    def test_json_roundtrip(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record(analyze_developer_console(
            render_plan=_realistic_plan(), turn_index=0, total_turns=2
        ))
        summary.add_record(analyze_developer_console(
            render_plan=_all_expanded_plan(), turn_index=1, total_turns=2
        ))
        restored = DeveloperConsoleAuditSummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())
        self.assertEqual(restored.average_sections_per_turn(), summary.average_sections_per_turn())

    def test_from_empty_dict(self):
        self.assertEqual(DeveloperConsoleAuditSummary.from_dict({}).total_turns(), 0)
        self.assertEqual(DeveloperConsoleAuditSummary.from_dict(None).total_turns(), 0)

    def test_to_display(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record(analyze_developer_console(render_plan=_realistic_plan()))
        summary.add_record(analyze_developer_console(render_plan=_realistic_plan()))
        display = summary.to_display()
        self.assertEqual(display["Turns Analyzed"], 2)
        self.assertEqual(display["Navigation → Avg Sections per Turn"], 7.0)
        self.assertEqual(display["Navigation → Max Content Depth"], 3)
        self.assertIn("Discoverability → Prompt Visible Rate", display)
        self.assertIn("Layout → Total Scroll Lines", display)
        json.dumps(display)

    def test_add_none_and_garbage_ignored(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record(None)
        summary.add_record("garbage")
        summary.add_record(123)
        self.assertEqual(summary.total_turns(), 0)

    def test_add_dict_record(self):
        summary = DeveloperConsoleAuditSummary()
        summary.add_record({
            "turn_index": 0,
            "sections_per_turn": 3,
            "maximum_depth": 1,
        })
        self.assertEqual(summary.total_turns(), 1)
        self.assertEqual(summary.average_sections_per_turn(), 3.0)


# ---------------------------------------------------------------------------
# Observation-only + determinism
# ---------------------------------------------------------------------------


class TestObservationOnly(unittest.TestCase):
    def test_analyzer_does_not_mutate_plan(self):
        plan = _realistic_plan()
        snapshot = json.dumps(plan, sort_keys=True)
        analyze_developer_console(render_plan=plan)
        self.assertEqual(json.dumps(plan, sort_keys=True), snapshot)

    def test_analysis_is_deterministic(self):
        a = analyze_developer_console(render_plan=_realistic_plan()).to_dict()
        b = analyze_developer_console(render_plan=_realistic_plan()).to_dict()
        self.assertEqual(a, b)

    def test_rendered_output_unchanged(self):
        """Building the audit must not change the render plan text."""
        before = [entry["text"] for entry in _realistic_plan()]
        plan = _realistic_plan()
        analyze_developer_console(render_plan=plan)
        after = [entry["text"] for entry in plan]
        self.assertEqual(after, before)


# ---------------------------------------------------------------------------
# Integration with a real developer_console render plan
# ---------------------------------------------------------------------------


class TestRealisticIntegration(unittest.TestCase):
    def test_realistic_plan_all_fields_populated(self):
        audit = analyze_developer_console(
            render_plan=_realistic_plan(), turn_index=4, total_turns=12
        )
        self.assertEqual(audit.turn_index, 4)
        self.assertEqual(audit.total_turns, 12)
        self.assertEqual(audit.sections_per_turn, 7)
        self.assertEqual(audit.maximum_depth, 3)
        self.assertEqual(audit.expandable_count, 7)
        self.assertEqual(audit.collapsed_count, 5)
        self.assertEqual(audit.prompt_location_depth, 7)
        self.assertTrue(audit.prompt_visible_without_expanding)
        self.assertFalse(audit.project_state_visible)
        self.assertFalse(audit.extraction_visible)
        self.assertTrue(audit.performance_visible)
        self.assertEqual(audit.clicks_to_prompt, 5)
        self.assertEqual(audit.clicks_to_project_state, 2)
        self.assertEqual(audit.clicks_to_memory, 4)
        self.assertEqual(audit.clicks_to_conversation_failure, 5)
        self.assertGreater(audit.average_characters_per_section, 0)
        self.assertIsNotNone(audit.largest_section)
        self.assertIsNotNone(audit.smallest_section)
        self.assertGreater(audit.panel_width_estimate, 0)
        self.assertGreater(audit.estimated_scrolling_distance, 0)
        self.assertGreater(audit.visible_sections_without_scrolling, 0)
        self.assertEqual(len(audit.section_profiles), 7)

    def test_plan_without_prompt(self):
        diagnostics = {
            "Pipeline": {"Stage": "Empathize"},
            "ProjectState": {"Personas": ["X"]},
            "ConversationFailure": {"Primary Failure": "None"},
        }
        turn = build_turn(
            turn_index=1, ordinal=1, role="assistant", content="c",
            timing={"total_ms": 10}, diagnostics=diagnostics,
        )
        audit = analyze_developer_console(render_plan=build_render_plan(turn))
        self.assertIsNone(audit.prompt_location_depth)
        self.assertFalse(audit.prompt_visible_without_expanding)
        self.assertIsNone(audit.clicks_to_prompt)

    def test_custom_collapse_state_reflected(self):
        state = CollapseState(explicit={"ProjectState": True})
        plan = _realistic_plan(state)
        audit = analyze_developer_console(render_plan=plan)
        self.assertTrue(audit.project_state_visible)
        # Pipeline (above ProjectState) is collapsed, so reaching ProjectState
        # costs one click in the cumulative model.
        self.assertEqual(audit.clicks_to_project_state, 1)


if __name__ == "__main__":
    unittest.main()
