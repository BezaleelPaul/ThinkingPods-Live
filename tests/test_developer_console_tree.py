"""
tests/test_developer_console_tree.py — exhaustive regression tests for the
redesigned VS Code-style Developer Console tree (Conversation → Turn →
Category → Section) in ``developer_console.py``.

Covers the full redesign surface:

  * Conversation hierarchy — turns as the first level, sections as children,
    turn labels are ``Turn N`` only (never assistant text),
  * categories — ``Core Pipeline`` / ``Conversation`` / ``Quality Audits`` /
    ``Prompt`` / ``Visual Design`` grouping with a deterministic fallback,
  * search — filters turns, section names, keys, and category names, but
    never text and never modifies diagnostics,
  * expand / collapse — defaults, per-turn overrides, expand-all,
    collapse-all (pinned stay open),
  * pinning — pinned sections expand in every turn and survive turn switches,
  * diagnostics identity — ``build_tree_plan`` never mutates the underlying
    dictionaries and renders section text byte-identical to
    ``build_render_plan``.

These are pure-model tests (no Streamlit). None touch conversation behavior,
extraction, objectives, or live pipeline diagnostics.
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from developer_console import (  # noqa: E402
    CATEGORY_ORDER,
    DEFAULT_CATEGORY,
    DEFAULT_EXPANDED_SECTIONS,
    TurnNode,
    build_render_plan,
    build_tree_plan,
    build_turn,
    collect_turns,
    highlight_matches,
    section_category,
    turn_label,
)


def _make_diagnostics(extra=None):
    diag = {
        "Pipeline": {"Stage": "Empathize", "Objective": "PROBLEMS"},
        "ProjectState": {"personas": ["elderly people"]},
        "Memory": {"resolved": ["age"]},
        "Prompt": "You are a mentor. Celebrate discoveries briefly.",
    }
    if extra:
        diag.update(extra)
    return diag


def _single_turn():
    return build_turn(
        turn_index=0,
        ordinal=1,
        role="assistant",
        content="Hello from the mentor",
        timing={"total_ms": 125.0, "llm_ms": 80.0},
        diagnostics=_make_diagnostics(),
    )


def _multi_turns(n=3):
    turns = []
    for i in range(n):
        diag = dict(_make_diagnostics())
        diag["ProjectState"] = {"personas": [f"persona-{i}"]}
        turns.append(
            build_turn(
                turn_index=i,
                ordinal=i + 1,
                role="assistant",
                content=f"Reply {i + 1} with distinct text",
                timing={"total_ms": float(100 + i), "llm_ms": float(50 + i)},
                diagnostics=diag,
            )
        )
    return turns


def _all_sections(turn_node):
    roms = []
    for category in turn_node.categories:
        roms.extend(category.sections)
    return roms


class TestCategories(unittest.TestCase):

    def test_known_keys_map_to_core_pipeline(self):
        for key in ("Performance", "Pipeline", "ObjectiveTrace",
                    "Extraction", "HybridExtraction", "InsightDetection"):
            self.assertEqual(section_category(key), "Core Pipeline")

    def test_known_keys_map_to_conversation(self):
        for key in ("ProjectState", "StateChanges", "Memory", "Recovery",
                    "QuestionHistory", "QuestionFamilies", "MentorDecision",
                    "MentorDecisionSummary", "ChecklistReasoning",
                    "CoachingStrategy", "ConversationMove"):
            self.assertEqual(section_category(key), "Conversation")

    def test_known_keys_map_to_quality_audits(self):
        for key in ("ExtractionComparison", "ExtractionAccuracy",
                    "ExtractionAccuracySummary", "ConversationStyle",
                    "ConversationStyleSummary", "ProductExperience",
                    "ProductExperienceSummary", "ConversationFailure",
                    "ConversationFailureSummary", "SemanticComplexity",
                    "HybridAudit", "HybridSummary", "ConversationMetrics"):
            self.assertEqual(section_category(key), "Quality Audits")

    def test_known_keys_map_to_prompt(self):
        self.assertEqual(section_category("Prompt"), "Prompt")
        self.assertEqual(section_category("PromptEffectiveness"), "Prompt")

    def test_unknown_key_uses_default_category(self):
        self.assertEqual(section_category("TotallyNewSection"), DEFAULT_CATEGORY)
        self.assertEqual(DEFAULT_CATEGORY, "Conversation")

    def test_category_order_is_stable_and_complete(self):
        self.assertEqual(
            CATEGORY_ORDER,
            ("Core Pipeline", "Conversation", "Quality Audits", "Prompt", "Visual Design"),
        )


class TestTurnLabels(unittest.TestCase):

    def test_labels_are_turn_number_only(self):
        turns = _multi_turns(4)
        self.assertEqual([turn_label(t) for t in turns],
                         ["Turn 1", "Turn 2", "Turn 3", "Turn 4"])

    def test_assistant_text_never_appears_in_label(self):
        for t in _multi_turns(3):
            self.assertNotIn(t.content_preview, turn_label(t))


class TestTreeHierarchy(unittest.TestCase):

    def setUp(self):
        self.turn = _single_turn()
        self.plan = build_tree_plan([self.turn], selected_index=0)

    def test_empty_turns_produce_empty_plan(self):
        plan = build_tree_plan([])
        self.assertEqual(plan.turns, [])
        self.assertEqual(plan.total_turns, 0)
        self.assertEqual(plan.visible_turns, 0)
        self.assertEqual(plan.visible_sections, 0)

    def test_turns_are_the_first_level(self):
        turns = _multi_turns(3)
        plan = build_tree_plan(turns, selected_index=1)
        self.assertEqual([tn.label for tn in plan.turns],
                         ["Turn 1", "Turn 2", "Turn 3"])
        self.assertIsInstance(plan.turns[0], TurnNode)

    def test_sections_are_children_via_categories(self):
        categories = self.plan.turns[0].categories
        names = [c.name for c in categories]
        self.assertIn("Core Pipeline", names)
        self.assertIn("Conversation", names)
        self.assertIn("Prompt", names)
        # Every category has SectionNode children.
        for category in categories:
            self.assertTrue(category.sections)
            for section in category.sections:
                self.assertEqual(section.category, category.name)

    def test_every_section_is_grouped_and_present(self):
        sections = _all_sections(self.plan.turns[0])
        by_key = {s.key: s for s in sections}
        self.assertEqual(
            set(by_key),
            {"Performance", "Pipeline", "ProjectState", "Memory", "Prompt"},
        )
        self.assertEqual(by_key["Performance"].kind, "performance")
        self.assertEqual(by_key["Prompt"].kind, "prompt")
        self.assertEqual(by_key["Pipeline"].kind, "section")
        self.assertEqual(by_key["Prompt"].display_name, "Prompt Inspector")

    def test_selected_turn_and_fallback(self):
        turns = _multi_turns(3)
        plan = build_tree_plan(turns, selected_index=1)
        flags = [tn.selected for tn in plan.turns]
        self.assertEqual(flags, [False, True, False])
        # None / out-of-range falls back to the last turn.
        self.assertTrue(build_tree_plan(turns, selected_index=None).turns[-1].selected)
        self.assertTrue(build_tree_plan(turns, selected_index=999).turns[-1].selected)

    def test_all_turns_carry_their_children(self):
        turns = _multi_turns(3)
        plan = build_tree_plan(turns, selected_index=0)
        for tn in plan.turns:
            self.assertGreater(tn.section_count, 0)
            self.assertTrue(tn.categories)

    def test_category_order_respected_within_turn(self):
        order = [c.name for c in self.plan.turns[0].categories]
        self.assertEqual(order, [n for n in CATEGORY_ORDER if n in order])


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.turns = _multi_turns(3)

    def test_matches_turn_label_and_ordinal(self):
        plan = build_tree_plan(self.turns, query="turn 2")
        self.assertTrue(plan.turns[1].search_matched)
        self.assertTrue(plan.turns[1].visible)
        plan2 = build_tree_plan(self.turns, query="3")
        self.assertTrue(plan2.turns[2].search_matched)

    def test_matches_section_display_name(self):
        plan = build_tree_plan(self.turns, selected_index=0, query="project state")
        hit = next(tn for tn in plan.turns if tn.visible)
        hit_sec = next(s for s in _all_sections(hit) if s.visible)
        self.assertEqual(hit_sec.key, "ProjectState")

    def test_matches_section_key(self):
        plan = build_tree_plan(self.turns, selected_index=0, query="projectstate")
        hit = next(tn for tn in plan.turns if tn.visible)
        self.assertTrue(any(s.key == "ProjectState" and s.visible
                            for s in _all_sections(hit)))

    def test_matches_category_name(self):
        plan = build_tree_plan([_single_turn()], query="quality audits")
        self.assertEqual(plan.visible_sections, 0)  # no QA sections in fixture
        # A turn with a Quality Audits section surfaces via category match.
        diag = _make_diagnostics({"ConversationFailure": {"turns": 1}})
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics=diag)
        plan2 = build_tree_plan([turn], query="quality audits")
        self.assertTrue(plan2.turns[0].visible)
        self.assertEqual(plan2.visible_sections, 1)

    def test_section_text_is_searched(self):
        # 'mentor' only occurs inside the Prompt text — not the name/key/category.
        plan = build_tree_plan([_single_turn()], query="mentor")
        self.assertEqual(plan.visible_sections, 1)
        hit = next(s for s in _all_sections(plan.turns[0]) if s.visible)
        self.assertEqual(hit.key, "Prompt")
        self.assertTrue(hit.search_matched)
        # The (always-visible) selected turn row remains.
        self.assertEqual(plan.visible_turns, 1)

    def test_performance_text_is_searched(self):
        # 'Total' only appears in the rendered Performance timing block.
        plan = build_tree_plan([_single_turn()], query="Total")
        self.assertEqual(plan.visible_sections, 1)
        hit = next(s for s in _all_sections(plan.turns[0]) if s.visible)
        self.assertEqual(hit.key, "Performance")
        self.assertEqual(hit.kind, "performance")

    def test_content_search_never_modifies_diagnostics(self):
        diag = _make_diagnostics()
        snapshot = copy.deepcopy(diag)
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics=diag)
        build_tree_plan([turn], query="mentor")
        self.assertEqual(diag, snapshot)

    def test_selected_turn_stays_visible_under_query(self):
        # Turn 1 (selected) has no Memory-bearing section match for 'memory'?
        # It has Memory unselected... give it a non-matching-only set instead.
        turns = [
            build_turn(turn_index=0, ordinal=1, role="assistant",
                       diagnostics={"Pipeline": {"Stage": "E"}}),
            build_turn(turn_index=1, ordinal=2, role="assistant",
                       diagnostics={"Memory": {"resolved": ["age"]}}),
        ]
        plan = build_tree_plan(turns, selected_index=0, query="memory")
        # Selected turn always visible even though only the other matches.
        self.assertTrue(plan.turns[0].selected)
        self.assertTrue(plan.turns[0].visible)
        self.assertFalse(plan.turns[0].search_matched)
        self.assertTrue(plan.turns[1].visible)
        self.assertEqual(plan.visible_turns, 2)
        self.assertEqual(plan.visible_sections, 1)

    def test_search_never_modifies_diagnostics(self):
        diag = _make_diagnostics()
        snapshot = copy.deepcopy(diag)
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics=diag)
        build_tree_plan([turn], query="project")
        build_tree_plan([turn], query="")
        build_tree_plan([turn], query="zzz")
        self.assertEqual(diag, snapshot)

    def test_no_match_hides_everything_except_selection(self):
        turns = _multi_turns(2)
        plan = build_tree_plan(turns, selected_index=None, query="doesnotexist-xyz")
        self.assertEqual(plan.visible_sections, 0)
        self.assertEqual(plan.visible_turns, 1)  # only the fallback selection
        self.assertTrue(plan.turns[-1].selected)


class TestExpandCollapse(unittest.TestCase):

    def test_defaults_expand_performance_and_prompt(self):
        plan = build_tree_plan([_single_turn()])
        sections = {s.key: s for s in _all_sections(plan.turns[0])}
        for key in DEFAULT_EXPANDED_SECTIONS:
            self.assertTrue(sections[key].expanded)
        self.assertFalse(sections["ProjectState"].expanded)
        self.assertFalse(sections["Memory"].expanded)

    def test_per_turn_collapse_override(self):
        plan = build_tree_plan(
            [_single_turn()],
            collapse={0: {"Prompt": False, "ProjectState": True}},
        )
        sections = {s.key: s for s in _all_sections(plan.turns[0])}
        self.assertFalse(sections["Prompt"].expanded)
        self.assertTrue(sections["ProjectState"].expanded)

    def test_str_turn_keys_are_normalised(self):
        plan = build_tree_plan(
            [_single_turn()],
            collapse={"0": {"ProjectState": True}},
        )
        sections = {s.key: s for s in _all_sections(plan.turns[0])}
        self.assertTrue(sections["ProjectState"].expanded)

    def test_expand_all_turns_every_section_on(self):
        present = {s.key for s in _all_sections(
            build_tree_plan([_single_turn()]).turns[0])}
        plan = build_tree_plan(
            [_single_turn()],
            collapse={0: {k: True for k in present}},
        )
        sections = _all_sections(plan.turns[0])
        self.assertTrue(all(s.expanded for s in sections))

    def test_collapse_all_turns_everything_off(self):
        present = {s.key for s in _all_sections(
            build_tree_plan([_single_turn()]).turns[0])}
        plan = build_tree_plan(
            [_single_turn()],
            collapse={0: {k: False for k in present}},
        )
        sections = _all_sections(plan.turns[0])
        self.assertFalse(any(s.expanded for s in sections))


class TestPinning(unittest.TestCase):

    def test_pinned_section_expands_in_every_turn(self):
        turns = _multi_turns(2)
        plan = build_tree_plan(turns, pins=["ProjectState"])
        for tn in plan.turns:
            sections = {s.key: s for s in _all_sections(tn)}
            self.assertTrue(sections["ProjectState"].pinned)
            self.assertTrue(sections["ProjectState"].expanded)

    def test_pin_applies_across_all_turns_globally(self):
        turns = _multi_turns(2)
        plan = build_tree_plan(turns, pins=["Memory"])
        for tn in plan.turns:
            sections = {s.key: s for s in _all_sections(tn)}
            self.assertTrue(sections["Memory"].pinned)
            self.assertTrue(sections["Memory"].expanded)

    def test_pin_wins_over_collapse_all(self):
        plan = build_tree_plan(
            [_single_turn()],
            pins=["ProjectState"],
            collapse={0: {"ProjectState": False, "Memory": False}},
        )
        sections = {s.key: s for s in _all_sections(plan.turns[0])}
        self.assertTrue(sections["ProjectState"].expanded)  # pinned
        self.assertFalse(sections["Memory"].expanded)       # not pinned

    def test_unpinned_sections_stay_collapsed_by_default(self):
        plan = build_tree_plan([_single_turn()], pins=["ProjectState"])
        sections = {s.key: s for s in _all_sections(plan.turns[0])}
        self.assertFalse(sections["Memory"].pinned)
        self.assertFalse(sections["Memory"].expanded)


class TestDiagnosticsIdentityAndRendering(unittest.TestCase):

    def test_tree_does_not_mutate_diagnostics(self):
        diag = _make_diagnostics()
        snapshot = copy.deepcopy(diag)
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics=diag, timing={"total_ms": 10})
        build_tree_plan([turn], pins=["Memory"], query="project")
        self.assertEqual(diag, snapshot)

    def test_section_text_identical_to_render_plan(self):
        turn = _single_turn()
        flat = {e["key"]: e for e in build_render_plan(turn)}
        tree = _all_sections(build_tree_plan([turn]).turns[0])
        for section in tree:
            self.assertEqual(section.text, flat[section.key]["text"])

    def test_collect_turns_preserves_diagnostics_reference(self):
        diag = _make_diagnostics()
        messages = [{"role": "assistant", "content": "x",
                     "diagnostics": diag, "timing": {"total_ms": 1}}]
        turns = collect_turns(messages)
        self.assertIs(turns[0].diagnostics, diag)
        build_tree_plan(turns, query="project")
        self.assertIs(turns[0].diagnostics, diag)


class TestSearchDrivenExpansion(unittest.TestCase):
    """While a query is active the plan auto-expands matching sections and
    collapses unrelated ones; clearing the query restores defaults/toggles."""

    def _turn(self):
        return build_turn(
            turn_index=0,
            ordinal=1,
            role="assistant",
            content="Hello from the mentor",
            timing={"total_ms": 100.0, "llm_ms": 40.0},
            diagnostics={
                "Pipeline": {"Stage": "Empathize"},
                "Memory": {"Open Threads": [{"field": "frequency", "reason": "asked"}]},
                "Prompt": "You are a mentor. Ask about frequency.",
            },
        )

    def _sections(self, plan):
        node = plan.turns[0]
        out = {}
        for category in node.categories:
            for section in category.sections:
                out[section.key] = section
        return out

    def test_matching_sections_auto_expand(self):
        # 'frequency' appears in Memory (content) and Prompt (content).
        sections = self._sections(build_tree_plan([self._turn()], query="frequency"))
        self.assertTrue(sections["Memory"].visible)
        self.assertTrue(sections["Memory"].expanded)
        self.assertTrue(sections["Prompt"].visible)
        self.assertTrue(sections["Prompt"].expanded)

    def test_unrelated_sections_collapse_and_hide(self):
        sections = self._sections(build_tree_plan([self._turn()], query="frequency"))
        self.assertFalse(sections["Pipeline"].visible)
        self.assertFalse(sections["Pipeline"].expanded)
        # Performance is expanded by default but unrelated to the query.
        self.assertFalse(sections["Performance"].expanded)

    def test_title_match_expands_section(self):
        # 'pipeline' only matches the section title/key/category, not content.
        sections = self._sections(build_tree_plan([self._turn()], query="pipeline"))
        self.assertTrue(sections["Pipeline"].visible)
        self.assertTrue(sections["Pipeline"].expanded)

    def test_clearing_query_restores_defaults(self):
        sections = self._sections(build_tree_plan([self._turn()], query=""))
        self.assertEqual(sections["Performance"].expanded, True)
        self.assertEqual(sections["Prompt"].expanded, True)
        self.assertEqual(sections["Memory"].expanded, False)
        self.assertTrue(sections["Memory"].visible)

    def test_pinned_sections_stay_open_during_search(self):
        sections = self._sections(
            build_tree_plan([self._turn()], query="frequency", pins=["Pipeline"])
        )
        self.assertTrue(sections["Pipeline"].pinned)
        self.assertTrue(sections["Pipeline"].expanded)

    def test_manual_expand_state_ignored_while_searching(self):
        # A previously opened section that does NOT match still collapses.
        sections = self._sections(
            build_tree_plan([self._turn()], query="frequency",
                            collapse={0: {"Pipeline": True}})
        )
        self.assertFalse(sections["Pipeline"].expanded)

    def test_no_query_uses_explicit_collapse(self):
        sections = self._sections(
            build_tree_plan([self._turn()], collapse={0: {"Pipeline": True}})
        )
        self.assertTrue(sections["Pipeline"].expanded)


class TestHighlightMatches(unittest.TestCase):

    def test_wraps_match_in_mark(self):
        self.assertEqual(
            highlight_matches("frequency: rarely", "frequency"),
            '<mark class="dc-search-hit">frequency</mark>: rarely',
        )

    def test_highlights_every_occurrence(self):
        out = highlight_matches("llm then llm", "llm")
        self.assertEqual(out.count('<mark class="dc-search-hit">'), 2)

    def test_case_insensitive(self):
        self.assertIn(
            '<mark class="dc-search-hit">Prompt</mark>',
            highlight_matches("The Prompt is here", "prompt"),
        )

    def test_empty_query_returns_escaped_text_unchanged(self):
        self.assertEqual(highlight_matches("a < b & c", ""), "a &lt; b &amp; c")

    def test_multiple_terms(self):
        out = highlight_matches("rule and llm disagree", "rule llm")
        self.assertEqual(out.count('<mark class="dc-search-hit">'), 2)

    def test_escapes_html_inside_and_around_matches(self):
        self.assertEqual(
            highlight_matches("<b>frequency</b>", "frequency"),
            "&lt;b&gt;<mark class=\"dc-search-hit\">frequency</mark>&lt;/b&gt;",
        )

    def test_no_match_returns_escaped_text(self):
        self.assertEqual(highlight_matches("nothing here", "zzz"), "nothing here")


if __name__ == "__main__":
    unittest.main(verbosity=2)