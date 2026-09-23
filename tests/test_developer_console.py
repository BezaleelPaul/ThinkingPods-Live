"""
tests/test_developer_console.py — regression tests for the Developer Console
right-side inspector panel builder (``developer_console.py``).

The Streamlit frontend only executes UI primitives; this module is the sole,
unit-tested source of the panel logic. Covers the redesign requirements:

  * diagnostics displayed correctly
  * turn switching
  * collapsed/expanded state
  * prompt rendering
  * identical underlying diagnostic dictionaries

None of these tests touch conversation behavior, extraction, objectives, or
live pipeline diagnostics — they build deterministic fake diagnostics exactly
in the shape ``mentor._build_diagnostics`` produces and assert the pure
console model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from developer_console import (  # noqa: E402
    DEFAULT_EXPANDED_SECTIONS,
    CollapseState,
    build_render_plan,
    build_timeline,
    build_turn,
    collect_turns,
    highlight_prompt,
    prompt_stats,
    prompt_text,
    prompt_viewer_stats,
    render_prompt_text,
    render_section_text,
    render_timing_lines,
    select_turn,
    turn_label,
)


def _make_diagnostics():
    """Deterministic fake diagnostics in the pipeline's shape."""
    return {
        "Pipeline": {"Stage": "Empathize", "Objective": "PROBLEMS"},
        "ProjectState": {"personas": ["elderly people"]},
        "StateChanges": [{"operation": "ADD", "field": "personas", "value": "elderly people"}],
        "Prompt": (
            "You are a mentor. Celebrate genuinely useful discoveries "
            "briefly and never ask multiple questions."
        ),
    }


def _make_timing():
    return {
        "total_ms": 125.0,
        "llm_ms": 80.0,
        "extraction_ms": 20.0,
        "extraction_breakdown": {"total_ms": 20.0, "rules_ms": 12.0, "merge_ms": 3.0},
    }


def _assistant_message(index, content="Hello from the mentor", **extra):
    msg = {"role": "assistant", "content": content}
    msg.update(extra)
    return msg


class TestDiagnosticsDisplayedCorrectly(unittest.TestCase):

    def setUp(self):
        self.diag = _make_diagnostics()
        self.timing = _make_timing()

    def test_collect_turns_finds_all_data_carrying_turns(self):
        messages = [
            {"role": "user", "content": "hi"},
            _assistant_message(1, diagnostics=self.diag, timing=self.timing),
            {"role": "user", "content": "more"},
            _assistant_message(3, diagnostics=self.diag),
        ]
        turns = collect_turns(messages)
        self.assertEqual(len(turns), 2)
        self.assertEqual([t.turn_index for t in turns], [1, 3])
        self.assertEqual([t.ordinal for t in turns], [1, 2])

    def test_plan_includes_every_section_preserving_order(self):
        plan = build_render_plan(build_turn(turn_index=0, ordinal=1, role="assistant",
                                            diagnostics=self.diag, timing=self.timing))
        kinds = [e["kind"] for e in plan]
        section_keys = [e["key"] for e in plan if e["kind"] == "section"]
        # Prompt is its own block; every other section is preserved in order.
        self.assertEqual(section_keys, ["Pipeline", "ProjectState", "StateChanges"])
        self.assertIn("prompt", kinds)
        self.assertIn("performance", kinds)

    def test_section_text_contains_expected_values(self):
        plan = build_render_plan(build_turn(turn_index=0, ordinal=1, role="assistant",
                                            diagnostics=self.diag, timing=self.timing))
        by_key = {e["key"]: e for e in plan}
        self.assertIn("PROBLEMS", by_key["Pipeline"]["text"])
        self.assertIn("elderly people", by_key["ProjectState"]["text"])
        self.assertIn("ADD", by_key["StateChanges"]["text"])

    def test_performance_block_renders(self):
        text = render_timing_lines(self.timing)
        self.assertIn("Total", text)
        self.assertIn("125 ms", text)
        self.assertIn("Extraction", text)

    def test_empty_section_shows_placeholder(self):
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics={"Pipeline": {}})
        section = next(s for s in turn.sections if s.key == "Pipeline")
        self.assertEqual(render_section_text(section), "No state changes")


class TestTurnSwitching(unittest.TestCase):

    def setUp(self):
        self.diag = _make_diagnostics()
        self.diag2 = _make_diagnostics()
        self.diag2["ProjectState"] = {"personas": ["students"]}
        self.messages = [
            {"role": "user", "content": "a"},
            _assistant_message(1, content="First reply", diagnostics=self.diag),
            {"role": "user", "content": "b"},
            _assistant_message(3, content="Second reply", diagnostics=self.diag2),
        ]
        self.turns = collect_turns(self.messages)

    def test_select_turn_by_index(self):
        turn = select_turn(self.turns, 3)
        self.assertIsNotNone(turn)
        self.assertEqual(turn.turn_index, 3)
        self.assertEqual(turn.ordinal, 2)
        self.assertEqual(turn.content_preview, "Second reply")

    def test_select_first_turn(self):
        turn = select_turn(self.turns, self.turns[0].turn_index)
        self.assertEqual(turn.ordinal, 1)

    def test_select_out_of_range_returns_none(self):
        self.assertIsNone(select_turn(self.turns, 999))
        self.assertIsNone(select_turn(self.turns, None))

    def test_turn_labels_are_stable_and_distinct(self):
        labels = [turn_label(t) for t in self.turns]
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], "Turn 1")
        self.assertEqual(labels[1], "Turn 2")
        # Labels must never leak assistant text.
        self.assertNotIn("First reply", labels[0])
        self.assertNotIn("Second reply", labels[1])

    def test_select_each_turn_yields_its_own_plan(self):
        first = select_turn(self.turns, self.turns[0].turn_index)
        second = select_turn(self.turns, self.turns[1].turn_index)
        p1 = build_render_plan(first)
        p2 = build_render_plan(second)
        self.assertNotEqual(p1, p2)  # content previews differ by design
        keys1 = {e["key"] for e in p1}
        keys2 = {e["key"] for e in p2}
        self.assertEqual(keys1, keys2)  # same section set survives switching


class TestCollapsedExpandedState(unittest.TestCase):

    def test_defaults_expand_prompt_and_performance(self):
        plan = build_render_plan(build_turn(turn_index=0, ordinal=1, role="assistant",
                                            diagnostics=_make_diagnostics()))
        by_key = {e["key"]: e for e in plan}
        self.assertTrue(by_key["Prompt"]["expanded"])

    def test_collapse_state_toggle(self):
        state = CollapseState()
        self.assertFalse(state.is_expanded("ProjectState"))
        state.toggle("ProjectState")
        self.assertTrue(state.is_expanded("ProjectState"))
        state.toggle("ProjectState")
        self.assertFalse(state.is_expanded("ProjectState"))

    def test_collapse_state_override_flips_plan(self):
        diag = _make_diagnostics()
        state = CollapseState(defaults={"ProjectState": True})
        state.set_expanded("ProjectState", False)  # now collapsed
        plan = build_render_plan(
            build_turn(turn_index=0, ordinal=1, role="assistant", diagnostics=diag),
            collapse_state=state,
        )
        by_key = {e["key"]: e for e in plan}
        self.assertFalse(by_key["ProjectState"]["expanded"])
        # Defaults still apply for keys not toggled.
        self.assertTrue(by_key["Prompt"]["expanded"])

    def test_collapse_state_serialization(self):
        state = CollapseState()
        state.toggle("ProjectState")
        state.set_expanded("Memory", True)
        restored = CollapseState.from_dict(state.to_dict())
        self.assertEqual(restored.to_dict(), state.to_dict())
        self.assertTrue(restored.is_expanded("ProjectState"))
        self.assertTrue(restored.is_expanded("Memory"))

    def test_build_plan_without_state_uses_module_defaults(self):
        self.assertIn("Prompt", DEFAULT_EXPANDED_SECTIONS)
        plan = build_render_plan(build_turn(turn_index=0, ordinal=1, role="assistant",
                                            diagnostics=_make_diagnostics()))
        expanded = {e["key"] for e in plan if e["expanded"]}
        self.assertEqual(expanded, set(DEFAULT_EXPANDED_SECTIONS))


class TestPromptRendering(unittest.TestCase):

    def test_prompt_is_own_expandable_block(self):
        plan = build_render_plan(build_turn(turn_index=0, ordinal=1, role="assistant",
                                            diagnostics=_make_diagnostics()))
        prompt_entries = [e for e in plan if e["kind"] == "prompt"]
        self.assertEqual(len(prompt_entries), 1)
        self.assertEqual(prompt_entries[0]["key"], "Prompt")
        self.assertEqual(prompt_entries[0]["display_name"], "Prompt Inspector")

    def test_prompt_text_preserved_verbatim(self):
        diag = _make_diagnostics()
        turn = build_turn(turn_index=0, ordinal=1, role="assistant", diagnostics=diag)
        self.assertEqual(prompt_text(turn), diag["Prompt"])
        self.assertEqual(prompt_text(turn), diag["Prompt"])

    def test_structured_prompt_renders(self):
        diag = {"Prompt": {"system": "be a mentor", "user": "help students"}}
        turn = build_turn(turn_index=0, ordinal=1, role="assistant", diagnostics=diag)
        rendered = render_prompt_text(diag["Prompt"])
        self.assertIn("be a mentor", rendered)
        self.assertIn("help students", rendered)

    def test_render_section_text_uses_prompt_block_for_prompt(self):
        diag = _make_diagnostics()
        turn = build_turn(turn_index=0, ordinal=1, role="assistant", diagnostics=diag)
        prompt_sections = [s for s in turn.sections if s.is_prompt]
        self.assertEqual(len(prompt_sections), 1)
        self.assertEqual(
            render_section_text(prompt_sections[0]), diag["Prompt"]
        )

    def test_empty_prompt_text(self):
        turn = build_turn(turn_index=0, ordinal=1, role="assistant",
                          diagnostics={"Prompt": None})
        self.assertEqual(prompt_text(turn), "No state changes")


class TestIdenticalDiagnosticDictionaries(unittest.TestCase):

    def test_turn_holds_original_diagnostics_by_reference(self):
        diag = _make_diagnostics()
        message = _assistant_message(1, diagnostics=diag)
        turn = build_turn(turn_index=1, ordinal=1, role="assistant",
                          diagnostics=message["diagnostics"])
        self.assertIs(turn.diagnostics, diag)

    def test_section_items_are_the_original_objects(self):
        diag = _make_diagnostics()
        turn = build_turn(turn_index=0, ordinal=1, role="assistant", diagnostics=diag)
        self.assertEqual(len(turn.sections), len(diag))
        for section in turn.sections:
            self.assertIs(section.items, diag[section.key])

    def test_render_plan_does_not_mutate_diagnostics(self):
        import copy

        diag = _make_diagnostics()
        snapshot = copy.deepcopy(diag)
        plan = build_render_plan(
            build_turn(turn_index=0, ordinal=1, role="assistant",
                       diagnostics=diag, timing=_make_timing())
        )
        self.assertTrue(plan)
        self.assertEqual(diag, snapshot)  # untouched after rendering

    def test_collect_turns_preserves_message_diagnostics_reference(self):
        diag = _make_diagnostics()
        messages = [_assistant_message(1, diagnostics=diag)]
        turns = collect_turns(messages)
        self.assertIs(turns[0].diagnostics, diag)


class TestPromptStats(unittest.TestCase):

    def test_counts_words_and_characters(self):
        stats = prompt_stats("Hello world mentor")
        self.assertEqual(stats["words"], 3)
        self.assertEqual(stats["characters"], len("Hello world mentor"))

    def test_empty_text_is_zero(self):
        stats = prompt_stats("")
        self.assertEqual(stats, {"characters": 0, "words": 0})

    def test_none_treated_as_empty(self):
        stats = prompt_stats(None)
        self.assertEqual(stats, {"characters": 0, "words": 0})

    def test_counts_include_whitespace_characters(self):
        stats = prompt_stats("   one two  ")
        self.assertEqual(stats["words"], 2)
        self.assertEqual(stats["characters"], 12)


class TestPromptViewerStats(unittest.TestCase):

    def test_multi_line_counts(self):
        stats = prompt_viewer_stats("You are a mentor.\nAsk about {frequency}.\n")
        self.assertEqual(stats["characters"], 41)
        self.assertEqual(stats["words"], 7)
        self.assertEqual(stats["lines"], 2)
        self.assertEqual(stats["bytes"], 41)

    def test_empty_is_zero(self):
        stats = prompt_viewer_stats("")
        self.assertEqual(stats, {"characters": 0, "words": 0, "lines": 0, "bytes": 0})

    def test_none_treated_as_empty(self):
        stats = prompt_viewer_stats(None)
        self.assertEqual(stats["lines"], 0)

    def test_single_line(self):
        stats = prompt_viewer_stats("one two")
        self.assertEqual(stats["lines"], 1)
        self.assertEqual(stats["words"], 2)

    def test_bytes_count_utf8(self):
        stats = prompt_viewer_stats("café")
        self.assertEqual(stats["characters"], 4)
        self.assertEqual(stats["bytes"], 5)


class TestHighlightPrompt(unittest.TestCase):

    def test_role_header_is_highlighted(self):
        out = highlight_prompt("ROLE: Mentor")
        self.assertIn('<span class="pc-h">', out)

    def test_markdown_header_is_highlighted(self):
        self.assertIn('class="pc-h"', highlight_prompt("# Instructions"))

    def test_placeholders_get_token_marks(self):
        out = highlight_prompt("Ask about {frequency}.")
        self.assertIn('<span class="pc-tok">{frequency}</span>', out)
        self.assertIn('class="pc-tokline"', out)

    def test_list_bullets_are_highlighted(self):
        out = highlight_prompt("- first\n- second")
        self.assertEqual(out.count('class="pc-list"'), 2)

    def test_html_is_escaped(self):
        out = highlight_prompt("<b>{x}</b>")
        self.assertNotIn("<b>", out)
        self.assertIn("&lt;b&gt;", out)
        self.assertIn("<span class=\"pc-tok\">{x}</span>", out)

    def test_empty_prompt_is_empty(self):
        self.assertEqual(highlight_prompt(""), "")

    def test_plain_line_has_no_class(self):
        out = highlight_prompt("just a normal line")
        self.assertEqual(out, "just a normal line")

    def test_multiline_preserves_newlines(self):
        out = highlight_prompt("line one\nline two")
        self.assertEqual(out, "line one\nline two")


class TestTimeline(unittest.TestCase):
    """Fixed-stage pipeline timeline built from timing + diagnostics."""

    STAGE_KEYS = [
        "user_message",
        "memory_extraction",
        "objective_engine",
        "recovery",
        "question_family",
        "prompt_builder",
        "llm",
        "response",
        "diagnostics",
    ]

    def _diag(self, **extra):
        diag = {
            "Pipeline": {"Stage": "Empathize", "Objective": "PROBLEMS"},
            "Extraction": {"message_type": "MEANINGFUL",
                           "updates": [{"field": "audience"}]},
            "ExtractionComparison": {"Decision": {"Rule sufficient": "Yes"}},
            "ObjectiveTrace": {"Objective": "PROBLEMS",
                               "Confidence": 1.0,
                               "Advancement": {"Verdict": "CONTINUE"}},
            "Recovery": {"Category": "NONE"},
            "QuestionFamilies": {"Question Family": "consequences"},
            "Prompt": "Ask about the consequences.",
            "ConversationStyle": {"Questions": 1, "Begins with Acknowledgment": "Yes"},
            "ConversationMove": {"Move": "EXPLORE"},
            "ConversationFailure": {"Primary Failure": "None",
                                    "Secondary Failures": None},
        }
        diag.update(extra)
        return diag

    def _timing(self, **extra):
        timing = {
            "total_ms": 125.0,
            "llm_ms": 80.0,
            "extraction_ms": 20.0,
            "extraction_breakdown": {"total_ms": 20.0, "rules_ms": 12.0},
        }
        timing.update(extra)
        return timing

    def _turn(self, content="Hello mentor", diag=None, timing=None):
        return build_turn(
            turn_index=1, ordinal=1, role="assistant",
            content=content,
            timing=timing if timing is not None else self._timing(),
            diagnostics=diag if diag is not None else self._diag(),
        )

    def test_stages_in_fixed_order(self):
        stages = build_timeline(self._turn())
        self.assertEqual([s.key for s in stages], self.STAGE_KEYS)

    def test_none_turn_returns_empty(self):
        self.assertEqual(build_timeline(None), [])

    def test_duration_mapping_from_timing(self):
        stages = {s.key: s for s in build_timeline(self._turn())}
        self.assertEqual(stages["memory_extraction"].duration_ms, 20.0)
        self.assertEqual(stages["llm"].duration_ms, 80.0)
        self.assertEqual(stages["diagnostics"].duration_ms, 125.0)
        # Stages without dedicated timing stay None.
        self.assertIsNone(stages["recovery"].duration_ms)
        self.assertIsNone(stages["question_family"].duration_ms)
        self.assertIsNone(stages["user_message"].duration_ms)

    def test_user_message_carries_preview(self):
        stages = build_timeline(self._turn(content="Short hello"))
        self.assertEqual(stages[0].key, "user_message")
        self.assertEqual(stages[0].metadata, [("message", "Short hello")])

    def test_recovery_contradiction_is_problem(self):
        stages = {s.key: s for s in build_timeline(
            self._turn(diag=self._diag(Recovery={"Category": "CONTRADICTION",
                                                 "Detected Inconsistency": "conflict"})))}
        self.assertEqual(stages["recovery"].status, "problem")
        self.assertIn("conflict", stages["recovery"].status_reason)
        self.assertEqual(stages["recovery"].metadata, [("category", "CONTRADICTION")])

    def test_llm_zero_with_total_warns_rules_only(self):
        stages = {s.key: s for s in build_timeline(
            self._turn(timing=self._timing(llm_ms=0)))}
        self.assertEqual(stages["llm"].status, "warning")
        self.assertIn("rules-only", stages["llm"].status_reason)

    def test_diagnostics_node_aggregates_worst_status(self):
        stages = {s.key: s for s in build_timeline(
            self._turn(diag=self._diag(ConversationFailure={
                "Primary Failure": "MISUNDERSTOOD_RESPONSE",
                "Secondary Failures": None,
            })))}
        self.assertEqual(stages["diagnostics"].status, "problem")
        self.assertIn(("problems", 1), stages["diagnostics"].metadata)

    def test_extraction_metadata(self):
        stages = {s.key: s for s in build_timeline(self._turn())}
        meta = dict(stages["memory_extraction"].metadata)
        self.assertEqual(meta.get("type"), "MEANINGFUL")
        self.assertEqual(meta.get("updates"), 1)
        self.assertEqual(meta.get("rule sufficient"), "Yes")

    def test_prompt_metadata(self):
        stages = {s.key: s for s in build_timeline(self._turn())}
        meta = dict(stages["prompt_builder"].metadata)
        self.assertEqual(meta.get("chars"), 27)
        self.assertEqual(meta.get("words"), 4)

    def test_timeline_is_read_only(self):
        import copy

        diag = self._diag()
        timing = self._timing()
        snapshot_diag = copy.deepcopy(diag)
        snapshot_timing = copy.deepcopy(timing)
        stages = build_timeline(self._turn(diag=diag, timing=timing))
        self.assertTrue(stages)
        self.assertEqual(diag, snapshot_diag)
        self.assertEqual(timing, snapshot_timing)


if __name__ == "__main__":
    unittest.main(verbosity=2)