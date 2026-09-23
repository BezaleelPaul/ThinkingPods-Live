"""
tests/test_conversation_export.py — regression tests for the deterministic
conversation export utilities (``conversation_export.py``).

Covers the required export scenarios:

  * markdown generation          — Copy Conversation / + Diagnostics / full
                                  Markdown shape and content.
  * json generation              — complete-session payload and canonical JSON.
  * empty conversation           — exporters handle zero messages gracefully.
  * diagnostics inclusion        — Developer Diagnostics block carries timing
                                  + each section verbatim.
  * byte-identical reload        — export → reload → export round-trips to the
                                  exact same bytes.

No conversation logic is touched — these tests build deterministic fake
messages (the same shape the app stores) and exercise the pure exporters.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_export import (  # noqa: E402
    build_conversation_copy,
    build_conversation_diagnostics_copy,
    build_full_markdown,
    build_session_payload,
    conversation_metrics_from_messages,
    export_session_json,
    reload_session_json,
    serialize_session_json,
)


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content, diagnostics=None, timing=None):
    msg = {
        "role": "assistant",
        "content": content,
        "audio": "AAABQlAAA",
    }
    if diagnostics is not None:
        msg["diagnostics"] = diagnostics
    if timing is not None:
        msg["timing"] = timing
    return msg


def _diagnostics():
    return {
        "Pipeline": {"Stage": "Empathize", "Objective": "PROBLEMS"},
        "ConversationMetrics": {"Turns": 1, "Objectives Completed": 0},
    }


def _timing():
    return {"total_ms": 125.0, "llm_ms": 80.0}


def _messages(include_diagnostics=True):
    msgs = [_user("I help elderly people take medicine on time")]
    if include_diagnostics:
        msgs.append(_assistant("Got it — who is the target audience?", _diagnostics(), _timing()))
    else:
        msgs.append(_assistant("Got it — who is the target audience?"))
    return msgs


class TestMarkdownGeneration(unittest.TestCase):

    def test_conversation_copy_has_roles_and_content(self):
        md = build_conversation_copy(_messages())
        self.assertIn("**User:**", md)
        self.assertIn("I help elderly people", md)
        self.assertIn("**Mentor:**", md)
        self.assertIn("who is the target audience?", md)

    def test_conversation_copy_excludes_diagnostics(self):
        md = build_conversation_copy(_messages())
        self.assertNotIn("Developer Diagnostics", md)
        self.assertNotIn("Pipeline", md)
        self.assertNotIn("OBJECTIVES", md)

    def test_diagnostics_copy_includes_diagnostics(self):
        md = build_conversation_diagnostics_copy(_messages())
        self.assertIn("Developer Diagnostics", md)
        self.assertIn("Pipeline", md)
        self.assertIn("Empathize", md)
        self.assertIn("Conversation", md)

    def test_full_markdown_has_all_four_sections(self):
        md = build_full_markdown(
            _messages(),
            project_name="MediTrack",
            timestamp="2026-08-05T10:00:00",
            saved_missions=["ProjectA"],
        )
        for heading in (
            "## Conversation",
            "## Developer Diagnostics",
            "## Mission Memory",
            "## Conversation Metrics",
        ):
            self.assertIn(heading, md)
        self.assertIn("**Project:** MediTrack", md)
        self.assertIn("**Exported:** 2026-08-05T10:00:00", md)

    def test_metrics_section_present_in_full_markdown(self):
        md = build_full_markdown(_messages())
        self.assertIn("Turns", md)
        self.assertIn("Objectives Completed", md)

    def test_deterministic_output(self):
        m1 = _messages()
        m2 = _messages(include_diagnostics=True)
        self.assertEqual(build_full_markdown(m1), build_full_markdown(m2))


class TestJSONGeneration(unittest.TestCase):

    def test_payload_preserves_complete_session(self):
        payload = build_session_payload(
            _messages(),
            project_name="MediTrack",
            dt_phase="Empathize",
            session_id="abc123",
            timestamp="2026-08-05T10:00:00",
            saved_missions=["ProjectA"],
        )
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["project_name"], "MediTrack")
        self.assertEqual(payload["session_id"], "abc123")
        self.assertEqual(len(payload["messages"]), 2)
        # The diagnostics/timing/audio dicts are preserved whole.
        self.assertIn("diagnostics", payload["messages"][1])
        self.assertIn("audio", payload["messages"][1])
        self.assertIn("timing", payload["messages"][1])

    def test_serialize_json_is_valid_and_contains_everything(self):
        text = serialize_session_json(_messages(), session_id="abc123")
        data = json.loads(text)
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["session_id"], "abc123")
        self.assertEqual(data["conversation_metrics"]["Turns"], 1)

    def test_empty_messages_json(self):
        text = serialize_session_json([])
        data = json.loads(text)
        self.assertEqual(data["messages"], [])

    def test_canonical_keys_sorted(self):
        payload = build_session_payload(_messages())
        text = export_session_json(payload)
        # First top-level key alphabetically.
        self.assertTrue(text.startswith("{\n  \"conversation_metrics\""))


class TestEmptyConversation(unittest.TestCase):

    def test_conversation_copy_empty(self):
        md = build_conversation_copy([])
        self.assertIn("No messages", md)

    def test_diagnostics_copy_empty(self):
        md = build_conversation_diagnostics_copy([])
        self.assertIn("No messages", md)
        self.assertIn("No diagnostics", md)

    def test_full_markdown_empty(self):
        md = build_full_markdown(
            [], timestamp="2026-08-05T10:00:00", saved_missions=None
        )
        self.assertIn("No messages", md)
        self.assertIn("No diagnostics", md)
        self.assertIn("No conversation metrics", md)

    def test_json_empty_conversation(self):
        payload = build_session_payload([])
        self.assertEqual(payload["messages"], [])
        self.assertEqual(payload["conversation_metrics"], {})


class TestDiagnosticsInclusion(unittest.TestCase):

    def test_diagnostics_text_contains_section_values(self):
        md = build_conversation_diagnostics_copy(_messages())
        self.assertIn("Empathize", md)
        self.assertIn("PROBLEMS", md)

    def test_timing_included_when_present(self):
        md = build_conversation_diagnostics_copy(_messages())
        self.assertIn("Total:", md)
        self.assertIn("LLM:", md)

    def test_turn_ordinals_number_assistant_turns(self):
        md = build_conversation_diagnostics_copy(
            [
                _user("first"),
                _assistant("reply one", _diagnostics(), _timing()),
                _user("second"),
                _assistant("reply two", _diagnostics(), _timing()),
            ]
        )
        self.assertIn("### Turn 1", md)
        self.assertIn("### Turn 2", md)

    def test_metrics_derived_from_latest_turn(self):
        metrics = conversation_metrics_from_messages(_messages())
        self.assertEqual(metrics.get("Turns"), 1)


class TestByteIdenticalReload(unittest.TestCase):

    def test_export_reload_export_byte_identical(self):
        text = serialize_session_json(
            _messages(),
            project_name="MediTrack",
            dt_phase="Empathize",
            session_id="abc123",
            timestamp="2026-08-05T10:00:00",
            saved_missions=["ProjectA", "ProjectB"],
        )
        reloaded = reload_session_json(text)
        self.assertEqual(reload_session_json(text)["messages"], json.loads(text)["messages"])
        self.assertEqual(export_session_json(reloaded), text)

    def test_reload_respects_byte_layout_of_reloaded(self):
        a = serialize_session_json(_messages(), session_id="s1")
        b = export_session_json(reload_session_json(a))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)