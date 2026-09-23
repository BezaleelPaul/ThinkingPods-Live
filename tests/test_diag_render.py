"""
tests/test_diag_render.py — regression tests for the generic Developer
Console renderer (``diag_render.render_diag_items``).

Covers dict, nested dict, list, nested list, dict containing lists, list
containing dicts, and scalar values, plus tuple/set support and the
preserved long-standing appearance (including the extraction-update
format and empty-value "—" markers).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import diag_render  # noqa: E402


def _render(value):
    return "\n".join(diag_render.render_diag_items(value))


class TestDictRendering(unittest.TestCase):
    def test_flat_dict(self):
        out = _render({"Insight Type": "ROOT_CAUSE_HINT", "Confidence": "HIGH"})
        self.assertEqual(
            out,
            "  Insight Type: ROOT_CAUSE_HINT\n"
            "  Confidence: HIGH",
        )

    def test_nested_dict(self):
        out = _render({"Outer": {"Inner": "val", "Count": 3}})
        self.assertEqual(
            out,
            "  Outer\n"
            "      Inner: val\n"
            "      Count: 3",
        )

    def test_deep_nesting(self):
        out = _render({"A": {"B": {"C": "leaf"}}})
        self.assertEqual(
            out,
            "  A\n"
            "      B\n"
            "          C: leaf",
        )

    def test_empty_dict_renders_header_only(self):
        out = _render({"Empty": {}})
        self.assertEqual(out, "  Empty")

    def test_empty_dict_section_renders_nothing(self):
        self.assertEqual(_render({}), "")


class TestListRendering(unittest.TestCase):
    def test_flat_list(self):
        out = _render(["Question 1", "Question 2"])
        self.assertEqual(
            out,
            "  • Question 1\n"
            "  • Question 2",
        )

    def test_nested_list_renders_recursively(self):
        out = _render({"Grid": [[1, 2], [3]]})
        self.assertEqual(
            out,
            "  Grid:\n"
            "      • 1\n"
            "      • 2\n"
            "      • 3",
        )

    def test_dict_containing_list(self):
        out = _render({"Signals": ["Root cause", "Evidence"]})
        self.assertEqual(
            out,
            "  Signals:\n"
            "      • Root cause\n"
            "      • Evidence",
        )

    def test_list_containing_dicts(self):
        out = _render(
            [
                {"field": "personas", "value": "students"},
                {"field": "frequency", "value": "daily"},
            ]
        )
        self.assertIn("  field: personas", out)
        self.assertIn("  value: students", out)
        self.assertIn("  field: frequency", out)
        self.assertIn("  value: daily", out)

    def test_mixed_list_renders_scalars_and_dicts(self):
        out = _render({"Items": ["plain", {"k": "v"}]})
        self.assertIn("  Items:", out)
        self.assertIn("      • plain", out)
        self.assertIn("      k: v", out)

    def test_empty_list_renders_dash(self):
        out = _render({"updates": []})
        self.assertEqual(out, "  updates: —")

    def test_question_history_renders_without_changing_producer(self):
        history = ["Who would benefit?", "How often do they struggle?"]
        out = _render(history)
        self.assertEqual(
            out,
            "  • Who would benefit?\n"
            "  • How often do they struggle?",
        )


class TestTupleAndSetRendering(unittest.TestCase):
    def test_tuple(self):
        out = _render((1, 2, 3))
        self.assertEqual(out, "  • 1\n  • 2\n  • 3")

    def test_nested_tuple(self):
        out = _render({"Coords": (1, 2)})
        self.assertEqual(out, "  Coords:\n      • 1\n      • 2")

    def test_set(self):
        lines = _render({1, 2})
        self.assertIn("  • 1", lines)
        self.assertIn("  • 2", lines)

    def test_empty_tuple_and_set_render_dash(self):
        self.assertEqual(_render({"t": ()}), "  t: —")
        self.assertEqual(_render({"s": set()}), "  s: —")


class TestScalarRendering(unittest.TestCase):
    def test_string_scalar(self):
        self.assertEqual(_render("just a value"), "just a value")

    def test_numeric_scalar(self):
        self.assertEqual(_render(42), "42")

    def test_boolean_scalar(self):
        self.assertEqual(_render(True), "True")

    def test_none_renders_dash(self):
        self.assertEqual(_render({"k": None}), "  k: —")

    def test_empty_string_renders_dash(self):
        self.assertEqual(_render({"k": ""}), "  k: —")


class TestExtractionUpdateFormatPreserved(unittest.TestCase):
    def test_extraction_update_shape(self):
        out = _render(
            {
                "Extraction": {
                    "updates": [
                        {
                            "operation": "ADD",
                            "field": "personas",
                            "value": "students",
                        }
                    ]
                }
            }
        )
        self.assertIn("  Extraction", out)
        self.assertIn("      updates", out)
        self.assertIn("          ADD personas:", out)
        self.assertIn("              students", out)

    def test_extraction_update_with_confidence_and_label(self):
        out = _render(
            {
                "updates": [
                    {
                        "operation": "ADD",
                        "field": "frequency",
                        "value": "daily",
                        "confidence": 0.95,
                        "label": "MATCHED",
                        "reasons": ["direct answer"],
                    }
                ]
            }
        )
        self.assertIn("      ADD frequency [confidence 0.95]  →  MATCHED — direct answer:", out)
        self.assertIn("          daily", out)

    def test_stringified_value_is_kept_stringy(self):
        out = _render({"k": "123"})
        self.assertEqual(out, "  k: 123")


if __name__ == "__main__":
    unittest.main()
