"""
tests/test_golden_conversations.py

Golden Conversation Regression Suite.

Replays the data-only conversation fixtures in ``tests/goldens/conversations/``
against the live ``mentor.process_mentor_turn`` pipeline and asserts the full
per-turn contract: ProjectState mutations, objective selection, checklist
progression, question-family history, replies, and the final objective.

New regression scenarios are added by dropping a new JSON fixture into
``tests/goldens/conversations/`` — no test code required. A failure in the
runner is reported with the fixture name and a ``Turn N:`` prefix so the
exact failing turn is immediately identifiable.
"""

import os
import sys
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.goldens.fixtures import load_all_fixtures  # noqa: E402
from tests.goldens.runner import run_golden_conversation  # noqa: E402


class TestGoldenConversations(unittest.TestCase):
    """Every fixture replays deterministically and passes all per-turn checks."""

    def test_all_fixtures_replay_deterministically(self):
        fixtures = load_all_fixtures()
        self.assertTrue(fixtures, "no golden conversation fixtures found")

        for fixture in fixtures:
            with self.subTest(fixture=fixture.name, scenario=fixture.scenario):
                result = run_golden_conversation(fixture)
                if not result.passed:
                    self.fail(
                        "Golden conversation %r failed:\n  %s"
                        % (fixture.name, "\n  ".join(result.all_failures))
                    )

    def test_fixture_suite_has_required_scope(self):
        """The regression suite must ship >= 10 representative fixtures."""
        fixtures = load_all_fixtures()
        self.assertGreaterEqual(len(fixtures), 10)

    def test_fixture_names_are_unique(self):
        fixtures = load_all_fixtures()
        names = [f.name for f in fixtures]
        self.assertEqual(len(names), len(set(names)), f"duplicate fixture names: {names}")


if __name__ == "__main__":
    unittest.main()
