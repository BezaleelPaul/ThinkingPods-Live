"""tests/goldens/fixtures.py — golden conversation fixture model + loader.

A fixture is a plain JSON dict under ``tests/goldens/conversations/``. The
loader maps it onto the frozen dataclasses below so the runner stays
fully data-driven: a new regression scenario is added by dropping in a new
JSON file, never by writing test code.

Fixture schema
--------------
{
  "name": "unique_id",
  "scenario": 1,                       # the scenario number this covers
  "description": "human-readable intent",
  "username": "golden_user",
  "project_name": "GoldenProject",
  "seed_state": {                       # optional initial ProjectState
      "personas": [], "problems": [], "current_solutions": [],
      "pain_points": [], "evidence": [], "impacts": [],
      "frequency": null
  },
  "seed_asked_families": [],           # optional pre-asked question families
  "turns": [ { ... }, ... ],
  "expect_final_objective": null,      # optional final objective value
  "expect_final_stage": null           # optional final stage value
}

Turn schema
-----------
{
  "user": "the user message",
  "extraction": {                       # optional; when omitted the runner
      "message_type": "MEANINGFUL",     #   feeds AMBIGUOUS, so the
      "updates": [                      #   deterministic rule-based
          {"operation": "ADD", "field": "personas", "value": "students"}
      ]
  },                                    #   extractor runs instead
  "ollama_reply": null,                 # optional canned LLM reply; null
                                        #   (or absent) => raising stub =>
                                        #   deterministic fallback reply
  "expect_reply": null,                 # optional exact reply assertion
  "expect_state": null,                 # optional partial ProjectState dict
  "expect_objective": null,             # optional objective value string
  "expect_checklist": null,             # optional {checklist_key: bool}
  "expect_family": null,                # optional QuestionFamily value asked
  "expect_asked_families": null,        # optional full family-history list
  "expect_unknown_facts": null,         # optional full unknown-facts list
  "expect_stage": null,                 # optional current stage string
  "expect_reask": null                  # optional bool (re-ask sanctioned)
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

FIXTURE_DIR: Path = Path(__file__).parent / "conversations"

_STATE_KEYS: tuple[str, ...] = (
    "personas",
    "problems",
    "current_solutions",
    "pain_points",
    "evidence",
    "impacts",
    "frequency",
)


@dataclass(frozen=True)
class GoldenTurn:
    """One user turn of a golden conversation."""

    user: str
    extraction: Optional[dict] = None
    ollama_reply: Optional[str] = None
    expect_reply: Optional[str] = None
    expect_state: Optional[dict] = None
    expect_objective: Optional[str] = None
    expect_checklist: Optional[dict] = None
    expect_family: Optional[str] = None
    expect_asked_families: Optional[list] = None
    expect_unknown_facts: Optional[list] = None
    expect_stage: Optional[str] = None
    expect_reask: Optional[bool] = None


@dataclass(frozen=True)
class GoldenConversationFixture:
    """A complete golden conversation: seed + turns + final expectations."""

    name: str
    scenario: int
    description: str
    username: str
    project_name: str
    turns: list[GoldenTurn]
    seed_state: dict = field(default_factory=dict)
    seed_asked_families: list = field(default_factory=list)
    expect_final_objective: Optional[str] = None
    expect_final_stage: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldenConversationFixture":
        return cls(
            name=d["name"],
            scenario=d.get("scenario", 0),
            description=d.get("description", ""),
            username=d.get("username", "golden_user"),
            project_name=d.get("project_name", "GoldenProject"),
            turns=[GoldenTurn(**t) for t in d.get("turns", [])],
            seed_state=d.get("seed_state", {}),
            seed_asked_families=d.get("seed_asked_families", []),
            expect_final_objective=d.get("expect_final_objective"),
            expect_final_stage=d.get("expect_final_stage"),
        )


def load_fixture(name: str) -> GoldenConversationFixture:
    """Load a single fixture by file stem (no ``.json`` suffix)."""
    path = FIXTURE_DIR / f"{name}.json"
    return GoldenConversationFixture.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def load_all_fixtures() -> list[GoldenConversationFixture]:
    """Load every fixture file, sorted by filename for stable ordering."""
    fixtures: list[GoldenConversationFixture] = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        fixtures.append(
            GoldenConversationFixture.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        )
    return fixtures
