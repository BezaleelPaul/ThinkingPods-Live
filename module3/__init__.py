"""
module3 — Objective Engine for the Empathize v2 pipeline.

Architecture role
-----------------
::

    User
        ↓
    Memory Extractor (Module 1)
        ↓
    StateValidator + StateManager (Module 2)
        ↓
    ProjectState
        ↓
    **Objective Engine (Module 3)**   <-- this package
        ↓
    Prompt Builder (Module 4)
        ↓
    LLM

Module 3 is **100% deterministic** and **read-only**. It consumes only
``ProjectState`` and produces a structured ``ConversationObjective``
describing what the mentor should accomplish next. It must:

- never call the LLM
- never touch the network
- never mutate ``ProjectState``
- never reference ``MentorSession`` or any legacy compatibility bridge
- never build prompts or generate replies

The same ``ProjectState`` always produces the same ``ConversationObjective``.

Public interface
----------------
Importing from the package brings in the orchestration components only:

>>> from module3 import (
...     ObjectiveEngine,
...     CompletenessChecker,
...     PriorityEngine,
...     ConversationObjective,
...     CoverageReport,
...     Objective,
...     Rule,
...     DEFAULT_RULES,
... )
"""

from __future__ import annotations

from .completeness_checker import CompletenessChecker
from .objective import (
    ConversationObjective,
    CoverageReport,
    Objective,
)
from .objective_engine import ObjectiveEngine
from .priority_engine import PriorityEngine
from .question_families import (
    FAMILIES_BY_FIELD,
    FAMILY_FALLBACK_QUESTIONS,
    FamilyPlan,
    QuestionFamily,
    QuestionFamilyPlanner,
    classify_question,
    family_label,
    field_of,
)
from .rules import DEFAULT_RULES, Rule
from .scoring_engine import (
    CandidateScore,
    DISCUSSION_WEIGHT,
    ObjectiveContext,
    REPEAT_WEIGHT,
    UNLOCK_WEIGHT,
    build_candidate_scores,
)
from .sufficiency import SufficiencyChecker, SufficiencyLevel, SufficiencyReport

__all__ = [
    "CandidateScore",
    "CompletenessChecker",
    "ConversationObjective",
    "CoverageReport",
    "DEFAULT_RULES",
    "DISCUSSION_WEIGHT",
    "FAMILIES_BY_FIELD",
    "FAMILY_FALLBACK_QUESTIONS",
    "FamilyPlan",
    "Objective",
    "ObjectiveContext",
    "ObjectiveEngine",
    "PriorityEngine",
    "QuestionFamily",
    "QuestionFamilyPlanner",
    "REPEAT_WEIGHT",
    "Rule",
    "SufficiencyChecker",
    "SufficiencyLevel",
    "SufficiencyReport",
    "UNLOCK_WEIGHT",
    "build_candidate_scores",
    "classify_question",
    "family_label",
    "field_of",
]
