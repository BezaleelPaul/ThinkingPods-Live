"""
module4 — Response Strategy + Prompt Builder for the Empathize v2 pipeline.

Architecture role
-----------------
::

    User
        ↓
    Memory Extractor (Module 1)
        ↓
    State Manager (Module 2)
        ↓
    Objective Engine (Module 3)
        ↓
    **Response Strategy Engine (Module 4)**   <-- this package
        ↓
    **Prompt Builder (Module 4)**            <-- this package
        ↓
    LLM

Module 4 owns TWO responsibilities:

1. **Response Strategy** — convert a Module 3
   :class:`~module3.ConversationObjective` into a
   :class:`ResponseStrategy` (deterministically; no LLM).
2. **Prompt Builder** — given ``ProjectState``, ``ConversationObjective``,
   ``ResponseStrategy``, the previous assistant message, and the
   previous user message, render the mentor LLM prompt. Module 4 owns
   ALL prompt construction — no other module builds prompts.

Module 4 does NOT:
  - decide what information is missing (Module 3's job),
  - update memory (Module 1's job),
  - validate state (Module 2's job),
  - perform lifecycle decisions (Module 5's job — not implemented yet).

Determinism
-----------
Every public component here is a pure function of its inputs: no LLM,
no randomness, no I/O, no state mutation. ``process_mentor_turn`` can
reuse one process-wide engine instance safely.

Public interface
----------------
>>> from module4 import (
...     ResponseStrategy,
...     ResponseStrategyEngine,
...     PromptBuilder,
...     build_prompt,
... )
"""

from __future__ import annotations

from .prompt_builder import PromptBuilder, build_prompt
from .response_strategy import ResponseStrategy, ResponseStrategyEngine

__all__ = [
    "PromptBuilder",
    "ResponseStrategy",
    "ResponseStrategyEngine",
    "build_prompt",
]
