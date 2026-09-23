"""Golden Conversation Regression Suite.

Data-only fixtures under ``tests/goldens/conversations/`` + a reusable
runner (``tests/goldens/runner.py``) that replays complete mentor
conversations turn-by-turn against the live pipeline and verifies
ProjectState, state mutations, objective selection, checklist
progression, question-family history, and the final objective.
"""
