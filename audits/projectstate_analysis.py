import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
#!/usr/bin/env python
"""ProjectState analysis for Category (2) conversational grounding."""

from memory_extractor import ProjectState, StateField, LIST_FIELDS, SCALAR_FIELDS, REQUIRED_FIELDS, NON_REQUIRED_FIELDS

print("=" * 70)
print("PROSTATESTATE ANALYSIS")
print("=" * 70)

print("""
ProjectState Fields (from memory_extractor.py, dataclass):

1. personas: list[str]  ← WHO experiences the problem (target audience)
   - LIST field, ADD operation
   - Examples: "students", "elderly people", "remote workers"
   - Captured when user says: "my target is students" / "elderly people are affected"

2. problems: list[str]  ← THE CORE PROBLEM / CHALLENGE
   - LIST field, ADD operation
   - Examples: "forgetting assignments", "late submissions", "medication non-adherence"
   - Captured when user says: "the problem is forgetting things"

3. current_solutions: list[str]  ← HOW PEOPLE CURRENTLY HANDLE THE PROBLEM
   - LIST field, ADD operation
   - Examples: "sticky notes", "Google Calendar", "phone alarms"
   - Captured when user says: "currently use sticky notes"

4. pain_points: list[str]  ← FRUSTRATIONS, IMPACT, NEGATIVE FEELINGS
   - LIST field, ADD operation
   - Examples: "stress", "anxiety", "feeling overwhelmed", "guilt"
   - Captured when user says: "it stresses me out" / "I feel anxious"

5. evidence: list[str]  ← OBSERVED PROOF, RESEARCH, DATA
   - LIST field, ADD operation
   - Examples: "I interviewed 5 students", "Research shows 70% forget"
   - Captured when user says: "I have seen students lose marks"

6. impacts: list[str]  ← Consequences / effects of the problem
   - LIST field, ADD operation
   - Supporting field, NOT in REQUIRED_FIELDS
   - Examples: "grades drop", "wasted time"
   - Distinct from problems (what goes wrong) and pain_points (emotional impact)

7. frequency: str | None  ← HOW OFTEN the problem occurs
   - SCALAR field, SET operation
   - Examples: "daily", "weekly", "every single day"
   - Captured when user says: "it happens daily"

8. previous_assistant_message: str | None  ← Conversation flow bookkeeping
   - NOT extracted from user messages

9. previous_user_message: str | None  ← Conversation flow bookkeeping
   - NOT extracted from user messages

Key Properties:

REQUIRED_FIELDS (gate Empathize completion/WRAP_UP):
   (StateField.PERSONAS, StateField.PROBLEMS, StateField.CURRENT_SOLUTIONS,
    StateField.PAIN_POINTS, StateField.EVIDENCE, StateField.FREQUENCY)

NON_REQUIRED_FIELDS (captured but don't gate completion):
   (StateField.IMPACTS,)

LIST_FIELDS: {personas, problems, current_solutions, pain_points, evidence, impacts}
SCALAR_FIELDS: {frequency}

How facts enter ProjectState:
- RuleBasedExtractor.extract() → pattern matching → dict → merge_extracted_to_state → StateManager.apply_extraction → project_state updated
- MemoryExtractor.extract() → LLM JSON → ExtractionResult → _apply_extraction_to_state → project_state updated
- _message_satisfies_field() → _make_field_satisfied() → objective re-computation
- Direct assignment via StateManager

Can user-provided facts exist without becoming objective-relevant?
- YES: The state can have values that are never the targeted field
- Example: User talks about their name ("chihuahua") → might match AUDIENCE patterns → added to personas
- But if not matching any pattern, stays out of state
- The objective engine only pursues fields in REQUIRED_FIELDS order
- Non-required fields (impacts) are captured but don't affect WRAP_UP

Does the state distinguish missing vs unknown vs not-applicable vs inferred?
- MISSING: field is in REQUIRED_FIELDS and not yet satisfied → objective will pursue it
- UNKNOWN: not explicitly modeled as a separate state; absence of value = missing
- NOT_APPLICABLE: not modeled; all required fields are always "applicable" in Empathize
- INFERRED: possible via _message_satisfies_field() + _make_field_satisfied(), which artificially marks a field satisfied
  - This is the answer-satisfaction mechanism
  - But it only works when: extraction missed the field + family was already asked + message heuristically satisfies the field
  - Does NOT happen for volunteered info before the question is asked (family_asked guard)

State serialization (to_state_dict()):
- Lists emitted as JSON arrays
- Scalar frequency emitted as string or null
- previous_* bookkeeping fields NOT included
- Used by MemoryExtractor to prompt the LLM with "what's already known"

State access methods:
- get_list(name): returns list for LIST fields (raises for scalar)
- get_scalar(name): returns scalar for FREQUENCY (raises for list)
- set_scalar(name, value): sets frequency value

Summary: ProjectState is a canonical knowledge store of extracted/project-relevant information.
It is the single source of truth for the objective engine and question planner.
It does NOT persist raw user messages or conversational context beyond what extraction captured.
It does NOT distinguish "user stated this but we didn't ask about it" from "we haven't asked about this yet".
Only the session_data.asked_question_families tracks what was asked; ProjectState only tracks what was extracted.
""")

# Now show the actual fields
print("\nActual ProjectState dataclass fields:")
for field in ProjectState.__dataclass_fields__.values():
    print(f"  {field.name}: {field.type} = {getattr(ProjectState, field.name, 'default')}")

print(f"\nLIST_FIELDS: {LIST_FIELDS}")
print(f"SCALAR_FIELDS: {SCALAR_FIELDS}")
print(f"REQUIRED_FIELDS: {REQUIRED_FIELDS}")
print(f"NON_REQUIRED_FIELDS: {NON_REQUIRED_FIELDS}")