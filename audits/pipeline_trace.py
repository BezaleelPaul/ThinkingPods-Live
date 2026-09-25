import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
#!/usr/bin/env python
"""Pipeline trace and architectural analysis - Category (2) conversational grounding."""

import sys
sys.path.insert(0, ".")

from mentor import process_mentor_turn, _extract_and_update_state, _determine_objective
from memory_extractor import ProjectState, StateField, MessageType, ExtractionResult, ExtractionUpdate, Operation
from extraction_pipeline import RuleBasedExtractor
from module3 import ObjectiveEngine, QuestionFamilyPlanner, FAMILIES_BY_FIELD
from module4 import ResponseStrategyEngine, ResponseStrategy
from module5 import LifecycleManager, LifecycleDecision

print("=" * 70)
print("PIPELINE TRACE: User Message to Final Mentor Question")
print("=" * 70)

# Let me trace the actual flow by examining what happens
# I'll use the test infrastructure to trace a turn

print("\n" + "=" * 70)
print("STEP 1: _prepare_turn and _extract_and_update_state")
print("=" * 70)

# From reading the code, _prepare_turn (in session_pipeline) creates:
# - session_data (with conversation_history, asked_question_families, etc.)
# - project_state (ProjectState instance)
# - session_id
# - last_assistant_msg

# Then _extract_and_update_state runs:
# Step 1: RuleBasedExtractor.extract(user_message, project_name) → rule_observation
# Step 2: decide_hybrid_extraction(rule_observation, project_state, session_data, user_message) → hybrid_decision
#   - If llm_invoked: run LLM extraction via MemoryExtractor
#   - If not llm_invoked: apply rules through StateManager merge

# The key output: project_state is updated with extracted facts
# extraction_message_type is captured in _capture["extraction_message_type"]
# extraction_updates is captured in _capture["extraction_updates"]

print("   - RuleBasedExtractor extracts patterns from user message")
print("   - decide_hybrid_extraction decides whether LLM extraction runs")
print("   - If LLM runs: MemoryExtractor extracts → extraction_result")
print("   - extraction_result.message_type ∈ {MEANINGFUL, NO_UPDATE, AMBIGUOUS, END}")
print("   - extraction_result.updates ∈ [ExtractionUpdate]")
print("   - _apply_extraction_to_state applies updates to project_state")
print("   - _capture['extraction_message_type'] = extraction_result.message_type.value")
print("   - _capture['extraction_updates'] = [u.to_dict() for u in extraction_result.updates]")

print("\n" + "=" * 70)
print("STEP 2: _determine_objective")
print("=" * 70)

# From the code:
# 1. Build ObjectiveContext from session_data.conversation_history and asked_question_families
# 2. _OBJECTIVE_ENGINE.determine_next(project_state, context=context) → initial objective
# 3. Answer-satisfaction check:
#    if extraction_message_type == "MEANINGFUL" AND latest_user_message AND NOT wrap_up:
#       targeted = objective.targeted_field()
#       if targeted not in extracted_fields:
#          families_for_field = FAMILIES_BY_FIELD.get(targeted, ())
#          asked_families = set(session_data.asked_question_families)
#          family_asked = any(f.value in asked_families for f in families_for_field)
#          if family_asked AND _message_satisfies_field(latest_user_message, targeted, project_state):
#             satisfied_state = _make_field_satisfied(project_state, targeted)
#             objective = _OBJECTIVE_ENGINE.determine_next(satisfied_state, context=context)

# Key insight: The answer-satisfaction check can advance the objective past a field
# that extraction missed, BUT only if:
#   - extraction was MEANINGFUL (or now, after our fix, even after AMBIGUOUS + fallback)
#   - the question family for that field was already asked
#   - _message_satisfies_field recognizes the message as answering that field

print("   1. Build ObjectiveContext from conversation_history + asked_question_families")
print("   2. determine_next(project_state, context) → initial objective (information-gain scoring)")
print("   3. Answer-satisfaction check may re-compute objective with field artificially satisfied")
print("   4. Result: objective with targeted_field and missing_fields")

print("\n" + "=" * 70)
print("STEP 3: _plan_question_family")
print("=" * 70)

# From the code:
# 1. targeted = objective.targeted_field()
# 2. sufficiency = SufficiencyChecker.evaluate(project_state)
# 3. _QUESTION_FAMILY_PLANNER.plan(field=targeted, asked_family_values=session_data.asked_question_families, sufficiency=sufficiency)
# 4. Planner picks the FIRST family not yet asked; if all asked, re-asks least-recently-asked

# Key: The planner is stateless and only considers:
#   - The field the objective is targeting
#   - Which families for that field have been asked
#   - Whether the field is already sufficient
#   - It does NOT consider raw user messages or conversation context beyond "asked families"

print("   1. objective.targeted_field() → which StateField to pursue")
print("   2. SufficiencyChecker.evaluate(project_state) → is the field already sufficient?")
print("   3. For each family in FAMILIES_BY_FIELD[field]: if not in asked_families, select it")
print("   4. If all asked: re-ask the least-recently-asked family (reask=True)")
print("   5. CRITICAL: The planner knows NOTHING about what the user actually said")
print("      - It only knows which families were asked (asked_question_families)")
print("      - It does NOT know what answers those families received")

print("\n" + "=" * 70)
print("STEP 4: _generate_reply (with family_plan)")
print("=" * 70)

# From the code:
# 1. family_to_ask = family_plan.selected
# 2. asked_families = list(family_plan.asked_families)
# 3. build_module4_prompt(...) with question_family=family_to_ask.value
# 4. Generate reply via ollama.chat(model, messages=[{"role": "user", "content": prompt}])
# 5. enforce_mentor_reply(llm_reply, fallback_reply, allow_summary=...)
# 6. _guard_repeated_question(reply, family_plan, project_state) → guards against repeats

# The prompt built by build_module4_prompt includes:
# - question_family value
# - previously_asked_families
# - project_state values (what's been extracted)
# - conversation_objective
# - response_strategy
# - resume_hint from memory_to_prompt_bullets

# CRITICAL: The LLM receives the prompt with the question family and can produce
# a repeat. The _guard_repeated_question then blocks repeats and substitutes
# a family-aware fallback.

print("   1. family_plan.selected → which QuestionFamily to ask next")
print("   2. build_module4_prompt(project_state, conversation_objective, ...) → LLM prompt")
print("   3. ollama.chat(model, prompt) → LLM reply")
print("   4. enforce_mentor_reply(llm_reply, fallback) → may rewrite the reply")
print("   5. _guard_repeated_question → blocks LLM repeats, substitutes fallback")
print("   6. The final question is what the user hears")

print("\n" + "=" * 70)
print("STEP 5: Summary of Information Flow")
print("=" * 70)

print("""
PIPELINE FLOW:
   raw user message
         ↓
RuleBasedExtractor.extract() → dict of {field: value} from pattern matching
         ↓
MemoryExtractor.extract(user_message, project_state) → ExtractionResult with message_type + updates
         ↓
decide_hybrid_extraction() → llm_invoked True/False
         ↓
If llm_invoked:
    MemoryExtractor → ExtractionResult(message_type, updates[])
Else:
    Rules applied through StateManager.merge → project_state updated

         ↓
_extraction_updates captured, extraction_message_type captured

         ↓
_determine_objective(project_state, session_data, latest_user_message, extraction_message_type, extraction_updates)
   - Builds ObjectiveContext from conversation_history + asked_families
   - Runs ObjectiveEngine.determine_next() → initial objective
   - Answer-satisfaction check: may re-compute objective if user just answered
   - Result: objective with targeted_field, missing_fields, completion status

         ↓
_plan_question_family(objective, project_state, session_data)
   - Selects first unasked family for targeted field
   - If all asked: re-ask least-recently-asked
   - RESULT: FamilyPlan(selected=QuestionFamily, asked_families=tuple)

         ↓
_generate_reply(project_state, objective, response_strategy, family_plan, session_data)
   - build_module4_prompt(...) builds LLM prompt including:
     * question_family value
     * previously_asked_families
     * project_state values
     * objective
     * resume_hint (from conversation_memory)
   - ollama.chat() → LLM reply
   - enforce_mentor_reply() → template fallback if needed
   - _guard_repeated_question() → blocks repeats
   - RESULT: final mentor question delivered to user

         ↓
update_conversation_memory(...) → stores open/resolved threads, deferred topics, etc.
         ↓
Session JSON persisted
""")

print("\n" + "=" * 70)
print("KEY ARCHITECTURAL OBSERVATIONS")
print("=" * 70)

print("""
1. INFORMATION DISCARDED AT EACH STAGE:
   
   a) RuleBasedExtractor.extract():
      - Only captures pattern-matched facts
      - Discards: conversational context, intent, anything not matching patterns
      - Output: simple dict {field: value}
   
   b) MemoryExtractor.extract():
      - Reads project_state to avoid re-extracting known info
      - Returns ExtractionResult with message_type and updates[]
      - Discards: everything not in the JSON schema (personas, problems, etc.)
      - message_type classifies: MEANINGFUL / NO_UPDATE / AMBIGUOUS / END
   
   c) ObjectiveEngine.determine_next():
      - Reads project_state and ObjectiveContext (conversation_history + asked_families)
      - Uses information-gain scoring to select next objective
      - May miss implicit answers (this is the Category (1) problem, now partially fixed)
      - Considers: what's in state, what families were asked, but NOT raw message intent
   
   d) QuestionFamilyPlanner.plan():
      - ONLY knows: which families for the field were asked
      - Does NOT know: what the user said, what answers were given, conversation context
      - Purely mechanical: "first unasked family" or "re-ask least-recently-asked"
   
   e) _generate_reply / LLM prompt:
      - Receives: project_state values, question_family, asked_families, objective, resume_hint
      - The LLM generates the question, but may ignore the steering
      - _guard_repeated_question blocks obvious repeats but doesn't fix grounding
   
   f) conversation_memory:
      - Stores: open_threads, resolved_threads, deferred_topics, acknowledged_facts, summarized_facts
      - Partially used in _generate_reply via memory_to_prompt_bullets (resume_hint)
      - BUT the resume_hint is optional and may not contain the critical info
      - The memory is updated AFTER objective selection, so it doesn't feed back this turn

2. WHERE GROUNDING FAILS:
   
   The mentor can technically "follow its system" but fail to demonstrate understanding because:
   
   a) ProjectState only contains extracted facts, not the user's raw statements
   b) The objective engine scores based on information gain from state, not conversational relevance
   c) The question planner selects the next unasked family regardless of whether the user already answered
   d) The LLM prompt includes project_state values but the LLM may not use them coherently
   e) conversation_memory is updated too late (after this turn's objective/question is set)
   f) The resume_hint from memory_to_prompt_bullets is just additional prompt text, not structured intent

3. THE CENTRAL PROBLEM:
   
   The system is designed as: user → extraction → state → objective → question.
   It is NOT designed as: user → understanding → contextual planning → question.
   
   The Category (2) failures occur because the system never builds a coherent
   "what does the user actually mean and what have they established" model that
   persists across turns and feeds into question planning.
""")

print("\n" + "=" * 70)
print("END PIPELINE TRACE")
print("=" * 70)