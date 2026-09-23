"""
tests/test_prompt_builder.py

Unit tests for module4.prompt_builder (Module 4 — Prompt Builder).

Covers (per the Module 4 spec test categories):
  ✓ Prompt Builder formatting
  ✓ Previous conversation inclusion
  ✓ ProjectState formatting
  ✓ Empty ProjectState
  ✓ Full ProjectState
  ✓ ASK_QUESTION prompt
  ✓ GENERATE_SUMMARY prompt
  ✓ Deterministic output
  ✓ No duplicated sections

The Prompt Builder is a pure formatter — no business logic. These tests
assert that property by checking:
  * byte-identical output for identical inputs across calls;
  * section ordering and required section headers;
  * section absence of duplicates;
  * that state and previous-message values are interpolated exactly,
    without transformation, addition, or removal.
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, StateField  # noqa: E402

from module3 import ConversationObjective, Objective, ObjectiveEngine  # noqa: E402
from module4 import (  # noqa: E402
    PromptBuilder,
    ResponseStrategy,
    build_prompt,
)
from module4.response_strategy import ResponseStrategyEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SECTION_RULE = "--------------------------------"


def _empty_state() -> ProjectState:
    return ProjectState()


def _full_state() -> ProjectState:
    return ProjectState(
        personas=["students"],
        problems=["buried messages"],
        current_solutions=["whatsapp groups"],
        pain_points=["overwhelmed"],
        evidence=["user interviews"],
        frequency="daily",
    )


def _empty_objective() -> ConversationObjective:
    return ObjectiveEngine().determine_next(_empty_state())


def _wrap_up_objective() -> ConversationObjective:
    return ObjectiveEngine().determine_next(_full_state())


def _strategy_for(obj: ConversationObjective, state: ProjectState) -> ResponseStrategy:
    return ResponseStrategyEngine().determine_strategy(obj, state)


# ---------------------------------------------------------------------------
# Structural sanity — five sections, no duplication
# ---------------------------------------------------------------------------


class TestPromptStructure(unittest.TestCase):

    def test_has_exactly_five_required_sections(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        for header in ("Role", "Current Objective", "Known Project State",
                       "Latest Conversation", "Instructions"):
            self.assertIn(header + "\n" + SECTION_RULE, prompt,
                          f"section header missing: {header}")

    def test_section_headers_appear_exactly_once(self):
        """No duplicated sections — each header appears once."""
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        for header in ("Role", "Current Objective", "Known Project State",
                       "Latest Conversation", "Instructions"):
            self.assertEqual(
                prompt.count(header + "\n" + SECTION_RULE), 1,
                f"section header '{header}' must appear exactly once"
            )

    def test_section_rule_separator_appears_once_per_section(self):
        """The ---------- separator appears exactly five times (one per section)."""
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertEqual(prompt.count(SECTION_RULE), 5)

    def test_section_order_matches_specification(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message="assistant",
            previous_user_message="user",
        )
        # Spec order: Role, Current Objective, Known Project State,
        # Latest Conversation, Instructions.
        header = lambda name: name + "\n" + SECTION_RULE
        i_role = prompt.index(header("Role"))
        i_obj = prompt.index(header("Current Objective"))
        i_state = prompt.index(header("Known Project State"))
        i_conv = prompt.index(header("Latest Conversation"))
        i_inst = prompt.index(header("Instructions"))
        self.assertLess(i_role, i_obj)
        self.assertLess(i_obj, i_state)
        self.assertLess(i_state, i_conv)
        self.assertLess(i_conv, i_inst)


# ---------------------------------------------------------------------------
# Role section
# ---------------------------------------------------------------------------


class TestRoleSection(unittest.TestCase):

    def test_role_section_contains_mentor_role(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("You are an experienced Design Thinking mentor.", prompt)


# ---------------------------------------------------------------------------
# Current Objective section
# ---------------------------------------------------------------------------


class TestObjectiveSection(unittest.TestCase):

    def test_objective_section_renders_personas_for_empty_state(self):
        obj = _empty_objective()
        self.assertEqual(obj.objective, Objective.PERSONAS)
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Understand the target users.", prompt)

    def test_objective_section_renders_wrap_up_for_full_state(self):
        obj = _wrap_up_objective()
        self.assertEqual(obj.objective, Objective.WRAP_UP)
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=obj,
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Produce the empathy summary.", prompt)


# ---------------------------------------------------------------------------
# Known Project State section — empty / full / partial formatting
# ---------------------------------------------------------------------------


class TestProjectStateSection(unittest.TestCase):

    def test_empty_state_renders_none_for_every_field(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # Six fields, each rendered as "<label>\n  None".
        for label in ("Personas", "Problems", "Current Solutions",
                      "Pain Points", "Evidence", "Frequency"):
            self.assertIn(f"{label}\n  None", prompt,
                          f"empty state field '{label}' must render as None")

    def test_full_state_renders_each_field_value(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # List-field items render as "- <value>" bullets.
        for value in ("students", "buried messages", "whatsapp groups",
                      "overwhelmed", "user interviews"):
            self.assertIn(f"- {value}", prompt,
                          f"known state value '{value}' must appear as a bullet")
        # Scalar frequency rendered verbatim (no bullet).
        self.assertIn("Frequency\n  daily", prompt)

    def test_partial_state_renders_filled_and_empty_appropriately(self):
        # Populated persons, problems; the rest empty.
        state = ProjectState(
            personas=["students"],
            problems=["buried messages"],
        )
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=_strategy_for(obj, state),
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Personas\n  - students", prompt)
        self.assertIn("Problems\n  - buried messages", prompt)
        # Empty fields render as None.
        for empty_label in ("Current Solutions", "Pain Points",
                            "Evidence", "Frequency"):
            self.assertIn(f"{empty_label}\n  None", prompt)

    def test_multi_item_list_renders_one_bullet_per_item(self):
        """A list field with multiple items renders all of them as bullets."""
        state = ProjectState(
            personas=["students", "parents", "teachers"],
            problems=["x"], current_solutions=["x"], pain_points=["x"],
            evidence=["x"], frequency="daily",
        )
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=_strategy_for(obj, state),
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # All three personas appear in the state section, as separate bullets.
        for persona in ("students", "parents", "teachers"):
            self.assertIn(f"- {persona}", prompt)


# ---------------------------------------------------------------------------
# Latest Conversation section — previous-assistant / previous-user
# ---------------------------------------------------------------------------


class TestLatestConversationSection(unittest.TestCase):

    def test_none_messages_render_as_None(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # Assistant and User labels present.
        self.assertIn("Assistant\n  None", prompt)
        self.assertIn("User\n  None", prompt)

    def test_messages_are_interpolated_verbatim(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="I'm helping students.",
        )
        self.assertIn("Assistant\n  Tell me about your users.", prompt)
        self.assertIn("User\n  I'm helping students.", prompt)

    def test_previous_user_appears_after_previous_assistant(self):
        """Conversation order: assistant's previous turn, then user's turn."""
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="ASSISTANT_MARKER",
            previous_user_message="USER_MARKER",
        )
        i_assistant = prompt.index("ASSISTANT_MARKER")
        i_user = prompt.index("USER_MARKER")
        self.assertLess(i_assistant, i_user)

    def test_whitespace_only_messages_treated_as_None(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="   ",
            previous_user_message="\n\t",
        )
        self.assertIn("Assistant\n  None", prompt)
        self.assertIn("User\n  None", prompt)


# ---------------------------------------------------------------------------
# Instructions section — varies by ResponseStrategy
# ---------------------------------------------------------------------------


class TestInstructionsSection(unittest.TestCase):

    def test_ask_question_instructions_block(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # Canonical ASK_QUESTION imperatives — mentoring behaviours plus the
        # one-question / no-invention / no-summarize constraints.
        self.assertIn("Acknowledge what the user just shared in one short sentence.", prompt)
        self.assertIn(
            "Ask ONE natural follow-up question that builds on their most recent answer.",
            prompt,
        )
        self.assertIn("Do not restart the topic - stay focused on the current objective.", prompt)
        self.assertIn(
            "Do not ask for information already in the Known Project State or the Latest Conversation - treat what has already been shared as remembered conversation, not as isolated turns.",
            prompt,
        )
        self.assertIn(
            "Never ask essentially the same question twice, even reworded - avoid semantically identical questions.",
            prompt,
        )
        self.assertIn("Do not summarize.", prompt)
        self.assertIn("Do not ask multiple questions.", prompt)
        self.assertIn("Do not invent information.", prompt)
        self.assertIn("Do not repeat previous questions.", prompt)
        self.assertIn(
            "Keep the response concise and mentor-like - no filler, no excess words.",
            prompt,
        )

    def test_generate_summary_instructions_block(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # Canonical GENERATE_SUMMARY imperatives.
        self.assertIn(
            "Summarize what you understand about the people affected and the problem.",
            prompt,
        )
        self.assertIn(
            "Ask the user whether the summary is accurate or whether they would refine it.",
            prompt,
        )
        self.assertIn("Do not propose solutions or product ideas.", prompt)
        self.assertIn("Keep the summary concise and easy to scan - no filler.", prompt)


# ---------------------------------------------------------------------------
# Mentoring-behaviour guidance — the improved instructions
# ---------------------------------------------------------------------------


def _instructions_bullets(prompt: str):
    """Extract the "- <bullet>" lines of the Instructions section, in order.
    Mirrors how the Developer Console prompt inspector surfaces the prompt."""
    lines = prompt.splitlines()
    try:
        idx = lines.index("Instructions")
    except ValueError:
        return []
    bullets = []
    for ln in lines[idx + 2:]:
        if ln.startswith("- "):
            bullets.append(ln[2:])
        elif ln.strip():
            break
    return bullets


class TestMentoringBehaviorGuidance(unittest.TestCase):
    """The prompt instructions must teach an experienced-mentor conversation:
    acknowledge, build on previous answers, avoid restarts and known facts,
    avoid semantic repeats, and stay concise."""

    def _ask_prompt(self):
        return build_prompt(
            project_state=ProjectState(personas=["students"], problems=["buried messages"]),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="Students lose marks when they miss deadlines.",
        )

    def test_acknowledges_what_user_shared(self):
        self.assertIn("Acknowledge what the user just shared", self._ask_prompt())

    def test_builds_on_previous_answer(self):
        self.assertIn("builds on their most recent answer", self._ask_prompt())

    def test_avoids_restarting_topics(self):
        self.assertIn("Do not restart the topic", self._ask_prompt())

    def test_avoids_known_information(self):
        self.assertIn(
            "Do not ask for information already in the Known Project State",
            self._ask_prompt(),
        )

    def test_avoids_semantically_identical_questions(self):
        self.assertIn(
            "avoid semantically identical questions", self._ask_prompt(),
        )

    def test_concise_mentor_style(self):
        self.assertIn("concise and mentor-like", self._ask_prompt())

    def test_concise_summary_style(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Keep the summary concise", prompt)


class TestCoachingQualityGuidance(unittest.TestCase):
    """The prompt instructions must teach an experienced Design Thinking
    mentor's conversation: acknowledge insights naturally, celebrate useful
    discoveries briefly, avoid a questionnaire feel, vary openings, recap
    progress, connect answers, avoid filler transitions, stay concise, and
    never ask multiple unrelated questions."""

    def _ask_prompt(self):
        return build_prompt(
            project_state=ProjectState(personas=["students"], problems=["buried messages"]),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="Students lose marks when they miss deadlines.",
        )

    def test_acknowledge_insights_naturally(self):
        self.assertIn(
            "Acknowledge what the user just shared in one short sentence.",
            self._ask_prompt(),
        )

    def test_celebrate_useful_discoveries_briefly(self):
        self.assertIn(
            "Celebrate genuinely useful discoveries briefly", self._ask_prompt()
        )

    def test_avoids_questionnaire_feel(self):
        self.assertIn("not a questionnaire", self._ask_prompt())

    def test_varies_sentence_openings(self):
        self.assertIn("Vary your sentence openings", self._ask_prompt())

    def test_recaps_progress_occasionally(self):
        self.assertIn("briefly recap what you understand so far", self._ask_prompt())

    def test_connects_answers_together(self):
        self.assertIn("Connect the new answer to something they already told you", self._ask_prompt())

    def test_avoids_unnecessary_transitions(self):
        self.assertIn("Avoid unnecessary transitions", self._ask_prompt())

    def test_maintains_concise_responses(self):
        self.assertIn("concise and mentor-like", self._ask_prompt())

    def test_never_asks_multiple_unrelated_questions(self):
        self.assertIn("Do not ask multiple questions.", self._ask_prompt())
        self.assertIn("Ask ONE natural follow-up question", self._ask_prompt())

    def test_summary_opens_warmly(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Open warmly - acknowledge the time and insights", prompt)

    def test_clarify_acknowledges_before_asking(self):
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.CLARIFY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn(
            "First acknowledge the part you did understand", prompt
        )
        self.assertIn("Ask only ONE clarifying question.", prompt)


class TestConversationalFlowGuidance(unittest.TestCase):
    """The conversational-flow instructions: observation-first openings,
    a reason for each question, small inferences, no filler, brief replies,
    a facilitator tone — while still requiring exactly ONE question and no
    topic restarts."""

    def _ask_prompt(self):
        return build_prompt(
            project_state=ProjectState(personas=["students"], problems=["buried messages"]),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="Students lose marks when they miss deadlines.",
        )

    def test_prompt_opens_with_observation_not_question(self):
        self.assertIn("never lead with the question", self._ask_prompt())

    def test_prompt_explains_why_question_matters(self):
        self.assertIn("why the next question matters", self._ask_prompt())

    def test_prompt_teaches_small_inferences(self):
        self.assertIn("make one small inference", self._ask_prompt())

    def test_prompt_forbids_filler(self):
        self.assertIn("'Thank you for sharing' or 'Great question.'", self._ask_prompt())

    def test_prompt_targets_brief_replies(self):
        self.assertIn("2-4 short sentences", self._ask_prompt())

    def test_prompt_facilitator_not_survey(self):
        self.assertIn("never like a survey", self._ask_prompt())

    def test_prompt_still_requires_exactly_one_question(self):
        self.assertIn("Ask ONE natural follow-up question", self._ask_prompt())
        self.assertIn("Do not ask multiple questions.", self._ask_prompt())

    def test_prompt_still_forbids_topic_restart(self):
        self.assertIn("Do not restart the topic", self._ask_prompt())

    def test_flow_guidance_is_ask_question_only(self):
        """The conversational-flow directives belong to the ask path, not the
        summary block (a summary already acknowledges by construction)."""
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertNotIn("never lead with the question", prompt)


class TestRepeatQuestionReductionGuidance(unittest.TestCase):
    """The ASK_QUESTION instructions must steer the mentor away from repeated
    questions and toward advancing the discussion: never re-ask what was
    already provided, never reword a question, advance when the objective is
    satisfied, avoid fishing for more examples, and build on elaborations —
    while still preserving the one-question-per-turn / concise / ack-first
    constraints and the existing coaching and conversation-move guidance."""

    def _ask_prompt(self):
        return build_prompt(
            project_state=ProjectState(personas=["students"], problems=["buried messages"]),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="Students lose marks when they miss deadlines.",
        )

    def test_never_asks_for_already_provided_information(self):
        self.assertIn(
            "Do not ask for information already in the Known Project State",
            self._ask_prompt(),
        )

    def test_treats_prior_shares_as_remembered_conversation(self):
        self.assertIn(
            "treat what has already been shared as remembered conversation, not as isolated turns",
            self._ask_prompt(),
        )

    def test_never_rewords_a_question(self):
        self.assertIn(
            "Never ask essentially the same question twice, even reworded",
            self._ask_prompt(),
        )

    def test_advances_when_objective_satisfied(self):
        self.assertIn(
            "If the user's previous answer already satisfies the current objective, naturally advance to the next objective",
            self._ask_prompt(),
        )

    def test_avoids_more_examples_fishing(self):
        self.assertIn(
            "Do not ask for more examples or further elaboration unless the previous answer was genuinely insufficient",
            self._ask_prompt(),
        )

    def test_builds_on_elaboration(self):
        self.assertIn(
            "If the user has already elaborated on a point, build on that elaboration",
            self._ask_prompt(),
        )

    def test_prefers_advancing_over_exhausting(self):
        self.assertIn(
            "Prefer advancing the discussion over exhausting a topic",
            self._ask_prompt(),
        )

    def test_does_not_leak_into_summary_block(self):
        """The advance-the-discussion guidance belongs to the ask path only;
        the summary block is unchanged."""
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertNotIn("naturally advance to the next objective", prompt)

    def test_preserves_one_question_and_concise_ack_first(self):
        for phrase in (
            "Ask ONE natural follow-up question",
            "Do not ask multiple questions.",
            "Keep the reply to typically 2-4 short sentences",
            "never lead with the question",
        ):
            self.assertIn(phrase, self._ask_prompt())


class TestConversationalTransitionGuidance(unittest.TestCase):
    """The ASK_QUESTION instructions must teach the mentor to connect each new
    question to the user's thinking — observation → inference → reason → one
    question — so it feels like following a train of thought, not ticking off
    a checklist, while preserving every existing constraint."""

    def _ask_prompt(self):
        return build_prompt(
            project_state=ProjectState(personas=["students"], problems=["buried messages"]),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message="Tell me about your users.",
            previous_user_message="Students lose marks when they miss deadlines.",
        )

    def test_new_question_follows_from_what_was_shared(self):
        self.assertIn(
            "Make every new question clearly follow from what the user just shared",
            self._ask_prompt(),
        )

    def test_question_feels_like_curiosity_not_checklist(self):
        self.assertIn(
            "feel like curiosity, not progression through a checklist",
            self._ask_prompt(),
        )

    def test_connects_previous_answer_to_next_objective(self):
        self.assertIn(
            "Connect the previous answer to the next objective in one natural sentence.",
            self._ask_prompt(),
        )

    def test_explains_what_was_noticed_before_focus_change(self):
        self.assertIn(
            "Before changing focus, briefly explain what you noticed or inferred",
            self._ask_prompt(),
        )

    def test_explains_why_new_topic_matters_now(self):
        self.assertIn(
            "When moving to a new topic, explain why that topic matters now.",
            self._ask_prompt(),
        )

    def test_teaches_observation_inference_reason_question_form(self):
        self.assertIn(
            "Prefer transitions of the form: observation, then inference, then reason, then one question.",
            self._ask_prompt(),
        )

    def test_acknowledges_significance_of_important_insight(self):
        self.assertIn(
            "If the user has revealed an important insight, spend one sentence acknowledging its significance",
            self._ask_prompt(),
        )

    def test_avoids_abrupt_jumps(self):
        self.assertIn(
            "Avoid abrupt jumps between unrelated questions, even when the objective changes.",
            self._ask_prompt(),
        )

    def test_never_exposes_internal_process(self):
        prompt = self._ask_prompt()
        self.assertIn(
            "Never expose internal process - do not say 'next objective', 'next step', or 'I need to ask about...'.",
            prompt,
        )

    def test_includes_positive_example_pairs(self):
        prompt = self._ask_prompt()
        self.assertIn("instead of 'How do people handle this today?' say 'Since this seems to happen almost every day", prompt)
        self.assertIn("Instead of 'How often does this happen?' say 'From what you've described, this sounds like a recurring challenge", prompt)

    def test_transition_guidance_does_not_leak_into_summary(self):
        prompt = build_prompt(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertNotIn("observation, then inference, then reason", prompt)
        self.assertNotIn("checklist item", prompt)
        self.assertNotIn("Never expose internal process", prompt)

    def test_preserves_prior_constraints(self):
        for phrase in (
            "Acknowledge what the user just shared in one short sentence.",
            "Ask ONE natural follow-up question",
            "Do not ask multiple questions.",
            "Never ask essentially the same question twice, even reworded",
            "treat what has already been shared as remembered conversation",
            "Keep the response concise and mentor-like",
            "never lead with the question",
        ):
            self.assertIn(phrase, self._ask_prompt())


class TestPromptInstructionSnapshot(unittest.TestCase):
    """Regression snapshot of the full per-strategy instruction blocks.

    These are the de-facto prompt snapshots: any intentional change to the
    instruction wording MUST update this test and the Developer Console
    prompt inspector output together."""

    ASK_QUESTION_SNAPSHOT = (
        "Acknowledge what the user just shared in one short sentence.",
        "Ask ONE natural follow-up question that builds on their most recent answer.",
        "Do not restart the topic - stay focused on the current objective.",
        "Do not ask for information already in the Known Project State or the Latest Conversation - treat what has already been shared as remembered conversation, not as isolated turns.",
        "Never ask essentially the same question twice, even reworded - avoid semantically identical questions.",
        "Do not summarize.",
        "Do not ask multiple questions.",
        "Do not invent information.",
        "Do not repeat previous questions.",
        "Keep the response concise and mentor-like - no filler, no excess words.",
        "Celebrate genuinely useful discoveries briefly - one short warm remark (for example, 'That is really helpful to know.') - then continue.",
        "Vary your sentence openings - never open two consecutive replies the same way.",
        "Avoid unnecessary transitions and filler such as 'Great question', 'Interesting', or 'Let me ask you'.",
        "Connect the new answer to something they already told you - reference a shared detail so it feels like one flowing conversation, not a questionnaire.",
        "Every few turns, briefly recap what you understand so far in one or two lines before asking the next question.",
        "Acknowledge any open threads, deferred topics, or previously resolved topics the user mentioned but you skipped earlier - gently reconnect them to the current question context.",
        "If the user shared new information that naturally advances the current objective, reference it specifically in your follow-up question to show direct engagement.",
        "Open every reply with a brief observation or acknowledgment of what the user just shared - never lead with the question.",
        "Briefly explain why the next question matters before asking it.",
        "When the user's information supports it, make one small inference or observation instead of immediately asking another question.",
        "Avoid filler such as 'Thank you for sharing' or 'Great question.' - use a genuine observation instead.",
        "Keep the reply to typically 2-4 short sentences - observation, reason, then one focused question.",
        "Sound like an experienced Design Thinking facilitator who guides the user's thinking - never like a survey.",
        "If the user's previous answer already satisfies the current objective, naturally advance to the next objective instead of continuing to probe the same topic.",
        "Do not ask for more examples or further elaboration unless the previous answer was genuinely insufficient for the current objective.",
        "If the user has already elaborated on a point, build on that elaboration toward the current objective rather than requesting another elaboration.",
        "Prefer advancing the discussion over exhausting a topic - once enough is known, move forward.",
        "Make every new question clearly follow from what the user just shared - it should feel like the next thought, not the next checklist item.",
        "Make the next question feel like curiosity, not progression through a checklist.",
        "Connect the previous answer to the next objective in one natural sentence.",
        "Before changing focus, briefly explain what you noticed or inferred from their answer.",
        "When moving to a new topic, explain why that topic matters now.",
        "Prefer transitions of the form: observation, then inference, then reason, then one question.",
        "If the user has revealed an important insight, spend one sentence acknowledging its significance before continuing.",
        "Avoid abrupt jumps between unrelated questions, even when the objective changes.",
        "Never expose internal process - do not say 'next objective', 'next step', or 'I need to ask about...'.",
        "Prefer questions that grow from the user's own words - instead of 'How do people handle this today?' say 'Since this seems to happen almost every day, I'm curious how people currently cope with it. What do they usually do today?'",
        "Instead of 'How often does this happen?' say 'From what you've described, this sounds like a recurring challenge rather than an isolated incident. How often would you say it happens?'",
    )

    GENERATE_SUMMARY_SNAPSHOT = (
        "Summarize what you understand about the people affected and the problem.",
        "Reference the known project state only; do not invent information.",
        "Ask the user whether the summary is accurate or whether they would refine it.",
        "Keep the summary concise and easy to scan - no filler.",
        "Do not ask multiple questions.",
        "Do not propose solutions or product ideas.",
        "Open warmly - acknowledge the time and insights the user shared before presenting the summary.",
        "Vary how you phrase each part so the summary reads naturally, not like a filled-in form.",
    )

    ACKNOWLEDGE_SNAPSHOT = (
        "Briefly acknowledge what the user just shared.",
        "Then ask ONE follow-up question that builds on their answer.",
        "Do not restart the topic.",
        "Do not ask for information already in the Known Project State or the Latest Conversation.",
        "Do not invent information.",
        "Do not ask multiple questions.",
        "Keep the response brief.",
        "Vary your sentence openings - do not repeat the same opening pattern.",
        "Connect their answer to what they shared earlier so the conversation flows.",
    )

    CLARIFY_SNAPSHOT = (
        "Ask the user to clarify their most recent answer.",
        "Reference the specific part of their answer that was unclear.",
        "Ask only ONE clarifying question.",
        "Do not invent information.",
        "Do not propose solutions.",
        "Keep the response brief.",
        "First acknowledge the part you did understand - then ask about the unclear part.",
        "Use their own words when pointing at what was unclear, so it feels like a conversation, not an interrogation.",
    )

    def _render(self, strategy):
        state = _full_state() if strategy is ResponseStrategy.GENERATE_SUMMARY else _empty_state()
        obj = _wrap_up_objective() if strategy is ResponseStrategy.GENERATE_SUMMARY else _empty_objective()
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=strategy,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        return tuple(_instructions_bullets(prompt))

    def test_ask_question_snapshot(self):
        self.assertEqual(
            self._render(ResponseStrategy.ASK_QUESTION),
            self.ASK_QUESTION_SNAPSHOT,
        )

    def test_generate_summary_snapshot(self):
        self.assertEqual(
            self._render(ResponseStrategy.GENERATE_SUMMARY),
            self.GENERATE_SUMMARY_SNAPSHOT,
        )

    def test_acknowledge_snapshot(self):
        self.assertEqual(
            self._render(ResponseStrategy.ACKNOWLEDGE),
            self.ACKNOWLEDGE_SNAPSHOT,
        )

    def test_clarify_snapshot(self):
        self.assertEqual(
            self._render(ResponseStrategy.CLARIFY),
            self.CLARIFY_SNAPSHOT,
        )


# ---------------------------------------------------------------------------
# ASK_QUESTION prompt — full shape
# ---------------------------------------------------------------------------


class TestAskQuestionPrompt(unittest.TestCase):

    def test_ask_question_prompt_for_empty_state_renders_personas_objective(self):
        state = _empty_state()
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Current Objective", prompt)
        self.assertIn("Understand the target users.", prompt)
        # No objective-specific summary-only markers leaked into the ask prompt.
        self.assertNotIn("Produce the empathy summary.", prompt)

    def test_ask_question_prompt_for_partial_state_targeted_field(self):
        state = ProjectState(
            personas=["x"],
        )
        obj = ObjectiveEngine().determine_next(state)
        # PERSONAS covered -> next objective is PROBLEMS (priority 90).
        self.assertEqual(obj.objective, Objective.PROBLEMS)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        # The objective in the prompt correlates to the targeted field.
        self.assertIn("Understand the core problem they face.", prompt)


# ---------------------------------------------------------------------------
# GENERATE_SUMMARY prompt — full shape
# ---------------------------------------------------------------------------


class TestGenerateSummaryPrompt(unittest.TestCase):

    def test_generate_summary_prompt_renders_wrap_up_objective(self):
        state = _full_state()
        obj = _wrap_up_objective()
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Current Objective", prompt)
        self.assertIn("Produce the empathy summary.", prompt)
        # The ask-question-only imperatives are NOT in the summary prompt.
        self.assertNotIn("Stay focused on the current objective.", prompt)

    def test_generate_summary_prompt_includes_full_state_bullets(self):
        state = _full_state()
        prompt = build_prompt(
            project_state=state,
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        for value in ("students", "buried messages", "whatsapp groups",
                      "overwhelmed", "user interviews", "daily"):
            self.assertIn(value, prompt)


# ---------------------------------------------------------------------------
# Determinism — identical inputs -> byte-identical output
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):

    def test_same_inputs_produce_identical_prompt(self):
        args = dict(
            project_state=_full_state(),
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message="assistant message",
            previous_user_message="user message",
        )
        a = build_prompt(**args)
        b = build_prompt(**args)
        self.assertEqual(a, b)

    def test_state_mutation_after_build_does_not_change_prior_output(self):
        """Prompt builder never mutates state; a subsequent state edit
        doesn't retroactively change a previously-built prompt."""
        state = _full_state()
        prompt_a = build_prompt(
            project_state=state,
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        state.personas.append("teachers")  # mutate after the prompt is built
        prompt_b = build_prompt(
            project_state=_full_state(),  # fresh, unchanged
            conversation_objective=_wrap_up_objective(),
            response_strategy=ResponseStrategy.GENERATE_SUMMARY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertEqual(prompt_a, prompt_b)

    def test_two_builder_invocations_via_class_and_module_alias_match(self):
        """PromptBuilder.build_prompt and the module-level build_prompt
        alias are the same callable."""
        self.assertIs(PromptBuilder.build_prompt, build_prompt)
        args = dict(
            project_state=_empty_state(),
            conversation_objective=_empty_objective(),
            response_strategy=ResponseStrategy.ASK_QUESTION,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertEqual(PromptBuilder.build_prompt(**args), build_prompt(**args))


# ---------------------------------------------------------------------------
# Type rejections — strict Module 4 boundary
# ---------------------------------------------------------------------------


class TestTypeRejection(unittest.TestCase):

    def test_rejects_non_project_state(self):
        obj = _empty_objective()
        with self.assertRaises(TypeError):
            build_prompt(
                project_state={"personas": []},  # plain dict
                conversation_objective=obj,
                response_strategy=ResponseStrategy.ASK_QUESTION,
                previous_assistant_message=None,
                previous_user_message=None,
            )

    def test_rejects_non_conversation_objective(self):
        with self.assertRaises(TypeError):
            build_prompt(
                project_state=_empty_state(),
                conversation_objective=Objective.PERSONAS,  # bare enum
                response_strategy=ResponseStrategy.ASK_QUESTION,
                previous_assistant_message=None,
                previous_user_message=None,
            )

    def test_rejects_non_response_strategy(self):
        obj = _empty_objective()
        with self.assertRaises(TypeError):
            build_prompt(
                project_state=_empty_state(),
                conversation_objective=obj,
                response_strategy="ASK_QUESTION",  # bare string
                previous_assistant_message=None,
                previous_user_message=None,
            )


# ---------------------------------------------------------------------------
# Reserved strategies (ACKNOWLEDGE / CLARIFY) — render stable instructions
# ---------------------------------------------------------------------------


class TestReservedStrategies(unittest.TestCase):
    """ACKNOWLEDGE / CLARIFY are not emitted on the live Empathize path
    yet, but their instruction blocks must still render deterministically
    so future Module 5 hooks can use them without rewrites."""

    def test_acknowledge_renders_acknowledge_instructions(self):
        obj = _empty_objective()  # any objective is acceptable here
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ACKNOWLEDGE,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn("Briefly acknowledge what the user just shared.", prompt)

    def test_clarify_renders_clarify_instructions(self):
        obj = _empty_objective()
        prompt = build_prompt(
            project_state=_empty_state(),
            conversation_objective=obj,
            response_strategy=ResponseStrategy.CLARIFY,
            previous_assistant_message=None,
            previous_user_message=None,
        )
        self.assertIn(
            "Ask the user to clarify their most recent answer.",
            prompt,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
