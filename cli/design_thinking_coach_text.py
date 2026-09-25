#!/usr/bin/env python3
"""
Design Thinking Coach - Text-Based Version
A simple command-line interface for design thinking questioning
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import ollama
import re
from memory import SemanticHistoryRetriever

# --- Design Thinking Stages with Focus on Questioning ---
DESIGN_THINKING_STAGES = [
    {
        "name": "Empathize & Explore",
        "goal": "Deepen understanding of the problem space through questioning",
        "question_focus": "Explore the problem context, user experiences, and underlying needs through probing questions",
        "max_questions": 5
    },
    {
        "name": "Define & Reframe",
        "goal": "Challenge assumptions and reframe the problem through critical questioning",
        "question_focus": "Question assumptions, explore alternative problem definitions, and identify hidden assumptions",
        "max_questions": 5
    },
    {
        "name": "Ideate & Explore",
        "goal": "Stimulate creative thinking through provocative questions",
        "question_focus": "Generate 'How might we...' questions, challenge constraints, and explore unconventional approaches",
        "max_questions": 5
    },
    {
        "name": "Prototype & Test Thinking",
        "goal": "Explore testing and learning approaches through questioning",
        "question_focus": "Question how to learn quickly, what assumptions to test, and how to gather feedback effectively",
        "max_questions": 4
    },
    {
        "name": "Reflect & Iterate",
        "goal": "Encourage reflective thinking and continuous improvement",
        "question_focus": "Question what was learned, what surprised you, and how to improve moving forward",
        "max_questions": 3
    }
]

class TextDesignThinkingCoach:
    def __init__(self):
        print("Initializing Text-Based Design Thinking Coach (Question-Focused)...")

        # LLM Model
        self.ollama_model = "qwen2.5:3b"
        try:
            ollama.show(self.ollama_model)
            print(f"LLM Model verified: {self.ollama_model}")
        except Exception:
            print(f"Error: Model '{self.ollama_model}' not found in Ollama.")
            print(f"Please pull the model: 'ollama pull {self.ollama_model}' and make sure Ollama is running.")
            self.ollama_model = "qwen2.5:3b"

        # State & History
        self.history_retriever = SemanticHistoryRetriever(max_history=30)
        self.current_stage = 0
        self.question_count = 0
        self.session_topic = ""

        # Opening message - pure question to start
        self.opening_question = "What challenge or opportunity would you like to explore together through questioning?"
        print(f"\n{self.opening_question}\n")

    def generate_question_prompt(self, user_input):
        """Generate a prompt that focuses ONLY on asking questions, never providing answers"""

        # Get current stage info
        stage_info = DESIGN_THINKING_STAGES[self.current_stage]

        # Build context from history
        history_context = ""
        if self.history_retriever.history:
            recent_history = self.history_retriever.history[-4:]  # Last 2 exchanges
            if recent_history:
                history_context = "\nRecent conversation context:\n"
                for turn in recent_history:
                    role = "User" if turn['role'] == 'user' else "Coach"
                    history_context += f"{role}: {turn['content']}\n"

        # Determine if we should advance stage
        should_advance = False
        if self.question_count >= stage_info["max_questions"] and self.current_stage < len(DESIGN_THINKING_STAGES) - 1:
            should_advance = True
            self.question_count = 0
            self.current_stage += 1
            stage_info = DESIGN_THINKING_STAGES[self.current_stage]
            print(f"\nAdvancing to Stage {self.current_stage}: {stage_info['name']}\n")

        # Create the core instruction - STRICTLY QUESTION-ONLY
        system_prompt = f"""You are a Design Thinking Coach whose ONLY job is to ask thoughtful, open-ended questions that stimulate critical thinking and exploration.

CURRENT STAGE: {stage_info['name']}
STAGE GOAL: {stage_info['goal']}
QUESTION FOCUS: {stage_info['question_focus']}

ABSOLUTE RULES - YOU MUST FOLLOW THESE:
1. ONLY ASK QUESTIONS. NEVER provide answers, suggestions, solutions, opinions, or information.
2. Your responses must END with a question mark (?).
3. Do NOT use statements like "You should..." or "Consider..." or "Try..."
4. Do NOT provide explanations, summaries, or background information.
5. Do NOT say "Interesting..." or "That's a good point..." or any form of feedback.
6. Do NOT share your own thoughts or perspectives.
7. Ask ONLY one question per response unless they are clearly related sub-questions.
8. Keep questions open-ended to encourage exploration (avoid yes/no questions when possible).
9. Focus on helping the user think deeper, not on providing value through your speech.
10. If you find yourself about to give information, STOP and rephrase as a question instead.

Current conversation topic: "{self.session_topic if self.session_topic else 'Not yet established'}"

{history_context}

Remember: Your value comes solely from asking questions that help the user think more deeply. Speak only to ask your next question."""

        # Add user input to the prompt
        prompt = f"{system_prompt}\n\nUser's last input: \"{user_input}\"\n\nYour next question (remember: ONLY a question, no statements):"

        return prompt

    def run(self):
        """Main conversation loop"""
        self.project_name = input("Enter Project Name [MyProject]: ").strip()
        if not self.project_name:
            self.project_name = "MyProject"
        self.username = "User"

        print("=" * 60)
        print("DESIGN THINKING MENTOR - QUESTION-BASED ASSISTANT")
        print("=" * 60)
        print("Type your thoughts or 'quit' to exit")
        print("-" * 60)

        from mentor import process_mentor_turn, ChecklistManager, MentorSession, print_ascii_dashboard, build_mentor_session
        session = build_mentor_session(self.project_name)

        # Start with opening question
        print(f"Coach: {self.opening_question}")

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check for exit
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThanks for exploring with questions. Keep thinking!")
                    break

                # Check for empty input
                if not user_input:
                    print("Coach: What's on your mind to explore?")
                    continue

                if user_input.lower() in ('reset', 'start over', 'clear'):
                    session = MentorSession(self.project_name)
                    print("\nSession reset! Let's start fresh.")
                    print(f"Coach: {self.opening_question}")
                    continue

                # Process turn using process_mentor_turn
                reply, session, _, _ = process_mentor_turn(
                    user_message=user_input, 
                    username=self.username, 
                    project_name=self.project_name, 
                    model_name=self.ollama_model
                )

                print(f"\nCoach: {reply}")

                # Display checklist and tracked findings via ASCII dashboard
                print_ascii_dashboard(session)

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Thanks for exploring with questions!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                continue

if __name__ == "__main__":
    coach = TextDesignThinkingCoach()
    coach.run()