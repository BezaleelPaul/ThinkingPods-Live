#!/usr/bin/env python3
"""
Design Thinking Coach - Pure Question-Based Assistant
Focuses exclusively on asking critical questions to stimulate
design thinking and brainstorming, without providing direct answers or solutions.
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import threading
import queue
import re
import numpy as np
import sounddevice as sd
import torch
import ollama
from faster_whisper import WhisperModel
from memory import SemanticHistoryRetriever
from mentor import process_mentor_turn, ChecklistManager, MentorSession, print_ascii_dashboard

# --- Configuration for Low-End Laptops ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")  # Default to tiny for better performance
OLLAMA_MODEL = os.getenv("MENTOR_MODEL", "qwen2.5:3b")
SILERO_SAMPLE_RATE = 48000
MIC_SAMPLE_RATE = 16000
MIC_THRESHOLD = float(os.getenv("MIC_THRESHOLD", "0.02"))
SILENCE_DURATION = 1.0
MIN_UTTERANCE_SECONDS = 0.5

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

class DesignThinkingCoach:
    def __init__(self):
        print("🚀 Initializing Design Thinking Coach (Question-Focused)...")

        # Dynamically determine optimal CPU threads based on hardware
        import multiprocessing
        logical_cores = multiprocessing.cpu_count()
        optimal_threads = max(1, logical_cores // 2)
        optimal_threads = min(optimal_threads, 8)  # Memory bandwidth cap
        print(f" -> Configured thread allocation: {optimal_threads} threads")

        # 1. Ears (Whisper) - Dynamic CPU threads to prevent freezing while maintaining speed
        self.stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=optimal_threads)

        # 2. Mouth (Silero) - Loaded dynamically based on optimal threads
        self.device = torch.device('cpu')
        torch.set_num_threads(optimal_threads)  # Limit PyTorch threads dynamically
        self.tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                          model='silero_tts',
                                          language='en',
                                          speaker='v3_en',
                                          trust_repo=True)
        self.tts_model.to(self.device)

        # 3. LLM Model verification
        self.ollama_model = OLLAMA_MODEL
        try:
            ollama.show(self.ollama_model)
            print(f"LLM Model verified: {self.ollama_model}")
        except Exception:
            print(f"Error: Model '{self.ollama_model}' not found in Ollama.")
            print(f"Please pull the model: 'ollama pull {self.ollama_model}' and make sure Ollama is running.")
            self.ollama_model = OLLAMA_MODEL

        # 4. State & Threading
        self.is_speaking = False
        self.interrupt_flag = False
        self.audio_queue = queue.Queue()  # Thread-safe audio passing
        self.last_speech_time = time.time()
        self.has_spoken = False
        self.cooldown_until = 0
        self.history_retriever = SemanticHistoryRetriever(max_history=int(os.getenv("MAX_HISTORY", "30")))

        # 5. Design Thinking Tracking
        self.current_stage = 0
        self.question_count = 0
        self.session_topic = ""

        # Opening message - pure question to start
        self.opening_question = "What challenge or opportunity would you like to explore together through questioning?"

    def play_audio(self, audio_data):
        self.is_speaking = True
        self.interrupt_flag = False

        try:
            with sd.OutputStream(samplerate=SILERO_SAMPLE_RATE, channels=1, dtype='int16') as stream:
                chunk_size = 2000
                for i in range(0, len(audio_data), chunk_size):
                    if self.interrupt_flag:
                        break
                    chunk = audio_data[i:i+chunk_size]
                    stream.write(chunk)
        except Exception as e:
            print(f"Playback Error: {e}")

        self.cooldown_until = time.time() + 1.2  # Safety cooldown
        self.is_speaking = False

    def speak(self, text):
        clean_text = text.replace("*", "").replace("#", "").replace("_", "")
        if not clean_text.strip():
            return

        self.is_speaking = True
        print(f"🤔 Design Thinking Coach: {clean_text}")
        try:
            # Generate audio
            audio = self.tts_model.apply_tts(text=clean_text, speaker='en_21', sample_rate=SILERO_SAMPLE_RATE)
            audio_int16 = (audio.numpy() * 32767).astype(np.int16)

            # Play in background
            threading.Thread(target=self.play_audio, args=(audio_int16,), daemon=True).start()
        except Exception as e:
            print(f"TTS Error: {e}")
            self.is_speaking = False

    def audio_callback(self, indata, frames, time_info, status):
        """ This runs in a high-priority C-thread. It must be FAST. """
        if status or time.time() < self.cooldown_until:
            return

        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))

        # Interruption Logic
        if self.is_speaking:
            if volume_norm > MIC_THRESHOLD * 2.5:
                self.interrupt_flag = True
            return

        # Voice Activity
        if volume_norm > MIC_THRESHOLD:
            self.audio_queue.put(indata.copy())
            self.last_speech_time = time.time()
            self.has_spoken = True
        elif self.has_spoken:
             self.audio_queue.put(indata.copy())  # Keep recording the silence tail

    def generate_question_prompt(self, user_input):
        """Generate a prompt that focuses ONLY on asking questions, never providing answers"""

        # Get current stage info
        stage_info = DESIGN_THINKING_STAGES[self.current_stage]

        # Build context from history (limited to maintain focus on questioning)
        history_context = ""
        if self.history_retriever.history:
            # Get recent exchanges for context, but limit to avoid influencing too much
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
            print(f"📢 Advancing to Stage {self.current_stage}: {stage_info['name']}")

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

    def listen_and_process_loop(self):
        """Main background thread that manages recording and AI processing"""
        print("\n🎤 Design Thinking Coach is listening... (You can talk or interrupt)")
        print(f"💡 Starting with: {self.opening_question}")

        # Start with opening question
        self.speak(self.opening_question)

        audio_buffer = []

        # Start the microphone stream
        stream = sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, callback=self.audio_callback)
        with stream:
            while True:
                # If speaking or in cooldown, constantly clear any microphone input queue/buffer to prevent feedback loops
                if self.is_speaking or time.time() < self.cooldown_until:
                    audio_buffer = []
                    self.has_spoken = False
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    time.sleep(0.1)
                    continue

                # Pull audio from the C-thread safely
                try:
                    while not self.audio_queue.empty():
                        audio_buffer.append(self.audio_queue.get_nowait())
                except queue.Empty:
                    pass

                # Check if user has stopped speaking
                if self.has_spoken and (time.time() - self.last_speech_time > SILENCE_DURATION):
                    self.has_spoken = False  # Reset immediately

                    if not audio_buffer:
                        continue

                    recording = np.concatenate(audio_buffer).flatten()
                    audio_buffer = []  # Clear for next utterance

                    # Ignore tiny noises
                    if len(recording) < MIC_SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
                        continue

                    print("🧠 Thinking about next question...")

                    # 1. Transcribe (STT)
                    try:
                        segments, _ = self.stt.transcribe(
                            recording,
                            beam_size=3,  # Balanced speed and accuracy
                            language="en",
                            initial_prompt="This is a voice recording of a user discussing the Sathnur software village project, rural to urban development, and design thinking. Please transcribe names like Sathnur, software village, rural-urban, and technical terms accurately.",
                            condition_on_previous_text=False,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            temperature=0.0
                        )
                        # Evaluate segments and filter out those that are likely background noise / silence
                        valid_segments = []
                        for s in segments:
                            if s.no_speech_prob < 0.45 and s.avg_logprob > -1.0 and s.compression_ratio < 2.4:
                                valid_segments.append(s.text)
                        user_text = " ".join(valid_segments).strip()

                        # Filter out common Whisper hallucinations on silence/noise/mic checks
                        hallucinations = ["1, 2, 3", "one, two, three", "thank you", "thanks for watching", "subtitle by", "1, 2, 3, 1, 1, 2, 3", "test", "testing", "check"]
                        lower_text = user_text.lower()
                        if not user_text or (any(h in lower_text for h in hallucinations) and len(user_text) < 30):
                            # If it looks like a test, just acknowledge with a question and continue
                            self.speak("I can hear you. What would you like to explore?")
                            continue

                        # Clean out punctuation and spacing to count actual letters/digits
                        cleaned_user_text = re.sub(r'[^a-zA-Z0-9]', '', user_text)

                        # If it's a hallucination, or too short (no actual content), ignore it silently.
                        if len(cleaned_user_text) < 2:
                            continue

                        # Voice command handling - even these should be questioned, not answered directly
                        if any(cmd in lower_text for cmd in ["next stage", "skip ahead", "move forward"]):
                            if self.current_stage < len(DESIGN_THINKING_STAGES) - 1:
                                self.question_count = 0
                                self.current_stage += 1
                                stage_name = DESIGN_THINKING_STAGES[self.current_stage]['name']
                                # Respond with a question about the new stage, not a statement
                                self.speak(f"What aspects of {stage_name.lower()} would you like to explore through questioning?")
                                continue
                            else:
                                self.speak("What final aspects would you like to question before we conclude?")
                                continue

                        elif any(cmd in lower_text for cmd in ["previous stage", "go back"]):
                            if self.current_stage > 0:
                                self.current_stage -= 1
                                self.question_count = 0
                                stage_name = DESIGN_THINKING_STAGES[self.current_stage]['name']
                                self.speak(f"What would you like to re-explore in {stage_name.lower()} through questioning?")
                                continue
                            else:
                                self.speak("We're at the beginning. What aspect would you like to start questioning?")
                                continue

                        elif any(cmd in lower_text for cmd in ["start over", "reset", "begin again"]):
                            self.session_topic = ""
                            self.history_retriever.history = []
                            self.history_retriever._rebuild_indices()
                            # Reset mentor session
                            session = MentorSession("VoiceProject")
                            self.speak(self.opening_question)
                            continue

                        print(f"👤 You: {user_text}")

                        # Set or update session topic if not set
                        if not self.session_topic and len(user_text.strip()) > 10:
                            self.session_topic = re.sub(r'[^a-zA-Z0-9_-]', '', user_text.strip()[:30]) or "VoiceProject"

                        project_name = self.session_topic if self.session_topic else "VoiceProject"

                        # Add user input to memory index
                        self.history_retriever.add_turn('user', user_text)

                        # Process turn using process_mentor_turn
                        try:
                            reply, session, _, _ = process_mentor_turn(
                                user_message=user_text,
                                username="User",
                                project_name=project_name,
                                model_name=self.ollama_model
                            )
                            
                            print(f"🤔 Coach: {reply}")
                            self.speak(reply)
                            self.history_retriever.add_turn('assistant', reply)
                            
                            # Print checklist and findings in console via ASCII dashboard
                            print_ascii_dashboard(session)
                                
                        except Exception as e:
                            print(f"Question Generation Error: {e}")
                            fallback = "What is most important to understand about this situation?"
                            self.speak(fallback)
                            self.history_retriever.add_turn('assistant', fallback)

                    except Exception as e:
                        print(f"Processing Error: {e}")

                time.sleep(0.1)  # Small sleep to prevent CPU hogging in the loop

    def run(self):
        """Main entry point"""
        try:
            self.listen_and_process_loop()
        except KeyboardInterrupt:
            print("\n👋 Design Thinking Coach stopping...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Cleanup
            pass

if __name__ == "__main__":
    coach = DesignThinkingCoach()
    coach.speak("Design Thinking Coach ready. I'll only ask questions to help you think deeper. What would you like to explore?")
    coach.run()
