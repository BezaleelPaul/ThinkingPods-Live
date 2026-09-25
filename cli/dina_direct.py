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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")  # Default to tiny for better performance
OLLAMA_MODEL = os.getenv("MENTOR_MODEL", "qwen2.5:3b")
SILERO_SAMPLE_RATE = 48000
MIC_SAMPLE_RATE = 16000
MIC_THRESHOLD = float(os.getenv("MIC_THRESHOLD", "0.02"))
SILENCE_DURATION = 1.0
MIN_UTTERANCE_SECONDS = 0.5

# --- Design Thinking Empathize Phase Stages ---
EMPATHIZE_STAGES = [
    {
        "name": "Problem Statement",
        "goal": "Get the user to clearly define what their project/idea is, what problem it aims to solve, and why it is important to solve.",
        "guidelines": "Ask probing questions about the core problem. Focus on the 'what' and 'why'. Do not let the user jump to solutions yet.",
        "max_turns": 2
    },
    {
        "name": "Stakeholders",
        "goal": "Identify all the stakeholders involved (direct users, indirect users, decision makers, daily operators).",
        "guidelines": "Ask about who experiences the problem. What are their roles? Who would benefit most if this is solved?",
        "max_turns": 2
    },
    {
        "name": "Pain Points",
        "goal": "Uncover frustrations, emotional and financial impacts, and how often this pain occurs.",
        "guidelines": "Ask about concrete frustrations. How does this make them feel? How often does it happen? What is the impact?",
        "max_turns": 2
    },
    {
        "name": "Current Workflow",
        "goal": "Map out the step-by-step process, existing tools, workarounds, and major bottlenecks or delays.",
        "guidelines": "Ask about their current workflow. What tools or manual steps do they use to get around the problem today?",
        "max_turns": 2
    },
    {
        "name": "Root Causes",
        "goal": "Ask 'Why?' repeatedly to dig deep and identify the fundamental cause of the problem.",
        "guidelines": "Ask 'Why?' to uncover root causes. Challenge surface-level explanations to find the base issue.",
        "max_turns": 2
    },
    {
        "name": "Evidence Collection",
        "goal": "Validate assumptions with facts, data, reports, or direct observations.",
        "guidelines": "Challenge assumptions. Ask if they have data, metrics, or observed proof, or if they are guessing.",
        "max_turns": 2
    },
    {
        "name": "Empathy Summary",
        "goal": "Provide a comprehensive summary of all details gathered (Problem, Stakeholders, Pain Points, Workflow, Root Cause, Evidence) and ask the user to confirm or refine it.",
        "guidelines": "Present a clear, structured summary of all findings. Do not ask a new question; instead, ask the user to verify if this summary is accurate.",
        "max_turns": 1
    }
]

class DinaDirect:
    def __init__(self):
        print("🚀 Initializing Dina Direct (Hardware Optimized)...")
        
        # Dynamically determine optimal CPU threads based on hardware
        import multiprocessing
        logical_cores = multiprocessing.cpu_count()
        optimal_threads = max(1, logical_cores // 2)
        optimal_threads = min(optimal_threads, 8) # Memory bandwidth cap
        print(f" -> Configured thread allocation: {optimal_threads} threads")
        
        # 1. Ears (Whisper) - Dynamic CPU threads to prevent freezing while maintaining speed
        self.stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=optimal_threads)
        
        # 2. Mouth (Silero) - Loaded dynamically based on optimal threads
        self.device = torch.device('cpu')
        torch.set_num_threads(optimal_threads) # Limit PyTorch threads dynamically
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
        self.audio_queue = queue.Queue() # Thread-safe audio passing
        self.last_speech_time = time.time()
        self.has_spoken = False
        self.cooldown_until = 0
        self.history_retriever = SemanticHistoryRetriever(max_history=int(os.getenv("MAX_HISTORY", "50")))
        
        # 5. Design Thinking Stage Tracking
        self.current_stage = 0
        self.stage_turns = 0

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
            
        self.cooldown_until = time.time() + 1.2 # Safety cooldown
        self.is_speaking = False

    def speak(self, text):
        clean_text = text.replace("*", "").replace("#", "").replace("_", "")
        if not clean_text.strip():
            return
            
        self.is_speaking = True
        print(f"🤖 Dina: {clean_text}")
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
             self.audio_queue.put(indata.copy()) # Keep recording the silence tail

    def listen_and_process_loop(self):
        """ Main background thread that manages recording and AI processing """
        print("\n🎤 Dina is listening... (You can talk or interrupt)")
        
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
                    self.has_spoken = False # Reset immediately
                    
                    if not audio_buffer:
                        continue
                        
                    recording = np.concatenate(audio_buffer).flatten()
                    audio_buffer = [] # Clear for next utterance
                    
                    # Ignore tiny noises
                    if len(recording) < MIC_SAMPLE_RATE * MIN_UTTERANCE_SECONDS:
                        continue

                    print("🧠 Thinking...")
                    
                    # 1. Transcribe (STT)
                    try:
                        segments, _ = self.stt.transcribe(
                            recording, 
                            beam_size=3, # Balanced speed and accuracy
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
                        lower_text = user_text.lower()
                        # Real test requests by the user
                        user_test_patterns = [
                            "testing 1", "testing one", "check 1", "check one", 
                            "mic check", "can you hear me", "is it working", "hello hello"
                        ]
                        # Whisper hallucinations / silence markers to ignore
                        hallucination_patterns = [
                            "thank you", "thanks for watching", "subtitle by", "viewing"
                        ]
                        
                        is_user_test = any(p in lower_text for p in user_test_patterns)
                        is_hallucination = any(p in lower_text for p in hallucination_patterns)
                        
                        if is_user_test:
                            reply = "I can hear you loud and clear! What project or idea are we exploring today?"
                            print(f"🤖 Dina (System): {reply}")
                            self.speak(reply)
                            continue
                            
                        # Clean out punctuation and spacing to count actual letters/digits
                        cleaned_user_text = re.sub(r'[^a-zA-Z0-9]', '', user_text)
                        
                        # If it's a hallucination, or too short (no actual content), ignore it silently.
                        if is_hallucination or len(cleaned_user_text) < 2:
                            continue
                            
                        # Voice command detection
                        if "next stage" in lower_text or "skip stage" in lower_text:
                            if self.current_stage < len(EMPATHIZE_STAGES) - 1:
                                self.current_stage += 1
                                self.stage_turns = 0
                                reply = f"Advancing to Stage {self.current_stage}: {EMPATHIZE_STAGES[self.current_stage]['name']}."
                                print(f"📢 System: {reply}")
                                self.speak(reply)
                                continue
                            else:
                                reply = "We are already at the final stage (Empathy Summary)."
                                print(f"📢 System: {reply}")
                                self.speak(reply)
                                continue
                        
                        elif "previous stage" in lower_text or "go back a stage" in lower_text:
                            if self.current_stage > 0:
                                self.current_stage -= 1
                                self.stage_turns = 0
                                reply = f"Going back to Stage {self.current_stage}: {EMPATHIZE_STAGES[self.current_stage]['name']}."
                                print(f"📢 System: {reply}")
                                self.speak(reply)
                                continue
                            else:
                                reply = "We are already at the first stage."
                                print(f"📢 System: {reply}")
                                self.speak(reply)
                                continue
                        elif "reset conversation" in lower_text or "start fresh" in lower_text:
                            self.current_stage = 0
                            self.stage_turns = 0
                            self.history_retriever.history = []
                            self.history_retriever._rebuild_indices()
                            # Reset mentor session
                            session = MentorSession("DinaProject")
                            reply = "Conversation reset. Let's start fresh with Stage 0: Problem Statement. What is your project idea?"
                            print(f"📢 System: {reply}")
                            self.speak(reply)
                            continue

                        print(f"👤 You: {user_text}")
                        
                        # Add user input to memory index
                        self.history_retriever.add_turn('user', user_text)
                        
                        # Process turn using process_mentor_turn
                        try:
                            reply, session, _, _ = process_mentor_turn(
                                user_message=user_text,
                                username="User",
                                project_name="DinaProject",
                                model_name=self.ollama_model
                            )
                            print(f"🤖 Dina: {reply}")
                            self.speak(reply)
                            self.history_retriever.add_turn('assistant', reply)
                            
                            # Print checklist and findings in console via ASCII dashboard
                            print_ascii_dashboard(session)
                                
                        except Exception as e:
                            print(f"Processing Error: {e}")
                            fallback = "What is most important to understand about this situation?"
                            self.speak(fallback)
                            self.history_retriever.add_turn('assistant', fallback)
                            
                    except Exception as e:
                        print(f"Processing Error: {e}")
                            
                time.sleep(0.1) # Small sleep to prevent CPU hogging in the loop

if __name__ == "__main__":
    dina = DinaDirect()
    dina.speak("System optimized. I am ready when you are.")
    dina.listen_and_process_loop()
