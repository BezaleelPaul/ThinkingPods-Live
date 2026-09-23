from fastapi import FastAPI, Request, Header, Body
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import os
import io
import re
import urllib.parse
import ollama
import numpy as np
from memory import SemanticHistoryRetriever
from mentor import process_mentor_turn, ChecklistManager, MentorSession, strip_model_output, build_mentor_session
from session_manager import get_session_manager, SessionMetadata
from session_lifecycle import SessionLifecycle

import threading
from collections import defaultdict
import time
from constants import MERMAID_KEYWORDS, LLM_LOADING_MSG_VERBOSE, LLM_LOADING_MSG_CONCISE, BRAIN_MODEL_LOADING_MSG, VOICE_OR_BRAIN_MODEL_LOADING_MSG

# --- Startup optimization ---------------------------------------------------
# torch / faster_whisper / scipy are heavy (they transitively pull in
# transformers + ctranslate2, ~10-15 s of synchronous import). They are now
# imported lazily, ONLY where the STT/TTS runtime actually needs them (see
# load_models, get_speech, voice_input), so the FastAPI server binds and can
# answer /health almost immediately while the models load in the background
# thread. Conversation quality and all mentor logic are untouched.

app = FastAPI()

# --- Models ---
MODELS = {
    "llm": None,
    "stt": None,
    "tts": None
}

# Conversation history per user (username -> SemanticHistoryRetriever)
conversation_histories = defaultdict(lambda: SemanticHistoryRetriever(max_history=int(os.getenv("MAX_HISTORY", "50"))))
history_lock = threading.Lock()

def load_models():
    try:
        import torch

        # --- Optimization for Intel HD 620 / Laptop CPUs ---
        torch.set_num_threads(4)

        print("Loading models...")

        # 1. LLM (Ollama) — single model for both mentor and extraction
        mentor_model = os.getenv("MENTOR_MODEL", "qwen2.5:3b")
        print(f"\n  Checking Ollama model: {mentor_model}")
        try:
            ollama.show(mentor_model)
            MODELS["llm"] = mentor_model
            print(f"  [OK] Mentor model ready: {mentor_model}")
        except Exception:
            print(f"  [FAIL] Model '{mentor_model}' not found.")
            print(f"  Please ensure Ollama is running and the model is pulled:")
            print(f"    ollama pull {mentor_model}")
            MODELS["llm"] = None

        # 2. STT (Whisper)
        try:
            from faster_whisper import WhisperModel

            whisper_model_name = os.getenv("WHISPER_MODEL", "tiny.en")
            print(f"\n  Loading Whisper model: {whisper_model_name}...")
            MODELS["stt"] = WhisperModel(whisper_model_name, device="cpu", compute_type="int8", cpu_threads=4)
            print(f"  [OK] Whisper ready")
        except Exception as e:
            print(f"  [FAIL] STT model failed to load: {e}")

        # 3. TTS (Silero)
        try:
            language = 'en'
            model_id = 'v3_en'
            device = torch.device('cpu')
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                    model='silero_tts',
                                                    language=language,
                                                    speaker=model_id,
                                                    trust_repo=True,
                                                    force_reload=False)
            model.to(device)
            MODELS["tts"] = model
            print(f"  [OK] TTS model ready")
        except Exception as e:
            print(f"  [FAIL] TTS model failed to load: {e}")

        print("\n  Application ready.\n")
    except Exception as e:
        print(f"Fatal error during model loading: {e}")

# Load models in background to not block startup
threading.Thread(target=load_models, daemon=True).start()

# --- Helper Functions ---
def get_history(username):
    with history_lock:
        return conversation_histories[username].history

def add_to_history(username, role, content):
    with history_lock:
        conversation_histories[username].add_turn(role, content)

def clear_history(username):
    with history_lock:
        if username in conversation_histories:
            del conversation_histories[username]

def should_use_mentor(interaction_mode, pod):
    """Route to mentor pipeline during Empathize/Discovery in Coach mode."""
    normalized_pod = (pod or "").strip().lower()
    if normalized_pod in {"empathize", "discovery", "general"}:
        return True
    return interaction_mode == "Design Thinking Coach"


def mentor_model_name():
    """Use the single configured mentor LLM model."""
    return os.getenv("MENTOR_MODEL", "qwen2.5:3b")

def generate_llm_response(text, pod, username, context_doc="", interaction_mode="Helpful Assistant"):
    if not MODELS["llm"]:
        return LLM_LOADING_MSG_VERBOSE

    # Interaction Mode System Prompts
    if interaction_mode == "Design Thinking Coach":
        # Original strict DT approach - no direct solutions
        if pod == "general" or any(k in text.lower() for k in ["help", "journey", "document", "documentation", "presentation", "commercialization"]):
            system_prompt = """You are ThinkingPods, a professional Design Thinking Consultant.
Your tone is encouraging, insightful, highly structured, and natural.
When the user shares their project journey or asks for feedback on documentation, analyze it by highlighting **problem-solving under constraints** (e.g., budget limits, hardware restrictions, and optimization iterations).
Advise them on how to structure their journey professionally for documentation, presentations, evaluations, and commercialization (e.g., breaking it down into Phases, Outcomes, Optimizations, and Vision).
Your primary role is still to guide the user by asking sharp, critical, probing questions. Do NOT offer direct ideas, suggestions, or answers—help them refine their own thoughts.
Do NOT use hashtags (#) or excessive emojis."""
        elif pod in ("empathize", "Discovery"):
            system_prompt = """You are ThinkingPods, acting as a professional interviewer and investigator in the **Empathize Phase**.
Your goal is to deeply understand the problem, people, and context—NOT to design solutions, offer suggestions, or discuss implementation.
Do NOT offer direct answers, ideas, or solutions; the user must identify the answers themselves. Focus on asking critical questions.

Follow this framework structure:
- Stage 0: Problem Statement (Define what, why, who, and interest)
- Stage 1: Stakeholders (Direct, indirect, daily users, decision makers)
- Stage 2: Pain Points (Frustrations, impact, frequency—Emotional/Financial)
- Stage 3: Current Workflow (Process steps, delays, tools used)
- Stage 4: Root Causes (Ask 'Why?' repeatedly until reaching the base cause)
- Stage 5: Evidence Collection (Challenge assumptions, look for data/reports)
- Stage 6: Empathy Summary (A full recap of the above sections)

CRITICAL RULES:
1. Do NOT discuss implementation, solutions, or suggestions. If the user tries, redirect them back to the problem context.
2. Ask 3–5 follow-up questions per topic before moving to the next stage.
3. Challenge unsupported claims respectfully (e.g., 'What evidence supports that belief?').
4. Focus on people and their experiences, not products.
5. Prevent the user from ending the phase too early; ensure all exit criteria are met.
6. Use verbal cues to show you are listening (Hmm, I see, etc.)."""
        else:
            system_prompt = f"""You are a professional Requirements Engineer helping the user with the '{pod}' stage.
Your goal is to guide the user toward high-quality software requirements (ISO 29148 standard) through natural conversation.
Be precise and professional, but keep the tone conversational and human.
Show personality and professional passion for the project.
IMPORTANT: Do NOT use hashtags (#) and focus on being helpful.
CRITICAL: Do NOT offer direct solutions or answers—guide the user to discover requirements themselves through questioning."""

    elif interaction_mode == "Helpful Assistant":
        # ChatGPT-like: conversational, solution-oriented, helpful
        if pod == "general" or any(k in text.lower() for k in ["help", "journey", "document", "documentation", "presentation", "commercialization"]):
            system_prompt = """You are ThinkingPods, a helpful and knowledgeable assistant. You are conversational, solution-oriented, and eager to help the user achieve their goals.
You can provide direct suggestions, ideas, and answers when appropriate, while still maintaining a thoughtful and structured approach.
When discussing projects or documentation, offer practical advice on structure, content, and presentation.
Be encouraging, insightful, and natural in your communication.
Do NOT use hashtags (#) excessively."""
        elif pod in ("empathize", "Discovery"):
            system_prompt = """You are ThinkingPods, assisting in the **Empathize Phase**. While understanding the problem is crucial, you can also help explore potential directions and ideas.
Ask insightful questions to understand stakeholders, pain points, and current workflows, but feel free to share relevant examples or suggestions when helpful.
Guide the user toward understanding root causes and gathering evidence, while being supportive and conversational.
Help them create an empathy summary that captures the essence of what they've learned."""
        else:
            system_prompt = f"""You are a helpful Requirements Engineering assistant helping the user with the '{pod}' stage.
You can provide guidance on software requirements best practices while being conversational and solution-oriented.
Offer clear, actionable suggestions for improving requirements, and help the user understand ISO 29148 standards through practical examples.
Be precise, professional, and helpful—don't hesitate to provide direct input when it advances the user's goals."""

    else:  # "Balanced" - mix of both approaches (default/original behavior)
        if pod == "general" or any(k in text.lower() for k in ["help", "journey", "document", "documentation", "presentation", "commercialization"]):
            system_prompt = """You are ThinkingPods, a professional Design Thinking Consultant.
Your tone is encouraging, insightful, highly structured, and natural.
When the user shares their project journey or asks for feedback on documentation, analyze it by highlighting **problem-solving under constraints** (e.g., budget limits, hardware restrictions, and optimization iterations).
Advise them on how to structure their journey professionally for documentation, presentations, evaluations, and commercialization (e.g., breaking it down into Phases, Outcomes, Optimizations, and Vision).
Your primary role is still to guide the user by asking sharp, critical, probing questions. Do NOT offer direct ideas, suggestions, or answers—help them refine their own thoughts.
Do NOT use hashtags (#) or excessive emojis."""
        elif pod in ("empathize", "Discovery"):
            system_prompt = """You are ThinkingPods, acting as a professional interviewer and investigator in the **Empathize Phase**.
Your goal is to deeply understand the problem, people, and context—NOT to design solutions, offer suggestions, or discuss implementation.
Do NOT offer direct answers, ideas, or solutions; the user must identify the answers themselves. Focus on asking critical questions.

Follow this framework structure:
- Stage 0: Problem Statement (Define what, why, who, and interest)
- Stage 1: Stakeholders (Direct, indirect, daily users, decision makers)
- Stage 2: Pain Points (Frustrations, impact, frequency—Emotional/Financial)
- Stage 3: Current Workflow (Process steps, delays, tools used)
- Stage 4: Root Causes (Ask 'Why?' repeatedly until reaching the base cause)
- Stage 5: Evidence Collection (Challenge assumptions, look for data/reports)
- Stage 6: Empathy Summary (A full recap of the above sections)

CRITICAL RULES:
1. Do NOT discuss implementation, solutions, or suggestions. If the user tries, redirect them back to the problem context.
2. Ask 3–5 follow-up questions per topic before moving to the next stage.
3. Challenge unsupported claims respectfully (e.g., 'What evidence supports that belief?').
4. Focus on people and their experiences, not products.
5. Prevent the user from ending the phase too early; ensure all exit criteria are met.
6. Use verbal cues to show you are listening (Hmm, I see, etc.)."""
        else:
            system_prompt = f"""You are a professional Requirements Engineer helping the user with the '{pod}' stage.
Your goal is to guide the user toward high-quality software requirements (ISO 29148 standard) through natural conversation.
Be precise and professional, but keep the tone conversational and human.
Show personality and professional passion for the project.
IMPORTANT: Do NOT use hashtags (#) and focus on being helpful."""

    doc_context = f"\n[LOCAL VAULT DATA]:\n{context_doc}\n" if context_doc else ""

    # Retrieve conversation history and search memory
    with history_lock:
        retriever = conversation_histories[username]
        relevant_indices = retriever.retrieve_relevant_context(text, top_k=2)
        history = retriever.history

    # Retrieve recalled context
    recalled_messages = []
    for idx in relevant_indices:
        # Check that it's not within the immediate sliding window of the last 4 turns (2 user exchanges)
        if idx < len(history) - 4:
            user_msg = history[idx]
            recalled_messages.append(f"User said: \"{user_msg['content']}\"")
            if idx + 1 < len(history):
                ast_msg = history[idx + 1]
                if ast_msg['role'] == 'assistant':
                    recalled_messages.append(f"ThinkingPods replied: \"{ast_msg['content']}\"")

    memory_context = ""
    if recalled_messages:
        memory_context = (
            "\n[RECALLED CONTEXT FROM EARLIER IN THIS DIALOGUE]:\n" +
            "\n".join(recalled_messages) +
            "\n(Use this recalled context only if the user is referring back to a past topic or project details. Otherwise, proceed with the current topic.)\n"
        )

    # Format for Ollama API
    messages = [{"role": "system", "content": f"{system_prompt}\n\n{doc_context}\n{memory_context}"}]

    # Add short-term history (sliding window of the last 4 turns)
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = ollama.chat(
            model=MODELS["llm"],
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        reply = response['message']['content'].strip()
        reply = strip_model_output(reply)
        if not reply:
            reply = "I completed my reasoning but didn't produce a final response. Could you please prompt me again?"
        return reply
    except Exception as e:
        print(f"Ollama generation error: {e}")
        return "I encountered an error while thinking. Is Ollama running?"

def get_speech(text):
    if not MODELS["tts"]:
        return None
    try:
        import scipy.io.wavfile as wavfile

        # Clean text for smoother TTS
        clean_text = text.replace("*", "").replace("#", "").replace("_", "")

        sample_rate = 48000
        speaker = 'en_21' # Using a more expressive speaker
        # silero apply_tts returns a torch tensor
        audio = MODELS["tts"].apply_tts(text=clean_text, speaker=speaker, sample_rate=sample_rate)

        # Explicitly convert to numpy and then to int16 for maximum compatibility
        audio_numpy = (audio.numpy() * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_numpy)
        return buffer.getvalue()
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

def generate_mermaid_diagram(username, context_doc=""):
    if not MODELS["llm"]:
        return LLM_LOADING_MSG_CONCISE

    system_prompt = """You are an Expert System Architect.
Your ONLY task is to generate valid Mermaid.js diagram code based on the conversation.
Valid types: graph TD, flowchart LR, sequenceDiagram, stateDiagram-v2.
CRITICAL: Output ONLY the code inside a ```mermaid block. No conversation, no explanations."""

    doc_context = f"\n[LOCAL VAULT DATA]:\n{context_doc}\n" if context_doc else ""
    history = get_history(username)

    messages = [{"role": "system", "content": f"{system_prompt}\n\n{doc_context}"}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = ollama.chat(model=MODELS["llm"], messages=messages)
        raw_text = response['message']['content'].strip()
        raw_text = strip_model_output(raw_text)
        print(f"DEBUG: Raw Mermaid Response: {raw_text[:200]}...")

        # Regex to extract content inside mermaid blocks
        match = re.search(r"```mermaid\n?(.*?)(?:\n```|$)", raw_text, re.DOTALL)
        if match:
            diagram = match.group(1).strip()
        else:
            # Fallback: find standard mermaid keywords if tags are missing
            lines = raw_text.split("\n")
            clean_lines = []
            started = False
            for line in lines:
                if any(k in line for k in MERMAID_KEYWORDS):
                    started = True
                if started:
                    clean_lines.append(line)
            diagram = "\n".join(clean_lines).replace("```", "").strip()

        # Validation: Ensure it starts with a keyword
        if not any(diagram.startswith(k) for k in MERMAID_KEYWORDS):
            print("DEBUG: Missing keyword, prepending 'graph TD'")
            diagram = "graph TD\n" + diagram

        print(f"DEBUG: Final Cleaned Diagram: {diagram[:200]}...")
        return diagram
    except Exception as e:
        print(f"Mermaid generation error: {e}")
        return "graph TD\n  A[Error] --> B[Could not generate diagram]"

# --- API Endpoints ---

@app.post("/visualize")
def visualize(data: dict = Body(...)):
    username = data.get("username", "User")
    context_doc = data.get("context_doc", "")

    print(f"[{username}] Generating Visualization...")
    diagram_code = generate_mermaid_diagram(username, context_doc=context_doc)

    # We return it as text; frontend will handle rendering
    headers = {
        "X-Reply": urllib.parse.quote(diagram_code),
        "Access-Control-Expose-Headers": "X-Reply"
    }
    return Response(content=diagram_code, media_type="text/plain", headers=headers)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "platform": "Windows",
        "is_apple_silicon": False,
        "stt_loaded": MODELS["stt"] is not None,
        "llm_ready": MODELS["llm"] is not None,
        "tts_loaded": MODELS["tts"] is not None,
        "turns": {},
        "ctx": {}
    }

@app.post("/generate_requirements")
def generate_requirements(prompt: str = Body(..., embed=True)):
    """Generate requirements using Ollama (Optimized for Intel HD 620)."""
    if not MODELS["llm"]:
        return JSONResponse({"error": BRAIN_MODEL_LOADING_MSG}, status_code=503)

    try:
        system_prompt = """You are a Requirements Engineering expert following ISO 29148 standards.
Generate specific, measurable, and verifiable functional requirements.
Output as a numbered list of clear requirements.
Be precise and avoid ambiguous language like "should" — use "shall"."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {prompt}\n\nRequirements:"}
        ]

        response = ollama.chat(model=MODELS["llm"], messages=messages)
        result = response['message']['content'].strip()
        result = strip_model_output(result)
        return {"requirement": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/start")
def start(data: dict = Body(...)):
    name = data.get("name", "User")
    pod = data.get("pod", "empathize")
    # Initialize or retrieve history for this user
    history = get_history(name)
    if not history:
        # Optionally add a system message to start
        add_to_history(name, "system", f"User started session in {pod} stage.")
    reply = f"Hello {name}, welcome to ThinkingPods. I am ready to help you with the {pod} stage."
    audio = get_speech(reply)

    headers = {
        "X-Reply": urllib.parse.quote(reply),
        "Access-Control-Expose-Headers": "X-Reply, X-Transcript"
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)

@app.post("/text")
def text_input(data: dict = Body(...)):
    user_text = data.get("text", "")
    pod = data.get("pod", "general")
    username = data.get("username", "User")
    project_name = data.get("project_name", "MyProject")
    context_doc = data.get("context_doc", "")
    interaction_mode = data.get("interaction_mode", "Design Thinking Coach")
    enable_voice = data.get("enable_voice", False)

    mentor_route = should_use_mentor(interaction_mode, pod)

    if not MODELS["llm"] and not mentor_route:
        return JSONResponse({"error": BRAIN_MODEL_LOADING_MSG}, status_code=503)

    print(f"[{username}] Text Input: {user_text[:50]}...")
    add_to_history(username, "user", user_text)
    
    timing_dict = None
    diagnostics_dict = None
    if mentor_route:
        reply, _, _timing, _diag = process_mentor_turn(user_text, username=username, project_name=project_name, model_name=mentor_model_name())
        timing_dict = _timing.to_dict() if _timing else None
        diagnostics_dict = _diag if _diag else None
    else:
        reply = generate_llm_response(user_text, pod, username, context_doc=context_doc, interaction_mode=interaction_mode)
        
    add_to_history(username, "assistant", reply)
    
    audio = get_speech(reply) if enable_voice else None

    headers = {
        "X-Reply": urllib.parse.quote(reply),
        "Access-Control-Expose-Headers": "X-Reply, X-Transcript"
    }
    if timing_dict:
        headers["X-Timing"] = urllib.parse.quote(json.dumps(timing_dict))
        headers["Access-Control-Expose-Headers"] += ", X-Timing"
    if diagnostics_dict:
        headers["X-Diagnostics"] = urllib.parse.quote(json.dumps(diagnostics_dict))
        headers["Access-Control-Expose-Headers"] += ", X-Diagnostics"
    return Response(content=audio if audio else b"", media_type="audio/wav", headers=headers)

@app.post("/voice")
async def voice_input(
    request: Request,
    x_pod: str = Header("general"),
    x_username: str = Header("User"),
    x_project_name: str = Header("MyProject"),
    x_context_doc: str = Header(""),
    x_interaction_mode: str = Header("Design Thinking Coach"),
    x_enable_voice: str = Header("True")
):
    audio_bytes = await request.body()
    if not MODELS["stt"]:
         return JSONResponse({"error": VOICE_OR_BRAIN_MODEL_LOADING_MSG}, status_code=503)

    enable_voice = x_enable_voice.lower() == "true"
    start_time = time.time()
    # Parse WAV bytes in-memory (skip temp file I/O ~100ms per turn)
    import scipy.io.wavfile as wavfile
    from scipy import signal as scipy_signal

    rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)
    # Normalize to float32 [-1, 1] for faster-whisper (handles both int and float PCM)
    if audio_data.dtype.kind == 'f':
        audio_float = audio_data.astype(np.float32)
    else:
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    # Resample to 16kHz if needed (faster-whisper expects 16kHz mono float32)
    if rate != 16000:
        n_samples = int(len(audio_float) * 16000 / rate)
        audio_float = scipy_signal.resample(audio_float, n_samples)

    # Transcribe
    print(f"[{x_username}] Transcribing voice...")
    try:
        stt_start = time.time()
        segments, _ = MODELS["stt"].transcribe(
            audio_float,
            beam_size=5, 
            language="en",
            initial_prompt="This is a voice recording of a user discussing the Sathnur software village project, rural to urban development, and design thinking. Please transcribe names like Sathnur, software village, rural-urban, and technical terms accurately.",
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500), 
            condition_on_previous_text=False,
            temperature=0.0
        )
        # Evaluate segments inside the timing block and filter out background noise/silence
        valid_segments = []
        for s in segments:
            if s.no_speech_prob < 0.45 and s.avg_logprob > -1.0 and s.compression_ratio < 2.4:
                valid_segments.append(s.text)
        transcript = " ".join(valid_segments).strip()
        stt_end = time.time()
        print(f"[{x_username}] Transcript: '{transcript}'")

        # Filter out common Whisper hallucinations on silence/noise
        hallucinations = ["1, 2, 3", "one, two, three", "thank you", "thanks for watching", "subtitle by", "1, 2, 3, 1, 1, 2, 3", "test", "testing", "check"]
        lower_transcript = transcript.lower()
        if not transcript or (any(h in lower_transcript for h in hallucinations) and len(transcript) < 30):
             # If it looks like a test, just acknowledge it briefly without over-analyzing
             reply = "Mic check received! I can hear you clearly now. What shall we work on?"
             audio = get_speech(reply) if enable_voice else None
             headers = {
                "X-Transcript": urllib.parse.quote(transcript),
                "X-Reply": urllib.parse.quote(reply),
                "Access-Control-Expose-Headers": "X-Reply, X-Transcript"
             }
             return Response(content=audio if audio else b"", media_type="audio/wav", headers=headers)

        # Determine context
        pod = x_pod
        username = x_username
        project_name = urllib.parse.unquote(x_project_name)
        context_doc = urllib.parse.unquote(x_context_doc)
        interaction_mode = x_interaction_mode

        add_to_history(username, "user", transcript)
        print(f"[{username}] Generating AI response...")
        llm_start = time.time()
        
        timing_dict = None
        diagnostics_dict = None
        if should_use_mentor(interaction_mode, pod):
            reply, _, _timing, _diag = process_mentor_turn(transcript, username=username, project_name=project_name, model_name=mentor_model_name())
            timing_dict = _timing.to_dict() if _timing else None
            diagnostics_dict = _diag if _diag else None
        else:
            if not MODELS["llm"]:
                return JSONResponse({"error": BRAIN_MODEL_LOADING_MSG}, status_code=503)
            reply = generate_llm_response(transcript, pod, username, context_doc=context_doc, interaction_mode=interaction_mode)
            
        llm_end = time.time()
        add_to_history(username, "assistant", reply)

        audio = get_speech(reply) if enable_voice else None
        total_end = time.time()

        stt_time = f"{stt_end - stt_start:.2f}"
        llm_time = f"{llm_end - llm_start:.2f}"
        total_time = f"{total_end - start_time:.2f}"
        print(f"[{username}] Done in {total_time}s (STT: {stt_time}s, LLM: {llm_time}s)")

        headers = {
            "X-Transcript": urllib.parse.quote(transcript),
            "X-Reply": urllib.parse.quote(reply),
            "X-Timing-STT": stt_time,
            "X-Timing-LLM": llm_time,
            "X-Timing-Total": total_time,
            "Access-Control-Expose-Headers": "X-Reply, X-Transcript, X-Timing-STT, X-Timing-LLM, X-Timing-Total"
        }
        if timing_dict:
            headers["X-Timing"] = urllib.parse.quote(json.dumps(timing_dict))
            headers["Access-Control-Expose-Headers"] += ", X-Timing"
        if diagnostics_dict:
            headers["X-Diagnostics"] = urllib.parse.quote(json.dumps(diagnostics_dict))
            headers["Access-Control-Expose-Headers"] += ", X-Diagnostics"
        return Response(content=audio if audio else b"", media_type="audio/wav", headers=headers)
    except Exception as e:
        print(f"Error in voice_input: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/session/status")
async def session_status(request: Request):
    data = await request.json()
    project_name = data.get("project_name", "MyProject")
    session = build_mentor_session(project_name)
    mgr = get_session_manager()
    sd = mgr.get_active_session_data()
    status_dict = ChecklistManager.get_status_dict(sd.project_state)
    return JSONResponse({
        "project_name": session.project_name,
        "current_stage": session.current_stage,
        "checklist": status_dict,
        "known_facts": session.known_facts,
        "assumptions": session.assumptions,
        "unknown_facts": session.unknown_facts,
        "open_questions": session.open_questions
    })

# --- New Session Management Endpoints ---

@app.post("/session/start")
async def session_start(request: Request):
    """
    Deterministic app-opening session bind (see ``session_lifecycle.py``).

    mode="fresh":    archive any current session and begin a pristine one.
                     Returns a brand-new session_id. Opening the application
                     without an explicit restore request always uses this.
    mode="restore":  bind to the persisted active session WITHOUT resetting
                     or archiving. Returns its current session_id ("" when
                     no persisted session exists).

    The response includes ``manual_refresh_needed`` — always ``false`` for a
    successful bind, so the frontend never needs a manual browser refresh
    after opening the application.
    """
    data = await request.json()
    mode = data.get("mode", SessionLifecycle.MODE_FRESH)
    username = data.get("username", "User")
    project_name = data.get("project_name", "MyProject")

    decision = SessionLifecycle.decide_startup(
        explicit_restore=(
            data.get("restore_project") if mode == SessionLifecycle.MODE_RESTORE else None
        ),
        project_name=project_name,
    )

    session_mgr = get_session_manager()
    if decision.backend_mode == SessionLifecycle.MODE_FRESH:
        clear_history(username)

    result = SessionLifecycle.bind_session(session_mgr, decision, username=username)
    return JSONResponse({
        "action": result["action"],
        "mode": result["backend_mode"],
        "session_id": result["session_id"],
        "manual_refresh_needed": result["manual_refresh_needed"],
        "message": f"{result['action']} session bound",
    })


@app.post("/session/new")
async def session_new(request: Request):
    """
    Create a new mentoring session.
    
    Archives the current session (if any) and starts a fresh session with
    empty state. Returns the new session ID.
    
    This clears ALL runtime state:
    - Conversation history
    - Legacy mentor session file (belt-and-suspenders)
    - New session folder (via reset_runtime_state)
    """
    data = await request.json()
    username = data.get("username", "User")
    project_name = data.get("project_name", "MyProject")
    
    # Clear conversation history for this user
    clear_history(username)
    
    # Create new session via SessionManager (archives old, creates fresh)
    session_mgr = get_session_manager()
    metadata = session_mgr.reset_runtime_state(username=username)
    
    return JSONResponse({
        "session_id": metadata.session_id,
        "status": "created",
        "created_at": metadata.created_at,
        "project_title": metadata.project_title,
    })


@app.get("/session/list")
async def session_list():
    """List all available sessions."""
    session_mgr = get_session_manager()
    sessions = session_mgr.list_sessions()
    return JSONResponse({
        "sessions": [s.to_dict() for s in sessions]
    })


@app.post("/session/switch")
async def session_switch(request: Request):
    """Switch to a different session."""
    data = await request.json()
    session_id = data.get("session_id", "")
    
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    
    session_mgr = get_session_manager()
    success = session_mgr.switch_session(session_id)
    
    if success:
        # Also clear conversation history since we're switching context
        username = data.get("username", "User")
        clear_history(username)
        return JSONResponse({"status": "ok", "session_id": session_id})
    else:
        return JSONResponse({"error": "Session not found"}, status_code=404)


@app.post("/session/archive")
async def session_archive(request: Request):
    """Archive the current active session."""
    session_mgr = get_session_manager()
    success = session_mgr.archive_current_session()
    
    if success:
        return JSONResponse({"status": "ok", "message": "Current session archived"})
    else:
        return JSONResponse({"error": "No active session to archive"}, status_code=404)


@app.get("/session/active")
async def session_active():
    """Get the currently active session metadata."""
    session_mgr = get_session_manager()
    meta = session_mgr.get_active_metadata()
    
    if meta:
        return JSONResponse(meta.to_dict())
    else:
        return JSONResponse({"active": False})

# End new session management endpoints

@app.post("/reset")
async def reset(request: Request):
    data = await request.json()
    username = data.get("username", "User")
    project_name = data.get("project_name", "MyProject")
    clear_history(username)

    # Reset mentor session
    session_mgr = get_session_manager()
    session_mgr.reset_runtime_state(username=username)

    return {"status": "ok", "message": f"History and mentor session cleared for {username}"}

# Additional endpoints expected by frontend
@app.post("/archive/check")
async def archive_check(request: Request):
    data = await request.json()
    name = data.get("name", "")
    # stub: no archive found
    return JSONResponse({"exists": False, "sessions": 0})

@app.post("/archive/load")
async def archive_load(request: Request):
    data = await request.json()
    name = data.get("name", "")
    # stub: return empty sessions
    return JSONResponse({"exists": False, "sessions": [], "projects": []})

@app.post("/archive/delete")
async def archive_delete(request: Request):
    data = await request.json()
    name = data.get("name", "")
    return JSONResponse({"message": f"Archive for {name} deleted (stub)"})

@app.get("/debug/llm")
async def debug_llm():
    # stub info
    return JSONResponse({
        "llm_ready": MODELS["llm"] is not None,
        "llm_url": "local",
        "status": "ready" if MODELS["llm"] else "not loaded"
    })

@app.post("/session/save")
async def session_save(request: Request):
    # stub
    return JSONResponse({"status": "ok", "message": "Session saved (stub)"})

@app.post("/session/load")
async def session_load(request: Request):
    # stub
    return JSONResponse({"turns": {}})

@app.get("/tts/test")
async def tts_test():
    # generate a simple test audio
    if not MODELS["tts"]:
        return Response(status_code=503, content="TTS not loaded")
    sample_text = "Testing text to speech."
    audio = get_speech(sample_text)
    if audio is None:
        return Response(status_code=500, content="Failed to generate speech")
    return Response(content=audio, media_type="audio/wav")

@app.get("/")
async def get_index():
    return {"message": "ReqGPT Backend is running! Use the Streamlit app to interact."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
