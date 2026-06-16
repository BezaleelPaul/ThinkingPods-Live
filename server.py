from fastapi import FastAPI, Request, Header, Body
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import os
import io
import torch
import urllib.parse
import scipy.io.wavfile as wavfile
import ollama
from faster_whisper import WhisperModel

import threading
from collections import defaultdict
import time

# --- Optimization for Intel HD 620 / Laptop CPUs ---
torch.set_num_threads(4) 

app = FastAPI()

# --- Models ---
MODELS = {
    "llm": None,
    "stt": None,
    "tts": None
}

# Conversation history per user (username -> list of {role, content})
conversation_histories = defaultdict(list)
history_lock = threading.Lock()

def load_models():
    try:
        print("Loading models...")
        # 1. LLM (Ollama - Optimized for Intel HD 620 / Laptop CPUs)
        target_model = "llama3.2:1b" # Ultra-fast on iGPUs
        try:
            print(f"Checking for Ollama model: {target_model}...")
            ollama.pull(target_model)
            MODELS["llm"] = target_model
            print(f"LLM ready via Ollama: {target_model}")
        except Exception as e:
            print(f"Ollama connection error or pull failed: {e}")
            print("Make sure Ollama is running (ollama serve).")

        # 2. STT (Whisper)
        try:
            # cpu_threads=4 helps balance load on laptop CPUs
            MODELS["stt"] = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)
        except Exception as e:
            print(f"STT model failed to load: {e}")

        # 4. TTS (Silero)
        try:
            language = 'en'
            model_id = 'v3_en'
            device = torch.device('cpu')
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                 model='silero_tts',
                                                 language=language,
                                                 speaker=model_id,
                                                 trust_repo=True)
            model.to(device)
            MODELS["tts"] = model
        except Exception as e:
            print(f"TTS model failed to load: {e}")

        print("Models loaded.")
    except Exception as e:
        print(f"Fatal error during model loading: {e}")

# Load models in background to not block startup
threading.Thread(target=load_models).start()

# --- Helper Functions ---
def get_history(username):
    with history_lock:
        return conversation_histories[username]

def add_to_history(username, role, content):
    with history_lock:
        conversation_histories[username].append({"role": role, "content": content})
        # Keep only last 10 exchanges (20 messages) to stay within context
        if len(conversation_histories[username]) > 20:
            conversation_histories[username] = conversation_histories[username][-20:]

def clear_history(username):
    with history_lock:
        if username in conversation_histories:
            del conversation_histories[username]

def generate_llm_response(text, pod, username, context_doc=""):
    if not MODELS["llm"]:
        return "Model still loading, please wait! I'll be ready in just a second! 🚀"

    # Grounded System Prompts
    if pod == "general" or "help" in text.lower():
        system_prompt = """You are ThinkingPods, a professional and helpful Design Thinking Consultant. 
Your tone is encouraging, insightful, and natural. 
Help the user refine their ideas by asking sharp, critical, yet friendly questions. 
IMPORTANT: Speak like a real person. Use verbal cues like 'Hmm', 'I see', or 'That's interesting' to show you are listening. 
Express empathy and excitement where appropriate. Do NOT use hashtags (#) or excessive emojis."""
    else:
        system_prompt = f"""You are a professional Requirements Engineer helping the user with the '{pod}' stage. 
Your goal is to guide the user toward high-quality software requirements (ISO 29148 standard) through natural conversation. 
Be precise and professional, but keep the tone conversational and human. 
Show personality and professional passion for the project.
IMPORTANT: Do NOT use hashtags (#) and focus on being helpful."""

    doc_context = f"\n[LOCAL VAULT DATA]:\n{context_doc}\n" if context_doc else ""

    # Retrieve conversation history
    history = get_history(username)
    
    # Format for Ollama API
    messages = [{"role": "system", "content": f"{system_prompt}\n\n{doc_context}"}]
    
    # Add history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = ollama.chat(
            model=MODELS["llm"],
            messages=messages,
            options={
                "temperature": 0.8, # Slightly higher for more variety
                "top_p": 0.9,
                "num_ctx": 4096
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Ollama generation error: {e}")
        return "⚠️ I encountered an error while thinking. Is Ollama running? 🤖"

def get_speech(text):
    if not MODELS["tts"]:
        return None
    try:
        # Clean text for smoother TTS
        clean_text = text.replace("*", "").replace("#", "").replace("_", "")
        
        sample_rate = 48000
        speaker = 'en_21' # Using a more expressive speaker
        # silero apply_tts returns a torch tensor
        audio = MODELS["tts"].apply_tts(text=clean_text, speaker=speaker, sample_rate=sample_rate)
        
        # Explicitly convert to numpy and then to int16 for maximum compatibility
        import numpy as np
        audio_numpy = (audio.numpy() * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_numpy)
        return buffer.getvalue()
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

# --- API Endpoints ---

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
        return JSONResponse({"error": "Brain model still loading... please wait! 🚀"}, status_code=503)
    
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
    context_doc = data.get("context_doc", "")

    if not MODELS["llm"]:
        return JSONResponse({"error": "Brain model still loading... please wait! 🚀"}, status_code=503)

    print(f"[{username}] Text Input: {user_text[:50]}...")
    add_to_history(username, "user", user_text)
    reply = generate_llm_response(user_text, pod, username, context_doc=context_doc)
    add_to_history(username, "assistant", reply)
    audio = get_speech(reply)

    headers = {
        "X-Reply": urllib.parse.quote(reply),
        "Access-Control-Expose-Headers": "X-Reply, X-Transcript"
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)

@app.post("/voice")
async def voice_input(
    request: Request,
    x_pod: str = Header("general"), 
    x_username: str = Header("User"), 
    x_context_doc: str = Header("")
):
    audio_bytes = await request.body()
    if not MODELS["stt"] or not MODELS["llm"]:
         return JSONResponse({"error": "Voice or Brain models still loading... please wait! 🚀"}, status_code=503)

    start_time = time.time()
    temp_filename = f"temp_{x_username}_voice.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)

    # Transcribe
    print(f"[{x_username}] Transcribing voice...")
    try:
        stt_start = time.time()
        segments, _ = MODELS["stt"].transcribe(temp_filename)
        stt_end = time.time()
        transcript = " ".join([s.text for s in segments]).strip()
        print(f"[{x_username}] Transcript: '{transcript}'")

        # Filter out common Whisper hallucinations on silence/noise
        hallucinations = ["1, 2, 3", "one, two, three", "thank you", "thanks for watching", "subtitle by", "1, 2, 3, 1, 1, 2, 3", "test", "testing", "check"]
        lower_transcript = transcript.lower()
        if not transcript or (any(h in lower_transcript for h in hallucinations) and len(transcript) < 30):
             # If it looks like a test, just acknowledge it briefly without over-analyzing
             reply = "Mic check received! I can hear you clearly now. What shall we work on?"
             audio = get_speech(reply)
             headers = {
                "X-Transcript": urllib.parse.quote(transcript),
                "X-Reply": urllib.parse.quote(reply),
                "Access-Control-Expose-Headers": "X-Reply, X-Transcript"
             }
             return Response(content=audio, media_type="audio/wav", headers=headers)

        # Determine context
        pod = x_pod
        username = x_username
        context_doc = urllib.parse.unquote(x_context_doc)

        add_to_history(username, "user", transcript)
        print(f"[{username}] Generating AI response...")
        llm_start = time.time()
        reply = generate_llm_response(transcript, pod, username, context_doc=context_doc)
        llm_end = time.time()
        add_to_history(username, "assistant", reply)

        audio = get_speech(reply)
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
        return Response(content=audio, media_type="audio/wav", headers=headers)
    except Exception as e:
        print(f"Error in voice_input: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.post("/reset")
async def reset(request: Request):
    data = await request.json()
    username = data.get("username", "User")
    clear_history(username)
    return {"status": "ok", "message": f"History cleared for {username}"}

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