import streamlit as st
import requests
import os
import io
import json
from datetime import datetime
import urllib.parse
from audio_recorder_streamlit import audio_recorder

# --- Backend Configuration ---
BACKEND_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(
    page_title="ReqGPT | Mission Control",
    page_icon="🚀",
    layout="centered"
)

# --- Persistence Helpers ---
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def save_session(name):
    data = {
        "messages": st.session_state.messages,
        "dt_phase": st.session_state.dt_phase,
        "timestamp": datetime.now().isoformat()
    }
    filepath = os.path.join(SESSION_DIR, f"{name}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.error(f"Failed to save session: {e}")

def load_session(name):
    filepath = os.path.join(SESSION_DIR, f"{name}.json")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            st.session_state.messages = data["messages"]
            st.session_state.dt_phase = data["dt_phase"]
    except FileNotFoundError:
        st.error(f"Session '{name}' not found.")
    except Exception as e:
        st.error(f"Failed to load session: {e}")

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ThinkingPods, your Design Thinking Consultant. Tell me, what project or idea are we working on today?"}
    ]
if "dt_phase" not in st.session_state:
    st.session_state.dt_phase = "Discovery"
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "audio_played" not in st.session_state:
    st.session_state.audio_played = False

# --- Backend Communication ---

def call_backend_text(text, phase, doc_ctx="", username="User"):
    try:
        payload = {"text": text, "pod": phase, "username": username, "context_doc": doc_ctx}
        response = requests.post(f"{BACKEND_URL}/text", json=payload)
        
        if response.status_code == 200:
            reply = urllib.parse.unquote(response.headers.get("X-Reply", ""))
            return reply, response.content # reply text and audio bytes
        else:
            try:
                err_msg = response.json().get("error", f"Error {response.status_code}")
            except:
                err_msg = f"Error {response.status_code}"
            return f"⚠️ {err_msg}", None
    except Exception as e:
        return f"⚠️ Backend Unreachable: {e}", None

def call_backend_voice(audio_bytes, phase, doc_ctx="", username="User"):
    try:
        headers = {
            "X-Pod": phase, 
            "X-Username": username,
            "X-Context-Doc": urllib.parse.quote(doc_ctx[:2000]) # Limit context size for headers
        }
        response = requests.post(f"{BACKEND_URL}/voice", data=audio_bytes, headers=headers)
        
        if response.status_code == 200:
            transcript = urllib.parse.unquote(response.headers.get("X-Transcript", ""))
            reply = urllib.parse.unquote(response.headers.get("X-Reply", ""))
            return transcript, reply, response.content
        else:
            try:
                err_msg = response.json().get("error", f"Error {response.status_code}")
            except:
                err_msg = f"Error {response.status_code}"
            return None, f"⚠️ {err_msg}", None
    except Exception as e:
        return None, f"⚠️ Backend Unreachable: {e}", None

def call_backend_reqgpt(prompt):
    try:
        payload = {"prompt": prompt}
        response = requests.post(f"{BACKEND_URL}/generate_requirements", json=payload)
        if response.status_code == 200:
            return response.json().get("requirement", "")
        else:
            return f"⚠️ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"⚠️ Backend Unreachable: {e}"

# --- Brain Logic (Moved to Backend) ---
# generate_chat_response is now handled by call_backend_text and call_backend_voice

# --- Mermaid Rendering ---
def render_mermaid(code):
    # Strip markdown code blocks if present
    code = code.replace("```mermaid", "").replace("```", "").strip()
    if "graph" not in code and "flowchart" not in code:
        return # Not mermaid code

    html_code = f"""
    <div style="background: white; padding: 10px; border-radius: 5px;">
        <pre class="mermaid">
            {code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    st.components.v1.html(html_code, height=400, scrolling=True)

# --- PRD Formatting ---
def format_prd():
    doc = f"# Product Requirements Document (PRD)\n"
    doc += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    doc += f"## Project Log\n"
    for m in st.session_state.messages:
        role = "Consultant" if m["role"] == "assistant" else "User"
        doc += f"**{role}:** {m['content']}\n\n"
    return doc

# --- UI Layout ---
st.title("🚀 ThinkingPods: Mission Control")

with st.sidebar:
    st.header("💾 Mission Memory")

    # Save/Load UI
    project_name = st.text_input("Project Name", value="MyProject")
    if st.button("Save Mission"):
        save_session(project_name)
        st.success(f"Saved to {project_name}")

    try:
        existing_sessions = [f.replace(".json", "") for f in os.listdir(SESSION_DIR) if f.endswith(".json")]
    except FileNotFoundError:
        existing_sessions = []
    if existing_sessions:
        selected_session = st.selectbox("Load Mission", ["None"] + existing_sessions)
        if selected_session != "None" and st.button("Load Now"):
            load_session(selected_session)
            st.rerun()

    st.divider()
    st.header("🛤️ Journey Status")
    st.markdown(f"**Current Phase:** `{st.session_state.dt_phase}`")

    if st.button("Move to Requirements Phase"):
        st.session_state.dt_phase = "Requirements"
        st.rerun()
    if st.button("Back to Discovery"):
        st.session_state.dt_phase = "Discovery"
        st.rerun()

    st.divider()
    st.header("🔍 Analysis Tools")
    col_a, col_b = st.columns(2)
    trigger_critique = col_a.button("🛡️ Critique")
    trigger_visualize = col_b.button("🗺️ Visualize")
    
    st.divider()
    st.header("✨ AI Power Tools")
    trigger_reqgpt = st.button("💎 Generate ISO Requirements")
    if trigger_reqgpt:
        with st.spinner("🚀 Calling specialized ReqGPT model..."):
            # Use the last user message or a generic prompt
            last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "Requirement: The system shall")
            specialized_req = call_backend_reqgpt(last_user_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"**Specialized Requirement:**\n\n{specialized_req}"})
            st.rerun()

    st.divider()
    st.header("📄 Export")
    if st.button("Prepare PRD"):
        prd_content = format_prd()
        st.download_button(
            label="Download .md PRD",
            data=prd_content,
            file_name=f"{project_name}_PRD.md",
            mime="text/markdown"
        )

    st.divider()
    st.header("🗄️ Knowledge Vault")
    uploaded_file = st.file_uploader("Upload Project Context", type=["txt", "md", "csv"])
    doc_context = ""
    if uploaded_file:
        doc_context = uploaded_file.read().decode("utf-8")
        st.success(f"Document Ingested (Limited to ~2000 chars for memory safety)")

    st.divider()
    if st.button("🗑️ Clear Mission / Start Over"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ThinkingPods. Let's start fresh. What's the idea?"}]
        st.session_state.dt_phase = "Discovery"
        st.rerun()

    st.divider()
    st.header("🎤 Voice Interface")
    audio_bytes = audio_recorder(text="Click to Speak", icon_size="2x", key="recorder")
    enable_voice = st.checkbox("Enable AI Voice Replies", value=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph TD" in message["content"] or "flowchart" in message["content"]:
            render_mermaid(message["content"])

# --- Process Specialized Tasks ---
if trigger_critique:
    with st.chat_message("assistant"):
        with st.spinner("🔍 Auditing requirements..."):
            ai_reply, audio_reply = call_backend_text("Critique the requirements for weak words.", st.session_state.dt_phase, doc_ctx=doc_context)
            st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            if enable_voice and audio_reply:
                st.audio(audio_reply, format="audio/wav", autoplay=True)

if trigger_visualize:
    with st.chat_message("assistant"):
        with st.spinner("🗺️ Mapping logic..."):
            ai_reply, audio_reply = call_backend_text("Visualize the system logic as mermaid code.", st.session_state.dt_phase, doc_ctx=doc_context)
            st.markdown(ai_reply)
            render_mermaid(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            if enable_voice and audio_reply:
                st.audio(audio_reply, format="audio/wav", autoplay=True)

# --- Input Handlers ---
if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
    with st.spinner("👂 Hearing and thinking..."):
        transcript, ai_reply, audio_reply = call_backend_voice(audio_bytes, st.session_state.get("dt_phase", "Discovery"), doc_ctx=doc_context)
        if transcript:
            st.session_state.last_audio_bytes = audio_bytes
            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            st.session_state.last_audio = audio_reply
            st.session_state.audio_played = False
            st.rerun()
        elif ai_reply and "⚠️" in ai_reply:
            st.session_state.last_audio_bytes = audio_bytes
            st.error(ai_reply)
        else:
            st.session_state.last_audio_bytes = audio_bytes
            st.warning("I couldn't quite hear that. Could you try again? 🎤")

user_input = st.chat_input("Type your reply here...")

# --- Process Input ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            ai_reply, audio_reply = call_backend_text(user_input, st.session_state.get("dt_phase", "Discovery"), doc_ctx=doc_context)
            st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            st.session_state.last_audio = audio_reply
            st.session_state.audio_played = False
            st.rerun()

# --- Render Persistent Audio ---
if st.session_state.last_audio and not st.session_state.audio_played:
    if enable_voice:
        st.audio(st.session_state.last_audio, format="audio/wav", autoplay=True)
        st.session_state.audio_played = True