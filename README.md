# ReqGPT / ThinkingPods — AI Design Thinking Assistant

An offline-first, local AI assistant for Design Thinking sessions (Empathize, Discovery, and Coaching).

> **Core Philosophy:** *The application does the thinking; the LLM does the talking.*
> Deterministic state, checklists, and question planning ensure consistency, while local LLMs (Ollama) convert facts into conversational responses.

---

## 📁 Repository Organization

```text
thinkingpodsreamke/
├── app.py                      # Streamlit frontend UI (Port 8501) with Developer Console
├── server.py                   # FastAPI backend server (Port 8000) for LLM, STT, TTS
├── mentor.py                   # Central mentor pipeline orchestrator
├── autotweak.py                # Hardware detection & Ollama model tuner
├── diagnose.py                 # System & environment health check
├── run.bat / run.py            # Primary launchers for backend + frontend
│
├── track_mistakes.py           # 🔍 Unified Mistake & Failure Tracking Engine
├── run_track_mistakes.bat      # 🚀 1-Click Mistake Tracker launcher for Windows
│
├── cli/                        # 🎙️ Standalone CLI voice & text loops
│   ├── dina_direct.py          # Direct Dina voice loop
│   ├── design_thinking_coach.py# Coach voice loop
│   └── design_thinking_coach_text.py # Coach text loop
│
├── audits/                     # 🔬 Evaluation, replay & pipeline audit runners
│   ├── replay_lab.py           # Replay lab regression harness
│   ├── run_conversation_failure.py # Failure classification runner
│   ├── phase4_replay.py        # Phase 4 replay harness
│   ├── phase7_evaluation.py    # Phase 7 evaluation suite
│   └── pipeline_trace.py       # Pipeline execution tracer
│
├── module3/                    # Module 3: Objective Engine & Completeness Checker
├── module4/                    # Module 4: Response Strategy & Prompt Builder
├── module5/                    # Module 5: Lifecycle Manager & Summary Builder
│
├── docs/                       # 📚 Documentation & Reference Materials
│   ├── reports/                # 28 deep-dive analysis, architecture & audit reports
│   └── reference/              # Criteria PDFs, training materials, PRD documents
│
├── benchmarks/                 # ⏱️ Latency & startup profiling scripts
│   ├── benchmark_latency.py
│   ├── measure_latency.py
│   └── measure_startup.py
│
├── archive/                    # 📦 Retired legacy code & scratch scripts
│   ├── legacy_code/            # Deprecated HF Mistral & unused Ollama client
│   └── scratch/                # One-off development & test scripts
│
├── tests/                      # 🧪 Test suite (50+ tests & golden fixtures)
│   └── goldens/                # Golden conversation fixture test cases
│
└── sessions/                   # 💾 Persisted user session data (JSON)
```

---

## 🔍 How to Track Mistakes & Failures

ReqGPT includes a built-in **Mistake & Conversation Failure Tracking Engine** designed to identify whenever the mentor makes an error or drifts out of sync with the user.

### Quick Start
To run a complete audit across all golden scenarios and saved sessions:
```bash
# Windows 1-Click
run_track_mistakes.bat

# Or via Command Line
python track_mistakes.py --verbose --export
```

### Options
| Command | Description |
|---|---|
| `python track_mistakes.py` | Run standard global mistake audit |
| `python track_mistakes.py --goldens` | Audit all golden test scenarios |
| `python track_mistakes.py --sessions` | Audit real user sessions in `sessions/` |
| `python track_mistakes.py --session <file>` | Inspect a specific session file |
| `python track_mistakes.py -v` / `--verbose` | Print turn-by-turn root causes & fixes |
| `python track_mistakes.py --export` | Generate `docs/reports/Mistakes_Audit_Report.md` |

### Mistake Categories Tracked
- 🔴 **MISUNDERSTOOD_RESPONSE**: Mentor re-asks for a field that the user already answered.
- 🔴 **ABRUPT_TRANSITION**: Mentor skips ahead or summarizes without acknowledging the user's thread.
- 🔴 **MEMORY_FAILURE**: Conversational memory contradicts itself (e.g. topic open and closed).
- 🔴 **MOVE_SELECTION**: Mentor selected an elicit move without actually asking a question.
- 🟡 **REPEATED_INFORMATION**: User is forced to repeat details because the mentor ignored them.
- 🟡 **OVER_EXPLORATION**: Mentor pesters user on a topic that was already resolved.
- 🟡 **UNSUPPORTED_INFERENCE**: Mentor makes assumptions without supporting facts.
- 🟡 **MISSED_INSIGHT**: State contains extracted data that memory dropped.
- ℹ️ **META_INTENT**: User provides non-substantive input (greeting, validation check).

---

## 🚀 Running the System

### Prerequisites
1. **Ollama running**: `ollama serve`
2. **Models downloaded**: `llama3.2:1b` (or `qwen2.5:3b`), faster-whisper `tiny.en`

### Launch Backend & Frontend
```cmd
run.bat
```
*(Runs `autotweak.py` for hardware optimization, then starts FastAPI on port 8000 and Streamlit on port 8501).*

### Diagnostics
Check environment and dependencies:
```cmd
python diagnose.py
```

### Tests
Run pytest across all modules:
```cmd
pytest tests
```

---

## 🌐 Free Deployment Options

ReqGPT requires running local AI models (Ollama LLM, faster-whisper STT, and Silero TTS), which require at least 2–4 GB RAM. Here are the two **100% free** ways to deploy and share it:

### Method 1: Instant Free Public Link (Zero Cost, No Account Needed)
Use the included 1-click Cloudflare Tunnel script to generate a secure, temporary `https://....trycloudflare.com` URL pointing directly to your running instance:
1. Start the system: `run.bat`
2. In a separate window, run: `run_public_tunnel.bat`
3. Copy the generated `https://....trycloudflare.com` link and open it on your phone or share it with anyone.

### Method 2: Free 24/7 Cloud Hosting (Hugging Face Spaces)
Hugging Face offers free 2 vCPU / 16 GB RAM Docker Spaces:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space SDK to **Docker** (Blank).
3. Connect your GitHub repository (`VenkataAdityaS125/ThinkingPods`).
4. Hugging Face will automatically build using the included `Dockerfile` and `start.sh`, boot Ollama, download the model, and launch the UI on the web for free 24/7.

