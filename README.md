# ReqGPT / ThinkingPods — AI Design Thinking Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BezaleelPaul/ThinkingPods-Live/blob/main/ThinkingPods_Cloud.ipynb)
[![Live Web App](https://img.shields.io/badge/Live_App-GitHub_Pages-blue?logo=github)](https://bezaleelpaul.github.io/ThinkingPods-Live/)

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

### Method 2: 1-Click Google Cloud Hosting (100% Free · 12.7 GB RAM & GPU)
Run ThinkingPods entirely on Google Cloud without using your personal computer's CPU or memory:
1. Click the **[Open in Colab](https://colab.research.google.com/github/BezaleelPaul/ThinkingPods-Live/blob/main/ThinkingPods_Cloud.ipynb)** badge at the top of the repository.
2. In Google Colab, click **Runtime ➡️ Run all**.
3. It will provision Google's high-speed cloud environment, start Ollama, boot the FastAPI backend and Streamlit UI, and output a live `https://....trycloudflare.com` link.
4. Your computer usage is 0%, and it runs on Google's free 12.7 GB RAM and NVIDIA GPU.


