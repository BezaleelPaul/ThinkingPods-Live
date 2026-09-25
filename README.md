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
