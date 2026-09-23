# Startup Performance Report — ReqGPT

Optimization of backend startup using measurements from the **Product Experience
Audit** (`product_experience_audit` / `run_product_experience.py`) and a direct
boot benchmark (`measure_startup.py`).

## Change

`server.py` now **lazy-imports** the heavyweight runtime dependencies that were
previously pulled in synchronously at module import:

- `import torch`
- `from faster_whisper import WhisperModel`  (transitively imports `transformers`
  + `ctranslate2`, ~5 s)
- `import scipy.io.wavfile`
- `from scipy import signal`

These are imported locally, exactly where they are used at runtime:

- `load_models` — `torch`, `faster_whisper.WhisperModel` (STT/TTS setup thread)
- `get_speech` — `scipy.io.wavfile`
- `voice_input` — `scipy.io.wavfile`, `scipy.signal`

`torch.set_num_threads(4)` moved inside `load_models`. The background model-load
thread (`load_models` → `threading.Thread`) already existed and is unchanged, so
models still load asynchronously after the server is reachable.

No conversation, extraction, objective, lifecycle, prompting, or reply logic was
changed. The mentor pipeline is completely untouched.

## Measured results (before → after)

| Metric | Before | After | Reduction |
| --- | ---: | ---: | --- |
| `import server` wall-clock | 13.9 s | 3.4 s | **−75%** |
| Boot → first `GET /health` 200 | 14.2 s | 3.7 s | **−74%** |
| RSS at first `/health` response | 326.9 MB | 105.5 MB | **−68%** |

All measured via `measure_startup.py` (fresh subprocess, psutil, poll `/health`).

- **Startup reduction:** the FastAPI server now binds port 8000 and answers
  `/health` in ~3.7 s instead of ~14 s, and first-response latency (time until
  the backend can serve a request) drops by ~10.5 s.
- **First-response latency:** the server responds almost immediately while
  STT/TTS models still finish loading in the background (verified: all of
  `llm_ready` / `stt_loaded` / `tts_loaded` reach `true` and `/tts/test`
  returns a valid WAV; the mentor `/text` route returns 200).
- **Memory usage:** at first response, RSS drops ~68% because `torch`,
  `faster_whisper`, `transformers`, and `ctranslate2` are no longer imported
  eagerly. If voice/STT is never used, that memory is never allocated at all.
  If voice is used, memory converges to the same steady-state once the models
  load (bandwidth of the optimization is the *startup window*, not steady state).

## Verification

- **Product Experience Audit** (`run_product_experience.py`, deterministic, 32
  turns across 12 fixtures):
  - Startup (audit's own measurement): first-message 36 ms → 27 ms, mentor
    creation 5 ms → 4 ms, avg turn 50 ms → 45 ms — within normal run-to-run
    wall-clock noise; no regression (pipeline untouched).
- **Replay Lab identical:** two independent captures compared
  (`replay_lab.py compare`) → **Changed: 0 across all 12 fixtures / 32 turns**.
  The only reported deltas are `extraction_ms` microbenchmark jitter; replies,
  objectives, transitions, state, prompts, and memory are byte-identical.
- **Golden fixtures identical:** covered by `tests/test_golden_conversations.py`
  in the suite — passed.
- **Full suite:** `python -m pytest tests -q` → **1047 passed, 140 subtests
  passed**.

## Determinism

The optimization is confined to *when* third-party libraries are imported, not
what the application computes. Because `server.py` is not in the import graph of
the mentor pipeline, Replay Lab, the goldens, or the developer audits, all
deterministic outputs are unchanged. Runtime import caching (Python
`sys.modules`) means the lazy imports still run exactly once before their first
use.