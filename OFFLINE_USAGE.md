# ReqGPT Offline Usage Guide

This guide explains how to use ReqGPT in offline mode (without internet connection) after initial setup.

## Prerequisites for Offline Use

Before going offline, you need to:
1. Download all required models while online
2. Ensure Ollama is installed and running locally
3. Cache the Silero TTS model

## Initial Online Setup

Run these steps ONCE while connected to the internet:

### 1. Install Ollama
- Download and install Ollama from https://ollama.com
- Start Ollama service: `ollama serve`

### 2. Pull Required Models
```bash
# Pull the LLM model (optimized for Intel HD 620)
ollama pull llama3.2:1b

# Verify the model is available
ollama list
```

### 3. Cache Silero TTS Model
The Silero TTS model will be automatically cached the first time you run the application. To pre-cache it, you can run:
```python
# Run this Python snippet to pre-cache the TTS model
import torch
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en',
    trust_repo=True
)
print("TTS model cached successfully")
```

## Offline Usage Instructions

Once the above setup is complete, you can use ReqGPT entirely offline:

### Starting the System
```bash
# Start the backend server
python server.py

# In another terminal, start the frontend
streamlit run app.py
```

Or use the provided batch files:
- `run_offline.bat` - Starts backend, frontend, and voice client
- `run_dina_voice.bat` - Starts the full voice assistant system

### What Works Offline
- ✅ Natural language conversations with ThinkingPods AI
- ✅ Voice input and output (speech-to-text and text-to-speech)
- ✅ Requirement generation following ISO 29148 standards
- ✅ System visualization (Mermaid diagrams)
- ✅ Session saving and loading
- ✅ All AI-powered features (critique, visualization, etc.)

### What Requires Internet (Limitations)
- ❌ Initial model downloads (handled in setup above)
- ❌ Mermaid diagram rendering library (loaded from CDN in browser)
  - Note: Once loaded in browser, Mermaid is typically cached and may work offline
  - For completely offline Mermaid, consider bundling the library locally

## Optimization Features for Intel HD 620/Laptops

The system includes specific optimizations for laptop CPUs and Intel HD 620 graphics:

1. **LLM Optimization**: Uses `llama3.2:1b` model which is optimized for iGPU acceleration
2. **CPU Thread Limits**: `torch.set_num_threads(4)` to balance load on laptop CPUs
3. **Efficient Models**:
   - STT: `tiny.en` Whisper model with `int8` quantization
   - TTS: Silero model optimized for CPU execution
4. **Context Management**: Limited conversation history (20 messages) to reduce memory usage
5. **Efficient Processing**: Background model loading to prevent startup blocking

## Troubleshooting Offline Issues

### "Model not found" Errors
If you see errors about missing models:
1. Ensure Ollama is running: `ollama serve`
2. Verify model is pulled: `ollama list` should show `llama3.2:1b`
3. For TTS issues, ensure the Silero model is cached

### Audio Issues
- Check microphone and speaker permissions
- Ensure LiveKit servers are running properly
- Verify audio drivers are working

### Performance Issues
- The system is already optimized for laptops
- If experiencing slowdowns, close other applications
- Consider reducing browser tabs if using the Streamlit interface

## Directory Structure for Offline Use
```
ReqGPT/
├── server.py              # Backend API (optimized for offline)
├── app.py                 # Frontend interface
├── offline_agent.py       # Voice agent (optimized for offline)
├── use_ollama.py          # Ollama utility (optimized for offline)
├── run_offline.bat        # Easy startup script
├── run_dina_voice.bat     # Voice assistant startup
├── sessions/              # Session storage (created automatically)
└── OFFLINE_USAGE.md       # This file
```

## Notes on Model Storage
Models are stored in:
- Ollama models: `~/.ollama/models/` (managed by Ollama)
- Silero TTS model: `~/.cache/torch/hub/snakers4_silero-models/` (managed by PyTorch Hub)
- Whisper model: `~/.cache/whisper/` (managed by faster-whisper)

These directories should be preserved for continued offline use.