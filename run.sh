#!/usr/bin/env bash
export OLLAMA_BASE_MODEL="qwen2.5:3b"
export MENTOR_MODEL="optimized-pods"
export LLM_NUM_CTX="1024"
export REPLY_CACHE_ANCHOR="false"
export FAST_SHORT_REPLIES="false"
export DYNAMIC_SHORT_PHRASING="false"
export COMPLEXITY_GATE="false"
export WHISPER_MODEL="tiny.en"
export FAST_SHORT_REPLIES_MAX_WORDS="30"
export MENTOR_USE_LLM_EXTRACTION="false"
export SENTIMENT_ENABLED="true"
export CLASSIFIER_ENABLED="true"

python autotweak.py
python server.py &
python -m streamlit run app.py &
echo "Both servers launched. Press Ctrl+C to stop."
wait
