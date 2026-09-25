#!/bin/bash
set -e

echo "=== Starting Ollama Service ==="
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama service to become responsive
echo "Waiting for Ollama to be ready..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready."

# Pull the lightweight model if not already present
MODEL="${OLLAMA_MODEL:-llama3.2:1b}"
echo "Ensuring model '$MODEL' is downloaded..."
ollama pull "$MODEL"

echo "=== Starting FastAPI Backend on Port 8000 ==="
python server.py &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend health check..."
until curl -s http://127.0.0.1:8000/health > /dev/null; do
    sleep 1
done
echo "Backend is ready."

echo "=== Starting Streamlit Frontend ==="
PORT="${STREAMLIT_SERVER_PORT:-7860}"
streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --browser.serverAddress 0.0.0.0
