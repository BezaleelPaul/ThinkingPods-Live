#!/usr/bin/env bash

# =====================================================================
#  DESIGN THINKING PODS - macOS INSTALLATION & LAUNCH SCRIPT
# =====================================================================

# 1. Environment Setup
echo "=== Step 1: Checking Python Environment ==="
PYTHON_EXEC="python3"

if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "[ERROR] Python 3 is not installed on this system."
    echo "Please install Python 3 via Homebrew: brew install python"
    exit 1
fi

# Locate project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR" || exit

# 2. Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo " -> Creating virtual environment..."
    $PYTHON_EXEC -m venv venv
fi

echo " -> Activating virtual environment..."
source venv/bin/activate

echo " -> Installing dependencies from requirements.txt (this may take a moment)..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify Ollama Installation on macOS
echo -e "\n=== Step 2: Verifying Ollama Service ==="
if ! command -v ollama &> /dev/null; then
    echo "[WARNING] Ollama is not installed or not in your PATH."
    echo " -> You can install it using Homebrew: brew install --cask ollama"
    echo " -> Or download it directly from: https://ollama.com/download"
else
    # Try to start Ollama App if it is not running
    if ! pgrep -x "Ollama" > /dev/null && ! curl -s http://localhost:11434 &> /dev/null; then
        echo " -> Ollama service is not running. Attempting to start the desktop app..."
        open -a Ollama
        # Wait a few seconds for Ollama to spin up
        sleep 5
    fi
fi

# 4. Set Configurations
export WHISPER_MODEL="tiny.en"
export MENTOR_MODEL="qwen2.5:3b"

# 5. Clean Port 8000 (Prevents backend launch block)
echo -e "\n=== Step 3: Checking Port 8000 ==="
PORT_PID=$(lsof -ti:8000)
if [ -not -z "$PORT_PID" ]; then
    echo " -> Process $PORT_PID is using port 8000. Terminating..."
    kill -9 $PORT_PID 2>/dev/null
fi

# 6. Run Dynamic Auto-Tweaker
echo -e "\n=== Step 4: Running Dynamic Hardware Auto-Tweaker ==="
python autotweak.py

if [ $? -ne 0 ]; then
    echo "[WARNING] Auto-tweaker encountered an issue. Falling back to default model."
fi

# 7. Launch Servers
echo -e "\n=== Step 5: Launching Applications ==="
python server.py &
BACKEND_PID=$!

python -m streamlit run app.py &
FRONTEND_PID=$!

echo "====================================================================="
echo " Backend running at -> http://localhost:8000"
echo " Streamlit interface running at -> http://localhost:8501"
echo " Press Ctrl+C to shutdown both servers."
echo "====================================================================="

# Graceful cleanup on terminate
cleanup() {
    echo -e "\n -> Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
