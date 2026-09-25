FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OLLAMA_MODEL=llama3.2:1b \
    OLLAMA_BASE_MODEL=llama3.2:1b \
    WHISPER_MODEL=tiny.en \
    BACKEND_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies, curl, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libportaudio2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set work directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x start.sh

# Expose Streamlit port (7860 is default for Hugging Face Spaces, 8501 for standard)
EXPOSE 7860 8000

CMD ["./start.sh"]
