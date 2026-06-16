#!/usr/bin/env bash
python server.py &
python -m streamlit run app.py &
echo "Both servers launched. Press Ctrl+C to stop."
wait
