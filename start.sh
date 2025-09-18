#!/bin/sh
set -e

echo "Running core setup..."
python core.py

echo "Starting Uvicorn and Streamlit..."
# Run uvicorn in the background
uvicorn backend:app --host 0.0.0.0 --port 8000 &

# Run streamlit in the foreground (container tracks this process)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0