#!/usr/bin/env bash
# ============================================================
#  ALMA - start the app on macOS and Linux
#  Run this after ./setup.sh has finished.
#  Settings menu first:  ./run.sh --config
# ============================================================
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] No virtual environment found."
    echo "  Run ./setup.sh first."
    exit 1
fi

echo "Starting ALMA ..."
echo "  (first run downloads the embedding + reranker models from"
echo "   Hugging Face - this needs internet and takes a few minutes)"
echo

exec .venv/bin/python main.py "$@"
