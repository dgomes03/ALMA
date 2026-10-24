#!/usr/bin/env bash
# ============================================================
#  ALMA - open the settings menu (macOS and Linux)
#  Run this after ./setup.sh has finished.
# ============================================================
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] No virtual environment found."
    echo "  Run ./setup.sh first."
    exit 1
fi

exec .venv/bin/python -m core.config_menu
