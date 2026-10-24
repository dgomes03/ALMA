#!/usr/bin/env bash
# ============================================================
#  ALMA - one-time setup for macOS and Linux
#  Run once:  ./setup.sh        (add --benchmark for benchmark/evaluation extras)
#  Then start the app with:  ./run.sh
# ============================================================
set -e
cd "$(dirname "$0")"

REQS="requirements.txt"
[ "${1:-}" = "--benchmark" ] && REQS="requirements-benchmark.txt"

echo
echo "============================================================"
echo "  ALMA - setup"
echo "============================================================"
echo

# --- 1. Find Python 3.10-3.13 --------------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "[ERROR] Python 3.10 to 3.13 was not found."
    echo "  macOS:  brew install python@3.12"
    echo "  Linux:  sudo apt install python3.12 python3.12-venv   (or your distro's equivalent)"
    exit 1
fi
echo "Using: $("$PY" --version) ($(command -v "$PY"))"
echo

# --- 2. Virtual environment ----------------------------------
if [ -x ".venv/bin/python" ]; then
    echo "Virtual environment already exists (.venv) - reusing it."
else
    echo "Creating virtual environment in .venv ..."
    "$PY" -m venv .venv || {
        echo "[ERROR] Could not create the virtual environment."
        echo "  On Debian/Ubuntu: sudo apt install python3-venv"
        exit 1
    }
fi
echo

# --- 3. Packages ---------------------------------------------
echo "Upgrading pip ..."
.venv/bin/python -m pip install --upgrade pip
echo
echo "Installing packages from $REQS ..."
echo "(this can take several minutes and downloads ~2-3 GB)"
.venv/bin/python -m pip install -r "$REQS"
echo

# --- 4. NLTK data --------------------------------------------
echo "Downloading NLTK data ..."
.venv/bin/python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"
echo

mkdir -p documents model

# --- 5. Remaining manual steps -------------------------------
IS_APPLE_SILICON=0
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && IS_APPLE_SILICON=1

echo "============================================================"
echo "  Setup finished successfully."
echo "============================================================"
echo
echo "  STILL TO DO before the first run:"
echo
if [ $IS_APPLE_SILICON -eq 1 ]; then
    echo "  1) Nothing for the language model: on the first start ALMA asks which"
    echo "     Ministral 3 model to download into the model folder."
else
    echo "  1) Install Ollama and keep it running:  https://ollama.com/download"
    echo "     On the first start ALMA asks which Ministral 3 model to pull."
fi
echo
echo "  2) Put the files you want to search in the documents folder."
echo
echo "  3) (Optional) For OCR of scanned PDFs/images, install Tesseract"
echo "     with the English and Portuguese language data:"
echo "         macOS:  brew install tesseract tesseract-lang"
echo "         Linux:  sudo apt install tesseract-ocr tesseract-ocr-por"
echo "     Without it, text documents still work fine."
echo
if [ "$(uname -s)" = "Linux" ]; then
    echo "  4) (Optional) The web UI's folder picker needs tkinter:"
    echo "         sudo apt install python3-tk"
    echo
fi
echo "  When ready, run:  ./run.sh"
if [ "$(uname -s)" = "Darwin" ]; then
    echo "  To change settings later, double-click:  settings_mac.command"
else
    echo "  To change settings later:  ./settings.sh"
fi
echo
if [ -t 0 ]; then
    read -r -p "Open the settings menu now? [y/N] " OPEN_MENU
    case "$OPEN_MENU" in
        [yY]*) .venv/bin/python -m core.config_menu ;;
    esac
fi
