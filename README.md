# ALMA

ALMA is a modular agentic RAG framework built to run on consumer hardware. It combines sparse (BM25) and dense (FAISS) retrieval with optional cross-encoder reranking, handles European Portuguese and English, cites its sources, can search the web and can query SQL databases. Three backends are supported: MLX and Ollama, which keep everything on your own computer for privacy, and an API backend.

It runs on macOS, Windows and Linux:

| Platform | LLM backend |
|---|---|
| macOS (Apple Silicon) | MLX (default), Ollama or API  |
| macOS (Intel), Windows, Linux | Ollama (default) or API  |

The **API backend** uses a hosted model instead of a local one: OpenAI, Anthropic, Gemini, Mistral, OpenRouter, DeepSeek, or any OpenAI-compatible server such as LM Studio or vLLM. Your questions and the retrieved passages are then sent to that provider.

## Install

You need Python 3.10 to 3.13.

**macOS**: double-click `setup_mac.command`. The first time, macOS blocks scripts downloaded from the internet: right-click the file, choose **Open**, then **Open** again. Do the same for `run_mac.command`.

**Windows**: double-click `setup_windows.bat`.

**Linux** (or macOS from Terminal):

```bash
./setup.sh
```

The setup script creates a virtual environment in `.venv`, installs the packages (about 2-3 GB) and lists what is left to do:

1. **Language model.** On the first start ALMA looks in the `model` folder. If it's empty, it asks which Ministral 3 model (4-bit) to download.
   - **3B** (2.8 GB). For very simple, quick questions. Fast, but more likely to hallucinate. Works with 8 GB of graphics memory.
   - **8B** (5.6 GB). For questions of medium complexity. Handles more complex searches. Needs 16 GB of graphics memory.
   - **14B** (8.5 GB). The minimum recommended for an accurate RAG system with few hallucinations. Searches long documents and reasons over them. Needs 16 to 24 GB of graphics memory.

   Graphics memory means the memory of the graphics card (VRAM), not the computer's RAM. On Apple Silicon Macs the two are shared, so it is the Mac's total memory. On a PC without enough graphics memory, Ollama runs the model on the processor and RAM instead, which works but is much slower.

   On Apple Silicon the model runs with MLX and downloads into `model/`. To use another MLX model, put its folder in `model/` (or set `MODEL_PATH`). Elsewhere ALMA uses [Ollama](https://ollama.com/download): install it and keep it running, and ALMA pulls the chosen model into it. To use a hosted model instead, see [API backend](#api-backend).
2. **Documents.** Put the files you want to search in `documents/`. Supported: PDF, DOCX, PPTX, XLSX/XLS, HTML, Markdown, CSV, TXT and images (PNG, JPG, TIFF, BMP).
3. **OCR (optional).** To read scanned PDFs and images, install Tesseract with English and Portuguese data:
   - macOS: `brew install tesseract tesseract-lang`
   - Linux: `sudo apt install tesseract-ocr tesseract-ocr-por`
   - Windows: the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki); tick Portuguese under "Additional language data"
4. **Folder picker on Linux (optional).** The web UI's "select folder" button needs tkinter: `sudo apt install python3-tk`.

## Run

Double-click `run_mac.command` (macOS) or `run_windows.bat` (Windows). On Linux, or from a terminal:

```bash
./run.sh
```

ALMA opens a chat in the terminal. The first run downloads the models and builds the index from `documents/`, which can take a while for a large folder.

Type a question and press Enter. Commands start with `/` (type `/` to see them):

| Command | What it does |
|---|---|
| `/help` | List the commands |
| `/config` | Open the settings menu |
| `/clear` | Start a new conversation |
| `/save`, `/list`, `/load <id>`, `/delete <id>` | Save, list, reopen or delete conversations |
| `/exit` | Quit (or press Ctrl+C twice) |

Shift+Enter (or Ctrl+J) adds a new line, up and down recall earlier messages, Esc stops an answer.

To chat in the browser instead, turn on `ENABLE_WEB_INTERFACE` (settings menu, **Interface & logging**) and open http://127.0.0.1:5050 after starting ALMA.

## API backend

1. Open the settings menu (see [Settings](#settings)) and go to **Model & backend**.
2. Set `LLM_BACKEND` to `api` and pick `API_PROVIDER`.
3. Choose **Set API key** and paste your key. It is stored in `.env` in the project folder, which git ignores, and never in `core/config.py`.
4. Optionally set `API_MODEL`; left empty, the provider's default is used:

| `API_PROVIDER` | Default model | Key variable |
|---|---|---|
| `openai` | `gpt-5-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-3.8-flash` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `mistral` | `mistral-small-latest` | `MISTRAL_API_KEY` |
| `openrouter` | `openai/gpt-5-mini` | `OPENROUTER_API_KEY` |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `custom` | set `API_MODEL` | `API_KEY` (optional) |

For `custom`, also set `API_BASE_URL`, for example `http://localhost:1234/v1` for LM Studio. Without the menu, put the key in `.env` yourself (`OPENAI_API_KEY=sk-...`) and edit the `API_*` settings in `core/config.py`.

The model must support tool calling for document search to work. Saved chats are reopened from their text, since the KV-cache that MLX stores is not available over an API.

## Settings

All settings live in `core/config.py`. To change them without editing the file, open the settings menu:

- **macOS**: double-click `settings_mac.command`
- **Windows**: double-click `settings_windows.bat`
- **Linux**: `./settings.sh`

`./run.sh --config` (or `run_windows.bat --config`) opens the same menu and then offers to start ALMA. You can also type `config` in the terminal chat. The menu groups the settings into sections (model and backend, generation, documents and indexing, retrieval, research agent and web, interface and logging, SQL databases, benchmarks) and writes your changes back into `core/config.py`. Changes apply the next time ALMA starts. If you change an indexing setting, the menu offers to delete `Indexes/` so the index is rebuilt.

The settings you are most likely to change:

| Setting | What it does |
|---|---|
| `LLM_BACKEND` | `'mlx'`, `'ollama'` or `'api'`. Defaults to MLX on Apple Silicon, Ollama elsewhere |
| `API_PROVIDER`, `API_MODEL` | Hosted provider and model for the API backend |
| `MODEL_PATH` | MLX model. Empty means the model in the `model` folder; or set a folder path or a Hugging Face repo id |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | Ollama model name and server address |
| `ENABLE_WEB_INTERFACE` | Chat in the browser instead of the terminal |
| `WEB_INTERFACE_HOST`, `WEB_INTERFACE_PORT` | Web interface address. `0.0.0.0` makes it reachable from your network, with no login |
| `PROMPT_LANGUAGE` | `'pt'` or `'en'`: the language of every prompt sent to the model. Answers still follow the language of the question |
| `RERANK` | Turn cross-encoder reranking on or off |
| `MAX_TOOL_ROUNDS` | Rounds of tool calls (searches, document reads) per answer before the model must reply. Default 5 |
| `MAX_DOCUMENT_CHARS` | Longest document opened in full. Longer ones are opened as an overview of their opening, key sections and samples. Default 16,000 characters |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking, in characters. Changing these needs an index rebuild |
| `ADVANCED_RESEARCH` | Lets the LLM call the multi-step research pipeline as a tool. Off by default (still in development) |
| `ENABLE_SQL_DATABASES`, `SQL_DATABASE_CONFIGS` | SQL databases the LLM can query. SQLite works out of the box; PostgreSQL needs `psycopg2-binary`, MySQL needs `mysql-connector-python`. Add databases in `core/config.py` itself |

To index new or changed documents, delete `Indexes/` and restart, or set `CHECK_NEW_DOCUMENTS_ON_START = True`.

## Example databases

`data/` holds two SQLite databases that are wired up in `SQL_DATABASE_CONFIGS`:

- `demo_data.db`: products, customers and orders
- `enterprise_data.db`: a synthetic retailer with 16 tables and two years of orders

## Benchmarks and evaluation

Install the extra packages first:

```bash
./setup.sh --benchmark                  # macOS / Linux
setup_windows.bat --benchmark           # Windows, from a terminal
```

The **Benchmarks** section of the settings menu sets the dataset, methods, cut-offs and query limits, shows which datasets are downloaded, and can start each run. From the command line, run from the repository root with the virtual environment's Python; flags override the settings:

```bash
# Retrieval on BEIR datasets (downloaded to benchmark/datasets/ on first use)
python -m benchmark.run_benchmark
python -m benchmark.run_benchmark --dataset nfcorpus --max-queries 50
python -m benchmark.plot_results          # figures -> benchmark/figures/

# RRF parameter sweep
python -m benchmark.sweep_rrf

# RAGAS generation quality
python -m evaluation.run_evaluation --mode benchmark
python -m evaluation.run_evaluation --mode reference-free --queries "Q1?" "Q2?"

# Unit tests
python -m pytest tests
```

See `benchmark/README.md` for the datasets and methods. The local RAGAS judge (`EVAL_RAGAS_JUDGE = 'mlx'`) needs Apple Silicon, so other platforms default to Claude, which needs `ANTHROPIC_API_KEY`.

## About

ALMA was developed as part of the master's thesis *ALMA: An Efficient and Modular Framework for Advanced Agentic Retrieval-Augmented Generation* by Diogo Henrique Martins Gomes (Master in Data Science in Agriculture, Food, Forest and Environment, Instituto Superior de Agronomia, University of Lisbon, 2026), supervised by Manuel Campagnolo. A link to the thesis will be added once it is published.

To cite ALMA:

```bibtex
@mastersthesis{gomes2026alma,
  author = {Gomes, Diogo Henrique Martins},
  title  = {{ALMA}: An Efficient and Modular Framework for Advanced Agentic Retrieval-Augmented Generation},
  school = {Instituto Superior de Agronomia, University of Lisbon},
  year   = {2026}
}
```

## Acknowledgements and Disclaimers

The terminal chat interface adapts code from [Mistral Vibe](https://github.com/mistralai/mistral-vibe) v2.26.1 (Copyright 2025 Mistral AI, Apache License 2.0) and runs on [Textual](https://github.com/Textualize/textual) (MIT License). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what was adapted and how.

The advanced research mode is still in development, so it is disabled by default. To try it, turn on `ADVANCED_RESEARCH` in the settings menu, under **Research agent & web**. When it is on, the model decides by itself when to call the research pipeline, so it is not guaranteed to use it. To make that more likely, ask for it in your message, for example "use advanced research to compare these reports".

Parts of the code were written with AI assistance because of the project's short academic timeframe. They will be reviewed and refactored by hand in the coming months.

## License

MIT, see [LICENSE](LICENSE). The files adapted from Mistral Vibe (`core/tui/`) remain under the Apache License 2.0, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).