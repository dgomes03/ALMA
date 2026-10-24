import os
import platform
import sys

# Set environment variables for single-threaded performance
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["PYTORCH_NUM_THREADS"] = num_threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Project Root ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === API keys ===
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


def _load_env_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                os.environ.setdefault(key, value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env_file(ENV_FILE)

# === Constants ===
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")   # the files ALMA indexes and searches

# === CLI ===
SHOW_CLI_SPINNER = True           # Animated indicator while the LLM prefills the prompt

# === Prompt language ===
# Language of every prompt sent to the model (core/prompts.py): 'pt' (European
# Portuguese) or 'en'. Answers still follow the language of the question.
PROMPT_LANGUAGE = 'pt'

# === LLM Backend Selection ===
# 'mlx'    — Apple Silicon via mlx-lm. Loads the model at MODEL_PATH.
#            Supports persistent KV-cache (fast multi-turn, cached chats on disk).
# 'ollama' — any GGUF model served by Ollama (macOS/Linux/Windows).
#            Uses OLLAMA_MODEL / OLLAMA_HOST below. Ollama manages its own KV-cache
#            server-side, so the on-disk prompt-cache feature is disabled.
# 'api'    — a hosted model: OpenAI, Anthropic, Gemini, Mistral, OpenRouter, DeepSeek,
#            or any OpenAI-compatible server. Uses API_PROVIDER / API_MODEL below and
#            the provider's key from .env. Your questions and the retrieved passages
#            are sent to that provider, so documents no longer stay on this machine.
# Default: mlx on Apple Silicon, ollama everywhere else. Set any of them explicitly to override.
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
LLM_BACKEND = 'mlx' if IS_APPLE_SILICON else 'ollama'
# MLX model. None = the model in the `model` folder (on first start ALMA offers
# to download one). Or a folder path, or a Hugging Face repo id.
MODEL_PATH = None
OLLAMA_MODEL = 'hf.co/lmstudio-community/Ministral-3-14B-Instruct-2512-GGUF:Q4_K_M'
OLLAMA_HOST = 'http://localhost:11434'
OLLAMA_NUM_CTX = 16384                 # context window; Ollama's default (4096) truncates RAG prompts
OLLAMA_KEEP_ALIVE = '30m'              # keep the model loaded between requests
API_PROVIDER = 'openai'                # openai | anthropic | gemini | mistral | openrouter | deepseek | custom
API_MODEL = None                       # None = the provider's default (see API_PROVIDERS in core/llm_backend.py)
API_BASE_URL = None                    # None = the provider's standard address; required for 'custom',
                                       # e.g. 'http://localhost:1234/v1' for LM Studio
API_TIMEOUT = 120                      # seconds to wait for the provider to start answering
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
MULTIVECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "FAISS_index.pkl")
BM25_DATA_PATH = os.path.join(PROJECT_ROOT, "Indexes", "bm25s_index")  # directory (bm25s native format)
METADATA_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "metadata_index.pkl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "faiss_index.faiss")
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MAX_RESPONSE_TOKENS = 5000
MAX_TOOL_ROUNDS = 5          # Tool-calling rounds per answer; after that the model must answer

# === Chunking Configuration ===
CHUNK_SIZE = 800     # CHARACTERS, not tokens. Smaller chunks fragment tables;
                     # larger ones dilute pinpoint facts. Worth re-testing on
                     # your own corpus.
CHUNK_OVERLAP = 200  # 25% overlap to preserve boundary information
MIN_CHUNK_CHARS = 20  # Minimum chunk length; filters junk OCR/table fragments
PARENT_CHUNK_SIZE = 2048  # Parent chunks for hierarchical indexing

# --- PDF structure handling ---
# Both are INDEX-TIME switches: changing either requires deleting Indexes/.
PDF_TEXT_BACKEND = 'pymupdf'   # 'pymupdf' | 'unstructured'.
                               # unstructured's strategy="auto" mis-orders text
                               # on multi-column / table-heavy PDFs (a table row
                               # can be paired with the previous row's value) and
                               # can swallow numbered headings into sentences.
                               # PyMuPDF keeps the layout reading order and is
                               # much faster. Scanned PDFs automatically fall
                               # back to unstructured.
PDF_CROSS_PAGE_CHUNKS = True   # Let a chunk span a PDF page break.
                               # False = legacy per-page chunking, where a table
                               # row ending page N and its value starting page
                               # N+1 can never share a chunk.
PDF_SECTION_PATH = True        # Track the heading stack and prepend the
                               # ancestor path ("5 Methods > 5.3 Sampling ...") to
                               # the EMBEDDED chunk text, not just metadata.
PDF_SECTION_REQUIRE_NUMBER = True  # Only accept numbered headings ("5.3 Sampling ...").
                               # unstructured labels wrapped body lines as Title on
                               # PDFs; the section number is what separates them.
PDF_SECTION_MAX_CHARS = 120    # A Title candidate longer than this, or ending in
                               # a full stop, is treated as body text
                               # (unstructured over-emits Title on PDFs).

# --- Front matter ---
# Cover pages, title pages and tables of contents crowd out the body text: a
# question about a long report can end up answered from its cover and contents.
FRONT_MATTER_MAX_PAGE = 2      # only pages 0..N can be front matter (0-based)
FRONT_MATTER_MIN_TOKENS = 100  # a short page early in a PDF is front matter
FRONT_MATTER_UPPER_RATIO = 0.30  # ...so is a heavily upper-case one
FRONT_MATTER_PENALTY = 0.7     # RRF score multiplier; 1.0 or None disables.
                               # A demotion, not a filter: early pages sometimes
                               # do hold the answer.

HIERARCHICAL_INDEXING = False  # Toggle: True = hierarchical, False = standard flat chunking; False is recommended for most use cases due to increased focus on relevant chunks and reduced noise. Hierarchical indexing is more complex and may not always yield better results.
PARENT_STORE_PATH = os.path.join(PROJECT_ROOT, "Indexes", "parent_store.pkl")

# === BM25 Configuration ===
BM25_ENABLE_STEMMING = True
BM25_ENABLE_STOPWORDS = True
BM25_LANGUAGES = ['english', 'portuguese']  # Languages for stemmer and stop words

# === Retrieval Configuration ===
RERANK = True # toggle reranking on or off; in some cases reranking can decrease retrieval quality
RETRIEVAL_TOP_K = 100       # Candidates from FAISS + BM25, and the RRF pool.
                            # Fusion is cheap (~35 ms even at depth 500) and the
                            # pool sets the recall ceiling; the reranker cost is
                            # bounded separately by RERANK_CANDIDATES.
RERANKER_TOP_N = 10         # Number of results kept after reranking
RRF_K = 60                  # RRF constant (higher = less rank-sensitive; 60 is standard)
RERANK_CANDIDATES = 50      # How many fused candidates the cross-encoder scores.
                            # None = all of them (legacy: same number as
                            # RETRIEVAL_TOP_K). Decoupling these two lets the
                            # candidate pool grow -- which sets the recall
                            # ceiling and costs ~20 ms -- without paying the
                            # cross-encoder, which costs ~1-3 s.
PARENT_TOP_N = 10            # Max parent chunks returned after dedup (hierarchical mode only)
RERANKER_CONFIDENCE_THRESHOLD = 0.3  # None = disabled; float in [0,1] to drop low-relevance chunks. Applied via sigmoid on raw reranker scores.
                            # 0.5 discarded too much context with a small
                            # cross-encoder on cross-lingual pairs.
MAX_DOCUMENT_CHARS = 16000   # Longest document the open-document tool returns in full (~4k tokens);
                             # longer ones get an overview: opening, key sections and samples
QUERY_EXPANSION_MIN_WORDS = 5  # Queries with fewer words skip LLM expansion (set to 0 to always expand)

# === FAISS Configuration ===
FAISS_INDEX_TYPE = 'flat'  # Options: 'flat', 'ivf', 'hnsw', 'auto'
                           # 'auto' switches to IVF above 10k vectors, which
                           # searches only part of the clusters and costs recall.
                           # Exact search over tens of thousands of chunks takes
                           # milliseconds, so 'flat' is the safe default.
FAISS_IVF_NPROBE_RATIO = 4  # nprobe = nlist // this ratio

# === Embedding Configuration ===
EMBEDDING_USE_PREFIX = True  # Add E5 instruction prefixes (query:/passage:)


# === Document Watcher Configuration ===
CHECK_NEW_DOCUMENTS_ON_START = False  # Check for new/modified documents at startup

# === Web Interface ===
ENABLE_WEB_INTERFACE = False      # True = chat in the browser, False = chat in the terminal
WEB_INTERFACE_HOST = '127.0.0.1'  # '0.0.0.0' exposes it to your whole network (no login)
WEB_INTERFACE_PORT = 5050

# === Chat Settings ===
# prints the full chat template on every turn
DEBUG_MESSAGES = False
# === Logging ===
ENABLE_LOGGING = False             # False silences all log output (app + libraries)
LOG_LEVEL = "INFO"                # DEBUG | INFO | WARNING | ERROR
LOG_FILE = None                   # Set to a path to also write logs to disk
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# === Chat Persistence ===
# Shared by CLI and server. Holds <chat_id>.json + <chat_id>.safetensors per chat.
CHATS_DIR = os.path.join(PROJECT_ROOT, "server", "chats")

# === User-built Index ===
# Where indexes built from a custom folder via the web UI are stored.
USER_INDEX_DIR = os.path.join(PROJECT_ROOT, "User_indexes")
USER_INDEX_PATHS = {
    "faiss_index_path":    os.path.join(USER_INDEX_DIR, "FAISS_index.pkl"),
    "bm25_data_path":      os.path.join(USER_INDEX_DIR, "bm25s_index"),
    "metadata_index_path": os.path.join(USER_INDEX_DIR, "metadata_index.pkl"),
    "faiss_binary_path":   os.path.join(USER_INDEX_DIR, "faiss_index.faiss"),
    "parent_store_path":   os.path.join(USER_INDEX_DIR, "parent_store.pkl"),
}

# Web search controls
MAX_FETCH_URL_CHARS = 5000       # Max characters kept from each fetched URL
DUCKDUCKGO_RESULTS_COUNT = 6      # Results per DDG search call

# === Research Configuration ===
ADVANCED_RESEARCH = False # still in development, off by default. True exposes the research pipeline as a tool the LLM can call; queries use the standard pipeline either way.
MAX_RESEARCH_STEPS = 5
RESEARCH_SATURATION_ROUNDS = 2   # Stop the research loop after this many
                                 # consecutive iterations that surface no new
                                 # (document, page). None disables.
MAX_RESEARCH_GOALS = 4           # Max sub-goals the planner can create

# === Advanced Research Tool Selection ===
AGENTIC_USE_DOCUMENT_SEARCH = True   # search_documents in advanced mode
AGENTIC_USE_WEB_SEARCH = True        # duckduckgo_search + fetch_url_content in advanced mode
AGENTIC_USE_DOC_RETRIEVAL = True     # list_available_documents + retrieve_document_by_name in advanced mode

# Web search loop controls
MAX_URLS_PER_SEARCH_LOOP = 3      # Max URLs fetched per search loop

MAX_EVAL_CHARS_PER_SOURCE = 800  # Max characters per source sent to the evaluator
MAX_SEARCH_LOOPS = 1              # Max search+fetch loops per goal (TODO: values above 1 tend to repeat the same search)

# Replanning limits
MAX_REPLANS_PER_GOAL = 3          # Per-goal: after N failed replans → cancel task
MAX_TOTAL_REPLANS = 5             # Total replans allowed across all goals


# === Language Reasoning Model (LRM) Configuration ===
REASONING_MODEL = False

# === SQL Database Configuration ===
ENABLE_SQL_DATABASES = True
SQL_DATABASE_CONFIGS = {
    "demo_db": {
        "db_type": "sqlite",
        "connection_string": os.path.join(PROJECT_ROOT, "data", "demo_data.db"),
        "description": "Demo database with products, customers, and orders for testing SQL queries",
        "max_rows": 100,
        "timeout": 30,
        "allowed_tables": ["products", "customers", "orders"],
    },
    "enterprise_db": {
        "db_type": "sqlite",
        "connection_string": os.path.join(PROJECT_ROOT, "data", "enterprise_data.db"),
        "description": (
            "NovaTech Retail enterprise database: multinational electronics/office retailer. "
            "Tables: departments, employees (manager hierarchy), suppliers, categories "
            "(parent/child tree), products, warehouses, inventory (per-warehouse stock), "
            "customers (consumer/business/enterprise segments), marketing_campaigns, orders, "
            "order_items, payments, shipments, returns, product_reviews, support_tickets. "
            "Views: v_order_totals, v_product_performance. Two years of order history."
        ),
        "max_rows": 200,
        "timeout": 30,
        "allowed_tables": [
            "departments", "employees", "suppliers", "categories", "products",
            "warehouses", "inventory", "customers", "marketing_campaigns", "orders",
            "order_items", "payments", "shipments", "returns", "product_reviews",
            "support_tickets", "v_order_totals", "v_product_performance",
        ],
    },
}

# === LLM Generation Sampling Configuration ===
# These are used by both the main runtime (standard.py) and benchmark evaluation.
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.0 # reduce from 1 to penalize
LLM_REPETITION_CONTEXT_SIZE = 128

# === Retrieval Benchmark (BEIR) ===
# Defaults for `python -m benchmark.run_benchmark` and `benchmark.sweep_rrf`;
# command-line flags override them. Datasets download to benchmark/datasets/.
BENCHMARK_DATASET = 'scifact'
BENCHMARK_METHODS = ['bm25', 'dense', 'hybrid', 'hybrid_rerank']
BENCHMARK_K_VALUES = [1, 3, 5, 10, 100]
BENCHMARK_MAX_QUERIES = None   # None = all queries; e.g. 50 for a quick run
BENCHMARK_MAX_CORPUS = None    # None = full corpus (subsampled automatically when
                               # BENCHMARK_MAX_QUERIES is set); 0 forces the full corpus

# === RAGAS Evaluation Configuration ===
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")
EVAL_DATASET = 'amnesty_qa'    # default for --mode benchmark: amnesty_qa | hotpotqa | scifact | ms_marco
EVAL_RAGAS_JUDGE = 'mlx' if IS_APPLE_SILICON else 'claude'  # 'mlx' (local) | 'claude' (needs ANTHROPIC_API_KEY)
EVAL_BENCHMARK_DATASET = "explodinggradients/amnesty_qa"
EVAL_BENCHMARK_DATASET_SUBSET = "english_v3"
EVAL_MAX_QUESTIONS = None  # None = all, or int to limit
EVAL_LLM_MAX_TOKENS = 2048