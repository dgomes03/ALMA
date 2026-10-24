"""
Logging setup for the RAG Framework.

Call setup_logging() once at process start (main.py / server entrypoint), then
use get_logger(__name__) in any module. Only the 'ALMA' logger is set
to LOG_LEVEL; the root logger stays at WARNING so third-party libraries
(faiss, urllib3, sentence_transformers, ...) don't flood the output at INFO.
"""

import logging
import sys

from core.config import (
    ENABLE_LOGGING, LOG_LEVEL, LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT
)

APP_LOGGER_NAME = "ALMA"

_configured = False


def setup_logging(level=None):
    # Idempotent: safe to call from more than one entrypoint.
    global _configured
    if _configured:
        return logging.getLogger(APP_LOGGER_NAME)

    app_logger = logging.getLogger(APP_LOGGER_NAME)

    if not ENABLE_LOGGING:
        # NullHandler on root stops logging's lastResort handler from writing
        # library warnings to stderr, so the toggle really means silence.
        logging.getLogger().addHandler(logging.NullHandler())
        app_logger.disabled = True
        quiet_libraries()
        _configured = True
        return app_logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # stdout, not stderr, so log lines stay in order with the CLI's own output.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(console)

    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    app_logger.setLevel(level or LOG_LEVEL)

    _configured = True
    return app_logger


def quiet_libraries(load_transformers=True):
    """Silence third-party output that bypasses the root logger.

    torch and transformers attach their own handlers, Python warnings go
    straight to stderr, and transformers draws a "Loading weights" bar on every
    model load. Download progress bars stay on: a first run fetches several GB
    and should show it. load_transformers=False skips the transformers import
    (worker processes that only read documents).
    """
    import os
    import warnings

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings("ignore")
    for name in ("torch", "transformers", "sentence_transformers", "huggingface_hub",
                 "urllib3", "httpx", "unstructured", "pdfminer", "PIL", "faiss"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # torch resets the level of its "torch" logger when it is imported, so the
    # one macOS note it prints on every start is switched off by name.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True
    if not load_transformers:
        return
    try:
        from transformers.utils import logging as hf_logging
        from huggingface_hub.utils import enable_progress_bars
    except ImportError:
        return
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()   # also turns off huggingface_hub's bars...
    enable_progress_bars()              # ...so turn the download bars back on


def quiet_worker():
    """Pool initializer: worker processes start fresh, so repeat the quiet setup."""
    if not ENABLE_LOGGING:
        logging.getLogger().addHandler(logging.NullHandler())
        logging.getLogger(APP_LOGGER_NAME).disabled = True
        quiet_libraries(load_transformers=False)


def get_logger(name):
    # Entrypoints run as __main__, which sits outside the app logger
    # tree and would miss its level; re-anchor those under the app logger.
    if name == "__main__":
        name = f"{APP_LOGGER_NAME}.main"
    elif not name.startswith(APP_LOGGER_NAME):
        name = f"{APP_LOGGER_NAME}.{name.rsplit('.', 1)[-1]}"
    return logging.getLogger(name)
