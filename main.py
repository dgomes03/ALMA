import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__" and "--config" in sys.argv[1:]:
    import subprocess
    from core.config_menu import run_menu, confirm
    run_menu()
    if confirm("\nStart ALMA now?", default=True):
        rest = [a for a in sys.argv[1:] if a != "--config"]
        sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__), *rest]))
    sys.exit(0)

import gc
import torch
import faiss
import sklearn

from core.config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, LLM_BACKEND,
    ENABLE_WEB_INTERFACE, WEB_INTERFACE_HOST, WEB_INTERFACE_PORT, num_threads,
    ADVANCED_RESEARCH, REASONING_MODEL, ENABLE_SQL_DATABASES, SQL_DATABASE_CONFIGS,
    CHECK_NEW_DOCUMENTS_ON_START, HIERARCHICAL_INDEXING, DOCUMENTS_DIR
)

torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

from components.indexer import Indexer, HierarchicalIndexer, start_document_check
from components.retrievers import Retriever
from components.generators import Generator
from core.chat_session import ChatSession
from core import cancel
from core.logging_config import setup_logging, get_logger

if ENABLE_WEB_INTERFACE:
    from server import run_server

logger = get_logger(__name__)


def build_indices_if_needed():
    logger.info("Checking for saved indices...")
    if HIERARCHICAL_INDEXING:
        indexer = HierarchicalIndexer(enable_ocr=True)
    else:
        indexer = Indexer(enable_ocr=True)
    indices = indexer.load_indices()
    if indices[0] is not None and indices[1] is not None:
        logger.info("Saved indices found.")
        return

    logger.info("No saved indices found. Loading and chunking documents...")
    if HIERARCHICAL_INDEXING:
        chunks, chunk_metadata, parent_store = indexer.load_and_chunk_documents_parallel()
    else:
        chunks, chunk_metadata = indexer.load_and_chunk_documents_parallel()
        parent_store = None

    if not chunks:
        alternative = ", or pick a folder with the web interface's folder button" if ENABLE_WEB_INTERFACE else ""
        print(f"\nNo documents could be read from {DOCUMENTS_DIR}.\n"
              f"ALMA will start without document search. To add documents, put "
              f"PDF, Word, PowerPoint, Excel, HTML, Markdown, CSV, text or image files "
              f"in the documents folder and restart{alternative}.\n")
        return

    from core.utils import load_sentence_transformer
    embedding_model = load_sentence_transformer(EMBEDDING_MODEL_NAME)

    chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
    multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)
    bm25 = indexer.build_bm25_index(chunks)
    faiss_index = indexer.build_faiss_index(multi_vector_index)
    metadata_index = indexer.build_metadata_index(chunk_metadata)
    
    if HIERARCHICAL_INDEXING:
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index, parent_store=parent_store)
    else:
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
    logger.info("Indices built and saved successfully.")
    n_docs = len({meta.get("document_name") for meta in chunk_metadata})
    print(f"Indexed {n_docs} document{'s' if n_docs != 1 else ''} ({len(chunks)} passages).")

    del embedding_model
    gc.collect()


def init_sql_databases():
    if not (ENABLE_SQL_DATABASES and SQL_DATABASE_CONFIGS):
        return
    logger.info("Initializing SQL databases...")
    from tools.SQL_database import initialize_sql_connector

    connector = initialize_sql_connector(SQL_DATABASE_CONFIGS)
    if connector.configs:
        logger.info("SQL connector initialized with %d database(s)", len(connector.configs))


def run_query(query, retriever, stream_callback=None):
    session = retriever.session
    if REASONING_MODEL and not ADVANCED_RESEARCH:
        from components.generators.LRM import LRMGenerator
        generator_cls = LRMGenerator
    else:
        generator_cls = Generator
    return cancel.run_cancellable(
        session,
        generator_cls.answer_query_with_llm,
        query,
        retriever.llm_model,
        retriever.llm_tokenizer,
        retriever,
        session.prompt_cache,
        stream_callback=stream_callback,
        conversation_manager=session.conversation_manager,
    )


if __name__ == "__main__":
    setup_logging()
    build_indices_if_needed()

    logger.info("Loading LLM (%s backend)...", LLM_BACKEND)
    print("Loading the language model...")
    from core.llm_backend import load_llm
    llm_model, llm_tokenizer = load_llm()

    logger.info("Initializing Retriever...")
    retriever = Retriever(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        reranker_model_name=RERANKER_MODEL_NAME,
    )
    retriever.session = ChatSession(llm_model, llm_tokenizer, reasoning_model=REASONING_MODEL)

    if CHECK_NEW_DOCUMENTS_ON_START:
        start_document_check(retriever)

    init_sql_databases()

    if ENABLE_WEB_INTERFACE:
        logger.info("Starting the web interface on http://%s:%s", WEB_INTERFACE_HOST, WEB_INTERFACE_PORT)
        run_server(retriever, host=WEB_INTERFACE_HOST, port=WEB_INTERFACE_PORT)
    else:
        from core.chat_ui import run_chat
        run_chat(retriever, run_query)
