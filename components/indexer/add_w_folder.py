"""End-to-end indexing pipeline. Used by the web UI's "build index from folder" flow.

The pipeline emits progress through a callback so the WebSocket layer can relay it
to the browser without owning the indexing logic itself.
"""
import gc
import os

from core.config import EMBEDDING_MODEL_NAME, HIERARCHICAL_INDEXING, USER_INDEX_DIR
from components.indexer import Indexer, HierarchicalIndexer


def build_index_from_folder(folder_path: str, output_paths: dict, progress_callback=None):
    """Build a complete index (chunks → embeddings → FAISS/BM25/metadata) from `folder_path`.

    `progress_callback(stage, percent, message)` is fired at every major step. Stages:
    'loading' (0-40), 'embedding' (40-75), 'building' (75-90), 'saving' (90-100), 'done' (100).
    Returns the number of chunks built. Raises on failure.
    """
    def emit(stage, percent, message):
        if progress_callback:
            progress_callback(stage, percent, message)

    os.makedirs(USER_INDEX_DIR, exist_ok=True)

    if HIERARCHICAL_INDEXING:
        indexer = HierarchicalIndexer(documents_dir=folder_path, **output_paths)
    else:
        flat_paths = {k: v for k, v in output_paths.items() if k != 'parent_store_path'}
        indexer = Indexer(documents_dir=folder_path, **flat_paths)

    # Loading (0-40%)
    def loading_cb(stage, current, total):
        pct = int(current / total * 40) if total > 0 else 0
        emit('loading', pct, f'Loading documents... ({current}/{total})')

    emit('loading', 0, 'Loading and chunking documents...')

    if HIERARCHICAL_INDEXING:
        chunks, chunk_metadata, parent_store = indexer.load_and_chunk_documents_parallel(
            progress_callback=loading_cb
        )
    else:
        chunks, chunk_metadata = indexer.load_and_chunk_documents_parallel(
            progress_callback=loading_cb
        )
        parent_store = None

    if not chunks:
        raise RuntimeError('No documents found or no content extracted.')

    emit('loading', 40, f'Loaded {len(chunks)} chunks from documents.')

    # Embeddings (40-75%)
    emit('embedding', 40, 'Loading embedding model...')
    from core.utils import load_sentence_transformer
    embedding_model = load_sentence_transformer(EMBEDDING_MODEL_NAME)

    def embedding_cb(stage, current, total):
        pct = 40 + int(current / total * 35) if total > 0 else 40
        emit('embedding', pct, f'Generating embeddings... ({current}/{total} chunks)')

    chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model, progress_callback=embedding_cb)

    # Index building (75-90%)
    emit('building', 75, 'Building indexes...')
    multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)
    bm25 = indexer.build_bm25_index(chunks)
    faiss_index = indexer.build_faiss_index(multi_vector_index)
    metadata_index = indexer.build_metadata_index(chunk_metadata)

    # Saving (90-100%)
    emit('saving', 90, 'Saving indexes to disk...')
    if HIERARCHICAL_INDEXING:
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index, parent_store=parent_store)
    else:
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)

    del embedding_model
    gc.collect()

    emit('done', 100, 'Indexing complete!')
    return len(chunks)
