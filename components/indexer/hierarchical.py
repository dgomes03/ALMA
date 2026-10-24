from core.config import PARENT_CHUNK_SIZE, PARENT_STORE_PATH
from core.utils import save_pickle, load_pickle
from components.indexer.standard import Indexer
from core.logging_config import get_logger

logger = get_logger(__name__)


class HierarchicalIndexer(Indexer):
    """
    Hierarchical chunking indexer: indexes small child chunks (identical to
    flat mode) for precise retrieval, but groups them into larger parent
    chunks for richer LLM context.

    Key design: children are chunked at CHUNK_SIZE exactly like standard mode,
    then grouped into parents. This guarantees identical retrieval quality
    while providing more context to the LLM.
    """

    def __init__(self, parent_store_path=PARENT_STORE_PATH, **kwargs):
        super().__init__(**kwargs)
        self.parent_store_path = parent_store_path

    @staticmethod
    def build_hierarchical_chunks(child_chunks, child_metadata,
                                  parent_chunk_size=PARENT_CHUNK_SIZE,
                                  parent_idx_offset=0):
        """
        Group consecutive same-page child chunks into larger parent chunks.

        Children are kept as-is (identical to flat mode). Parents are formed
        by joining consecutive children from the same page/document until
        parent_chunk_size is exceeded.

        Mutates child_metadata in-place to add parent_idx and child_idx.

        Args:
            child_chunks: List of child-sized text chunks (from flat chunking)
            child_metadata: List of metadata dicts for each child
            parent_chunk_size: Max character size for parent chunks
            parent_idx_offset: Offset for parent_idx (for incremental indexing)

        Returns:
            (child_chunks, child_metadata, parent_store)
        """
        if not child_chunks:
            return [], [], []

        parent_store = []
        group_start = 0
        group_length = 0
        prev_page_key = None

        def flush_group(start, end):
            parent_text = " ".join(child_chunks[start:end])
            parent_meta = child_metadata[start].copy()
            parent_meta.pop('parent_idx', None)
            parent_meta.pop('child_idx', None)
            parent_store.append({"text": parent_text, "metadata": parent_meta})
            parent_idx = parent_idx_offset + len(parent_store) - 1
            for pos, idx in enumerate(range(start, end)):
                child_metadata[idx]['parent_idx'] = parent_idx
                child_metadata[idx]['child_idx'] = pos

        for i, (c_text, c_meta) in enumerate(zip(child_chunks, child_metadata)):
            page_key = (c_meta.get('filename'), c_meta.get('page', 0))

            # Start new parent group on page change or size overflow
            if i > group_start and (
                page_key != prev_page_key or
                group_length + len(c_text) > parent_chunk_size
            ):
                flush_group(group_start, i)
                group_start = i
                group_length = 0

            prev_page_key = page_key
            group_length += len(c_text)

        # Flush last group
        if group_start < len(child_chunks):
            flush_group(group_start, len(child_chunks))

        return child_chunks, child_metadata, parent_store

    def save_parent_store(self, parent_store):
        """Save parent store to disk."""
        save_pickle(parent_store, self.parent_store_path)

    def load_parent_store(self):
        """Load parent store from disk. Returns None if file missing."""
        return load_pickle(self.parent_store_path)

    def load_and_chunk_documents_parallel(self, max_chunk_length=None, chunk_overlap=None, progress_callback=None):
        """
        Load documents, chunk at child size (identical to flat mode),
        then group children into parents.

        Returns:
            (child_chunks, child_metadata, parent_store) — 3-tuple
        """
        # Get flat chunks identical to standard mode (CHUNK_SIZE)
        child_chunks, child_metadata = super().load_and_chunk_documents_parallel(progress_callback=progress_callback)

        if not child_chunks:
            return [], [], []

        # Group flat chunks into parents
        child_chunks, child_metadata, parent_store = self.build_hierarchical_chunks(
            child_chunks, child_metadata
        )
        logger.info(f"Grouped {len(child_chunks)} child chunks into {len(parent_store)} parents")

        return child_chunks, child_metadata, parent_store

    def save_indices(self, multi_vector_index, bm25, metadata_index, faiss_index=None,
                     parent_store=None):
        """Save all indices plus parent store."""
        super().save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        if parent_store is not None:
            self.save_parent_store(parent_store)

    def load_indices(self):
        """Load all indices plus parent store.

        Returns:
            (multi_vector_index, bm25, metadata_index, faiss_index, parent_store) — 5-tuple
        """
        mvi, bm25, metadata_index, faiss_index = super().load_indices()
        parent_store = self.load_parent_store()
        return mvi, bm25, metadata_index, faiss_index, parent_store

    def index_new_documents(self, multi_vector_index, bm25, metadata_index,
                            faiss_index, embedding_model, parent_store=None):
        """
        Detect and index new documents, handling hierarchical parent/child chunks.

        Returns:
            (mvi, bm25, metadata_index, faiss_index, new_filenames, parent_store)
            or None if no new documents found.
        """
        processed = self._find_and_process_new_documents(metadata_index)
        if processed is None:
            return None
        new_chunks, new_metadata, new_filenames = processed

        # Group flat chunks into parents with offset
        if parent_store is None:
            parent_store = []
        new_chunks, new_metadata, new_parent_store = self.build_hierarchical_chunks(
            new_chunks, new_metadata, parent_idx_offset=len(parent_store)
        )
        logger.info(f"Extracted {len(new_chunks)} child chunks grouped into {len(new_parent_store)} parents")
        parent_store.extend(new_parent_store)

        bm25 = self._merge_new_chunks(
            multi_vector_index, faiss_index, metadata_index,
            new_chunks, new_metadata, embedding_model)

        self.save_indices(multi_vector_index, bm25, metadata_index, faiss_index,
                          parent_store=parent_store)
        logger.info(f"Indices updated and saved. Total child chunks: {len(multi_vector_index)}")

        return multi_vector_index, bm25, metadata_index, faiss_index, new_filenames, parent_store
