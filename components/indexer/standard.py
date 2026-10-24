import os
import numpy as np
import faiss
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import bm25s

from core.config import (
    DOCUMENTS_DIR, EMBEDDING_MODEL_NAME,
    MULTIVECTOR_INDEX_PATH, BM25_DATA_PATH,
    METADATA_INDEX_PATH, FAISS_INDEX_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS,
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    FAISS_INDEX_TYPE, FAISS_IVF_NPROBE_RATIO,
    EMBEDDING_USE_PREFIX, ENABLE_LOGGING
)
from core.utils import (
    save_pickle, load_pickle,
    tokenize_for_bm25, prepare_for_embedding, split_sentences,
)
from core.logging_config import get_logger, quiet_worker

logger = get_logger(__name__)

# Same width and label column for every indexing bar.
_BAR = dict(ncols=80, bar_format="{desc:<18} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}")


class Indexer:
    def __init__(self, documents_dir=DOCUMENTS_DIR, embedding_model_name=EMBEDDING_MODEL_NAME,
                 faiss_index_path=MULTIVECTOR_INDEX_PATH, bm25_data_path=BM25_DATA_PATH,
                 metadata_index_path=METADATA_INDEX_PATH, faiss_binary_path=FAISS_INDEX_PATH,
                 enable_ocr=True, ocr_min_width=50, ocr_min_height=50):
        self.documents_dir = documents_dir
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.bm25_data_path = bm25_data_path
        self.metadata_index_path = metadata_index_path
        self.faiss_binary_path = faiss_binary_path
        self.enable_ocr = enable_ocr
        self.ocr_min_width = ocr_min_width
        self.ocr_min_height = ocr_min_height

    @staticmethod
    def chunk_text_with_overlap(text, max_chunk_length, chunk_overlap):
        """
        Split text into chunks with overlap to preserve boundary information.

        Args:
            text: Text to chunk
            max_chunk_length: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Split into sentences (syntok — abbreviation/decimal aware)
        sentences = split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_sentences = []  # Sentences to carry over to next chunk

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds max length and we have content
            if current_length + sentence_len > max_chunk_length and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Determine overlap: keep sentences from the end that fit within overlap size
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent) + 1
                    else:
                        break

                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + 1

        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def chunk_text_with_overlap_paged(pieces, max_chunk_length, chunk_overlap):
        """Chunk across piece boundaries while keeping page provenance.

        Same sentence-atomic algorithm as chunk_text_with_overlap, but the input
        is a list of (text, page) pieces that are chunked as one continuous
        stream. Without this a PDF chunk can never span a page break, so a table
        row ending one page and its value starting the next never co-occur in a
        chunk.

        Args:
            pieces: list of (text, page) in document order; `page` is whatever
                    should land in metadata (0-based page index for PDFs)
            max_chunk_length: maximum characters per chunk
            chunk_overlap: characters of overlap between chunks

        Returns:
            List of (chunk_text, start_page) — start_page is the page of the
            first sentence in the chunk, so citations point at where the chunk
            begins. Passing a single piece is equivalent to
            chunk_text_with_overlap on that text.
        """
        # Flatten to (sentence, page) so the carry-over overlap can cross pages
        sentences = []
        for text, page in pieces:
            if not text:
                continue
            sentences.extend((sent, page) for sent in split_sentences(text))
        if not sentences:
            return []

        chunks = []
        current_chunk = []          # list of (sentence, page)
        current_length = 0

        def _flush(group):
            chunk_text = " ".join(sent for sent, _ in group).strip()
            if chunk_text:
                chunks.append((chunk_text, group[0][1]))

        for sentence, page in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > max_chunk_length and current_chunk:
                _flush(current_chunk)

                overlap_sentences = []
                overlap_length = 0
                for sent, sent_page in reversed(current_chunk):
                    if overlap_length + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, (sent, sent_page))
                        overlap_length += len(sent) + 1
                    else:
                        break

                current_chunk = overlap_sentences + [(sentence, page)]
                current_length = sum(len(s) + 1 for s, _ in current_chunk)
            else:
                current_chunk.append((sentence, page))
                current_length += sentence_len + 1

        if current_chunk:
            _flush(current_chunk)

        return chunks

    @staticmethod
    def _add_to_metadata_index(metadata_index, metadata_list, offset=0):
        """Register chunks in the reverse index under both their document name
        and full_document_id keys. `offset` shifts indices for incremental adds."""
        for idx, meta in enumerate(metadata_list):
            for key in (meta.get('document_name'), meta.get('full_document_id')):
                if key:
                    metadata_index.setdefault(key, []).append(offset + idx)

    def build_metadata_index(self, all_metadata):
        """
        Build a reverse index mapping document names to chunk indices
        Returns: dict with structure {document_name: [list_of_chunk_indices]}
        """
        metadata_index = {}
        self._add_to_metadata_index(metadata_index, all_metadata)
        return metadata_index

    def get_embeddings(self, texts, model, batch_size=32, is_query=False, progress_callback=None):
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of texts to embed
            model: Sentence transformer model
            batch_size: Batch size for encoding
            is_query: If True, use "query: " prefix; if False, use "passage: " prefix
            progress_callback: Optional callable(stage, current, total) for progress reporting

        Returns:
            numpy array of embeddings
        """
        # Defensive: HF tokenizer (Rust) rejects strings with invalid UTF-8
        # (lone surrogates) or NUL bytes even though they're valid Python str.
        # Round-trip through UTF-8 to scrub any such inputs.
        def _scrub(t):
            if not isinstance(t, str):
                t = str(t) if t is not None else ''
            return t.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')
        texts = [_scrub(t) for t in texts]

        # Apply E5 prefixes if enabled
        if EMBEDDING_USE_PREFIX:
            texts = [prepare_for_embedding(text, is_query=is_query) for text in texts]

        # Cap sequence length so a batch that happens to contain very long chunks
        # can't blow up the padded attention matrix (e5-base only uses 512 tokens).
        try:
            model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
        except Exception:
            pass

        # Release the Metal (MPS) cache between batches; without this, memory
        # fragments and climbs until it hits Metal's ceiling and OOMs mid-build.
        import torch
        mps_available = getattr(
            getattr(torch.backends, "mps", None), "is_available", lambda: False
        )()
        mps_cache_clear = getattr(getattr(torch, "mps", None), "empty_cache", None) if mps_available else None

        all_embeddings = []
        total = len(texts)
        for i in tqdm(range(0, total, batch_size), desc="Indexing", **_BAR):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch, batch_size=batch_size)
            all_embeddings.extend(embeddings)
            if mps_cache_clear is not None:
                mps_cache_clear()
            if progress_callback is not None:
                progress_callback('embedding', min(i + batch_size, total), total)
        return np.array(all_embeddings, dtype=np.float32)

    def build_faiss_index(self, multi_vector_index, index_type=None):
        """
        Build FAISS index with automatic selection based on corpus size.

        Args:
            multi_vector_index: List of dicts with 'embedding' key
            index_type: Override index type ('flat', 'ivf', 'hnsw', 'auto')
                       If None, uses FAISS_INDEX_TYPE from config

        Index selection:
            - < 10,000 vectors: IndexFlatIP (exact search)
            - 10,000 - 1,000,000: IndexIVFFlat (inverted file)
            - > 1,000,000: IndexHNSWFlat (graph-based)

        Returns:
            FAISS index
        """
        if not multi_vector_index:
            logger.warning("Warning: Multi-vector index is empty. Cannot build FAISS index.")
            return None

        if index_type is None:
            index_type = FAISS_INDEX_TYPE

        embeddings = np.array([item["embedding"] for item in multi_vector_index]).astype('float32')
        n_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]

        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)

        # Auto-select index type based on corpus size
        if index_type == 'auto':
            if n_vectors < 10000:
                index_type = 'flat'
            elif n_vectors < 1000000:
                index_type = 'ivf'
            else:
                index_type = 'hnsw'

        logger.info(f"Building FAISS index: type={index_type}, vectors={n_vectors}, dimension={dimension}")

        if index_type == 'flat':
            # Exact search - O(n) but 100% recall
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

        elif index_type == 'ivf':
            # IVF for medium scale - O(sqrt(n)) with ~95-99% recall
            # nlist = number of clusters (sqrt(n) is a good default)
            nlist = max(int(np.sqrt(n_vectors)), 1)
            nlist = min(nlist, n_vectors)  # Can't have more clusters than vectors

            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # Training required for IVF
            logger.info(f"Training IVF index with nlist={nlist}...")
            index.train(embeddings)
            index.add(embeddings)

            # Set nprobe for search (tradeoff: higher = better recall, slower)
            index.nprobe = max(nlist // FAISS_IVF_NPROBE_RATIO, 1)
            logger.info(f"IVF index built: nlist={nlist}, nprobe={index.nprobe}")

        elif index_type == 'hnsw':
            # HNSW for large scale - O(log n) with ~95% recall
            # M = number of neighbors per node (32 is a good default)
            M = 32
            index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40  # Build-time accuracy
            index.hnsw.efSearch = 16  # Search-time accuracy (can adjust later)
            index.add(embeddings)
            logger.info(f"HNSW index built: M={M}, efConstruction=40, efSearch=16")

        else:
            raise ValueError(f"Unknown FAISS index type: {index_type}. Use 'flat', 'ivf', 'hnsw', or 'auto'")

        return index

    def build_bm25_index(self, texts):
        """
        Build BM25 index with improved tokenization.

        Uses stemming and stop word removal for better matching.
        """
        tokenized_corpus = [
            tokenize_for_bm25(
                doc,
                enable_stemming=BM25_ENABLE_STEMMING,
                enable_stopwords=BM25_ENABLE_STOPWORDS,
                languages=BM25_LANGUAGES
            )
            for doc in tqdm(texts, desc="Tokenizing for BM25", disable=not ENABLE_LOGGING)
        ]
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_corpus, show_progress=False)
        return bm25_index

    def build_multi_vector_index(self, embeddings, metadata, texts):
        index = []
        for emb, meta, text in zip(embeddings, metadata, texts):
            index.append({
                "embedding": np.array(emb, dtype=np.float32),
                "text": text,
                "metadata": meta
            })
        return index

    def save_indices(self, multi_vector_index, bm25, metadata_index, faiss_index=None):
        """Save all indices including the new metadata index"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        save_pickle(multi_vector_index, self.faiss_index_path)
        # bm25s uses its own directory-based save format (scipy arrays + json)
        if bm25 is not None:
            bm25.save(self.bm25_data_path)
        save_pickle(metadata_index, self.metadata_index_path)  # Save metadata index
        if faiss_index:
            faiss.write_index(faiss_index, self.faiss_binary_path)

    def load_indices(self):
        """Load all indices including metadata index"""
        multi_vector_index = load_pickle(self.faiss_index_path)
        bm25 = None
        if os.path.isdir(self.bm25_data_path):
            bm25 = bm25s.BM25.load(self.bm25_data_path)
        metadata_index = load_pickle(self.metadata_index_path)
        faiss_index = None
        if os.path.exists(self.faiss_binary_path):
            faiss_index = faiss.read_index(self.faiss_binary_path)
        return multi_vector_index, bm25, metadata_index, faiss_index

    def _find_and_process_new_documents(self, metadata_index):
        """Detect unindexed files, process them in parallel and filter junk chunks.

        Returns (new_chunks, new_metadata, new_filenames), or None when there
        are no new files or no content could be extracted.
        """
        from components.indexer.processors import process_single_document, SUPPORTED_EXTENSIONS as _SE

        # List all supported files in the documents directory
        all_files = [
            f for f in os.listdir(self.documents_dir)
            if os.path.splitext(f)[1].lower() in _SE
        ]

        # Determine which are already indexed by checking metadata_index keys
        # (keys that have a file extension are document names)
        indexed_filenames = {key for key in metadata_index if os.path.splitext(key)[1]}

        new_filenames = [f for f in all_files if f not in indexed_filenames]

        if not new_filenames:
            return None

        logger.info(f"New documents detected: {new_filenames}")

        existing_doc_count = len(indexed_filenames)
        file_infos = [
            (
                os.path.join(self.documents_dir, f),
                existing_doc_count + i,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                self.enable_ocr,
                self.ocr_min_width,
                self.ocr_min_height,
            )
            for i, f in enumerate(new_filenames)
        ]

        num_processes = max(1, min(cpu_count() - 1, 8))
        logger.info(f"Processing {len(new_filenames)} new document(s) with {num_processes} processes...")

        with Pool(processes=num_processes, initializer=quiet_worker) as pool:
            results = list(tqdm(
                pool.imap(process_single_document, file_infos),
                total=len(file_infos),
                desc="Reading new documents", **_BAR
            ))

        new_chunks = []
        new_metadata = []
        for chunks, metadata in results:
            for chunk, meta in zip(chunks, metadata):
                if len(chunk) >= MIN_CHUNK_CHARS:
                    new_chunks.append(chunk)
                    new_metadata.append(meta)

        if not new_chunks:
            logger.info("No content extracted from new documents.")
            return None

        return new_chunks, new_metadata, new_filenames

    def _merge_new_chunks(self, multi_vector_index, faiss_index, metadata_index,
                          new_chunks, new_metadata, embedding_model):
        """Embed new chunks and merge them into the in-memory indices (mutates
        multi_vector_index, faiss_index and metadata_index). Returns the
        rebuilt BM25 index."""
        logger.info("Generating embeddings for new chunks...")
        new_embeddings = self.get_embeddings(new_chunks, model=embedding_model)

        # Offset for new chunk indices
        offset = len(multi_vector_index)

        new_entries = self.build_multi_vector_index(new_embeddings, new_metadata, new_chunks)
        multi_vector_index.extend(new_entries)

        # Add new vectors to FAISS index
        embeddings_array = np.array([item["embedding"] for item in new_entries]).astype('float32')
        faiss.normalize_L2(embeddings_array)
        faiss_index.add(embeddings_array)

        # Rebuild BM25 from all texts (fast — just tokenization)
        all_texts = [item["text"] for item in multi_vector_index]
        bm25 = self.build_bm25_index(all_texts)

        self._add_to_metadata_index(metadata_index, new_metadata, offset)
        return bm25

    def index_new_documents(self, multi_vector_index, bm25, metadata_index, faiss_index, embedding_model):
        """
        Detect new documents in DOCUMENTS_DIR, process only those, and merge into existing indices.

        Returns:
            Tuple of (multi_vector_index, bm25, metadata_index, faiss_index, new_filenames)
            or None if no new documents were found.
        """
        processed = self._find_and_process_new_documents(metadata_index)
        if processed is None:
            return None
        new_chunks, new_metadata, new_filenames = processed

        logger.info(f"Extracted {len(new_chunks)} chunks from {len(new_filenames)} new document(s)")

        bm25 = self._merge_new_chunks(
            multi_vector_index, faiss_index, metadata_index,
            new_chunks, new_metadata, embedding_model)

        self.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        logger.info(f"Indices updated and saved. Total chunks: {len(multi_vector_index)}")

        return multi_vector_index, bm25, metadata_index, faiss_index, new_filenames

    def load_and_chunk_documents_parallel(self, max_chunk_length=None, chunk_overlap=None, progress_callback=None):
        """Load and process all supported documents in parallel.

        Args:
            max_chunk_length: Override chunk size (default: CHUNK_SIZE from config)
            chunk_overlap: Override chunk overlap (default: CHUNK_OVERLAP from config)
            progress_callback: Optional callable(stage, current, total) for progress reporting
        """
        from components.indexer.processors import process_single_document, SUPPORTED_EXTENSIONS as _SE

        # Use config defaults if not specified
        if max_chunk_length is None:
            max_chunk_length = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP

        doc_files = [
            os.path.join(self.documents_dir, f)
            for f in os.listdir(self.documents_dir)
            if os.path.splitext(f)[1].lower() in _SE
        ]
        logger.info(f"Found {len(doc_files)} document(s) to process.")
        if not doc_files:
            return [], []

        # Pass OCR configuration and chunk overlap to each process
        file_infos = [
            (f, i, max_chunk_length, chunk_overlap, self.enable_ocr, self.ocr_min_width, self.ocr_min_height)
            for i, f in enumerate(doc_files)
        ]

        # Use optimal number of processes (cpu_count - 1, minimum 1, maximum 8)
        num_processes = max(1, min(cpu_count() - 1, 8))
        logger.info(f"Processing with {num_processes} parallel processes (OCR {'enabled' if self.enable_ocr else 'disabled'})")

        # Process files in parallel with progress bar
        total_files = len(file_infos)
        all_chunks = []
        all_metadata = []
        with Pool(processes=num_processes, initializer=quiet_worker) as pool:
            for i, (chunks, metadata) in enumerate(tqdm(
                pool.imap(process_single_document, file_infos),
                total=total_files,
                desc="Reading documents", **_BAR
            )):
                for chunk, meta in zip(chunks, metadata):
                    if len(chunk) >= MIN_CHUNK_CHARS:
                        all_chunks.append(chunk)
                        all_metadata.append(meta)
                if progress_callback is not None:
                    progress_callback('loading', i + 1, total_files)

        logger.info(f"Extracted {len(all_chunks)} chunks from {len(doc_files)} documents")
        return all_chunks, all_metadata
