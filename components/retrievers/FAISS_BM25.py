import math
import os
import re
import threading
import unicodedata
import numpy as np
import faiss
from collections import defaultdict
from core import prompts
from core.llm_backend import generate

from core.config import (
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    EMBEDDING_USE_PREFIX, RETRIEVAL_TOP_K, RERANK, RERANKER_TOP_N, PARENT_TOP_N,
    RERANKER_CONFIDENCE_THRESHOLD, QUERY_EXPANSION_MIN_WORDS, RRF_K, MAX_DOCUMENT_CHARS,
    RERANK_CANDIDATES, FRONT_MATTER_PENALTY,
)
from core.utils import tokenize_for_bm25, prepare_for_embedding
from core.logging_config import get_logger

logger = get_logger(__name__)


# A chunk opening with one of these is the start of a section worth showing in
# a document overview ("Abstract", "1 Introduction", "6. Conclusões", ...).
_KEY_SECTION_RE = re.compile(
    r"^\W*(\d+(\.\d+)*\.?\s+)?(abstract|resumo|summary|sum[aá]rio|introduction|introdu[cç][aã]o|"
    r"results|resultados|findings|discussion|discuss[aã]o|conclusions?|conclus[oõ]es|conclus[aã]o|"
    r"contributions?|contribui[cç][oõ]es)\b", re.IGNORECASE)


def _overview_indices(texts, budget):
    """Pick chunks for an overview of a long document, within `budget` characters:
    the opening (title, abstract, contents), chunks that start key sections
    (introduction, results, conclusion, ...) with the chunk after each, then an
    even sample of the rest. Returns indices in reading order."""
    picked, used = set(), 0

    def take(i):
        nonlocal used
        if 0 <= i < len(texts) and i not in picked and used < budget:
            picked.add(i)
            used += len(texts[i])

    i = 0
    while i < len(texts) and used < 0.3 * budget:
        take(i)
        i += 1
    for i, text in enumerate(texts):
        if used >= 0.75 * budget:
            break
        if _KEY_SECTION_RE.search(text[:120]):
            take(i)
            take(i + 1)
    rest = [i for i in range(len(texts)) if i not in picked]
    if rest and used < budget:
        average = max(sum(len(texts[i]) for i in rest) / len(rest), 1)
        slots = max(int((budget - used) / average), 1)
        for i in rest[::max(len(rest) // slots, 1)]:
            take(i)
    return sorted(picked)


def _fold(text):
    """Lower-case without accents, for forgiving name matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class Retriever:
    def __init__(self, llm_model, llm_tokenizer,
                 embedding_model_name=None, reranker_model_name=None,
                 index_paths=None):
        # LLM needed immediately for query expansion
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

        # Store config for lazy loading
        self._embedding_model_name = embedding_model_name
        self._reranker_model_name = reranker_model_name
        self._index_paths = index_paths

        # Private backing fields (None until loaded)
        self._embedding_model = None
        self._reranker = None
        self._multi_vector_index = None
        self._bm25 = None
        self._metadata_index = None
        self._faiss_index = None
        self._parent_store = None
        self._indices_loaded = False

        # Thread safety for lazy loading
        self._load_lock = threading.Lock()

        self.last_retrieved_metadata = []
        self.current_user_query = None
        # Cumulative (document_name, page) seen across an agentic run, so the
        # research loop can detect source saturation. Reset per run.
        self.seen_sources = set()
        # Retrieved chunk texts for this turn, consumed by the post-generation
        # numeric check. Bounded; cleared by the generator on each new query.

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            with self._load_lock:
                if self._embedding_model is None:
                    logger.info(f"Loading embedding model: {self._embedding_model_name}...")
                    from core.utils import load_sentence_transformer
                    self._embedding_model = load_sentence_transformer(self._embedding_model_name)
        return self._embedding_model

    @property
    def reranker(self):
        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    logger.info(f"Loading reranker: {self._reranker_model_name}...")
                    from core.utils import load_cross_encoder
                    self._reranker = load_cross_encoder(self._reranker_model_name)
        return self._reranker

    def ensure_indices_loaded(self):
        if not self._indices_loaded:
            with self._load_lock:
                if not self._indices_loaded:
                    logger.info("Loading indices...")
                    from core.config import HIERARCHICAL_INDEXING
                    kwargs = self._index_paths or {}
                    if HIERARCHICAL_INDEXING:
                        from components.indexer import HierarchicalIndexer
                        indexer = HierarchicalIndexer(**kwargs)
                        self._multi_vector_index, self._bm25, self._metadata_index, self._faiss_index, self._parent_store = indexer.load_indices()
                    else:
                        from components.indexer import Indexer
                        indexer = Indexer(**kwargs)
                        self._multi_vector_index, self._bm25, self._metadata_index, self._faiss_index = indexer.load_indices()
                    self._indices_loaded = True

    def reset_indices(self, index_paths):
        """Reset all in-memory indices and set new load paths (None = default config)."""
        with self._load_lock:
            self._index_paths = index_paths
            self._indices_loaded = False
            self._multi_vector_index = None
            self._bm25 = None
            self._metadata_index = None
            self._faiss_index = None
            self._parent_store = None

    def switch_to_user_index(self, index_paths: dict):
        """Reset indices and reload from user-provided paths on next retrieval."""
        self.reset_indices(index_paths)

    def switch_to_default_index(self):
        """Reset to default config paths."""
        self.reset_indices(None)

    @property
    def multi_vector_index(self):
        self.ensure_indices_loaded()
        return self._multi_vector_index

    @property
    def faiss_index(self):
        self.ensure_indices_loaded()
        return self._faiss_index

    @property
    def bm25(self):
        self.ensure_indices_loaded()
        return self._bm25

    @property
    def metadata_index(self):
        self.ensure_indices_loaded()
        return self._metadata_index

    @property
    def parent_store(self):
        self.ensure_indices_loaded()
        return self._parent_store

    def update_indices(self, multi_vector_index, bm25, metadata_index, faiss_index,
                       parent_store=None):
        """Thread-safe swap of all in-memory indices."""
        with self._load_lock:
            self._multi_vector_index = multi_vector_index
            self._bm25 = bm25
            self._metadata_index = metadata_index
            self._faiss_index = faiss_index
            if parent_store is not None:
                self._parent_store = parent_store
            self._indices_loaded = True

    def resolve_parents(self, reranked_pairs, top_n=None):
        """
        Resolve child chunks to their parent chunks, deduplicating by parent.

        Args:
            reranked_pairs: List of (child_text, child_metadata) tuples
            top_n: Max number of results to return (applied after dedup)

        Returns:
            List of (text, metadata) tuples — parent chunks if available,
            otherwise children as-is.
        """
        if self._parent_store is None:
            return reranked_pairs

        resolved = []
        seen_parent_ids = set()

        for child_text, child_meta in reranked_pairs:
            parent_idx = child_meta.get('parent_idx')
            if parent_idx is None:
                # No parent mapping — return child as-is
                resolved.append((child_text, child_meta))
                continue

            if parent_idx in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_idx)

            if parent_idx < len(self._parent_store):
                parent = self._parent_store[parent_idx]
                resolved.append((parent["text"], parent["metadata"]))
            else:
                # Fallback if parent_idx is out of range
                resolved.append((child_text, child_meta))

        if top_n:
            resolved = resolved[:top_n]

        return resolved

    @staticmethod
    def format_chunk_with_source(text, metadata):
        """Prefix a chunk with a source tag so the LLM can cite it.

        E.g. "[Source: report.pdf, p. 12]" — pages are stored 0-based,
        displayed 1-based. Chunks without a document name are returned as-is.
        """
        doc = metadata.get('document_name')
        if not doc:
            return text

        doc_ext = os.path.splitext(doc)[1].lower()
        if doc_ext == '.pptx':
            loc = f", slide {metadata.get('slide', metadata.get('page', 0) + 1)}"
        elif doc_ext in ('.xlsx', '.xls'):
            sheet = metadata.get('sheet')
            loc = f", sheet '{sheet}'" if sheet else ""
        elif doc_ext == '.pdf':
            page = metadata.get('page')
            loc = f", p. {page + 1}" if isinstance(page, int) else ""
        else:
            loc = ""
        return f"[Source: {doc}{loc}]\n{text}"

    @staticmethod
    def retrieve_with_faiss(query_embedding, faiss_index, k=10):
        if faiss_index is None:
             logger.warning("Warning: FAISS index is not available.")
             return []
        query_vec = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        distances, indices = faiss_index.search(query_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((idx, float(dist)))
        return results

    @staticmethod
    def retrieve_with_bm25(tokenized_query, bm25, top_k=10):
        if bm25 is None:
             logger.warning("Warning: BM25 index is not available.")
             return []
        if not tokenized_query:
             # bm25s raises on empty queries (e.g. all tokens were stop words)
             return []
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]

    def retrieve_by_metadata(self, document_name, sort_by_page=True):
        """
        Retrieve all chunks for a specific document name
        Returns: list of chunks with their metadata, sorted by page number
        """
        original_query = document_name
        if document_name not in self.metadata_index:
            # Try to find partial matches for more flexible retrieval
            matching_docs = [doc for doc in self.metadata_index.keys()
                           if document_name.lower() in doc.lower()]
            if not matching_docs:
                return []
            # Use the first match, or could return all matches
            document_name = matching_docs[0]
            logger.info(f"Using partial match: '{document_name}' for query: '{original_query}'")

        chunk_indices = self.metadata_index[document_name]
        results = []
        for idx in chunk_indices:
            if idx < len(self.multi_vector_index):
                results.append({
                    'text': self.multi_vector_index[idx]["text"],
                    'metadata': self.multi_vector_index[idx]["metadata"],
                    'index': idx
                })

        # Sort by page number and chunk index for coherent reading order
        if sort_by_page and results:
            results.sort(key=lambda x: (
                x['metadata'].get('page', 0),
                x['metadata'].get('slide', 0),
                x['metadata'].get('sheet_index', 0),
                x['metadata'].get('chunk_idx', 0),
                x['metadata'].get('table_idx', 0),
                x['metadata'].get('img_idx', 0)
            ))
        return results

    def expand_query(self, query):
        if self.current_user_query:
            query = self.current_user_query
        if QUERY_EXPANSION_MIN_WORDS > 0 and len(query.split()) < QUERY_EXPANSION_MIN_WORDS:
            return []
        system_prompt = prompts.get("query_expansion_system")
        message = prompts.get("query_expansion_user", query=query)
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        prompt = self.llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        from core.llm_backend import make_sampler

        sampler = make_sampler(temp=0.2)

        response = generate(
            self.llm_model,
            self.llm_tokenizer,
            prompt=prompt,
            max_tokens=300,
            sampler=sampler,
            verbose=False)
        
        response = re.sub(r'(?i)\b(?:here\s+are\s+the\s+)?buzz\s*words?\s*:\s*', '', response)
        
        logger.info(f"Alternative questions: {response}")
        alternatives = []
        for line in response.split('\n'):
            line = line.strip('"-â ¢* 0123456789.').strip()
            line = self._strip_bare_numeric_keywords(line)
            if 2 < len(line):
                alternatives.append(line)
        if not alternatives:
            alternatives = [query]
        return alternatives

    @staticmethod
    def _strip_bare_numeric_keywords(line):
        """Drop standalone numbers from the `Keywords:` / `Palavras-chave:` expansion variant.

        A bare year is a terrible sparse-retrieval key: it matches every
        document that merely happens to carry that year, so a question about a
        law from a given year surfaces unrelated papers from that year. Numbers attached to a term ("artigo 22", "254/2009") are kept —
        only tokens that are nothing but digits are removed, and only from the
        keyword variant, where they carry no context.
        """
        prefix, sep, body = line.partition(':')
        if not sep or not any(w in prefix.lower() for w in ('palavra', 'keyword')):
            return line
        kept = [t.strip() for t in body.split(',')
                if t.strip() and not t.strip().isdigit()]
        # Return the terms only. Keeping the prefix put the words 'palavras' and
        # 'chave' (or 'keywords') into the query itself, where BM25 scored them
        # as ordinary search terms and matched any chunk using 'chave' in its
        # everyday sense.
        return ', '.join(kept)

    def rerank_chunks(self, query, chunks, metadata_list, top_n=None, threshold=None):
        def _sigmoid(s):
            return 1.0 / (1.0 + math.exp(-float(s)))

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(chunks, metadata_list, scores), key=lambda x: x[2], reverse=True)
        if top_n:
            reranked = reranked[:top_n]
        for chunk, _, score in reranked:
            logger.info(f"  score={_sigmoid(score):.3f}  {chunk[:60]}")
        if threshold is not None:
            filtered = [item for item in reranked if _sigmoid(item[2]) >= threshold]
            if not filtered and reranked:
                # Always keep at least the best result to avoid empty context
                best = reranked[0]
                logger.warning(f"Warning: All {len(reranked)} chunks below confidence threshold "
                    f"{threshold:.2f} (best: {_sigmoid(best[2]):.3f}). Keeping top result.")
                filtered = [best]
            reranked = filtered
        return [(chunk, meta) for chunk, meta, _ in reranked]

    def _record_sources(self, resolved):
        """Accumulate what this turn retrieved.

        `seen_sources` drives the research loop's saturation stop.
        It is created on demand so instances built with __new__ — as the
        tests and some server paths do — keep working.
        """
        if not hasattr(self, 'seen_sources'):
            self.seen_sources = set()
        self.seen_sources.update(
            (meta.get('document_name'), meta.get('page')) for _, meta in resolved)

    def _demote_front_matter(self, ranked, rrf_scores):
        """Down-weight cover/title/TOC chunks after fusion.

        A demotion, not a filter — front matter can still win when nothing
        better exists, and sometimes does hold the answer.
        """
        if not FRONT_MATTER_PENALTY or FRONT_MATTER_PENALTY >= 1.0:
            return ranked
        adjusted = {}
        for idx in ranked:
            meta = self.multi_vector_index[idx]["metadata"]
            score = rrf_scores[idx]
            if meta.get('is_front_matter'):
                score *= FRONT_MATTER_PENALTY
            adjusted[idx] = score
        return sorted(adjusted, key=adjusted.get, reverse=True)

    def fuse_candidates(self, query, k):
        """Query expansion → FAISS + BM25 over every variant → RRF fusion.

        Returns (ranked_indices, rrf_scores): indices sorted best-first and
        NOT truncated, plus the raw score map. Split out of retrieve_resolved
        so evaluation harnesses can measure the candidate pool before reranking
        without duplicating the fusion logic.
        """
        expanded_queries = [query] + self.expand_query(query)
        rrf_scores = defaultdict(float)

        # Dense retrieval (FAISS) — RRF over all query variants
        prefixed_queries = [
            prepare_for_embedding(q, is_query=True, use_prefix=EMBEDDING_USE_PREFIX)
            for q in expanded_queries
        ]
        query_embeddings = self.embedding_model.encode(prefixed_queries)
        for query_emb in query_embeddings:
            for rank, (idx, _) in enumerate(self.retrieve_with_faiss(query_emb, self.faiss_index, k=k)):
                rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        # Sparse retrieval (BM25) — RRF over all query variants
        for exp_query in expanded_queries:
            tokenized_query = tokenize_for_bm25(
                exp_query,
                enable_stemming=BM25_ENABLE_STEMMING,
                enable_stopwords=BM25_ENABLE_STOPWORDS,
                languages=BM25_LANGUAGES
            )
            for rank, (idx, _) in enumerate(self.retrieve_with_bm25(tokenized_query, self.bm25, top_k=k)):
                rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return ranked, rrf_scores

    def retrieve_resolved(self, query, k, rerank_top_n, r):
        """Runs the full retrieval pipeline: query expand → FAISS+BM25 → RRF → rerank → resolve parents.

        Returns list of (text, metadata) tuples and sets self.last_retrieved_metadata.
        """
        ranked, rrf_scores = self.fuse_candidates(query, k)
        ranked = self._demote_front_matter(ranked, rrf_scores)

        top_indices = ranked[:k]
        retrieved_chunks = [self.multi_vector_index[idx]["text"] for idx in top_indices]
        retrieved_metadata = [self.multi_vector_index[idx]["metadata"] for idx in top_indices]

        if r:
            # The candidate pool sets the recall ceiling; the cross-encoder only
            # reorders within it (measured: Recall@100 is identical with and
            # without reranking). Those are different jobs, so they get
            # different depths — reranking every candidate is what made
            # retrieval ~2.9 s per query.
            width = RERANK_CANDIDATES or len(retrieved_chunks)
            reranked = self.rerank_chunks(
            query, retrieved_chunks[:width], retrieved_metadata[:width],
            top_n=rerank_top_n, threshold=RERANKER_CONFIDENCE_THRESHOLD,
        )
            resolved = self.resolve_parents(reranked, top_n=PARENT_TOP_N)

        else:
            hybrid = list(zip(retrieved_chunks, retrieved_metadata))
            resolved = self.resolve_parents(hybrid, top_n=PARENT_TOP_N)

        self.last_retrieved_metadata = [meta for _, meta in resolved]
        self._record_sources(resolved)
        return resolved

    def combined_retrieval(self, query, k=RETRIEVAL_TOP_K, rerank_top_n=RERANKER_TOP_N, r=RERANK):
        """The final chunks from retrieve_resolved, each with its source, as one string."""
        if not self.faiss_index or not self.bm25 or not self.multi_vector_index:
            logger.warning("Warning: FAISS index, BM25 index, or multi-vector index is not available for retrieval.")
            return "No documents available for retrieval."

        resolved = self.retrieve_resolved(query, k, rerank_top_n, r)
        return "\n---\n".join(
            self.format_chunk_with_source(chunk, meta) for chunk, meta in resolved
        )

    def combined_retrieval_with_chunks(self, query, k=RETRIEVAL_TOP_K, rerank_top_n=RERANKER_TOP_N): # oq é isto?
        """Same as combined_retrieval but returns (context_string, individual_chunks_list)."""
        if not self.faiss_index or not self.bm25 or not self.multi_vector_index:
            return "No documents available for retrieval.", []

        resolved = self.retrieve_resolved(query, k, rerank_top_n, RERANK)
        # Context string carries source tags for citation; the raw chunk list
        # (used as RAGAS contexts) stays untagged.
        context = "\n---\n".join(
            self.format_chunk_with_source(chunk, meta) for chunk, meta in resolved
        )
        chunks_list = [chunk for chunk, _ in resolved]
        return context, chunks_list

    def search_documents_tool(self, query: str) -> str:
        """Tool for semantic search across all documents.

        Args:
            query: Search query string.
        """
        if not self.metadata_index:
            return prompts.get("no_documents")
        try:
            return self.combined_retrieval(query=query)
        except Exception as e:
            logger.warning("search_documents failed for %r", query, exc_info=True)
            return f"Error during search: {str(e)}"

    def retrieve_document_by_name_tool(self, document_name: str) -> str:
        """
        Tool for retrieving entire documents by name
        Returns all chunks from the specified document
        """
        if not self.metadata_index:
            return prompts.get("no_documents")
        try:
            results = self.retrieve_by_metadata(document_name)
            if not results:
                # More helpful error message with better matching suggestions
                all_docs = list(self.metadata_index.keys())
                # Try to find similar document names
                similar = [doc for doc in all_docs if document_name.lower() in doc.lower() or doc.lower() in document_name.lower()]

                error_msg = f"Document '{document_name}' not found.\n"
                if similar:
                    error_msg += f"\nDid you mean one of these?\n"
                    for doc in similar[:5]:
                        error_msg += f"  - {doc}\n"
                else:
                    error_msg += f"\nAvailable documents (first 10):\n"
                    for doc in sorted(all_docs)[:10]:
                        error_msg += f"  - {doc}\n"
                    if len(all_docs) > 10:
                        error_msg += f"\n... and {len(all_docs) - 10} more. Use list_available_documents to see all."

                return error_msg

            # Determine document format for citation style
            doc_ext = os.path.splitext(document_name)[1].lower() if '.' in document_name else ''

            def citation(metadata):
                if doc_ext == '.pptx':
                    return f"[Slide {metadata.get('slide', metadata.get('page', 0) + 1)}, Chunk {metadata.get('chunk_idx', metadata.get('table_idx', metadata.get('img_idx', 'N/A')))}]"
                if doc_ext in ('.xlsx', '.xls'):
                    return f"[Sheet '{metadata.get('sheet', 'Sheet1')}', Chunk {metadata.get('chunk_idx', 'N/A')}]"
                if doc_ext == '.csv':
                    return f"[Chunk {metadata.get('chunk_idx', 'N/A')}]"
                if doc_ext == '.pdf':
                    return f"[Page {metadata.get('page', 0) + 1}, Chunk {metadata.get('chunk_idx', metadata.get('table_idx', metadata.get('img_idx', 'N/A')))}]"
                return f"[Chunk {metadata.get('chunk_idx', metadata.get('img_idx', 'N/A'))}]"

            # A long document would fill the context window and take minutes to
            # read, so return an overview within MAX_DOCUMENT_CHARS instead.
            texts = [r['text'] for r in results]
            overview = sum(len(x) for x in texts) > MAX_DOCUMENT_CHARS
            shown = _overview_indices(texts, MAX_DOCUMENT_CHARS) if overview else range(len(results))

            document_content = []
            previous = -1
            for i in shown:
                if overview and i > previous + 1:
                    document_content.append(f"[... {i - previous - 1} chunk(s) not shown ...]")
                document_content.append(f"{texts[i]}\n{citation(results[i]['metadata'])}")
                previous = i

            full_document = "\n--- Page Break ---\n".join(document_content)

            # Add document summary header
            doc_metadata = results[0]['metadata']
            header = f"DOCUMENT: {doc_metadata.get('document_name', document_name)}\n"
            header += f"TOTAL CHUNKS: {len(results)}\n"
            if doc_ext == '.pptx':
                header += f"SLIDES: {max(r['metadata'].get('slide', r['metadata'].get('page', 0) + 1) for r in results)}\n"
            elif doc_ext in ('.xlsx', '.xls'):
                sheets = set(r['metadata'].get('sheet', '') for r in results)
                header += f"SHEETS: {', '.join(sorted(s for s in sheets if s))}\n"
            else:
                header += f"PAGES: {max(r['metadata'].get('page', 0) for r in results) + 1}\n"
            if overview:
                header += (f"This document is too long to read in full here, so this is an OVERVIEW: "
                           f"{len(shown)} of {len(results)} chunks (the opening, key sections such as "
                           f"the introduction and conclusion, and samples from the rest). "
                           f"For specific topics, call search_documents with focused terms.\n")
            header += "="*50 + "\n"

            return header + full_document
        except Exception as e:
            import traceback
            return f"Error retrieving document '{document_name}': {str(e)}\n{traceback.format_exc()}"

    def list_available_documents_tool(self, filter_keyword: str = "") -> str:
        """Tool to list all available documents in the index, optionally filtered by keyword"""
        if not self.metadata_index:
            return prompts.get("no_documents")
        try:
            doc_names = list(self.metadata_index.keys())

            # Prefer the plain filename over its `_N` full_document_id variants.
            def _base(name):
                base, _, suffix = name.rpartition('_')
                return base if base and suffix.isdigit() else name

            chosen = {}
            for d in sorted(doc_names):
                base = _base(d)
                if d == base or base not in chosen:
                    chosen[base] = d
            meaningful_docs = sorted(chosen.values()) if chosen else doc_names

            # Filter by name, ignoring case and accents. With no match, list
            # everything: the user's word ("thesis") rarely appears in the file
            # name ("Dissertacao_..."), and an empty answer costs another turn.
            no_match = False
            if filter_keyword:
                wanted = _fold(filter_keyword)
                filtered_docs = [doc for doc in meaningful_docs if wanted in _fold(doc)]
                if filtered_docs:
                    meaningful_docs = filtered_docs
                else:
                    no_match = True

            document_info = []
            for doc_name in sorted(meaningful_docs):
                chunk_count = len(self.metadata_index[doc_name])
                document_info.append(f"- {doc_name} ({chunk_count} chunks)")

            total_docs = len(meaningful_docs)
            if no_match:
                header = (f"No document name contains '{filter_keyword}'. "
                          f"All available documents ({total_docs}):\n")
            elif filter_keyword:
                header = f"Documents matching '{filter_keyword}' ({total_docs} found):\n"
            else:
                header = f"Available documents ({total_docs} total):\n"

            return header + "\n".join(document_info)
        except Exception as e:
            return f"Error listing documents: {str(e)}"
