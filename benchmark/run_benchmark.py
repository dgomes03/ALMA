#!/usr/bin/env python
"""Benchmark the ALMA indexer/retriever on standard IR benchmarks (BEIR).

Measures, using the framework's own indexing/retrieval components and config:

  Indexing:   BM25 build time, embedding throughput (docs/s), FAISS build
              time and resolved index type, on-disk index sizes, RAM usage.
  Retrieval:  quality (nDCG@k, MAP@k, Recall@k, P@k, MRR@k against qrels)
              and per-query latency (mean / p50 / p95 / p99, QPS) for:
                bm25          — sparse only (bm25s + framework tokenizer)
                dense         — embeddings + FAISS only
                hybrid        — dense + bm25 fused with RRF (as in Retriever)
                hybrid_rerank — hybrid + cross-encoder reranking

LLM query expansion and hierarchical parent resolution are deliberately
excluded: they require a loaded LLM / parent store and are orthogonal to
raw indexer/retriever performance.

Usage (from the repository root):
    python -m benchmark.run_benchmark \
        --dataset scifact --methods bm25 dense hybrid

Results are printed and saved as JSON under benchmark/results/.
"""
import argparse
import datetime
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict

# Allow running as a plain script (python run_benchmark.py) as well as with -m
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from tqdm import tqdm

from core.config import (
    EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, EMBEDDING_USE_PREFIX,
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    RETRIEVAL_TOP_K, RRF_K, FAISS_INDEX_TYPE, RERANK_CANDIDATES,
    BENCHMARK_DATASET, BENCHMARK_METHODS, BENCHMARK_K_VALUES,
    BENCHMARK_MAX_QUERIES, BENCHMARK_MAX_CORPUS,
)
from core.utils import (
    load_sentence_transformer, load_cross_encoder,
    tokenize_for_bm25, prepare_for_embedding,
)
from components.indexer import Indexer
from benchmark.beir_data import load_beir_dataset, KNOWN_DATASETS, DEFAULT_DATA_DIR
from benchmark.metrics import evaluate_run

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALL_METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]

# When --max-queries is set without --max-corpus, the corpus is auto-subsampled
# to max(queries * AUTO_DISTRACTORS_PER_QUERY, AUTO_MIN_CORPUS) docs.
AUTO_DISTRACTORS_PER_QUERY = 20
AUTO_MIN_CORPUS = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None


def _peak_rss_mb():
    try:
        import resource  # Unix only
    except ImportError:
        # Windows: psutil exposes the peak working set instead
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6
        except (ImportError, AttributeError):
            return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3


def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def _latency_stats(latencies):
    lat_sorted = sorted(latencies)

    def pct(p):
        return lat_sorted[min(int(len(lat_sorted) * p), len(lat_sorted) - 1)]

    total = sum(latencies)
    return {
        "queries": len(latencies),
        "mean_ms": round(statistics.mean(latencies) * 1000, 2),
        "p50_ms": round(pct(0.50) * 1000, 2),
        "p95_ms": round(pct(0.95) * 1000, 2),
        "p99_ms": round(pct(0.99) * 1000, 2),
        "max_ms": round(lat_sorted[-1] * 1000, 2),
        "qps": round(len(latencies) / total, 2) if total > 0 else None,
    }


def _subsample_corpus(corpus, qrels, max_corpus, seed):
    """Shrink the corpus for quick runs while keeping every judged document."""
    judged = {did for rels in qrels.values() for did in rels}
    if max_corpus <= len(judged):
        print(f"--max-corpus {max_corpus} <= {len(judged)} judged docs; keeping judged docs only.")
        return {did: corpus[did] for did in judged if did in corpus}
    rest = [did for did in corpus if did not in judged]
    random.Random(seed).shuffle(rest)
    keep = judged.union(rest[: max_corpus - len(judged)])
    return {did: doc for did, doc in corpus.items() if did in keep}


# ---------------------------------------------------------------------------
# Indexing benchmark
# ---------------------------------------------------------------------------

def benchmark_indexing(indexer, texts, metadata, embedding_model, args, scratch_dir):
    """Build BM25 + embeddings + FAISS with the framework's Indexer, timing each stage."""
    stats = {"corpus_docs": len(texts),
             "corpus_chars": sum(len(t) for t in texts)}
    rss_before = _rss_mb()

    print("\n=== Indexing benchmark ===")
    t0 = time.perf_counter()
    bm25 = indexer.build_bm25_index(texts)
    bm25_seconds = time.perf_counter() - t0
    stats["bm25"] = {
        "build_seconds": round(bm25_seconds, 2),
        "docs_per_second": round(len(texts) / bm25_seconds, 1),
    }

    t0 = time.perf_counter()
    embeddings = indexer.get_embeddings(texts, model=embedding_model, batch_size=args.batch_size)
    emb_seconds = time.perf_counter() - t0
    stats["embeddings"] = {
        "model": args.embedding_model,
        "dimension": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "build_seconds": round(emb_seconds, 2),
        "docs_per_second": round(len(texts) / emb_seconds, 1),
        "size_mb": round(embeddings.nbytes / 1e6, 1),
    }

    multi_vector_index = indexer.build_multi_vector_index(embeddings, metadata, texts)
    t0 = time.perf_counter()
    faiss_index = indexer.build_faiss_index(multi_vector_index, index_type=args.faiss_type)
    faiss_seconds = time.perf_counter() - t0

    # Write indices to a scratch dir only to measure on-disk size
    faiss_path = os.path.join(scratch_dir, "bench_faiss.faiss")
    faiss.write_index(faiss_index, faiss_path)
    bm25_dir = os.path.join(scratch_dir, "bench_bm25s")
    bm25.save(bm25_dir)

    stats["faiss"] = {
        "requested_type": args.faiss_type,
        "resolved_type": type(faiss_index).__name__,
        "build_seconds": round(faiss_seconds, 2),
        "vectors": int(faiss_index.ntotal),
        "size_on_disk_mb": round(os.path.getsize(faiss_path) / 1e6, 1),
    }
    stats["bm25"]["size_on_disk_mb"] = round(_dir_size(bm25_dir) / 1e6, 1)
    shutil.rmtree(scratch_dir, ignore_errors=True)

    rss_after = _rss_mb()
    if rss_before is not None and rss_after is not None:
        stats["rss_delta_mb"] = round(rss_after - rss_before, 1)

    total = bm25_seconds + emb_seconds + faiss_seconds
    stats["total_seconds"] = round(total, 2)
    stats["total_docs_per_second"] = round(len(texts) / total, 1)
    return stats, bm25, faiss_index


# ---------------------------------------------------------------------------
# Retrieval methods (mirror Retriever.retrieve_resolved without the LLM)
# ---------------------------------------------------------------------------

def make_searchers(embedding_model, faiss_index, bm25, candidates):
    """Return {method: search_fn(query) -> [(doc_idx, score), ...]} (descending)."""

    def bm25_search(query, k=candidates):
        tokens = tokenize_for_bm25(
            query,
            enable_stemming=BM25_ENABLE_STEMMING,
            enable_stopwords=BM25_ENABLE_STOPWORDS,
            languages=BM25_LANGUAGES,
        )
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top = np.argsort(scores)[-k:][::-1]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def dense_search(query, k=candidates):
        prefixed = prepare_for_embedding(query, is_query=True, use_prefix=EMBEDDING_USE_PREFIX)
        emb = embedding_model.encode([prefixed])
        vec = np.asarray(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        distances, indices = faiss_index.search(vec, k)
        return [(int(i), float(d)) for d, i in zip(distances[0], indices[0]) if i != -1]

    def hybrid_search(query, k=candidates):
        # Same fusion as Retriever.retrieve_resolved: RRF over dense + sparse
        rrf = defaultdict(float)
        for rank, (idx, _) in enumerate(dense_search(query, k)):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_search(query, k)):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]

    return {"bm25": bm25_search, "dense": dense_search, "hybrid": hybrid_search}


def benchmark_retrieval(method, search_fn, queries, doc_ids, run_depth,
                        reranker=None, texts=None, rerank_candidates=None):
    """Run every query through `search_fn`, timing each call.

    Returns (run, latency_stats) where run maps query ids to
    {doc_id: score} for metric evaluation.
    """
    run = {}
    latencies = []
    # Untimed warm-up so first-call costs (MPS kernel compilation, model
    # warm-up) don't skew the latency percentiles.
    first_query = next(iter(queries.values()), None)
    if first_query is not None:
        warm = search_fn(first_query)
        if reranker is not None and warm:
            reranker.predict([[first_query, texts[warm[0][0]]]])
    for qid, query in tqdm(queries.items(), desc=f"Retrieval [{method}]"):
        t0 = time.perf_counter()
        results = search_fn(query)
        if reranker is not None and results:
            # Rerank only the head of the candidate pool. The pool sets the
            # recall ceiling; the cross-encoder only reorders within it, and it
            # dominates latency (~99% of it). Keeping the untouched tail below
            # the reranked head leaves deep metrics (Recall@100) comparable
            # across widths instead of silently truncating the pool.
            width = rerank_candidates or len(results)
            head, tail = results[:width], results[width:]
            scores = reranker.predict([[query, texts[idx]] for idx, _ in head])
            head = sorted(
                ((idx, float(s)) for (idx, _), s in zip(head, scores)),
                key=lambda x: x[1], reverse=True,
            )
            # Keep the tail strictly below the head regardless of score scale
            floor = min((s for _, s in head), default=0.0)
            results = head + [(idx, floor - 1e-6 * (rank + 1))
                              for rank, (idx, _) in enumerate(tail)]
        latencies.append(time.perf_counter() - t0)
        run[qid] = {doc_ids[idx]: score for idx, score in results[:run_depth]}
    return run, _latency_stats(latencies)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results, k_values):
    headline = [f"nDCG@{min(10, max(k_values))}", f"Recall@{max(k_values)}",
                "MRR@10", "P@10", f"MAP@{max(k_values)}"]
    headline = [m for m in headline
                if any(m in r["quality"] for r in results["methods"].values())]
    cols = ["method"] + headline + ["p50 ms", "p95 ms", "QPS"]
    rows = []
    for method, r in results["methods"].items():
        row = [method]
        row += [f"{r['quality'].get(m, float('nan')):.4f}" for m in headline]
        row += [f"{r['latency']['p50_ms']:.1f}",
                f"{r['latency']['p95_ms']:.1f}",
                f"{r['latency']['qps']:.1f}"]
        rows.append(row)

    widths = [max(len(str(x)) for x in [c] + [row[i] for row in rows])
              for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print("\n=== Results:", results["dataset"], "===")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))

    idx = results["indexing"]
    print(f"\nIndexing: {idx['corpus_docs']} docs | "
          f"BM25 {idx['bm25']['build_seconds']}s | "
          f"embeddings {idx['embeddings']['build_seconds']}s "
          f"({idx['embeddings']['docs_per_second']} docs/s) | "
          f"FAISS [{idx['faiss']['resolved_type']}] {idx['faiss']['build_seconds']}s | "
          f"disk: faiss {idx['faiss']['size_on_disk_mb']} MB, "
          f"bm25 {idx['bm25']['size_on_disk_mb']} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the ALMA indexer/retriever on BEIR datasets.",
        epilog="Known datasets: " + "; ".join(
            f"{k} ({v[0]})" for k, v in KNOWN_DATASETS.items()),
    )
    parser.add_argument("--dataset", default=BENCHMARK_DATASET,
                        help=f"BEIR dataset name (default from config: {BENCHMARK_DATASET})")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="where datasets are downloaded/cached")
    parser.add_argument("--methods", nargs="+", default=list(BENCHMARK_METHODS),
                        choices=ALL_METHODS,
                        help="retrieval methods to benchmark (hybrid_rerank loads the cross-encoder)")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(BENCHMARK_K_VALUES),
                        help="metric cutoffs (default from config: "
                             f"{' '.join(map(str, BENCHMARK_K_VALUES))})")
    parser.add_argument("--candidates", type=int, default=None,
                        help="candidate pool per method "
                             "(default: max(RETRIEVAL_TOP_K, max k-value))")
    parser.add_argument("--rerank-candidates", type=int, default=RERANK_CANDIDATES,
                        help="how many of the fused candidates the cross-encoder "
                             "scores (default from config: "
                             f"{RERANK_CANDIDATES}); the rest keep their fused "
                             "order below the reranked head")
    parser.add_argument("--max-queries", type=int, default=BENCHMARK_MAX_QUERIES,
                        help="limit number of queries (quick runs)")
    parser.add_argument("--max-corpus", type=int, default=BENCHMARK_MAX_CORPUS,
                        help="subsample corpus to N docs, always keeping judged docs "
                             "(0 = force full corpus; default: full corpus, or an "
                             "automatic subsample when --max-queries is set)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="embedding batch size (default: 64)")
    parser.add_argument("--faiss-type", default=FAISS_INDEX_TYPE,
                        choices=["flat", "ivf", "hnsw", "auto"],
                        help=f"FAISS index type (default from config: {FAISS_INDEX_TYPE})")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME,
                        help="override the embedding model (A/B testing); note: "
                             "E5 query:/passage: prefixes still follow config "
                             "EMBEDDING_USE_PREFIX")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL_NAME,
                        help="override the cross-encoder reranker (A/B testing)")
    parser.add_argument("--output-dir", default=RESULTS_DIR,
                        help="where result JSON files are written")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = args.candidates or max(RETRIEVAL_TOP_K, max(args.k_values))
    run_depth = max(args.k_values)

    print(f"Loading BEIR dataset '{args.dataset}' (split={args.split})...")
    corpus, queries, qrels = load_beir_dataset(args.dataset, args.data_dir, args.split)
    # Filter queries first so corpus subsampling only needs to keep the
    # judged docs of the queries actually being evaluated.
    if args.max_queries and args.max_queries < len(queries):
        keep = random.Random(args.seed).sample(sorted(queries), args.max_queries)
        queries = {qid: queries[qid] for qid in keep}
        qrels = {qid: qrels[qid] for qid in keep}
    max_corpus = args.max_corpus
    if max_corpus is None and args.max_queries:
        # Evaluating a subset of queries: no point embedding millions of docs.
        # Keep all judged docs plus random distractors (20x queries, min 10K).
        auto = max(AUTO_DISTRACTORS_PER_QUERY * len(queries), AUTO_MIN_CORPUS)
        if len(corpus) > auto:
            max_corpus = auto
            print(f"--max-queries set: auto-subsampling corpus from {len(corpus)} to "
                  f"{max_corpus} docs (all judged docs kept + random distractors).\n"
                  f"Pass --max-corpus to override (0 = full corpus). "
                  f"Quality numbers are optimistic vs the full corpus.")
    if max_corpus:
        corpus = _subsample_corpus(corpus, qrels, max_corpus, args.seed)
    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | "
          f"Judged pairs: {sum(len(r) for r in qrels.values())}")

    # BEIR convention: retrieve over "title text"
    doc_ids = list(corpus)
    texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).strip() for d in doc_ids]
    metadata = [{"document_name": d} for d in doc_ids]

    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = load_sentence_transformer(args.embedding_model)

    os.makedirs(args.output_dir, exist_ok=True)
    scratch_dir = os.path.join(args.output_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    indexer = Indexer()  # only its build_* methods are used; nothing is saved to config paths
    indexing_stats, bm25, faiss_index = benchmark_indexing(
        indexer, texts, metadata, embedding_model, args, scratch_dir)

    searchers = make_searchers(embedding_model, faiss_index, bm25, candidates)
    reranker = None
    if "hybrid_rerank" in args.methods:
        print(f"Loading reranker: {args.reranker_model}")
        reranker = load_cross_encoder(args.reranker_model)

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "config": {
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model if reranker else None,
            "embedding_use_prefix": EMBEDDING_USE_PREFIX,
            "bm25_stemming": BM25_ENABLE_STEMMING,
            "bm25_stopwords": BM25_ENABLE_STOPWORDS,
            "bm25_languages": BM25_LANGUAGES,
            "rrf_k": RRF_K,
            "candidates": candidates,
            "rerank_candidates": args.rerank_candidates,
            "k_values": args.k_values,
            "max_queries": args.max_queries,
            "max_corpus": max_corpus,
            "corpus_docs": len(corpus),
        },
        "indexing": indexing_stats,
        "methods": {},
    }

    print("\n=== Retrieval benchmark ===")
    for method in args.methods:
        if method == "hybrid_rerank":
            run, latency = benchmark_retrieval(
                method, searchers["hybrid"], queries, doc_ids, run_depth,
                reranker=reranker, texts=texts,
                rerank_candidates=args.rerank_candidates)
        else:
            run, latency = benchmark_retrieval(
                method, searchers[method], queries, doc_ids, run_depth)
        quality = evaluate_run(run, qrels, args.k_values)
        results["methods"][method] = {"quality": quality, "latency": latency}

    peak = _peak_rss_mb()
    results["peak_rss_mb"] = round(peak, 1) if peak is not None else None

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_summary(results, args.k_values)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
