#!/usr/bin/env python
"""Hyperparameter sweep: RRF_K x candidate pool size -> retrieval quality.

Builds the indices once, computes each query's dense and BM25 rankings once
(at the largest candidate pool), then re-fuses them per grid cell — so the
whole sweep costs roughly one benchmark run. Each cell is the mean nDCG@10
of hybrid RRF retrieval at that (rrf_k, candidates) combination.

Results JSON goes to benchmark/results/sweeps/ (kept out of results/ so the
regular figure regeneration doesn't mistake it for a benchmark run), and the
heatmap benchmark_<dataset>_sweep.pdf is rendered immediately.

Usage (from the repository root):
    python -m benchmark.sweep_rrf \
        --dataset scifact
"""
import argparse
import datetime
import json
import os
import random
import sys
from collections import defaultdict

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from tqdm import tqdm

from core.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_USE_PREFIX,
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    RETRIEVAL_TOP_K, RRF_K, BENCHMARK_DATASET,
)
from core.utils import (
    load_sentence_transformer, tokenize_for_bm25, prepare_for_embedding,
)
from components.indexer import Indexer
from benchmark.beir_data import load_beir_dataset, DEFAULT_DATA_DIR
from benchmark.metrics import evaluate_run
from benchmark.run_benchmark import (
    _subsample_corpus, AUTO_DISTRACTORS_PER_QUERY, AUTO_MIN_CORPUS,
)

SWEEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "sweeps")

DEFAULT_RRF_K = [5, 10, 20, 30, 40, 60, 80, 120]
DEFAULT_CANDIDATES = [10, 20, 30, 50, 75, 100, 150, 200]


def compute_rankings(corpus, queries, batch_size, max_candidates):
    """Build indices once; return per-query dense and BM25 rankings
    (doc indices, best first) plus the doc id list."""
    doc_ids = list(corpus)
    texts = [(corpus[d]["title"] + " " + corpus[d]["text"]).strip() for d in doc_ids]
    metadata = [{"document_name": d} for d in doc_ids]

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = load_sentence_transformer(EMBEDDING_MODEL_NAME)

    indexer = Indexer()
    bm25 = indexer.build_bm25_index(texts)
    embeddings = indexer.get_embeddings(texts, model=model, batch_size=batch_size)
    multi_vector = indexer.build_multi_vector_index(embeddings, metadata, texts)
    faiss_index = indexer.build_faiss_index(multi_vector)

    qids = list(queries)
    print("Encoding queries...")
    prefixed = [prepare_for_embedding(queries[q], is_query=True,
                                      use_prefix=EMBEDDING_USE_PREFIX) for q in qids]
    q_vecs = np.asarray(model.encode(prefixed, batch_size=batch_size),
                        dtype="float32")
    faiss.normalize_L2(q_vecs)
    _, dense_idx = faiss_index.search(q_vecs, max_candidates)

    dense_rankings, bm25_rankings = {}, {}
    for row, qid in enumerate(tqdm(qids, desc="BM25 rankings")):
        dense_rankings[qid] = [int(i) for i in dense_idx[row] if i != -1]
        tokens = tokenize_for_bm25(
            queries[qid], enable_stemming=BM25_ENABLE_STEMMING,
            enable_stopwords=BM25_ENABLE_STOPWORDS, languages=BM25_LANGUAGES)
        if tokens:
            scores = bm25.get_scores(tokens)
            top = np.argsort(scores)[-max_candidates:][::-1]
            bm25_rankings[qid] = [int(i) for i in top if scores[i] > 0]
        else:
            bm25_rankings[qid] = []
    return doc_ids, dense_rankings, bm25_rankings


def fuse_and_evaluate(doc_ids, dense_rankings, bm25_rankings, qrels,
                      rrf_k, candidates, metric_k):
    """RRF-fuse the precomputed rankings at (rrf_k, candidates); return nDCG."""
    run = {}
    for qid in dense_rankings:
        scores = defaultdict(float)
        for ranking in (dense_rankings[qid], bm25_rankings[qid]):
            for rank, idx in enumerate(ranking[:candidates]):
                scores[idx] += 1.0 / (rrf_k + rank + 1)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:metric_k]
        run[qid] = {doc_ids[idx]: s for idx, s in top}
    return evaluate_run(run, qrels, [metric_k])[f"nDCG@{metric_k}"]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep RRF_K x candidate pool; heatmap the hybrid quality.")
    parser.add_argument("--dataset", default=BENCHMARK_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--rrf-k", nargs="+", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--candidates", nargs="+", type=int,
                        default=DEFAULT_CANDIDATES)
    parser.add_argument("--metric-k", type=int, default=10,
                        help="nDCG cutoff per cell (default 10)")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading BEIR dataset '{args.dataset}' (split={args.split})...")
    corpus, queries, qrels = load_beir_dataset(args.dataset, args.data_dir, args.split)
    if args.max_queries and args.max_queries < len(queries):
        keep = random.Random(args.seed).sample(sorted(queries), args.max_queries)
        queries = {qid: queries[qid] for qid in keep}
        qrels = {qid: qrels[qid] for qid in keep}
    max_corpus = args.max_corpus
    if max_corpus is None and args.max_queries:
        auto = max(AUTO_DISTRACTORS_PER_QUERY * len(queries), AUTO_MIN_CORPUS)
        if len(corpus) > auto:
            max_corpus = auto
            print(f"Auto-subsampling corpus to {max_corpus} docs.")
    if max_corpus:
        corpus = _subsample_corpus(corpus, qrels, max_corpus, args.seed)
    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)}")

    rrf_ks = sorted(args.rrf_k)
    cands = sorted(args.candidates)
    doc_ids, dense_rankings, bm25_rankings = compute_rankings(
        corpus, queries, args.batch_size, max(cands))

    print(f"Evaluating {len(rrf_ks)}x{len(cands)} grid...")
    cells = [
        [fuse_and_evaluate(doc_ids, dense_rankings, bm25_rankings, qrels,
                           K, c, args.metric_k) for c in cands]
        for K in tqdm(rrf_ks, desc="Sweep")
    ]

    flat = [(cells[i][j], rrf_ks[i], cands[j])
            for i in range(len(rrf_ks)) for j in range(len(cands))]
    best_v, best_k, best_c = max(flat)
    sweep = {
        "type": "sweep",
        "dataset": args.dataset,
        "split": args.split,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "metric": f"nDCG@{args.metric_k}",
        "rrf_k_values": rrf_ks,
        "candidate_values": cands,
        "cells": cells,  # rows = rrf_k, cols = candidates
        "baseline": {"rrf_k": RRF_K, "candidates": RETRIEVAL_TOP_K},
        "best": {"rrf_k": best_k, "candidates": best_c, "value": best_v},
        "queries": len(queries),
        "corpus_docs": len(corpus),
    }

    os.makedirs(SWEEPS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SWEEPS_DIR, f"{args.dataset}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    print(f"Sweep saved to {out_path}")
    print(f"Best cell: nDCG@{args.metric_k}={best_v:.4f} at "
          f"RRF_K={best_k}, candidates={best_c} "
          f"(config default: RRF_K={RRF_K}, candidates={RETRIEVAL_TOP_K})")

    from benchmark.plot_results import plot_sweep, DEFAULT_OUT
    from benchmark.figstyle import apply_style
    apply_style()
    print(f"  wrote {plot_sweep(sweep, DEFAULT_OUT)}")


if __name__ == "__main__":
    main()
