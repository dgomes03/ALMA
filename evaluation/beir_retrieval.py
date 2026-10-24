"""BEIR-style retrieval evaluation on the SciFact dataset.

Evaluates retriever quality purely at the retrieval level — no answer
generation or LLM judge is needed.  The SciFact corpus (~5 K passages,
300 test queries) is loaded from the HuggingFace Hub (mteb/scifact),
indexed into the retriever, and then each query is run through the
retrieval pipeline.  Retrieved document IDs are compared against the
official relevance judgements (qrels) to compute standard IR metrics.

Metrics computed for each configured k value:
  - nDCG@k        — ranking quality; standard for BEIR/MTEB comparisons
  - Recall@k      — did the relevant document appear in the top-k results?
  - Precision@k   — fraction of top-k results that are relevant
  - MRR           — mean reciprocal rank of the first relevant result

Usage (via run_evaluation.py):
    python -m evaluation.run_evaluation --mode beir-retrieval
    python -m evaluation.run_evaluation --mode beir-retrieval --max-questions 100
    python -m evaluation.run_evaluation --mode beir-retrieval --k-values 1 5 10 100
"""

import json
import os
import subprocess
from datetime import datetime

from huggingface_hub import hf_hub_download

from core.config import EVAL_OUTPUT_DIR
from evaluation.lm_free_metrics import compute_ir_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scifact_beir(max_questions=None):
    """Load SciFact queries, corpus, and qrels from mteb/scifact on HuggingFace.

    Returns:
        queries  — list of dicts  {query_id, question}
        corpus   — dict[doc_id -> str]  (title + abstract)
        qrels    — dict[query_id -> set[doc_id]]
    """
    print("Downloading SciFact queries...")
    qpath = hf_hub_download("mteb/scifact", "queries.jsonl", repo_type="dataset")
    print("Downloading SciFact corpus...")
    cpath = hf_hub_download("mteb/scifact", "corpus.jsonl", repo_type="dataset")

    queries = []
    with open(qpath) as f:
        for line in f:
            row = json.loads(line)
            queries.append({
                "query_id": str(row["_id"]),
                "question": row["text"],
            })

    corpus = {}
    with open(cpath) as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title", "")
            text = row.get("text", "")
            corpus[str(row["_id"])] = f"{title}. {text}" if title else text

    # Load relevance judgements (qrels/test.tsv)
    qrels = {}
    try:
        qrels_path = hf_hub_download("mteb/scifact", "qrels/test.tsv", repo_type="dataset")
        with open(qrels_path) as f:
            next(f)  # skip header row
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qid, doc_id = str(parts[0]), str(parts[1])
                    qrels.setdefault(qid, set()).add(doc_id)
    except Exception as exc:
        print(f"WARNING: Could not load SciFact qrels — IR metrics will be skipped. ({exc})")

    # Filter queries to those with qrels so we don't evaluate unjudged queries
    if qrels:
        queries = [q for q in queries if q["query_id"] in qrels]
        if max_questions:
            queries = queries[:max_questions]

    print(
        f"Loaded {len(queries)} queries with relevance labels, "
        f"{len(corpus)} corpus passages."
    )
    return queries, corpus, qrels


# ---------------------------------------------------------------------------
# Corpus indexing
# ---------------------------------------------------------------------------

_BEIR_CACHE_DIR = os.path.join(EVAL_OUTPUT_DIR, "beir_scifact_index_cache")


def _index_corpus(corpus, retriever):
    """Index the SciFact corpus into the retriever (FAISS + BM25).

    The built index is cached to disk so that subsequent runs skip the
    4-minute embedding step entirely.
    """
    import bm25s
    import faiss as _faiss
    from components.indexer import Indexer
    from core.utils import save_pickle, load_pickle

    # Disable query expansion — no LLM available in this mode.
    retriever.expand_query = lambda query: []

    cache_dir = _BEIR_CACHE_DIR
    mv_path   = os.path.join(cache_dir, "multi_vector_index.pkl")
    bm25_path = os.path.join(cache_dir, "bm25s_index")  # directory (bm25s native format)
    meta_path = os.path.join(cache_dir, "metadata_index.pkl")
    faiss_path = os.path.join(cache_dir, "faiss.index")

    cache_exists = all(os.path.exists(p) for p in (mv_path, bm25_path, meta_path, faiss_path))

    if cache_exists:
        print(f"Loading cached SciFact index from {cache_dir} ...")
        multi_vector_index = load_pickle(mv_path)
        bm25               = bm25s.BM25.load(bm25_path)
        metadata_index     = load_pickle(meta_path)
        faiss_index        = _faiss.read_index(faiss_path)
        print(f"Loaded {len(multi_vector_index)} passages from cache.")
    else:
        texts    = list(corpus.values())
        doc_ids  = list(corpus.keys())
        metadata = [
            {"document_name": did, "page": 0, "chunk_idx": i}
            for i, did in enumerate(doc_ids)
        ]

        indexer = Indexer()
        print(f"Indexing {len(texts)} SciFact passages (this runs once and is then cached)...")
        embeddings         = indexer.get_embeddings(texts, model=retriever.embedding_model)
        multi_vector_index = indexer.build_multi_vector_index(embeddings, metadata, texts)
        bm25               = indexer.build_bm25_index(texts)
        faiss_index        = indexer.build_faiss_index(multi_vector_index)
        metadata_index     = indexer.build_metadata_index(metadata)

        os.makedirs(cache_dir, exist_ok=True)
        save_pickle(multi_vector_index, mv_path)
        bm25.save(bm25_path)
        save_pickle(metadata_index, meta_path)
        _faiss.write_index(faiss_index, faiss_path)
        print(f"Index cached to {cache_dir}")

    retriever.update_indices(multi_vector_index, bm25, metadata_index, faiss_index)
    print("Corpus ready.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _get_git_info():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL
        ) != 0
        return sha, dirty
    except Exception:
        return "unknown", False


def run_beir_retrieval_evaluation(retriever, max_questions=None, k_values=(1, 5, 10, 100)):
    """Evaluate the retriever on SciFact using BEIR-style IR metrics.

    Args:
        retriever:     Initialised Retriever instance (no LLM required).
        max_questions: Optional cap on the number of queries to evaluate.
        k_values:      Iterable of k values for nDCG@k, Recall@k, Precision@k.
    """
    k_values = tuple(sorted(set(k_values)))  # deduplicate and sort

    print(f"\n{'='*60}")
    print(f"BEIR Retrieval Evaluation — SciFact")
    print(f"Metrics: nDCG, Recall, Precision @ {list(k_values)} + MRR")
    print(f"{'='*60}\n")

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load dataset ----
    queries, corpus, qrels = _load_scifact_beir(max_questions)
    if not qrels:
        print("ERROR: No relevance labels loaded. Cannot compute IR metrics.")
        return

    # ---- Index corpus ----
    _index_corpus(corpus, retriever)

    # ---- Retrieve for each query ----
    n = len(queries)
    retrieval_results = {}  # query_id -> [doc_id, ...]

    print(f"\nRunning retrieval for {n} queries...")
    for i, item in enumerate(queries):
        qid = item["query_id"]
        question = item["question"]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {question[:80]}...")

        try:
            retriever.combined_retrieval_with_chunks(question)
            # Each metadata entry has the doc_id stored as "document_name"
            retrieved_ids = [
                meta.get("document_name", "")
                for meta in retriever.last_retrieved_metadata
            ]
        except Exception as exc:
            print(f"    WARNING: retrieval failed for query {qid}: {exc}")
            retrieved_ids = []

        retrieval_results[qid] = retrieved_ids

    # ---- Compute IR metrics ----
    print("\nComputing IR metrics...")
    ir_metrics = compute_ir_metrics(retrieval_results, qrels, k_values=k_values)

    # ---- Build per-query results ----
    per_query = []
    for item in queries:
        qid = item["query_id"]
        retrieved = retrieval_results.get(qid, [])
        relevant = qrels.get(qid, set())
        query_metrics = compute_ir_metrics({qid: retrieved}, {qid: relevant}, k_values=k_values)
        per_query.append({
            "query_id": qid,
            "question": item["question"],
            "relevant_doc_ids": sorted(relevant),
            "retrieved_doc_ids": retrieved,
            "scores": query_metrics,
        })

    # ---- Output ----
    git_sha, git_dirty = _get_git_info()
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": "beir-retrieval",
            "dataset": "scifact",
            "num_queries": n,
            "k_values": list(k_values),
            "git_commit": git_sha,
            "git_dirty": git_dirty,
        },
        "aggregate_scores": ir_metrics,
        "per_query_results": per_query,
    }

    out_path = os.path.join(EVAL_OUTPUT_DIR, f"beir_scifact_{ts}_{git_sha}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ---- Print summary ----
    print(f"\n{'='*60}")
    dirty_tag = " (uncommitted changes)" if git_dirty else ""
    print(f"BEIR Retrieval Results — SciFact ({n} queries) — git: {git_sha}{dirty_tag}")
    print(f"{'='*60}")
    metric_order = []
    for k in k_values:
        metric_order += [f"ndcg@{k}", f"recall@{k}", f"precision@{k}"]
    metric_order.append("mrr")
    for metric in metric_order:
        score = ir_metrics.get(metric)
        if score is not None:
            print(f"  {metric:<18} {score:.4f}")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output
