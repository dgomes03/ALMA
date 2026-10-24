"""LLM-free evaluation metrics for RAG benchmarking.

Three categories:
  1. Proxy metrics  — no ground truth, always available
  2. Reference-based — need reference answers (ROUGE, semantic similarity)
  3. IR metrics      — need relevance labels / qrels (NDCG@K, MRR)
"""

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------------------------
# 1. Proxy metrics (no ground truth needed)
# ---------------------------------------------------------------------------

def answer_groundedness(answer: str, context: str) -> float:
    """Token F1 between answer and retrieved context.

    Proxy for faithfulness: measures how much of the answer is
    lexically grounded in the retrieved context (no LLM needed).
    Returns a value in [0, 1].
    """
    pred = Counter(_tokenize(answer))
    ref = Counter(_tokenize(context))
    if not pred or not ref:
        return 0.0
    common = sum((pred & ref).values())
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def context_relevance(query_embedding, chunk_embeddings) -> float:
    """Mean cosine similarity between query and retrieved chunk embeddings.

    Proxy for retrieval quality: how semantically close the retrieved
    chunks are to the query. Returns a value in [-1, 1].
    """
    import numpy as np
    if not chunk_embeddings:
        return 0.0
    q = np.array(query_embedding, dtype=float)
    q /= np.linalg.norm(q) + 1e-9
    sims = []
    for emb in chunk_embeddings:
        c = np.array(emb, dtype=float)
        c /= np.linalg.norm(c) + 1e-9
        sims.append(float(q @ c))
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# 2. Reference-based metrics (need ground truth answers)
# ---------------------------------------------------------------------------

def compute_rouge(candidates, references):
    """ROUGE-1, ROUGE-2, ROUGE-L F1 averaged over all (candidate, reference) pairs.

    Requires: pip install rouge-score

    Returns:
        dict with 'rouge1', 'rouge2', 'rougeL' — mean F1 floats (or None on error).
    """
    try:
        from rouge_score import rouge_scorer as _rs
    except ImportError:
        print("WARNING: rouge_score not installed. Run: pip install rouge-score")
        return {"rouge1": None, "rouge2": None, "rougeL": None}

    scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for cand, ref in zip(candidates, references):
        s = scorer.score(ref, cand)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1) if r1 else None,
        "rouge2": sum(r2) / len(r2) if r2 else None,
        "rougeL": sum(rl) / len(rl) if rl else None,
    }


def compute_semantic_similarity(candidates, references, embed_fn):
    """Cosine similarity between answer and reference using the existing embedding model.

    Semantically measures answer accuracy without a generative LLM judge.
    Reuses the retrieval embedding model so no extra model download is needed.

    Args:
        embed_fn: callable(str) -> list[float]  (e.g. ragas_embeddings.embed_text)

    Returns:
        dict with 'semantic_similarity' — mean cosine similarity.
    """
    import numpy as np

    def _cos(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 1e-9 else 0.0

    sims = [_cos(embed_fn(c), embed_fn(r)) for c, r in zip(candidates, references)]
    return {"semantic_similarity": sum(sims) / len(sims) if sims else None}


def compute_per_sample_rouge(candidate, reference):
    """ROUGE-L F1 for a single (candidate, reference) pair. Returns float or None."""
    try:
        from rouge_score import rouge_scorer as _rs
    except ImportError:
        return None
    scorer = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# 3. IR metrics (need qrels — relevance labels)
# ---------------------------------------------------------------------------

def _dcg_at_k(retrieved_ids, relevant_ids, k):
    return sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )


def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    """NDCG@K for a single query."""
    dcg = _dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def mrr_score(retrieved_ids, relevant_ids):
    """MRR for a single query."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@K for a single query: fraction of relevant docs found in top-K."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Precision@K for a single query: fraction of returned results that are relevant.

    The denominator is min(k, len(retrieved_ids)) rather than a fixed k, so that
    a retriever which deliberately returns fewer results (e.g. via a confidence
    threshold) is not penalised for unfilled slots.
    """
    actual_k = min(k, len(retrieved_ids))
    if actual_k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / actual_k


def compute_ir_metrics(retrieval_results, qrels, k_values=(10,)):
    """NDCG@K, Recall@K, Precision@K, and MRR across all queries.

    Args:
        retrieval_results: dict[query_id -> list[doc_id]] ordered by rank
        qrels:             dict[query_id -> set[doc_id]] of relevant doc IDs
        k_values:          tuple of K values for NDCG@K, Recall@K, and Precision@K

    Returns:
        dict with 'ndcg@K', 'recall@K', 'precision@K' per K value and 'mrr',
        all mean floats (or None when no queries with relevance judgements exist).
    """
    ndcg_acc = {k: [] for k in k_values}
    recall_acc = {k: [] for k in k_values}
    precision_acc = {k: [] for k in k_values}
    mrr_acc = []
    for qid, retrieved in retrieval_results.items():
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        for k in k_values:
            ndcg_acc[k].append(ndcg_at_k(retrieved, relevant, k))
            recall_acc[k].append(recall_at_k(retrieved, relevant, k))
            precision_acc[k].append(precision_at_k(retrieved, relevant, k))
        mrr_acc.append(mrr_score(retrieved, relevant))

    result = {}
    for k in k_values:
        result[f"ndcg@{k}"] = sum(ndcg_acc[k]) / len(ndcg_acc[k]) if ndcg_acc[k] else None
        result[f"recall@{k}"] = sum(recall_acc[k]) / len(recall_acc[k]) if recall_acc[k] else None
        result[f"precision@{k}"] = sum(precision_acc[k]) / len(precision_acc[k]) if precision_acc[k] else None
    result["mrr"] = sum(mrr_acc) / len(mrr_acc) if mrr_acc else None
    return result
