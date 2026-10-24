"""TREC-style retrieval quality metrics.

Implements the metrics reported by pytrec_eval / the BEIR benchmark
(ndcg_cut, map_cut, recall, P, recip_rank) so results are directly
comparable with published BEIR numbers, without requiring the
pytrec_eval C extension.

Conventions (matching trec_eval):
- A document is *relevant* when its qrel score is > 0.
- nDCG uses linear gain: rel / log2(rank + 1).
- MAP divides by the total number of relevant documents (not min(R, k)).
- Ties in run scores are broken by doc id (descending score, ascending id)
  for deterministic results.
- A document whose id equals the query id is dropped before scoring, matching
  BEIR's `ignore_identical_ids=True` default. This matters enormously on
  ArguAna, where 92% of queries ARE corpus documents (the query text and the
  same-id document have ~0.81 text similarity): without this the query's own
  text is retrieved at rank 1, is never in the qrels, and pushes the real
  counter-argument down. A query's own id is never in its own qrels on any
  dataset here, so dropping it can never discard a relevant document.
"""
import math
from collections import defaultdict


def _ranking(doc_scores):
    """Sort a {doc_id: score} run for one query into a ranked doc id list."""
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))]


def ndcg_at_k(ranked, rels, k):
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        rel = rels.get(doc, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked, relevant, k):
    return len(relevant.intersection(ranked[:k])) / k


def reciprocal_rank_at_k(ranked, relevant, k):
    for i, doc in enumerate(ranked[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(run, qrels, k_values, ignore_identical_ids=True):
    """Evaluate a retrieval run against relevance judgements.

    Args:
        run: {query_id: {doc_id: score}} retrieved results.
        qrels: {query_id: {doc_id: relevance}} ground truth.
        k_values: iterable of cutoffs, e.g. [1, 3, 5, 10, 100].
        ignore_identical_ids: drop a retrieved doc whose id equals the query id
            (BEIR's default). Required for comparability with published
            numbers; see the module docstring for why ArguAna depends on it.

    Returns:
        {metric_name: value} averaged over the queries present in qrels
        that have at least one relevant document.
    """
    sums = defaultdict(float)
    n_queries = 0

    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        n_queries += 1
        doc_scores = run.get(qid, {})
        if ignore_identical_ids and qid in doc_scores:
            doc_scores = {d: s for d, s in doc_scores.items() if d != qid}
        ranked = _ranking(doc_scores)
        for k in k_values:
            sums[f"nDCG@{k}"] += ndcg_at_k(ranked, rels, k)
            sums[f"MAP@{k}"] += average_precision_at_k(ranked, relevant, k)
            sums[f"Recall@{k}"] += recall_at_k(ranked, relevant, k)
            sums[f"P@{k}"] += precision_at_k(ranked, relevant, k)
            sums[f"MRR@{k}"] += reciprocal_rank_at_k(ranked, relevant, k)

    if n_queries == 0:
        return {}
    return {name: round(total / n_queries, 5) for name, total in sums.items()}
