# Indexer / Retriever Benchmark

Benchmarks the framework's indexing and retrieval pipeline on standard IR
benchmarks from [BEIR](https://github.com/beir-cellar/beir), using the
framework's **own components and config**: `Indexer.build_bm25_index`,
`Indexer.get_embeddings`, `Indexer.build_faiss_index`,
`tokenize_for_bm25`, E5 query/passage prefixes, and the same RRF fusion as
`Retriever.retrieve_resolved`.

Datasets are downloaded automatically from the official BEIR mirror on first
use and cached in `benchmark/datasets/` — no `beir` or `pytrec_eval`
installation required (metrics are implemented in `metrics.py` following
trec_eval/BEIR conventions, so numbers are comparable with published results).

## What is measured

**Indexing** — BM25 build time, embedding throughput (docs/s), FAISS build
time + resolved index type (flat/IVF/HNSW auto-selection), on-disk index
sizes, RAM delta and peak RSS.

**Retrieval** — for each method, quality against the qrels
(nDCG@k, MAP@k, Recall@k, P@k, MRR@k) and per-query latency
(mean/p50/p95/p99 ms, QPS; dense latency includes query encoding):

| method | pipeline |
|---|---|
| `bm25` | sparse only (bm25s + framework tokenizer) |
| `dense` | embedding model + FAISS only |
| `hybrid` | dense + bm25 fused with RRF (the framework default) |
| `hybrid_rerank` | hybrid + cross-encoder reranking of the candidate pool |

LLM query expansion and hierarchical parent resolution are intentionally
excluded — they need a loaded LLM / parent store and are orthogonal to raw
indexer/retriever performance. The reranker confidence threshold is also not
applied, since quality metrics require a full ranking.

## Usage

From the repository root:

```bash
# Default: SciFact (5K docs, 300 queries), bm25 + dense + hybrid
python -m benchmark.run_benchmark

# Include cross-encoder reranking
python -m benchmark.run_benchmark \
    --dataset scifact --methods bm25 dense hybrid hybrid_rerank

# Another dataset
python -m benchmark.run_benchmark --dataset nfcorpus

# Quick smoke test (subsampled corpus keeps all judged docs)
python -m benchmark.run_benchmark \
    --dataset scifact --max-corpus 1000 --max-queries 50
```

Results are printed as a table and written as JSON to `benchmark/results/`.

## Recommended datasets

| dataset | size | domain |
|---|---|---|
| `scifact` | 5K docs, 300 queries | scientific claim verification |
| `nfcorpus` | 3.6K docs, 323 queries | medical / nutrition |
| `arguana` | 8.7K docs, 1406 queries | argument retrieval |
| `scidocs` | 25K docs, 1000 queries | citation prediction |
| `fiqa` | 57K docs, 648 queries | financial QA |
| `trec-covid` | 171K docs, 50 queries | COVID literature |

Any other BEIR dataset name works too (`quora`, `nq`, `msmarco`, …) but the
larger ones take hours to embed locally. `msmarco` uses `--split dev`.

## Options

| flag | default | description |
|---|---|---|
| `--dataset` | `scifact` | BEIR dataset name |
| `--split` | `test` | qrels split (`train`/`dev`/`test`) |
| `--methods` | `bm25 dense hybrid` | methods to benchmark |
| `--k-values` | `1 3 5 10 100` | metric cutoffs |
| `--candidates` | `max(RETRIEVAL_TOP_K, max k)` | candidate pool per query |
| `--max-queries` | all | limit queries; also auto-subsamples a huge corpus (see below) |
| `--max-corpus` | all | subsample corpus (judged docs always kept); `0` forces the full corpus |
| `--batch-size` | 64 | embedding batch size |
| `--faiss-type` | config (`auto`) | `flat` / `ivf` / `hnsw` / `auto` |
| `--output-dir` | `benchmark/results` | JSON output location |

## Notes

- `core/config.py` pins all BLAS/OMP thread pools to 1 at import time; the
  benchmark inherits this on purpose, so timings reflect the framework as it
  actually runs.
- Corpus documents are indexed as `"title text"` (BEIR convention).
- With `--max-queries` and no explicit `--max-corpus`, the corpus is
  automatically subsampled to `max(20 × queries, 10 000)` docs so that e.g.
  `nq` (2.68M docs) doesn't require embedding the whole corpus for a
  500-query run. All judged docs of the kept queries are always retained;
  the rest are random distractors. Pass `--max-corpus 0` to force the full
  corpus.
- With any corpus subsampling, quality numbers are optimistic (fewer
  distractors); use the full corpus for reportable results.
