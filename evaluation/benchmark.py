"""Benchmark evaluation against standard datasets with ground truth.

Supported datasets:
  - amnesty_qa (explodinggradients/amnesty_qa): 20 Qs, comes with contexts + ground truth.
  - scifact (mteb/scifact): 1109 queries, ~5K passages. Requires corpus indexing.
  - hotpotqa (hotpotqa/hotpot_qa): multi-hop QA, comes with 10 distractor docs per question.
  - ms_marco (microsoft/ms_marco v2.1): real Bing queries with multi-sentence passage answers.
    Use this instead of hotpotqa when ROUGE/factual_correctness matter — hotpotqa's 1-3 word
    answers ("yes", "no", "1954") tank those metrics for systems that generate full explanations.

Metrics computed (no LLM judge required for starred items):
  * NonLLMContextPrecision / NonLLMContextRecall  — RAGAS non-LLM variants
  * ROUGE-1 / ROUGE-2 / ROUGE-L                  — n-gram overlap vs reference
  * Semantic similarity                            — cosine sim answer vs reference (E5)
  * NDCG@10 / MRR                                 — retrieval ranking (scifact only, uses qrels)
    Faithfulness / AnswerRelevancy                — RAGAS LLM-based (kept for reference)
    ContextRecall / ContextPrecision / FactualCorrectness — RAGAS LLM-based
"""

import json
import math
import os
import subprocess
import time
from datetime import datetime

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from core.llm_backend import (
    generate as mlx_generate, make_sampler, make_logits_processors
)
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from core.conversation_manager import ConversationManager
from core.config import (
    EVAL_BENCHMARK_DATASET,
    EVAL_BENCHMARK_DATASET_SUBSET,
    EVAL_MAX_QUESTIONS,
    EVAL_OUTPUT_DIR,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY,
    LLM_REPETITION_CONTEXT_SIZE,
    MAX_RESPONSE_TOKENS,
)
from evaluation.lm_free_metrics import (
    answer_groundedness,
    compute_ir_metrics,
    compute_per_sample_rouge,
    compute_rouge,
    compute_semantic_similarity,
    context_relevance,
)

# Try to import RAGAS non-LLM metrics (available in RAGAS 0.2+)
try:
    from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
    _HAS_NONLLM_RAGAS = True
except ImportError:
    _HAS_NONLLM_RAGAS = False
    print("INFO: NonLLMContextPrecisionWithReference / NonLLMContextRecall not available "
          "in this RAGAS version. Skipping.")


def _get_git_info():
    """Return current git commit SHA and whether the working tree is dirty."""
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


def _generate_answer(query, context, llm_model, llm_tokenizer, system_prompt=None):
    """Generate an answer for a query given retrieved context.

    Simulates the real runtime tool-call flow so the model receives context
    as a tool result (role=tool) rather than raw text in the user message.
    This matches exactly what the model sees in production.
    """
    from tools.tools import get_tools_for_standard_generator

    tools = get_tools_for_standard_generator()
    call_id = "benchmark000"

    conversation = [
        {"role": "system", "content": system_prompt or ConversationManager.SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": f'{{"query": {query!r}}}',
                },
            }],
        },
        {
            "role": "tool",
            "name": "search_documents",
            "content": context,
            "tool_call_id": call_id,
        },
    ]
    prompt = llm_tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tools=tools, tokenize=False
    )
    sampler = make_sampler(temp=LLM_TEMPERATURE, top_p=LLM_TOP_P)
    logits_processors = make_logits_processors(
        repetition_penalty=LLM_REPETITION_PENALTY,
        repetition_context_size=LLM_REPETITION_CONTEXT_SIZE,
    )
    response = mlx_generate(
        llm_model,
        llm_tokenizer,
        prompt=prompt,
        max_tokens=MAX_RESPONSE_TOKENS,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    return response


# ---------------------------------------------------------------------------
# amnesty_qa helpers
# ---------------------------------------------------------------------------

def _load_amnesty_qa(max_questions=None):
    """Load amnesty_qa dataset. Returns list of dicts with question, contexts, ground_truth."""
    ds = load_dataset(EVAL_BENCHMARK_DATASET, EVAL_BENCHMARK_DATASET_SUBSET, split="eval")
    samples = []
    for row in ds:
        samples.append({
            "question": row["user_input"],
            "contexts": row["retrieved_contexts"],
            "ground_truth": row["reference"],
        })
        if max_questions and len(samples) >= max_questions:
            break
    return samples


# ---------------------------------------------------------------------------
# hotpotqa helpers
# ---------------------------------------------------------------------------

def _load_hotpotqa(max_questions=None):
    """Load HotpotQA (distractor setting). Returns list of dicts with question, contexts, ground_truth.

    Each sample comes with 10 documents (2 supporting + 8 distractors), which we use
    directly as retrieved_contexts — no retriever needed.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples = []
    for row in ds:
        # Build one context string per document: "Title. Sentence1 Sentence2 ..."
        contexts = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            doc = title + ". " + " ".join(sentences)
            contexts.append(doc)
        samples.append({
            "question": row["question"],
            "contexts": contexts,
            "ground_truth": row["answer"],
        })
        if max_questions and len(samples) >= max_questions:
            break
    return samples


# ---------------------------------------------------------------------------
# ms_marco helpers
# ---------------------------------------------------------------------------

def _load_ms_marco(max_questions=None):
    """Load MS MARCO v2.1 (validation split). Returns samples with multi-sentence reference answers.

    Each sample has 10 candidate passages pre-provided, so no corpus indexing is needed.
    Unanswerable queries ("No Answer Present.") are skipped.
    """
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    samples = []
    for row in ds:
        answers = row.get("answers", [])
        if not answers or answers == ["No Answer Present."]:
            continue
        passages = row.get("passages", {})
        passage_texts = passages.get("passage_text", [])
        if not passage_texts:
            continue
        samples.append({
            "question": row["query"],
            "contexts": passage_texts,
            "ground_truth": answers[0],
        })
        if max_questions and len(samples) >= max_questions:
            break
    return samples


# ---------------------------------------------------------------------------
# scifact helpers
# ---------------------------------------------------------------------------

def _load_scifact(max_questions=None):
    """Load scifact queries, corpus, and qrels from mteb/scifact."""
    qpath = hf_hub_download("mteb/scifact", "queries.jsonl", repo_type="dataset")
    cpath = hf_hub_download("mteb/scifact", "corpus.jsonl", repo_type="dataset")

    queries = []
    with open(qpath) as f:
        for line in f:
            row = json.loads(line)
            queries.append({
                "query_id": str(row["_id"]),
                "question": row["text"],
                "ground_truth": row["text"],
            })
            if max_questions and len(queries) >= max_questions:
                break

    corpus = {}
    with open(cpath) as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title", "")
            text = row.get("text", "")
            corpus[str(row["_id"])] = f"{title}. {text}" if title else text

    # Load relevance judgments (qrels) for IR metrics
    qrels = {}
    try:
        qrels_path = hf_hub_download("mteb/scifact", "qrels/test.tsv", repo_type="dataset")
        with open(qrels_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    qid, doc_id = str(parts[0]), str(parts[1])
                    qrels.setdefault(qid, set()).add(doc_id)
    except Exception as e:
        print(f"WARNING: Could not load scifact qrels for IR metrics: {e}")

    return queries, corpus, qrels


def _index_scifact_corpus(corpus, retriever, embedding_model):
    """Index the scifact corpus into the retriever using existing Indexer methods."""
    from components.indexer import Indexer

    texts = list(corpus.values())
    doc_ids = list(corpus.keys())
    metadata = [{"document_name": f"scifact_{did}", "page": 0, "chunk_idx": i}
                 for i, did in enumerate(doc_ids)]

    indexer = Indexer()
    print(f"Indexing {len(texts)} scifact passages...")
    embeddings = indexer.get_embeddings(texts, model=embedding_model)
    multi_vector_index = indexer.build_multi_vector_index(embeddings, metadata, texts)
    bm25 = indexer.build_bm25_index(texts)
    faiss_index = indexer.build_faiss_index(multi_vector_index)
    metadata_index = indexer.build_metadata_index(metadata)

    retriever.update_indices(multi_vector_index, bm25, metadata_index, faiss_index)
    print("Scifact corpus indexed.")


# ---------------------------------------------------------------------------
# Metric scoring
# ---------------------------------------------------------------------------

def _run_llm_metrics(samples, ragas_llm, ragas_embeddings, has_reference):
    """Score all samples with RAGAS LLM-based metrics."""
    metric_instances = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    }
    if has_reference:
        metric_instances.update({
            "context_recall": ContextRecall(llm=ragas_llm),
            "context_precision": ContextPrecisionWithReference(llm=ragas_llm),
            "factual_correctness": FactualCorrectness(llm=ragas_llm),
        })

    inputs_map = {
        "faithfulness": [
            {"user_input": s["user_input"], "response": s["response"],
             "retrieved_contexts": s["retrieved_contexts"]}
            for s in samples
        ],
        "answer_relevancy": [
            {"user_input": s["user_input"], "response": s["response"]}
            for s in samples
        ],
    }
    if has_reference:
        inputs_map["context_recall"] = [
            {"user_input": s["user_input"], "retrieved_contexts": s["retrieved_contexts"],
             "reference": s["reference"]}
            for s in samples
        ]
        inputs_map["context_precision"] = [
            {"user_input": s["user_input"], "reference": s["reference"],
             "retrieved_contexts": s["retrieved_contexts"]}
            for s in samples
        ]
        inputs_map["factual_correctness"] = [
            {"response": s["response"], "reference": s["reference"]}
            for s in samples
        ]

    all_scores = {name: [] for name in metric_instances}
    for name, metric in metric_instances.items():
        print(f"  Scoring {name} (LLM-based)...")
        scores = []
        for i, inp in enumerate(inputs_map[name]):
            try:
                result = metric.score(**inp)
                scores.append(result.value)
            except Exception as e:
                print(f"    WARNING: {name} sample {i} failed: {e}")
                scores.append(float("nan"))
            time.sleep(2)  # small delay between samples to avoid rate limits
        all_scores[name] = scores
        time.sleep(15)  # longer pause between metrics

    return all_scores


def _run_nonllm_ragas_metrics(samples, has_reference):
    """Score with RAGAS non-LLM metrics (string-matching based, no LLM judge)."""
    if not _HAS_NONLLM_RAGAS or not has_reference:
        return {}

    from ragas.dataset_schema import SingleTurnSample

    results = {}
    for name, metric in [
        ("nonllm_context_precision", NonLLMContextPrecisionWithReference()),
        ("nonllm_context_recall", NonLLMContextRecall()),
    ]:
        print(f"  Scoring {name} (LLM-free)...")
        scores = []
        try:
            for s in samples:
                sample = SingleTurnSample(
                    user_input=s["user_input"],
                    retrieved_contexts=s["retrieved_contexts"],
                    reference=s["reference"],
                )
                scores.append(metric.single_turn_score(sample))
            results[name] = scores
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            results[name] = [float("nan")] * len(samples)

    return results


# ---------------------------------------------------------------------------
# Score merging / reporting (shared with run_evaluation's custom-labeled mode)
# ---------------------------------------------------------------------------

LLM_FREE_METRICS = {
    "nonllm_context_precision", "nonllm_context_recall",
    "rouge1", "rouge2", "rougeL", "semantic_similarity",
    "answer_groundedness", "context_relevance", "ndcg@10", "mrr",
}


def summarize_results(samples, all_scores, rouge_agg, ir_agg,
                      proxy_groundedness, proxy_context_rel, has_reference):
    """Merge per-metric score lists into per-sample results and aggregate means.

    Returns (per_sample_results, aggregate).
    """
    per_sample_results = []
    for i, s in enumerate(samples):
        scores = {}
        for metric_name, score_list in all_scores.items():
            val = score_list[i]
            scores[metric_name] = val if not (isinstance(val, float) and math.isnan(val)) else None
        if has_reference and s.get("reference"):
            scores["rougeL"] = compute_per_sample_rouge(s["response"], s["reference"])
        scores["answer_groundedness"] = proxy_groundedness[i]
        scores["context_relevance"] = proxy_context_rel[i]
        per_sample_results.append({**s, "scores": scores})

    aggregate = {}
    for metric_name, score_list in all_scores.items():
        valid = [v for v in score_list if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid:
            aggregate[metric_name] = sum(valid) / len(valid)
    aggregate.update({k: v for k, v in rouge_agg.items() if v is not None})
    aggregate.update({k: v for k, v in ir_agg.items() if v is not None})
    aggregate["answer_groundedness"] = sum(proxy_groundedness) / len(proxy_groundedness)
    aggregate["context_relevance"] = sum(proxy_context_rel) / len(proxy_context_rel)
    return per_sample_results, aggregate


def print_aggregate_scores(aggregate, header):
    """Print aggregate metric scores, tagging each as LLM-free or LLM-based."""
    print(f"\n{'='*60}")
    print(header)
    for metric, score in aggregate.items():
        tag = "(LLM-free)" if metric in LLM_FREE_METRICS else "(LLM-based)"
        print(f"  {metric}: {score:.4f}  {tag}")


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_benchmark_evaluation(
    dataset_name,
    llm_model,
    llm_tokenizer,
    retriever,
    ragas_llm,
    ragas_embeddings,
    max_questions=None,
    checkpoint_path=None,
    skip_llm_metrics=False,
    system_prompt=None,
    system_prompt_name="default",
):
    """Run benchmark evaluation with a standard dataset.

    Args:
        checkpoint_path:  Path to a previously saved checkpoint JSON. If provided,
                          skips generation and scores the saved answers directly.
        skip_llm_metrics: If True, skip LLM-based RAGAS metrics (faithfulness etc.)
                          and only compute LLM-free metrics. Useful when the local
                          LLM crashes during scoring due to GPU memory pressure.
    """
    if max_questions is None:
        max_questions = EVAL_MAX_QUESTIONS

    print(f"\n{'='*60}")
    print(f"Benchmark Evaluation — dataset: {dataset_name}")
    print(f"{'='*60}\n")

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load from checkpoint OR generate answers ----
    retrieval_results = {}

    if checkpoint_path:
        print(f"Loading answers from checkpoint: {checkpoint_path}")
        with open(checkpoint_path) as f:
            data = json.load(f)
        # Support both raw checkpoint (list) and final results file (dict with per_sample_results)
        if isinstance(data, dict):
            samples = [
                {k: v for k, v in s.items() if k != "scores"}
                for s in data["per_sample_results"]
            ]
        else:
            samples = data
        n = len(samples)
        print(f"Loaded {n} samples from checkpoint.\n")
        # Load qrels for IR metrics if scifact
        qrels = {}
        if dataset_name == "scifact":
            _, _, qrels = _load_scifact(max_questions)
    else:
        qrels = {}
        if dataset_name == "amnesty_qa":
            data = _load_amnesty_qa(max_questions)
            has_contexts = True
        elif dataset_name == "hotpotqa":
            data = _load_hotpotqa(max_questions)
            has_contexts = True
        elif dataset_name == "ms_marco":
            data = _load_ms_marco(max_questions)
            has_contexts = True
        elif dataset_name == "scifact":
            queries, corpus, qrels = _load_scifact(max_questions)
            _index_scifact_corpus(corpus, retriever, retriever.embedding_model)
            data = queries
            has_contexts = False
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'amnesty_qa', 'hotpotqa', 'ms_marco', or 'scifact'.")

        n = len(data)
        print(f"Loaded {n} samples.\n")

        samples = []
        for i, item in enumerate(data):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            print(f"[{i+1}/{n}] {question[:80]}...")

            if has_contexts:
                contexts = item["contexts"]
                context_str = "\n---\n".join(contexts)
            else:
                context_str, contexts = retriever.combined_retrieval_with_chunks(question)
                if qrels and dataset_name == "scifact":
                    query_id = item["query_id"]
                    retrieved_doc_ids = [
                        meta.get("document_name", "").replace("scifact_", "")
                        for meta in retriever.last_retrieved_metadata
                    ]
                    retrieval_results[query_id] = retrieved_doc_ids

            t0 = time.time()
            response = _generate_answer(question, context_str, llm_model, llm_tokenizer,
                                        system_prompt=system_prompt)
            gen_time = time.time() - t0
            print(f"  Generated in {gen_time:.1f}s")

            samples.append({
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            })

        # Save checkpoint
        new_checkpoint = os.path.join(EVAL_OUTPUT_DIR, f"_checkpoint_{dataset_name}_{ts}.json")
        with open(new_checkpoint, "w") as f:
            json.dump(samples, f, indent=2, default=str)
        print(f"\nAnswers checkpointed to {new_checkpoint}")

    # ---- LLM-based RAGAS metrics ----
    has_reference = any(s.get("reference") for s in samples)
    if skip_llm_metrics:
        print("\nSkipping LLM-based RAGAS metrics (--skip-llm-metrics).")
        llm_scores = {}
    else:
        print("\nRunning LLM-based RAGAS metrics...")
        llm_scores = _run_llm_metrics(samples, ragas_llm, ragas_embeddings, has_reference)

    # ---- LLM-free RAGAS metrics ----
    print("\nRunning LLM-free RAGAS metrics...")
    nonllm_scores = _run_nonllm_ragas_metrics(samples, has_reference)

    # ---- ROUGE ----
    rouge_agg = {}
    if has_reference:
        print("  Scoring ROUGE (LLM-free)...")
        candidates = [s["response"] for s in samples]
        references = [s["reference"] for s in samples]
        rouge_agg = compute_rouge(candidates, references)
        sem_agg = compute_semantic_similarity(candidates, references, ragas_embeddings.embed_text)
        rouge_agg.update(sem_agg)

    # ---- Proxy metrics (always available) ----
    print("  Scoring proxy metrics (LLM-free)...")
    embed_fn = ragas_embeddings.embed_text
    proxy_groundedness = []
    proxy_context_rel = []
    for s in samples:
        context_str = "\n".join(s["retrieved_contexts"]) if s["retrieved_contexts"] else ""
        proxy_groundedness.append(answer_groundedness(s["response"], context_str))
        chunk_embeddings = [embed_fn(c) for c in s["retrieved_contexts"]]
        query_embedding = embed_fn(s["user_input"])
        proxy_context_rel.append(context_relevance(query_embedding, chunk_embeddings))

    # ---- IR metrics (scifact only) ----
    ir_agg = {}
    if retrieval_results and qrels:
        print("  Scoring IR metrics (NDCG@10, MRR) (LLM-free)...")
        ir_agg = compute_ir_metrics(retrieval_results, qrels, k_values=(10,))

    # ---- Merge scores per sample & aggregate ----
    all_scores = {**llm_scores, **nonllm_scores}
    per_sample_results, aggregate = summarize_results(
        samples, all_scores, rouge_agg, ir_agg,
        proxy_groundedness, proxy_context_rel, has_reference)

    # ---- Output ----
    git_sha, git_dirty = _get_git_info()
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": "benchmark",
            "dataset": dataset_name,
            "num_samples": n,
            "git_commit": git_sha,
            "git_dirty": git_dirty,
            "system_prompt": system_prompt_name,
        },
        "aggregate_scores": aggregate,
        "per_sample_results": per_sample_results,
    }

    out_path = os.path.join(EVAL_OUTPUT_DIR, f"benchmark_{dataset_name}_{ts}_{git_sha}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Only delete checkpoint if we created it in this run
    new_checkpoint = os.path.join(EVAL_OUTPUT_DIR, f"_checkpoint_{dataset_name}_{ts}.json")
    if os.path.exists(new_checkpoint):
        os.remove(new_checkpoint)

    dirty_tag = " (uncommitted changes)" if git_dirty else ""
    print_aggregate_scores(
        aggregate,
        f"Aggregate Scores ({dataset_name}, {n} samples) — git: {git_sha}{dirty_tag}")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output
