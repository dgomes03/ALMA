"""CLI entry point for RAGAS evaluation.

Usage:
    # Quick benchmark (amnesty_qa, 30 questions)
    python -m evaluation.run_evaluation --mode benchmark --dataset amnesty_qa

    # Benchmark with limit
    python -m evaluation.run_evaluation --mode benchmark --dataset scifact --max-questions 50

    # Reference-free on own documents (LLM-free proxy metrics + LLM-based RAGAS)
    python -m evaluation.run_evaluation --mode reference-free --queries "Q1?" "Q2?"
    python -m evaluation.run_evaluation --mode reference-free --queries-file queries.txt

    # Custom labeled on own documents (all LLM-free metrics available)
    # JSON format: [{"question": "...", "reference": "...", "relevant_doc_ids": ["opt"]}]
    python -m evaluation.run_evaluation --mode custom-labeled --labeled-file my_labels.json

    # BEIR retrieval evaluation on SciFact (no LLM needed)
    python -m evaluation.run_evaluation --mode beir-retrieval
    python -m evaluation.run_evaluation --mode beir-retrieval --max-questions 100
    python -m evaluation.run_evaluation --mode beir-retrieval --k-values 1 5 10 100
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    EVAL_DATASET,
    EVAL_RAGAS_JUDGE,
)
from core.llm_backend import load_llm
from components.retrievers import Retriever
from evaluation.mlx_llm_wrapper import MLXRagasLLM
from evaluation.embeddings_wrapper import E5RagasEmbeddings


def _run_pipeline(fn, **kwargs):
    """Run an evaluation that touches the local LLM on the MLX worker thread.

    mlx-core 0.31.2 made GPU streams thread-local, and load_llm() loads the model
    on the dedicated mlx worker (core/mlx_runtime.py). Anything that then calls
    generate() — answer generation, and query expansion inside the retriever —
    must run on that same thread or mlx raises
    "RuntimeError: There is no Stream(gpu, N) in current thread".
    main.py already routes generation through run_on_mlx; the evaluation entry
    point did not, so every generating mode crashed on the first query.

    Only needed for the MLX backend; Ollama and the API backend generate over
    HTTP and are thread-agnostic.
    """
    from core.config import LLM_BACKEND
    if LLM_BACKEND != "mlx":
        return fn(**kwargs)
    from core.mlx_runtime import run_on_mlx
    return run_on_mlx(lambda: fn(**kwargs))


def _build_components(ragas_judge: str = "mlx", skip_llm: bool = False):
    """Load LLM, build retriever, and create RAGAS wrappers.

    Args:
        ragas_judge: Which LLM to use as the RAGAS judge.
                     "mlx"    — local MLX model (default, no API costs)
                     "claude" — Anthropic Claude API (requires ANTHROPIC_API_KEY)
        skip_llm:   If True, skip loading the local LLM and retriever entirely.
                    Only valid when ragas_judge="claude" and using --from-checkpoint.
    """
    if skip_llm:
        from core.utils import load_sentence_transformer
        print("Loading embedding model only (skipping LLM)...")
        embedding_model = load_sentence_transformer(EMBEDDING_MODEL_NAME)
        ragas_embeddings = E5RagasEmbeddings(embedding_model)
        llm_model, llm_tokenizer, retriever = None, None, None
    else:
        print("Loading LLM...")
        llm_model, llm_tokenizer = load_llm()

        print("Initialising retriever...")
        retriever = Retriever(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
        )
        ragas_embeddings = E5RagasEmbeddings(retriever.embedding_model)

    # Build RAGAS LLM judge
    if ragas_judge == "claude":
        from evaluation.claude_llm_wrapper import ClaudeRagasLLM
        print("Using Claude API as RAGAS judge.")
        ragas_llm = ClaudeRagasLLM()
    else:
        ragas_llm = MLXRagasLLM(model=llm_model, tokenizer=llm_tokenizer)

    return llm_model, llm_tokenizer, retriever, ragas_llm, ragas_embeddings


def _run_custom_labeled(labeled_file, retriever, llm_model, llm_tokenizer,
                        ragas_llm, ragas_embeddings):
    """Evaluate on a user-provided labeled dataset.

    JSON format (array of objects):
        [
          {
            "question": "What causes X?",
            "reference": "X is caused by Y.",
            "relevant_doc_ids": ["optional_chunk_id"]   // omit if unknown
          },
          ...
        ]

    Metrics computed:
      - ROUGE-1/2/L, semantic similarity, NonLLM RAGAS context metrics  (need 'reference')
      - NDCG@10, MRR                                                      (need 'relevant_doc_ids')
      - context_relevance, answer_groundedness                             (always, LLM-free)
      - Faithfulness, AnswerRelevancy                                      (LLM-based)
    """
    from evaluation.lm_free_metrics import (
        answer_groundedness,
        compute_ir_metrics,
        compute_rouge,
        compute_semantic_similarity,
        context_relevance,
    )
    from evaluation.benchmark import summarize_results, print_aggregate_scores
    from evaluation.reference_free import _generate_answer
    from core.config import EVAL_OUTPUT_DIR
    import time
    from datetime import datetime
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    try:
        from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall
        _has_nonllm = True
    except ImportError:
        _has_nonllm = False

    with open(labeled_file) as f:
        labeled_data = json.load(f)

    n = len(labeled_data)
    print(f"\n{'='*60}")
    print(f"Custom-Labeled Evaluation — {n} questions from {labeled_file}")
    print(f"{'='*60}\n")

    embed_fn = ragas_embeddings.embed_text
    samples = []
    proxy_groundedness = []
    proxy_context_rel = []
    retrieval_results = {}  # for NDCG/MRR if relevant_doc_ids provided

    for i, item in enumerate(labeled_data):
        question = item["question"]
        reference = item.get("reference", "")
        relevant_doc_ids = item.get("relevant_doc_ids", [])
        print(f"[{i+1}/{n}] {question[:80]}...")

        context_string, chunks = retriever.combined_retrieval_with_chunks(question)

        # Collect retrieved doc_ids for IR metrics
        if relevant_doc_ids:
            retrieved_doc_ids = [
                meta.get("document_name", "")
                for meta in retriever.last_retrieved_metadata
            ]
            retrieval_results[str(i)] = retrieved_doc_ids

        t0 = time.time()
        response = _generate_answer(question, context_string, llm_model, llm_tokenizer)
        print(f"  Generated in {time.time() - t0:.1f}s")

        groundedness = answer_groundedness(response, context_string)
        chunk_embeddings = [embed_fn(c) for c in chunks]
        query_embedding = embed_fn(question)
        ctx_rel = context_relevance(query_embedding, chunk_embeddings)
        proxy_groundedness.append(groundedness)
        proxy_context_rel.append(ctx_rel)

        samples.append({
            "user_input": question,
            "response": response,
            "retrieved_contexts": chunks,
            "reference": reference,
        })

    has_reference = any(s.get("reference") for s in samples)

    # ---- LLM-based RAGAS ----
    print("\nRunning LLM-based RAGAS metrics...")
    llm_scores = {}
    for name, metric, inputs in [
        ("faithfulness", Faithfulness(llm=ragas_llm), [
            {"user_input": s["user_input"], "response": s["response"],
             "retrieved_contexts": s["retrieved_contexts"]} for s in samples
        ]),
        ("answer_relevancy", AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings), [
            {"user_input": s["user_input"], "response": s["response"]} for s in samples
        ]),
    ]:
        print(f"  Scoring {name} (LLM-based)...")
        try:
            results = metric.batch_score(inputs)
            llm_scores[name] = [r.value for r in results]
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            llm_scores[name] = [float("nan")] * len(samples)

    # ---- NonLLM RAGAS ----
    nonllm_scores = {}
    if _has_nonllm and has_reference:
        print("\nRunning LLM-free RAGAS metrics...")
        for name, metric, inputs in [
            ("nonllm_context_precision", NonLLMContextPrecisionWithReference(), [
                {"user_input": s["user_input"], "reference": s["reference"],
                 "retrieved_contexts": s["retrieved_contexts"]} for s in samples
            ]),
            ("nonllm_context_recall", NonLLMContextRecall(), [
                {"user_input": s["user_input"], "reference": s["reference"],
                 "retrieved_contexts": s["retrieved_contexts"]} for s in samples
            ]),
        ]:
            print(f"  Scoring {name} (LLM-free)...")
            try:
                results = metric.batch_score(inputs)
                nonllm_scores[name] = [r.value for r in results]
            except Exception as e:
                print(f"    WARNING: {name} failed: {e}")
                nonllm_scores[name] = [float("nan")] * len(samples)

    # ---- Reference-based LLM-free metrics ----
    rouge_agg = {}
    if has_reference:
        print("\nRunning reference-based LLM-free metrics...")
        candidates = [s["response"] for s in samples]
        references = [s["reference"] for s in samples]
        rouge_agg = compute_rouge(candidates, references)
        rouge_agg.update(compute_semantic_similarity(candidates, references, embed_fn))

    # ---- IR metrics ----
    ir_agg = {}
    if retrieval_results:
        print("  Scoring IR metrics (NDCG@10, MRR) (LLM-free)...")
        qrels = {
            str(i): set(item.get("relevant_doc_ids", []))
            for i, item in enumerate(labeled_data)
            if item.get("relevant_doc_ids")
        }
        ir_agg = compute_ir_metrics(retrieval_results, qrels, k_values=(10,))

    # ---- Merge scores per sample & aggregate ----
    all_scores = {**llm_scores, **nonllm_scores}
    per_sample_results, aggregate = summarize_results(
        samples, all_scores, rouge_agg, ir_agg,
        proxy_groundedness, proxy_context_rel, has_reference)

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": "custom-labeled",
            "labeled_file": labeled_file,
            "num_samples": n,
        },
        "aggregate_scores": aggregate,
        "per_sample_results": per_sample_results,
    }
    out_path = os.path.join(EVAL_OUTPUT_DIR, f"custom_labeled_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_aggregate_scores(aggregate, f"Aggregate Scores (custom-labeled, {n} samples):")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for the RAG Framework")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "reference-free", "custom-labeled", "beir-retrieval"],
        required=True,
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--dataset",
        choices=["amnesty_qa", "hotpotqa", "scifact", "ms_marco"],
        default=EVAL_DATASET,
        help="Benchmark dataset (only for --mode benchmark).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions to evaluate.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Queries for reference-free mode.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to a text file with one query per line.",
    )
    parser.add_argument(
        "--labeled-file",
        type=str,
        default=None,
        help="Path to JSON file with labeled Q&A pairs for custom-labeled mode.",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint JSON. Skips generation and scores saved answers directly.",
    )
    parser.add_argument(
        "--skip-llm-metrics",
        action="store_true",
        help="Skip LLM-based RAGAS metrics (faithfulness etc.) and only run LLM-free metrics.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Text file whose contents replace the framework's system prompt when "
             "generating answers (benchmark mode only). The judge is unaffected.",
    )
    parser.add_argument(
        "--ragas-judge",
        choices=["mlx", "claude"],
        default=EVAL_RAGAS_JUDGE,
        help=(
            "LLM to use as the RAGAS judge for LLM-based metrics. "
            "'mlx' uses the local model; "
            "'claude' uses the Anthropic API (requires ANTHROPIC_API_KEY)."
        ),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10, 100],
        metavar="K",
        help=(
            "K values for nDCG@K, Recall@K, and Precision@K in beir-retrieval mode. "
            "Default: 1 5 10 100."
        ),
    )
    args = parser.parse_args()

    # Disable async to avoid issues with local model
    os.environ["RAGAS_RUN_ASYNC"] = "false"

    # beir-retrieval only needs the retriever — skip loading the LLM entirely
    if args.mode == "beir-retrieval":
        print("Initialising retriever (no LLM required for BEIR retrieval evaluation)...")
        retriever = Retriever(
            llm_model=None,
            llm_tokenizer=None,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
        )
        from evaluation.beir_retrieval import run_beir_retrieval_evaluation
        run_beir_retrieval_evaluation(
            retriever=retriever,
            max_questions=args.max_questions,
            k_values=tuple(args.k_values),
        )
        return

    skip_llm = args.ragas_judge == "claude" and args.from_checkpoint is not None
    llm_model, llm_tokenizer, retriever, ragas_llm, ragas_embeddings = _build_components(
        ragas_judge=args.ragas_judge,
        skip_llm=skip_llm,
    )

    if args.mode == "benchmark":
        from evaluation.benchmark import run_benchmark_evaluation

        system_prompt = None
        if args.system_prompt_file:
            with open(args.system_prompt_file, encoding="utf-8") as f:
                system_prompt = f.read().strip()
            print(f"System prompt for generation: {args.system_prompt_file}")

        _run_pipeline(
            run_benchmark_evaluation,
            dataset_name=args.dataset,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            retriever=retriever,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            max_questions=args.max_questions,
            checkpoint_path=args.from_checkpoint,
            skip_llm_metrics=args.skip_llm_metrics,
            system_prompt=system_prompt,
            system_prompt_name=(os.path.basename(args.system_prompt_file)
                                if args.system_prompt_file else "default"),
        )

    elif args.mode == "reference-free":
        from evaluation.reference_free import run_reference_free_evaluation

        queries = args.queries or []
        if args.queries_file:
            # Accept either a JSON array of strings or one query per line.
            # Reading a .json file line-by-line silently yields junk queries
            # ("[", "],", quoted fragments) and spends judge calls on them.
            with open(args.queries_file, encoding="utf-8") as f:
                raw = f.read()
            stripped = raw.lstrip()
            if stripped.startswith("["):
                loaded = json.loads(raw)
                if not all(isinstance(q, str) for q in loaded):
                    parser.error(f"{args.queries_file}: expected a JSON array of strings.")
                queries.extend(q.strip() for q in loaded if q.strip())
            else:
                queries.extend(line.strip() for line in raw.splitlines() if line.strip())
        if not queries and not args.from_checkpoint:
            parser.error("Provide --queries or --queries-file for reference-free mode.")

        _run_pipeline(
            run_reference_free_evaluation,
            queries=queries,
            retriever=retriever,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            checkpoint_path=args.from_checkpoint,
        )

    elif args.mode == "custom-labeled":
        if not args.labeled_file:
            parser.error("Provide --labeled-file for custom-labeled mode.")

        _run_pipeline(
            _run_custom_labeled,
            labeled_file=args.labeled_file,
            retriever=retriever,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )


if __name__ == "__main__":
    main()
