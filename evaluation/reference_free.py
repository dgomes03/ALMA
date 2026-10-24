"""Reference-free evaluation: runs queries through the RAG pipeline and evaluates
without any ground truth labels.

LLM-based metrics (require RAGAS judge LLM):
  - Faithfulness, AnswerRelevancy

LLM-free proxy metrics (always available, no ground truth):
  - context_relevance   — cosine similarity between query and retrieved chunks
  - answer_groundedness — token F1 between answer and retrieved context
"""

import json
import math
import os
import time
from datetime import datetime

from core.llm_backend import (
    generate as mlx_generate, make_sampler, make_logits_processors
)
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from core.config import (
    EVAL_OUTPUT_DIR,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY,
    LLM_REPETITION_CONTEXT_SIZE,
    MAX_RESPONSE_TOKENS,
)
from core.conversation_manager import ConversationManager
from evaluation.lm_free_metrics import answer_groundedness, context_relevance


def _generate_answer(query, context, llm_model, llm_tokenizer):
    """Generate an answer for a query given retrieved context.

    Simulates the real runtime tool-call flow so the model receives context
    as a tool result (role=tool) rather than raw text in the user message.
    """
    from tools.tools import get_tools_for_standard_generator

    # Expose ONLY search_documents. The document search has already been run and
    # its result is supplied below as a tool result, so the model has everything
    # it needs; leaving the web/database/advanced-research tools visible let it
    # reply with a further [TOOL_CALLS] request instead of an answer. This
    # single-step harness has no tool loop (unlike Generator.answer_query_with_llm),
    # so that raw marker was captured as the "answer" and scored 0 by RAGAS.
    #
    # Evaluation-side only: the live pipeline's tool set is untouched.
    tools = [
        t for t in get_tools_for_standard_generator()
        if (t.get("function", {}) or {}).get("name", t.get("name")) == "search_documents"
    ]
    if not tools:  # tool registry renamed/changed — fail loudly, don't score noise
        raise RuntimeError(
            "search_documents not found in get_tools_for_standard_generator(); "
            "the evaluation harness cannot build a document-only tool list."
        )
    call_id = "benchmark000"

    conversation = [
        {"role": "system", "content": ConversationManager.SYSTEM_PROMPT},
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


def run_reference_free_evaluation(queries, retriever, llm_model, llm_tokenizer,
                                  ragas_llm, ragas_embeddings, checkpoint_path=None):
    """Run reference-free evaluation on a list of queries using the live RAG pipeline.

    With ``checkpoint_path`` the (query, answer, contexts) triples are loaded
    from a checkpoint written by an earlier run and generation is skipped, so
    the same answers can be scored again, e.g. with a different judge. The
    LLM-free proxies are then recomputed from the untagged chunks, without the
    source tags the live context string carries.
    """
    embed_fn = ragas_embeddings.embed_text
    samples = []
    proxy_groundedness = []
    proxy_context_rel = []

    if checkpoint_path:
        with open(checkpoint_path, encoding="utf-8") as f:
            samples = json.load(f)
        queries = [s["user_input"] for s in samples]
        print(f"\n{'='*60}")
        print(f"Reference-Free Evaluation — {len(samples)} answers loaded from checkpoint")
        print(f"{os.path.basename(checkpoint_path)}")
        print(f"{'='*60}\n")
        for s_ in samples:
            chunks = s_["retrieved_contexts"]
            context_string = "\n---\n".join(chunks)
            proxy_groundedness.append(answer_groundedness(s_["response"], context_string))
            chunk_embeddings = [embed_fn(c) for c in chunks]
            proxy_context_rel.append(context_relevance(embed_fn(s_["user_input"]), chunk_embeddings))
    else:
        print(f"\n{'='*60}")
        print(f"Reference-Free Evaluation — {len(queries)} queries")
        print(f"{'='*60}\n")

    for i, query in enumerate([] if checkpoint_path else queries):
        print(f"[{i+1}/{len(queries)}] Processing: {query[:80]}...")

        context_string, chunks = retriever.combined_retrieval_with_chunks(query)

        t0 = time.time()
        response = _generate_answer(query, context_string, llm_model, llm_tokenizer)
        gen_time = time.time() - t0
        print(f"  Answer generated in {gen_time:.1f}s ({len(response)} chars)")

        # LLM-free proxy metrics
        groundedness = answer_groundedness(response, context_string)
        chunk_embeddings = [embed_fn(c) for c in chunks]
        query_embedding = embed_fn(query)
        ctx_rel = context_relevance(query_embedding, chunk_embeddings)
        proxy_groundedness.append(groundedness)
        proxy_context_rel.append(ctx_rel)

        print(f"  context_relevance={ctx_rel:.3f}  answer_groundedness={groundedness:.3f}")

        samples.append({
            "user_input": query,
            "response": response,
            "retrieved_contexts": chunks,
        })

    # ---- Save checkpoint (only when the answers were generated here) ----
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    loaded_from = checkpoint_path
    if not checkpoint_path:
        checkpoint_path = os.path.join(EVAL_OUTPUT_DIR, f"_checkpoint_reffree_{ts}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(samples, f, indent=2, default=str)

    # ---- LLM-based RAGAS metrics ----
    print("\nRunning LLM-based RAGAS metrics...")
    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    }

    faith_inputs = [
        {"user_input": s["user_input"], "response": s["response"],
         "retrieved_contexts": s["retrieved_contexts"]}
        for s in samples
    ]
    relevancy_inputs = [
        {"user_input": s["user_input"], "response": s["response"]}
        for s in samples
    ]

    all_scores = {}
    for name, (metric, inputs) in {
        "faithfulness": (metrics["faithfulness"], faith_inputs),
        "answer_relevancy": (metrics["answer_relevancy"], relevancy_inputs),
    }.items():
        print(f"  Scoring {name} (LLM-based)...")
        try:
            results = metric.batch_score(inputs)
            all_scores[name] = [r.value for r in results]
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            all_scores[name] = [float("nan")] * len(samples)

    # ---- Merge scores per sample ----
    per_sample_results = []
    for i, s in enumerate(samples):
        scores = {}
        for metric_name, score_list in all_scores.items():
            val = score_list[i]
            scores[metric_name] = val if not (isinstance(val, float) and math.isnan(val)) else None
        scores["context_relevance"] = proxy_context_rel[i]
        scores["answer_groundedness"] = proxy_groundedness[i]
        per_sample_results.append({**s, "scores": scores})

    # ---- Aggregate ----
    aggregate = {}
    for metric_name, score_list in all_scores.items():
        valid = [v for v in score_list if isinstance(v, (int, float)) and not math.isnan(v)]
        if valid:
            aggregate[metric_name] = sum(valid) / len(valid)
    aggregate["context_relevance"] = sum(proxy_context_rel) / len(proxy_context_rel)
    aggregate["answer_groundedness"] = sum(proxy_groundedness) / len(proxy_groundedness)

    # ---- Output ----
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": "reference-free",
            "num_samples": len(queries),
            "answers_from": os.path.basename(loaded_from) if loaded_from else "generated",
            "ragas_judge": getattr(ragas_llm, "_model", type(ragas_llm).__name__),
        },
        "aggregate_scores": aggregate,
        "per_sample_results": per_sample_results,
    }

    out_path = os.path.join(EVAL_OUTPUT_DIR, f"reference_free_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if not loaded_from:
        os.remove(checkpoint_path)

    print(f"\n{'='*60}")
    print("Aggregate Scores:")
    llm_free = {"context_relevance", "answer_groundedness"}
    for metric, score in aggregate.items():
        tag = "(LLM-free)" if metric in llm_free else "(LLM-based)"
        print(f"  {metric}: {score:.4f}  {tag}")
    print(f"\nResults saved to: {out_path}")
    print(f"{'='*60}\n")

    return output
