from core import cancel
from core.llm_backend import generate, is_remote
from core.logging_config import get_logger

logger = get_logger(__name__)


def run_batch_generate(llm_model, llm_tokenizer, prompts, max_tokens,
                        label="BATCH", on_complete=None, sampler=None):
    """Run generation for a list of prompts simultaneously.

    Returns a list of decoded strings in the same order as *prompts*.
    Falls back to a single generate() call for a single prompt.

    MLX uses mlx_lm's BatchGenerator for true batched decoding. Ollama and the
    API backend have no client-side batching, so prompts run sequentially (Ollama
    serializes requests per model anyway unless OLLAMA_NUM_PARALLEL is set).

    on_complete(idx, result): optional callback fired immediately when each
    individual sequence finishes, before the full batch completes.
    """
    if not prompts:
        return []

    if len(prompts) == 1:
        result = generate(
            llm_model, llm_tokenizer,
            prompt=prompts[0], max_tokens=max_tokens, verbose=False,
            sampler=sampler
        )
        if on_complete:
            on_complete(0, result)
        return [result]

    if is_remote(llm_model):
        logger.info(f"\n[{label}] Remote backend: {len(prompts)} prompts sequentially, max_tokens={max_tokens}")
        results = []
        for i, p in enumerate(prompts):
            text = generate(
                llm_model, llm_tokenizer,
                prompt=p, max_tokens=max_tokens, verbose=False,
                sampler=sampler
            )
            results.append(text)
            logger.info(f"[{label}] Sequence {i} done ({len(text)} chars)")
            if on_complete:
                on_complete(i, text)
        logger.info(f"[{label}] Remote batch complete")
        return results

    from mlx_lm.generate import BatchGenerator
    from core.llm_backend import _as_mlx_sampler

    logger.info(f"\n[{label}] BatchGenerator: {len(prompts)} prompts, max_tokens={max_tokens}")

    token_prompts = []
    for p in prompts:
        if isinstance(p, str):
            token_prompts.append(llm_tokenizer.encode(p, add_special_tokens=False))
        else:
            token_prompts.append(p)

    batch_kwargs = dict(max_tokens=max_tokens)
    mlx_sampler = _as_mlx_sampler(sampler)
    if mlx_sampler is not None:
        batch_kwargs["sampler"] = mlx_sampler

    batch_gen = BatchGenerator(llm_model, **batch_kwargs)
    uids = batch_gen.insert(token_prompts)
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    uid_to_token_ids = {uid: [] for uid in uids}
    active = set(uids)
    results = [None] * len(prompts)

    while active:
        cancel.check()
        # BatchGenerator.next() returns (prompt_responses, generation_responses).
        # Only generation responses carry .token / .finish_reason for decoding.
        _, generation_responses = batch_gen.next()
        for resp in generation_responses:
            if resp.uid in active:
                uid_to_token_ids[resp.uid].append(resp.token)
                if resp.finish_reason:
                    active.remove(resp.uid)
                    idx = uid_to_idx[resp.uid]
                    text = llm_tokenizer.decode(uid_to_token_ids[resp.uid])
                    results[idx] = text
                    logger.info(f"[{label}] Sequence {idx} done ({len(text)} chars)")
                    if on_complete:
                        on_complete(idx, text)

    batch_gen.close()
    logger.info(f"[{label}] BatchGenerator complete")
    return results
