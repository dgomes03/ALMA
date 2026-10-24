"""Utility for managing KV cache checkpoints to exclude tool results from cache."""

from core.logging_config import get_logger

logger = get_logger(__name__)

# Token ids currently held by each prompt cache, keyed by id(cache). With them,
# prepare_prompt_tokens reuses exactly the part of the cache that matches the
# new prompt instead of trusting the offset alone. An entry only counts while
# its length equals the cache's offset, which also guards against id reuse.
_cached_tokens = {}


class CacheManager:
    """
    Manages KV cache checkpoints to support selective caching.

    This allows saving the cache state before processing tool results,
    then trimming back after generation to exclude bulky tool results
    from the cache, reducing RAM usage significantly.
    """

    @staticmethod
    def get_checkpoint(cache):
        """
        Record current cache position for later restoration.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)

        Returns:
            List of offsets for each cache layer, or None if cache is None
        """
        if cache is None:
            return None
        return [layer.offset for layer in cache]

    @staticmethod
    def restore_checkpoint(cache, checkpoint):
        """
        Trim cache back to a saved checkpoint.

        This removes tokens added after the checkpoint was taken,
        effectively excluding tool results from the persistent cache.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)
            checkpoint: List of offsets from get_checkpoint()
        """
        if cache is None or checkpoint is None:
            return

        for layer, saved_offset in zip(cache, checkpoint):
            current_offset = layer.offset
            trim_amount = current_offset - saved_offset
            if trim_amount > 0:
                layer.trim(trim_amount)
        CacheManager._truncate_tokens(cache)

    @staticmethod
    def reset(cache):
        """Trim the cache back to empty."""
        if cache is None:
            return
        for layer in cache:
            if layer.offset > 0:
                layer.trim(layer.offset)
        _cached_tokens[id(cache)] = []

    @staticmethod
    def offset(cache):
        return cache[0].offset if cache else 0

    @staticmethod
    def tokens(cache):
        """The token ids the cache holds, or None if they are not known."""
        known = _cached_tokens.get(id(cache))
        return known if known is not None and len(known) == CacheManager.offset(cache) else None

    @staticmethod
    def set_tokens(cache, tokens):
        _cached_tokens[id(cache)] = list(tokens) if tokens is not None else None

    @staticmethod
    def _truncate_tokens(cache):
        known = _cached_tokens.get(id(cache))
        if known is not None:
            _cached_tokens[id(cache)] = known[:CacheManager.offset(cache)]

    @staticmethod
    def _trim_to(cache, length):
        excess = CacheManager.offset(cache) - length
        if excess > 0:
            for layer in cache:
                layer.trim(excess)

    @staticmethod
    def prepare_prompt_tokens(tokenizer, prompt, cache):
        """Return the prompt to feed generation, reusing as much of the cache as matches.

        Token-level cache logic is MLX-only: with no cache (Ollama or API
        backend) the prompt is returned untouched. When the cache's tokens are
        known, it is trimmed to the longest prefix it shares with the new prompt
        and only the rest is returned, so a turn whose stored text differs from
        what was generated (an interrupted answer, stripped whitespace, a
        reasoning model's hidden thinking) costs only the part that differs.
        Without known tokens, the cache is assumed to be a prefix of the prompt.
        """
        if cache is None:
            return prompt
        full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        cache_offset = CacheManager.offset(cache)
        known = CacheManager.tokens(cache)
        if known is not None:
            shared = 0
            for a, b in zip(known, full_tokens):
                if a != b:
                    break
                shared += 1
            shared = min(shared, len(full_tokens) - 1)      # always feed at least one token
            if shared < cache_offset:
                logger.info("[KV-Cache] reusing %d of %d cached tokens", shared, cache_offset)
                CacheManager._trim_to(cache, shared)
        elif 0 < cache_offset < len(full_tokens):
            shared = cache_offset
        else:
            if cache_offset > 0:
                logger.warning("[KV-Cache] Cache invalidated (offset %d >= prompt %d), resetting cache",
                               cache_offset, len(full_tokens))
                CacheManager.reset(cache)
            shared = 0
        _cached_tokens[id(cache)] = full_tokens[:shared]
        return full_tokens[shared:]

    @staticmethod
    def after_generation(cache, fed_tokens, generated_tokens):
        """Record what a generation added to the cache: the fed prompt tokens and
        the generated ones, cut to the cache's real length (prefill may have been
        interrupted, and the last sampled token is never fed back)."""
        known = _cached_tokens.get(id(cache))
        if known is None or not isinstance(fed_tokens, (list, tuple)):
            _cached_tokens[id(cache)] = None
            return
        sequence = known + [int(t) for t in fed_tokens] + list(generated_tokens)
        offset = CacheManager.offset(cache)
        _cached_tokens[id(cache)] = sequence[:offset] if len(sequence) >= offset else None

    @staticmethod
    def get_cache_size(cache):
        """
        Get total tokens currently in cache.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)

        Returns:
            Number of tokens in cache, or 0 if cache is None/empty
        """
        if cache and len(cache) > 0:
            return cache[0].offset
        return 0

    @staticmethod
    def log_cache_stats(cache, label="Cache"):
        """
        Log current cache statistics for debugging/monitoring.

        Args:
            cache: MLX-LM KVCache object
            label: Label for the log message
        """
        size = CacheManager.get_cache_size(cache)
        logger.info("%s: %d tokens", label, size)
        return size
