"""RAGAS LLM wrapper that uses the Anthropic (Claude) API as the judge.

Uses claude-haiku-4-5 by default (fast and cheap) but can be overridden
    with --ragas-model.
Requires the ANTHROPIC_API_KEY environment variable to be set.

Usage:
    from evaluation.claude_llm_wrapper import ClaudeRagasLLM
    ragas_llm = ClaudeRagasLLM()
"""

import asyncio
import json
import os
import time
import typing as t
from concurrent.futures import ThreadPoolExecutor

import anthropic
from pydantic import BaseModel
from ragas.llms import InstructorBaseRagasLLM

T = t.TypeVar("T", bound=BaseModel)

# Model IDs are complete as-is — never append a date suffix. The previous
# value ("claude-haiku-4-5-20251001") carried one.
_DEFAULT_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 8192


class ClaudeRagasLLM(InstructorBaseRagasLLM):
    """RAGAS judge backed by the Anthropic API (Claude)."""

    def __init__(self, model: str = _DEFAULT_MODEL, max_tokens: int = _MAX_TOKENS):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        # Serialize all API calls to avoid concurrent connection rate limits
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _generate_structured(self, prompt: str, response_model: t.Type[T]) -> T:
        schema = response_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a valid JSON object that matches this schema "
            f"(no markdown fences, no explanation):\n{json.dumps(schema)}"
        )
        for attempt in range(5):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                break
            except anthropic.RateLimitError as e:
                wait = 30 * (attempt + 1)
                print(f"    Rate limit hit, retrying in {wait}s... ({e})")
                time.sleep(wait)
        else:
            raise RuntimeError("Claude API rate limit: all retries exhausted.")

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return response_model.model_validate_json(raw)

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        return self._generate_structured(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._generate_structured, prompt, response_model
        )
