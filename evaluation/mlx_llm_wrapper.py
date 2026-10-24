"""Custom RAGAS LLM wrapper for local MLX-LM models.

RAGAS 0.4+ requires InstructorBaseRagasLLM which returns structured Pydantic
objects.  We prompt the local model to output JSON matching the expected schema,
then parse it (via core.parser's json_repair-backed utilities, which handle
markdown fences, surrounding prose and truncated output) into the requested
Pydantic model.
"""

import asyncio
import json
import typing as t
from concurrent.futures import ThreadPoolExecutor

from core.llm_backend import generate, make_sampler
from pydantic import BaseModel
from ragas.llms import InstructorBaseRagasLLM

from core.config import EVAL_LLM_MAX_TOKENS, LLM_TEMPERATURE as EVAL_LLM_TEMPERATURE
from core.parser import parse_json_safely

T = t.TypeVar("T", bound=BaseModel)


class MLXRagasLLM(InstructorBaseRagasLLM):
    """Wraps an already-loaded MLX-LM model/tokenizer for RAGAS 0.4+ evaluation."""

    def __init__(self, model, tokenizer, *,
                 max_tokens: int = EVAL_LLM_MAX_TOKENS,
                 temperature: float = EVAL_LLM_TEMPERATURE):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        # max_workers=1 ensures MLX inference calls are serialized even when
        # RAGAS fires multiple concurrent agenerate() calls via asyncio.gather.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _call_llm(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        sampler = make_sampler(temp=self.temperature)
        raw = generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return raw

    def _generate_structured(self, prompt: str, response_model: t.Type[T]) -> T:
        schema = response_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a valid JSON object (no markdown, no explanation). "
            f"Keep it concise. Schema:\n{json.dumps(schema)}"
        )
        raw = self._call_llm(full_prompt)
        data = parse_json_safely(raw)
        if data is not None:
            try:
                return response_model.model_validate(data)
            except Exception:
                pass
        # Retry once with stricter instruction
        retry_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL: Output ONLY valid JSON, nothing else. Be very concise. Schema:\n"
            f"{json.dumps(schema)}"
        )
        raw = self._call_llm(retry_prompt)
        return response_model.model_validate(parse_json_safely(raw) or {})

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        return self._generate_structured(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        # Single-worker executor serializes all MLX calls — prevents concurrent
        # Metal GPU command encoders which cause an assertion crash on Apple Silicon.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._generate_structured, prompt, response_model,
        )
