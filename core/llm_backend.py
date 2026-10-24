"""Backend-agnostic LLM interface: MLX (mlx-lm), Ollama (GGUF) or a hosted API.

Exposes drop-in replacements for the mlx_lm functions the framework uses —
``generate``, ``stream_generate``, ``make_sampler``, ``make_logits_processors`` —
plus ``load_llm()``. Call sites keep the exact same signatures; dispatch happens
on the model object's type, controlled by ``LLM_BACKEND`` in config.

MLX path: thin pass-through to mlx_lm (imported lazily so the framework runs on
machines without mlx installed). Prompt caching, batch generation and the
dedicated MLX thread all keep working exactly as before.

Ollama and API paths: raw HTTP (no extra dependency beyond ``requests``).
``apply_chat_template`` returns a ChatPrompt (messages + tools) instead of a
string; templating happens server-side. Two translations keep the rest of the
framework backend-agnostic:
  - native tool calls are re-emitted as ``[TOOL_CALLS][{"name":..,"arguments":..}]``
    text, the format core/parser.parse_tool_calls already handles;
  - reasoning/thinking output is wrapped in [THINK]...[/THINK], the format the
    LRM generator's streaming state machine expects.
The server manages its own KV-cache, so prompt_cache is always None on these
paths and the on-disk cache persistence in ChatSession is disabled.

The API path speaks two protocols: the OpenAI chat-completions format (OpenAI,
Gemini's OpenAI-compatible endpoint, Mistral, OpenRouter, DeepSeek, and local
servers such as LM Studio or vLLM) and Anthropic's Messages API.
"""
import json
import os

import requests

from core import cancel
from core.config import (
    LLM_BACKEND,
    OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE,
    API_PROVIDER, API_MODEL, API_BASE_URL, API_TIMEOUT,
)
from core.logging_config import get_logger

logger = get_logger(__name__)

# The model load_llm() loaded: a folder, repo id or Ollama/API model name (for display).
ACTIVE_MODEL = None

_TOOL_MARKER = "[TOOL_CALLS]"
_THINK_START = "[THINK]"
_THINK_END = "[/THINK]"


def supports_prompt_cache() -> bool:
    """Whether the active backend supports client-side persistent KV-caches."""
    return LLM_BACKEND == "mlx"


def load_llm():
    """Load the configured backend. Returns (model, tokenizer).

    For MLX the pair is the real mlx_lm model + tokenizer (loaded on the MLX
    thread). For Ollama and the API it's (OllamaModel | ApiModel, ChatTokenizer)
    — handles that the dispatch functions below understand.
    """
    global ACTIVE_MODEL
    from core import model_setup
    if LLM_BACKEND == "ollama":
        model = OllamaModel(OLLAMA_MODEL, OLLAMA_HOST)
        model.name = model_setup.ensure_ollama_model(model)   # asks and pulls if missing
        model.verify_available()
        ACTIVE_MODEL = model.name
        logger.info(f"Ollama backend ready: model '{OLLAMA_MODEL}' at {OLLAMA_HOST}")
        return model, ChatTokenizer()
    if LLM_BACKEND == "api":
        model = ApiModel(API_PROVIDER, API_MODEL, API_BASE_URL)
        model.verify_available()
        ACTIVE_MODEL = model.name
        print(f"API backend ready: {model.provider} model '{model.name}'. "
              f"Questions and retrieved passages are sent to {model.base_url}.")
        return model, ChatTokenizer()
    if LLM_BACKEND == "mlx":
        from mlx_lm import load
        from core.mlx_runtime import run_on_mlx
        path = model_setup.mlx_model_path()       # asks and downloads if there is none
        ACTIVE_MODEL = path
        # Load on the MLX thread so the weights share its stream affinity (see mlx_runtime).
        return run_on_mlx(load, path)
    raise ValueError(f"Unknown LLM_BACKEND '{LLM_BACKEND}' (expected 'mlx', 'ollama' or 'api')")


def is_remote(model) -> bool:
    """True for backends that generate over HTTP (no token-level access, no batching)."""
    return isinstance(model, (OllamaModel, ApiModel))


# ── Sampling specs ─────────────────────────────────────────────────
# Backend-neutral holders for sampling parameters. The MLX path converts them
# lazily into real mlx_lm sampler/processor objects; the Ollama path maps them
# onto request options.

class SamplerSpec:
    def __init__(self, temp=0.0, top_p=0.0, min_p=0.0, top_k=-1, **extra):
        self.temp = temp
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.extra = extra
        self._mlx = None

    def mlx_sampler(self):
        if self._mlx is None:
            from mlx_lm.sample_utils import make_sampler as _mlx_make_sampler
            self._mlx = _mlx_make_sampler(
                temp=self.temp, top_p=self.top_p, min_p=self.min_p,
                top_k=self.top_k, **self.extra
            )
        return self._mlx


class LogitsProcessorsSpec:
    def __init__(self, repetition_penalty=None, repetition_context_size=20, **extra):
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        self.extra = extra
        self._mlx = None

    def mlx_processors(self):
        if self._mlx is None:
            from mlx_lm.sample_utils import make_logits_processors as _mlx_make
            self._mlx = _mlx_make(
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                **self.extra
            )
        return self._mlx


def make_sampler(temp=0.0, top_p=0.0, min_p=0.0, top_k=-1, **extra):
    return SamplerSpec(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k, **extra)


def make_logits_processors(repetition_penalty=None, repetition_context_size=20, **extra):
    return LogitsProcessorsSpec(
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        **extra
    )


def _as_mlx_sampler(sampler):
    if sampler is None:
        return None
    if isinstance(sampler, SamplerSpec):
        return sampler.mlx_sampler()
    return sampler  # already a raw mlx sampler


def _as_mlx_processors(logits_processors):
    if logits_processors is None:
        return None
    if isinstance(logits_processors, LogitsProcessorsSpec):
        return logits_processors.mlx_processors()
    return logits_processors


# ── Ollama handles ─────────────────────────────────────────────────

class OllamaModel:
    """Handle passed around as `llm_model` when LLM_BACKEND == 'ollama'."""

    def __init__(self, name, host):
        self.name = name
        self.host = host.rstrip("/")
        self.session = requests.Session()
        # None = unknown, False = server said the model can't take a tools param
        self.tools_supported = None

    def has_model(self):
        """Whether Ollama already has this model (raises if Ollama can't be reached)."""
        try:
            r = self.session.post(f"{self.host}/api/show", json={"model": self.name}, timeout=10)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is the server running? "
                f"Start it with `ollama serve` (or open the Ollama app)."
            ) from e
        return r.status_code != 404

    def verify_available(self):
        try:
            r = self.session.post(f"{self.host}/api/show", json={"model": self.name}, timeout=10)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is the server running? "
                f"Start it with `ollama serve` (or open the Ollama app)."
            ) from e
        if r.status_code == 404:
            raise RuntimeError(
                f"Model '{self.name}' is not available in Ollama. "
                f"Pull it first with: ollama pull {self.name}"
            )
        r.raise_for_status()


class ChatPrompt:
    """Opaque prompt returned by ChatTokenizer.apply_chat_template.

    Holds the raw conversation + tool schemas; the actual chat template is
    applied server-side (Ollama or the API provider).
    """

    def __init__(self, messages, tools=None):
        self.messages = messages
        self.tools = tools

    def __str__(self):
        parts = []
        for m in self.messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if m.get("tool_calls"):
                content = f"{content} [tool_calls: {json.dumps(m['tool_calls'], ensure_ascii=False)}]"
            parts.append(f"<{m.get('role', '?')}> {content}")
        return "\n".join(parts)


class ChatTokenizer:
    """Tokenizer-shaped adapter so call sites keep their llm_tokenizer usage."""

    def apply_chat_template(self, conversation, add_generation_prompt=True,
                            tools=None, tokenize=False, **kwargs):
        # Copy so later conversation.append() calls don't mutate the prompt.
        return ChatPrompt(messages=list(conversation), tools=tools)

    def encode(self, text, add_special_tokens=False, **kwargs):
        """Rough token-count estimate (~4 chars/token). Only used for logging;
        all cache-suffix logic is skipped on remote backends (no prompt cache)."""
        return [0] * max(1, len(str(text)) // 4)

    def decode(self, token_ids, **kwargs):
        raise NotImplementedError(
            "ChatTokenizer cannot decode token ids; token-level operations "
            "are MLX-only (batch generation runs sequentially on remote backends)."
        )


class _StreamChunk:
    """Duck-typed stand-in for mlx_lm's GenerationResponse (only .text is used)."""

    def __init__(self, text):
        self.text = text


# ── Ollama request plumbing ────────────────────────────────────────

def _ollama_options(max_tokens, sampler, logits_processors):
    """Map sampling specs onto Ollama options.

    Everything is set explicitly: Ollama's server defaults (temperature 0.8,
    repeat_penalty 1.1, min_p 0.05, num_ctx 4096) differ from mlx_lm's
    "no sampler = greedy, no processors = no penalty" semantics.
    """
    s = sampler if isinstance(sampler, SamplerSpec) else SamplerSpec()
    lp = logits_processors if isinstance(logits_processors, LogitsProcessorsSpec) else None
    opts = {
        "num_predict": max_tokens,
        "num_ctx": OLLAMA_NUM_CTX,
        "temperature": s.temp,
        "top_p": s.top_p if 0.0 < s.top_p < 1.0 else 1.0,
        "top_k": s.top_k if s.top_k and s.top_k > 0 else 0,
        "min_p": s.min_p,
        "repeat_penalty": (lp.repetition_penalty if lp and lp.repetition_penalty else 1.0),
    }
    if opts["repeat_penalty"] != 1.0 and lp.repetition_context_size:
        opts["repeat_last_n"] = lp.repetition_context_size
    return opts


def _to_ollama_messages(conversation):
    """Convert the framework's conversation entries to Ollama chat messages.

    - list-form assistant content (LRM thinking) is flattened to its text parts
    - tool_calls arguments are JSON strings internally; Ollama wants objects
    - tool results keep the tool name via `tool_name` (ignored by older servers)
    """
    msgs = []
    for m in conversation:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        out = {"role": role, "content": content or ""}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            out["tool_calls"] = calls
        if role == "tool" and m.get("name"):
            out["tool_name"] = m["name"]
        msgs.append(out)
    return msgs


def _ollama_post_stream(model, endpoint, payload):
    try:
        resp = model.session.post(
            f"{model.host}{endpoint}", json=payload, stream=True, timeout=(10, 600)
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Lost connection to Ollama at {model.host}. Is the server still running?"
        ) from e
    return resp


def _ollama_chat_stream(model, prompt, options):
    """Stream /api/chat. Yields _StreamChunk objects; re-encodes native tool
    calls and thinking into the framework's text-marker formats."""
    payload = {
        "model": model.name,
        "messages": _to_ollama_messages(prompt.messages),
        "stream": True,
        "options": options,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    if prompt.tools and model.tools_supported is not False:
        payload["tools"] = prompt.tools

    resp = _ollama_post_stream(model, "/api/chat", payload)
    if resp.status_code == 400 and "tools" in payload:
        try:
            err = resp.json().get("error", "")
        except ValueError:
            err = resp.text
        if "does not support tools" in err:
            print(f"[Ollama] WARNING: model '{model.name}' does not support tool calling — "
                  f"retrying without tools. RAG tool use is disabled; switch to a "
                  f"tool-capable model (e.g. mistral, llama3.1, qwen3) to re-enable it.")
            model.tools_supported = False
            payload.pop("tools")
            resp = _ollama_post_stream(model, "/api/chat", payload)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", resp.text)
        except ValueError:
            err = resp.text
        raise RuntimeError(f"Ollama /api/chat error {resp.status_code}: {err}")
    if payload.get("tools"):
        model.tools_supported = True

    tool_calls = []
    think_open = False
    with resp:
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"Ollama error: {chunk['error']}")
            msg = chunk.get("message") or {}

            thinking = msg.get("thinking")
            if thinking:
                if not think_open:
                    yield _StreamChunk(_THINK_START)
                    think_open = True
                yield _StreamChunk(thinking)

            content = msg.get("content")
            if content:
                if think_open:
                    yield _StreamChunk(_THINK_END)
                    think_open = False
                yield _StreamChunk(content)

            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                tool_calls.append({
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or {},
                })

            if chunk.get("done"):
                break

    if think_open:
        yield _StreamChunk(_THINK_END)
    if tool_calls:
        yield _StreamChunk(_TOOL_MARKER + json.dumps(tool_calls, ensure_ascii=False))


def _ollama_generate_stream(model, prompt_text, options):
    """Stream /api/generate for plain-string prompts (no chat template client-side;
    Ollama wraps the text in the model's template as a user turn)."""
    payload = {
        "model": model.name,
        "prompt": prompt_text,
        "stream": True,
        "options": options,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    resp = _ollama_post_stream(model, "/api/generate", payload)
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error", resp.text)
        except ValueError:
            err = resp.text
        raise RuntimeError(f"Ollama /api/generate error {resp.status_code}: {err}")
    with resp:
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"Ollama error: {chunk['error']}")
            text = chunk.get("response")
            if text:
                yield _StreamChunk(text)
            if chunk.get("done"):
                break


# ── API backend ────────────────────────────────────────────────────

# provider -> (protocol, base URL, default model, environment variable holding the key)
API_PROVIDERS = {
    "openai":     ("openai", "https://api.openai.com/v1", "gpt-5-mini", "OPENAI_API_KEY"),
    "anthropic":  ("anthropic", "https://api.anthropic.com/v1", "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    "gemini":     ("openai", "https://generativelanguage.googleapis.com/v1beta/openai",
                   "gemini-3.8-flash", "GEMINI_API_KEY"),
    "mistral":    ("openai", "https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "openai/gpt-5-mini", "OPENROUTER_API_KEY"),
    "deepseek":   ("openai", "https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "custom":     ("openai", None, None, "API_KEY"),     # any OpenAI-compatible server
}


def _error_text(resp):
    """Best-effort error message from a provider's JSON error body."""
    try:
        err = resp.json().get("error", resp.text)
    except ValueError:
        return resp.text[:500]
    if isinstance(err, dict):
        return err.get("message") or json.dumps(err)
    return str(err)


class ApiModel:
    """Handle passed around as `llm_model` when LLM_BACKEND == 'api'."""

    def __init__(self, provider, name=None, base_url=None):
        if provider not in API_PROVIDERS:
            raise ValueError(f"Unknown API_PROVIDER '{provider}' "
                             f"(expected one of: {', '.join(API_PROVIDERS)})")
        protocol, default_url, default_model, key_env = API_PROVIDERS[provider]
        self.provider = provider
        self.protocol = protocol
        self.base_url = (base_url or default_url or "").rstrip("/")
        self.name = name or default_model
        self.key_env = key_env
        self.api_key = os.environ.get(key_env) or (
            os.environ.get("GOOGLE_API_KEY", "") if provider == "gemini" else "")
        self.session = requests.Session()
        if protocol == "anthropic":
            self.session.headers.update({"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        elif self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        # None = unknown, False = provider said the model can't take a tools param
        self.tools_supported = None
        # Sampling parameters the model rejected (e.g. temperature on reasoning models)
        self.dropped_params = set()

    def verify_available(self):
        if not self.base_url:
            raise RuntimeError("API_PROVIDER 'custom' needs API_BASE_URL in core/config.py, "
                               "e.g. 'http://localhost:1234/v1'.")
        if not self.name:
            raise RuntimeError("Set API_MODEL in core/config.py to the model name your server expects.")
        if not self.api_key and self.provider != "custom":
            raise RuntimeError(
                f"No API key for {self.provider}. Add {self.key_env}=<your key> to the .env file "
                f"in the project folder, or set it in the settings menu (python main.py --config).")
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=15)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot reach {self.base_url}: {e}") from e
        if r.status_code in (401, 403):
            raise RuntimeError(f"{self.provider} rejected the key in {self.key_env} "
                               f"({r.status_code}): {_error_text(r)}")
        # Anything else (e.g. a local server without /models) is left for the
        # first real request to report.


def _flatten_content(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _args_as_json(args):
    if isinstance(args, str):
        return args
    return json.dumps(args or {}, ensure_ascii=False)


def _args_as_dict(args):
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args) if args else {}
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_openai_messages(conversation):
    """Conversation entries are already OpenAI-shaped; flatten list content
    (LRM thinking) and keep only the fields the API accepts."""
    msgs = []
    for m in conversation:
        role = m.get("role", "user")
        out = {"role": role, "content": _flatten_content(m.get("content"))}
        if role == "assistant" and m.get("tool_calls"):
            out["tool_calls"] = [{
                "id": tc.get("id", ""),
                "type": "function",
                "function": {"name": tc.get("function", {}).get("name", ""),
                             "arguments": _args_as_json(tc.get("function", {}).get("arguments"))},
            } for tc in m["tool_calls"]]
        if role == "tool":
            out["tool_call_id"] = m.get("tool_call_id", "")
        msgs.append(out)
    return msgs


def _to_anthropic_messages(conversation):
    """-> (system text, messages) for the Messages API.

    System entries move to the top-level `system` field, assistant tool calls
    become tool_use blocks, tool results become tool_result blocks in a user
    turn, and consecutive turns of the same role are merged.
    """
    system, messages = [], []

    def add(role, blocks):
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in conversation:
        role = m.get("role", "user")
        text = _flatten_content(m.get("content"))
        if role == "system":
            if text:
                system.append(text)
        elif role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                          "content": text or "(no result)"}])
        elif role == "assistant":
            blocks = [{"type": "text", "text": text}] if text.strip() else []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""),
                               "input": _args_as_dict(fn.get("arguments"))})
            if blocks:
                add("assistant", blocks)
        elif text.strip():
            add("user", [{"type": "text", "text": text}])
    return "\n\n".join(system), messages


def _api_post_stream(model, url, payload):
    """POST a streaming request. Parameters the model rejects (temperature on
    reasoning models, tools on models without tool use) are dropped and the
    request retried, and remembered for later requests."""
    for _ in range(4):
        try:
            resp = model.session.post(url, json=payload, stream=True, timeout=(10, API_TIMEOUT))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Lost connection to {model.base_url}: {e}") from e
        if resp.status_code < 400:
            if payload.get("tools"):
                model.tools_supported = True
            return resp

        err = _error_text(resp)
        lowered = err.lower()
        retry = False
        if resp.status_code == 400:
            for param in ("temperature", "top_p"):
                if param in payload and param in lowered:
                    payload.pop(param)
                    model.dropped_params.add(param)
                    retry = True
            if "tools" in payload and "tool" in lowered and "support" in lowered:
                print(f"[API] WARNING: model '{model.name}' does not support tool calling — "
                      f"retrying without tools. Document search is disabled; pick a "
                      f"tool-capable model to re-enable it.")
                payload.pop("tools")
                model.tools_supported = False
                retry = True
        if not retry:
            raise RuntimeError(f"{model.provider} API error {resp.status_code}: {err}")
    raise RuntimeError(f"{model.provider} API kept rejecting the request: {err}")


def _sse_events(resp):
    """Yield the JSON payload of each `data:` line of a server-sent-events stream."""
    with resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def _openai_chat_stream(model, prompt, max_tokens, sampler):
    """Stream /chat/completions. Yields _StreamChunk objects; re-encodes native
    tool calls and reasoning text into the framework's text-marker formats."""
    s = sampler if isinstance(sampler, SamplerSpec) else SamplerSpec()
    if isinstance(prompt, ChatPrompt):
        messages, tools = _to_openai_messages(prompt.messages), prompt.tools
    else:
        messages, tools = [{"role": "user", "content": str(prompt)}], None
    payload = {"model": model.name, "messages": messages, "stream": True}
    # OpenAI's reasoning models only accept max_completion_tokens; other providers use max_tokens.
    payload["max_completion_tokens" if model.provider == "openai" else "max_tokens"] = max_tokens
    if "temperature" not in model.dropped_params:
        payload["temperature"] = s.temp
    if 0.0 < s.top_p < 1.0 and "top_p" not in model.dropped_params:
        payload["top_p"] = s.top_p
    if tools and model.tools_supported is not False:
        payload["tools"] = tools

    resp = _api_post_stream(model, f"{model.base_url}/chat/completions", payload)
    calls = {}
    think_open = False
    for event in _sse_events(resp):
        if event.get("error"):
            err = event["error"]
            raise RuntimeError(f"{model.provider} API error: "
                               f"{err.get('message', err) if isinstance(err, dict) else err}")
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            # DeepSeek and OpenRouter stream reasoning separately from the answer.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                if not think_open:
                    yield _StreamChunk(_THINK_START)
                    think_open = True
                yield _StreamChunk(reasoning)
            content = delta.get("content")
            if content:
                if think_open:
                    yield _StreamChunk(_THINK_END)
                    think_open = False
                yield _StreamChunk(content)
            for tc in delta.get("tool_calls") or []:
                index = tc.get("index")
                slot = calls.setdefault(len(calls) if index is None else index,
                                        {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += _args_as_json(fn["arguments"])

    if think_open:
        yield _StreamChunk(_THINK_END)
    if calls:
        parsed = [{"name": c["name"], "arguments": _args_as_dict(c["arguments"])}
                  for _, c in sorted(calls.items())]
        yield _StreamChunk(_TOOL_MARKER + json.dumps(parsed, ensure_ascii=False))


def _anthropic_stream(model, prompt, max_tokens, sampler):
    """Stream the Messages API, with the same re-encoding as the OpenAI path."""
    s = sampler if isinstance(sampler, SamplerSpec) else SamplerSpec()
    if isinstance(prompt, ChatPrompt):
        system, messages = _to_anthropic_messages(prompt.messages)
        tools = prompt.tools
    else:
        system, messages, tools = "", [{"role": "user", "content": str(prompt)}], None
    payload = {"model": model.name, "max_tokens": max_tokens, "messages": messages, "stream": True}
    if system:
        payload["system"] = system
    # Temperature only: recent Claude models reject temperature and top_p together.
    if "temperature" not in model.dropped_params:
        payload["temperature"] = min(max(s.temp, 0.0), 1.0)
    if tools and model.tools_supported is not False:
        payload["tools"] = [{
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
        } for t in tools if t.get("type") == "function"]

    resp = _api_post_stream(model, f"{model.base_url}/messages", payload)
    tool_blocks = {}
    think_open = False
    for event in _sse_events(resp):
        kind = event.get("type")
        if kind == "error":
            err = event.get("error") or {}
            raise RuntimeError(f"anthropic API error: {err.get('message', err)}")
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                tool_blocks[event.get("index")] = {"name": block.get("name", ""), "json": ""}
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                if think_open:
                    yield _StreamChunk(_THINK_END)
                    think_open = False
                yield _StreamChunk(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                if not think_open:
                    yield _StreamChunk(_THINK_START)
                    think_open = True
                yield _StreamChunk(delta.get("thinking", ""))
            elif delta.get("type") == "input_json_delta" and event.get("index") in tool_blocks:
                tool_blocks[event["index"]]["json"] += delta.get("partial_json", "")
        elif kind == "message_stop":
            break

    if think_open:
        yield _StreamChunk(_THINK_END)
    if tool_blocks:
        parsed = [{"name": b["name"], "arguments": _args_as_dict(b["json"])}
                  for _, b in sorted(tool_blocks.items())]
        yield _StreamChunk(_TOOL_MARKER + json.dumps(parsed, ensure_ascii=False))


# ── Dispatch functions (mlx_lm-compatible signatures) ──────────────

def _track_cache_tokens(chunks, prompt_cache, prompt):
    """Pass MLX chunks through, then tell CacheManager which tokens the cache now holds."""
    from core.cache_manager import CacheManager
    generated = []
    try:
        for chunk in chunks:
            token = getattr(chunk, "token", None)
            if token is not None:
                generated.append(int(token))
            yield chunk
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()
        CacheManager.after_generation(prompt_cache, prompt, generated)


def _cancellable(chunks):
    """Pass chunks through, stopping (and closing the stream) once a cancel is requested."""
    try:
        for chunk in chunks:
            cancel.check()
            yield chunk
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()


def stream_generate(model, tokenizer=None, prompt=None, max_tokens=256,
                    sampler=None, logits_processors=None, prompt_cache=None,
                    **kwargs):
    """Yield chunks with a .text attribute, whatever the backend. Every backend
    stops at the next chunk once core.cancel.request() is called."""
    if isinstance(model, OllamaModel):
        options = _ollama_options(max_tokens, sampler, logits_processors)
        if isinstance(prompt, ChatPrompt):
            return _cancellable(_ollama_chat_stream(model, prompt, options))
        return _cancellable(_ollama_generate_stream(model, str(prompt), options))
    if isinstance(model, ApiModel):
        # Repetition penalties have no portable API equivalent and are ignored here.
        if model.protocol == "anthropic":
            return _cancellable(_anthropic_stream(model, prompt, max_tokens, sampler))
        return _cancellable(_openai_chat_stream(model, prompt, max_tokens, sampler))

    from mlx_lm import stream_generate as _mlx_stream_generate
    mlx_kwargs = dict(kwargs)
    if prompt_cache is not None:
        mlx_kwargs["prompt_cache"] = prompt_cache
    s = _as_mlx_sampler(sampler)
    if s is not None:
        mlx_kwargs["sampler"] = s
    lp = _as_mlx_processors(logits_processors)
    if lp is not None:
        mlx_kwargs["logits_processors"] = lp
    # Called after every 2048-token prefill step: lets a cancel interrupt a long
    # prompt, which produces no tokens until it has been read.
    mlx_kwargs.setdefault("prompt_progress_callback", lambda done, total: cancel.check())
    chunks = _mlx_stream_generate(
        model=model, tokenizer=tokenizer, prompt=prompt,
        max_tokens=max_tokens, **mlx_kwargs
    )
    if prompt_cache is not None:
        chunks = _track_cache_tokens(chunks, prompt_cache, prompt)
    return _cancellable(chunks)


def generate(model, tokenizer=None, prompt=None, max_tokens=256, verbose=False,
             sampler=None, logits_processors=None, prompt_cache=None, **kwargs):
    """Blocking generation returning the full text, whatever the backend.

    Built on stream_generate (also for MLX) so it stops as soon as a cancel is
    requested, like every other generation.
    """
    pieces = []
    for chunk in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler,
        logits_processors=logits_processors, prompt_cache=prompt_cache, **kwargs,
    ):
        pieces.append(chunk.text)
        if verbose:
            print(chunk.text, end="", flush=True)
    if verbose:
        print(flush=True)
    return "".join(pieces)
