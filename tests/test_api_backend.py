"""
Tests for the API backend (core/llm_backend.py) against a local mock server
that speaks the OpenAI chat-completions and Anthropic Messages protocols.
No network access or API keys are needed.

Run from the repository root:
    pytest tests/test_api_backend.py -v
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.llm_backend import (
    ApiModel, ChatTokenizer, generate, is_remote, make_sampler, stream_generate,
    _TOOL_MARKER,
)
from core.parser import parse_tool_calls


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class _Mock:
    requests = []                 # (path, headers, json body) of every POST
    reject_temperature = False    # first chat request fails like an OpenAI reasoning model
    reply = "text"                # "text" | "tool" | "reasoning"


def _sse(handler, events, done=True):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    for event in events:
        if "type" in event:
            handler.wfile.write(f"event: {event['type']}\n".encode())
        handler.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
    if done:
        handler.wfile.write(b"data: [DONE]\n\n")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        key = self.headers.get("Authorization", "") + self.headers.get("x-api-key", "")
        if "bad-key" in key:
            self._json(401, {"error": {"message": "Incorrect API key provided"}})
        else:
            self._json(200, {"data": []})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        headers = {k.lower(): v for k, v in self.headers.items()}
        _Mock.requests.append((self.path, headers, body))

        if self.path.endswith("/chat/completions"):
            if _Mock.reject_temperature and "temperature" in body:
                return self._json(400, {"error": {"message":
                    "Unsupported value: 'temperature' does not support 0.3 with this model."}})
            if _Mock.reply == "tool":
                return _sse(self, [
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                        "function": {"name": "search_documents", "arguments": '{"que'}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0,
                        "function": {"arguments": 'ry": "forest fires"}'}}]}}]},
                ])
            if _Mock.reply == "reasoning":
                return _sse(self, [
                    {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
                    {"choices": [{"delta": {"content": "Answer."}}]},
                ])
            return _sse(self, [{"choices": [{"delta": {"content": "Hello"}}]},
                               {"choices": [{"delta": {"content": " world"}}]}])

        if self.path.endswith("/messages"):
            if _Mock.reply == "tool":
                return _sse(self, [
                    {"type": "message_start", "message": {}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me look."}},
                    {"type": "content_block_start", "index": 1,
                     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search_documents", "input": {}}},
                    {"type": "content_block_delta", "index": 1,
                     "delta": {"type": "input_json_delta", "partial_json": '{"query": '}},
                    {"type": "content_block_delta", "index": 1,
                     "delta": {"type": "input_json_delta", "partial_json": '"forest fires"}'}},
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_stop"},
                ], done=False)
            return _sse(self, [
                {"type": "message_start", "message": {}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi there"}},
                {"type": "message_stop"},
            ], done=False)
        self._json(404, {"error": "not found"})


@pytest.fixture(scope="module")
def server_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset_mock():
    _Mock.requests.clear()
    _Mock.reject_temperature = False
    _Mock.reply = "text"


def _conversation_with_tool_round():
    """System + user + an earlier tool call and its result, as the generators build it."""
    return [
        {"role": "system", "content": "You are ALMA."},
        {"role": "user", "content": "What causes forest fires?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "abc123XYZ", "type": "function",
             "function": {"name": "search_documents", "arguments": '{"query": "fire causes"}'}}]},
        {"role": "tool", "name": "search_documents", "content": "Lightning and arson.",
         "tool_call_id": "abc123XYZ"},
    ]


TOOLS = [{"type": "function", "function": {
    "name": "search_documents", "description": "Search the documents.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def _model(provider, url, monkeypatch, key="test-key"):
    env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "custom": "API_KEY"}[provider]
    monkeypatch.setenv(env, key)
    return ApiModel(provider, "test-model", url)


def _text(chunks):
    return "".join(c.text for c in chunks)


# ---------------------------------------------------------------------------
# OpenAI-compatible protocol
# ---------------------------------------------------------------------------

def test_openai_streams_text(server_url, monkeypatch):
    model = _model("openai", server_url, monkeypatch)
    prompt = ChatTokenizer().apply_chat_template([{"role": "user", "content": "hi"}])
    assert _text(stream_generate(model, prompt=prompt, max_tokens=50)) == "Hello world"
    assert is_remote(model)


def test_openai_request_shape(server_url, monkeypatch):
    model = _model("openai", server_url, monkeypatch)
    prompt = ChatTokenizer().apply_chat_template(_conversation_with_tool_round(), tools=TOOLS)
    generate(model, prompt=prompt, max_tokens=64, sampler=make_sampler(temp=0.3, top_p=0.9))

    path, headers, body = _Mock.requests[-1]
    assert path == "/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
    assert body["temperature"] == 0.3 and body["top_p"] == 0.9
    assert body["tools"] == TOOLS
    assistant, tool = body["messages"][2], body["messages"][3]
    assert assistant["tool_calls"][0]["id"] == "abc123XYZ"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"query": "fire causes"}'
    assert tool == {"role": "tool", "content": "Lightning and arson.", "tool_call_id": "abc123XYZ"}


def test_other_providers_send_max_tokens(server_url, monkeypatch):
    model = _model("custom", server_url, monkeypatch)
    generate(model, prompt="hi", max_tokens=10)
    body = _Mock.requests[-1][2]
    assert body["max_tokens"] == 10 and "max_completion_tokens" not in body


def test_openai_tool_calls_become_marker(server_url, monkeypatch):
    _Mock.reply = "tool"
    model = _model("openai", server_url, monkeypatch)
    prompt = ChatTokenizer().apply_chat_template([{"role": "user", "content": "fires?"}], tools=TOOLS)
    text = generate(model, prompt=prompt, max_tokens=50)
    assert text.startswith(_TOOL_MARKER)
    assert parse_tool_calls(text) == [{"name": "search_documents", "arguments": {"query": "forest fires"}}]


def test_reasoning_is_wrapped_in_think_markers(server_url, monkeypatch):
    _Mock.reply = "reasoning"
    model = _model("custom", server_url, monkeypatch)
    assert generate(model, prompt="q", max_tokens=50) == "[THINK]thinking...[/THINK]Answer."


def test_rejected_temperature_is_dropped_and_remembered(server_url, monkeypatch):
    _Mock.reject_temperature = True
    model = _model("openai", server_url, monkeypatch)
    sampler = make_sampler(temp=0.3)
    assert generate(model, prompt="hi", max_tokens=10, sampler=sampler) == "Hello world"
    assert "temperature" in model.dropped_params
    generate(model, prompt="again", max_tokens=10, sampler=sampler)
    assert "temperature" not in _Mock.requests[-1][2]


# ---------------------------------------------------------------------------
# Anthropic Messages protocol
# ---------------------------------------------------------------------------

def test_anthropic_request_shape(server_url, monkeypatch):
    model = _model("anthropic", server_url, monkeypatch)
    prompt = ChatTokenizer().apply_chat_template(_conversation_with_tool_round(), tools=TOOLS)
    assert generate(model, prompt=prompt, max_tokens=64,
                    sampler=make_sampler(temp=0.3, top_p=0.9)) == "Hi there"

    path, headers, body = _Mock.requests[-1]
    assert path == "/messages"
    assert headers["x-api-key"] == "test-key" and headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "You are ALMA."
    assert body["temperature"] == 0.3 and "top_p" not in body
    assert body["tools"] == [{"name": "search_documents", "description": "Search the documents.",
                              "input_schema": TOOLS[0]["function"]["parameters"]}]
    user, assistant, result = body["messages"]
    assert user == {"role": "user", "content": [{"type": "text", "text": "What causes forest fires?"}]}
    assert assistant["content"] == [{"type": "tool_use", "id": "abc123XYZ", "name": "search_documents",
                                     "input": {"query": "fire causes"}}]
    assert result["role"] == "user"
    assert result["content"] == [{"type": "tool_result", "tool_use_id": "abc123XYZ",
                                  "content": "Lightning and arson."}]


def test_anthropic_tool_use_becomes_marker(server_url, monkeypatch):
    _Mock.reply = "tool"
    model = _model("anthropic", server_url, monkeypatch)
    prompt = ChatTokenizer().apply_chat_template([{"role": "user", "content": "fires?"}], tools=TOOLS)
    text = generate(model, prompt=prompt, max_tokens=50)
    assert text.startswith("Let me look.")
    assert parse_tool_calls(text) == [{"name": "search_documents", "arguments": {"query": "forest fires"}}]


# ---------------------------------------------------------------------------
# Setup errors
# ---------------------------------------------------------------------------

def test_missing_key_explains_where_to_put_it(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ApiModel("openai").verify_available()


def test_rejected_key_is_reported(server_url, monkeypatch):
    model = _model("openai", server_url, monkeypatch, key="bad-key")
    with pytest.raises(RuntimeError, match="rejected the key"):
        model.verify_available()


def test_custom_provider_needs_base_url(monkeypatch):
    with pytest.raises(RuntimeError, match="API_BASE_URL"):
        ApiModel("custom", "some-model").verify_available()


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown API_PROVIDER"):
        ApiModel("nope")
