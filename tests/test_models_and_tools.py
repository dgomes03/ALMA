"""
Tests for model setup (core/model_setup.py), stopping an answer (core/cancel.py),
the document tools' long-document overview and name matching, and emoji removal.
No real model, download or network access is used.

Run from the repository root:
    pytest tests/test_models_and_tools.py -v
"""

import json
import os
import threading
import time
from unittest.mock import MagicMock

import pytest

import core.model_setup as model_setup
from core import cancel
from core.utils import strip_emoji


# ── Model setup ─────────────────────────────────────────────────

def _fake_model(folder):
    os.makedirs(folder)
    with open(os.path.join(folder, "config.json"), "w") as f:
        f.write("{}")


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    folder = tmp_path / "model"
    folder.mkdir()
    monkeypatch.setattr(model_setup, "MODEL_DIR", str(folder))
    monkeypatch.setattr(model_setup, "MODEL_PATH", None)
    return folder


def test_finds_the_model_in_the_folder_including_through_a_symlink(model_dir, tmp_path):
    real = tmp_path / "elsewhere" / "My-Model"
    _fake_model(str(real))
    os.symlink(real, model_dir / "My-Model")
    _fake_model(str(model_dir / "Unfinished.partial"))           # an interrupted download
    assert model_setup.installed_models() == [str(model_dir / "My-Model")]
    assert model_setup.mlx_model_path() == str(model_dir / "My-Model")


def test_model_path_setting_wins(model_dir, monkeypatch):
    monkeypatch.setattr(model_setup, "MODEL_PATH", "mlx-community/Some-Model-4bit")
    assert model_setup.mlx_model_path() == "mlx-community/Some-Model-4bit"


def test_without_a_terminal_it_explains_instead_of_downloading(model_dir, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError, match="No language model found"):
        model_setup.mlx_model_path()


def test_download_lands_in_partial_folder_then_is_renamed(model_dir, monkeypatch):
    seen = {}

    def fake_snapshot(repo, local_dir):
        seen["dir"] = local_dir
        _fake_model(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    path = model_setup.download_mlx(model_setup.CHOICES[0])
    assert seen["dir"].endswith(".partial")
    assert path == str(model_dir / "Ministral-3-3B-Instruct-2512-4bit")
    assert model_setup.installed_models() == [path]


def test_the_14b_is_offered_as_the_recommended_choice():
    assert "14B" in model_setup.CHOICES[model_setup.RECOMMENDED].name
    assert "minimum recommended" in model_setup.CHOICES[model_setup.RECOMMENDED].best_for


def test_ollama_pull_saves_the_choice(monkeypatch):
    model = MagicMock(host="http://localhost:11434")
    model.has_model.return_value = False
    lines = [json.dumps({"status": "pulling", "completed": 50, "total": 100}).encode(),
             json.dumps({"status": "success"}).encode()]
    resp = MagicMock()
    resp.__enter__.return_value.iter_lines.return_value = lines
    model.session.post.return_value = resp
    monkeypatch.setattr(model_setup, "_choose_download", lambda backend: model_setup.CHOICES[1])
    written = {}
    monkeypatch.setattr("core.config_menu.write_changes", lambda changes: written.update(changes))
    tag = model_setup.ensure_ollama_model(model)
    assert tag == model_setup.CHOICES[1].ollama_tag
    assert written == {"OLLAMA_MODEL": repr(tag)}


# ── Stopping an answer ──────────────────────────────────────────

class _Manager:
    def __init__(self):
        self.conversation = [{"role": "system", "content": "s"}]

    def get_conversation(self):
        return self.conversation

    def add_assistant_message(self, content):
        self.conversation.append({"role": "assistant", "content": content})


def _session():
    session = MagicMock(prompt_cache=None)
    session.conversation_manager = _Manager()
    return session


def test_cancel_stops_quickly_and_keeps_the_partial_answer():
    session = _session()

    def slow_answer():
        session.conversation_manager.conversation.append({"role": "user", "content": "q"})
        try:
            for _ in range(500):                # stands in for token generation
                cancel.check()
                time.sleep(0.01)
        except cancel.Cancelled as stop:
            stop.partial_text = "Foi lançado em 2015"
            raise
        return "never", None

    threading.Timer(0.1, cancel.request).start()
    started = time.time()
    with pytest.raises(cancel.Cancelled):
        cancel.run_cancellable(session, slow_answer)
    assert time.time() - started < 1.0
    conversation = session.conversation_manager.conversation
    assert [m["role"] for m in conversation] == ["system", "user", "assistant"]
    assert conversation[-1]["content"].startswith("Foi lançado em 2015 …")


def test_cancel_drops_a_tool_call_without_results():
    session = _session()

    def answer():
        conv = session.conversation_manager.conversation
        conv.append({"role": "user", "content": "q"})
        conv.append({"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]})
        conv.append({"role": "tool", "tool_call_id": "a", "content": "result a"})
        raise cancel.Cancelled()                # stopped while tool "b" was running

    with pytest.raises(cancel.Cancelled):
        cancel.run_cancellable(session, answer)
    assert [m["role"] for m in session.conversation_manager.conversation] == ["system", "user", "assistant"]
    assert "tool_calls" not in session.conversation_manager.conversation[-1]


# ── KV-cache reuse ──────────────────────────────────────────────

class _Layer:
    def __init__(self):
        self.offset = 0

    def trim(self, n):
        self.offset -= n


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _fill(cache, n):
    for layer in cache:
        layer.offset += n


def test_cache_reuses_only_the_matching_prefix():
    from core.cache_manager import CacheManager
    cache, tok = [_Layer(), _Layer()], _Tokenizer()
    fed = CacheManager.prepare_prompt_tokens(tok, "SYS USER1 ", cache)
    assert len(fed) == 10                                   # empty cache: everything
    _fill(cache, len(fed) + 6)                              # prompt + 7 generated, last never fed
    CacheManager.after_generation(cache, fed, [ord(c) for c in "ANSWER!"])
    assert CacheManager.tokens(cache) == [ord(c) for c in "SYS USER1 ANSWER"]

    # Next turn stores the answer differently (cut short + a note): reuse up to the difference.
    fed = CacheManager.prepare_prompt_tokens(tok, "SYS USER1 ANSW (stopped) USER2", cache)
    assert "".join(map(chr, fed)) == " (stopped) USER2"
    assert all(layer.offset == len("SYS USER1 ANSW") for layer in cache)


def test_cache_tokens_after_an_interrupted_prefill():
    from core.cache_manager import CacheManager
    cache, tok = [_Layer()], _Tokenizer()
    fed = CacheManager.prepare_prompt_tokens(tok, "ABCDEFGHIJ", cache)
    _fill(cache, 4)                                         # stopped after reading 4 tokens
    CacheManager.after_generation(cache, fed, [])
    assert CacheManager.tokens(cache) == [ord(c) for c in "ABCD"]
    fed = CacheManager.prepare_prompt_tokens(tok, "ABCDEFGHIJ more", cache)
    assert "".join(map(chr, fed)) == "EFGHIJ more"


def test_cancel_is_not_swallowed_by_ordinary_error_handlers():
    assert not issubclass(cancel.Cancelled, Exception)


# ── Document tools ──────────────────────────────────────────────

def test_overview_keeps_the_opening_and_key_sections_within_budget():
    from components.retrievers.FAISS_BM25 import _overview_indices
    texts = ["Title page " + "x" * 400] + ["filler " + "y" * 400] * 300
    texts[120] = "5 Conclusions " + "z" * 400
    picked = _overview_indices(texts, 8000)
    assert 0 in picked and 120 in picked
    assert sum(len(texts[i]) for i in picked) <= 8000 + 500
    assert picked == sorted(picked)


def test_document_names_match_without_case_or_accents():
    from components.retrievers.FAISS_BM25 import _fold
    assert _fold("Dissertação_Final.PDF") == "dissertacao_final.pdf"


# ── Emoji removal ───────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("De nada! 😊", "De nada!"), ("Ótimo 👍🏽 vamos lá", "Ótimo vamos lá"),
    ("🔥 Incêndios", "Incêndios"), ("Tabela: ✓ sim / ✗ não", "Tabela: ✓ sim / ✗ não"),
    ("valor : 3 ?", "valor : 3 ?"),
])
def test_strip_emoji(text, expected):
    assert strip_emoji(text) == expected
