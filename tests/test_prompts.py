"""
Tests for the prompt sets (core/prompts.py): both languages must offer the same
prompts with the same placeholders, keep JSON field and tool names in English,
and fill without errors.

Run from the repository root:
    pytest tests/test_prompts.py -v
"""

import importlib
import json
import string

import pytest

import core.config
import core.prompts as prompts
from components.retrievers.FAISS_BM25 import Retriever

FIELDS = {
    "query": "q", "datetime": "DT", "tools_block": "- tools", "max_goals": 4,
    "allowed": ["search_documents"], "description": "d", "tool": "t", "reasoning": "r",
    "goal": "g", "context": "c", "previous": "p", "n": 1, "text": "x",
}


def _placeholders(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


@pytest.fixture(params=["pt", "en"])
def language(request, monkeypatch):
    monkeypatch.setattr(core.config, "PROMPT_LANGUAGE", request.param)
    importlib.reload(prompts)
    yield request.param
    monkeypatch.undo()
    importlib.reload(prompts)


def test_both_languages_have_the_same_prompts_and_placeholders():
    pt, en = prompts._PROMPTS["pt"], prompts._PROMPTS["en"]
    assert pt.keys() == en.keys()
    for name in pt:
        assert _placeholders(pt[name]) == _placeholders(en[name]), name


def test_every_prompt_fills(language):
    for name, template in prompts._PROMPTS[language].items():
        prompts.get(name, **{k: FIELDS[k] for k in _placeholders(template)})


def test_json_examples_stay_parseable_and_in_english(language):
    evaluator = prompts.get("evaluator_system")
    schema = json.loads(evaluator[evaluator.index("{"):])
    assert set(schema) == {"is_complete", "confidence", "information_gain", "sparse_results",
                           "contradictory_info", "missing_aspects", "reasoning"}
    planner = prompts.get("planner_system", datetime="DT", tools_block="", max_goals=4)
    assert '"goals"' in planner and '"tool": "search_documents"' in planner


def test_date_line_follows_language(language):
    line = prompts.current_datetime()
    assert line.startswith("Data e hora atuais:" if language == "pt" else "Current date and time:")


@pytest.mark.parametrize("line, expected", [
    ("Palavras-chave: lei, 2009, artigo 22", "lei, artigo 22"),
    ("Keywords: fire, 2024, risk", "fire, risk"),
    ("Some rephrased question: with a colon", "Some rephrased question: with a colon"),
])
def test_keyword_line_is_stripped_in_both_languages(line, expected):
    assert Retriever._strip_bare_numeric_keywords(line) == expected
