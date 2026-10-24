"""
Tests for the terminal chat: which inputs count as commands (plain loop), and
the Textual app driven headlessly with a fake model.

Run from the repository root:
    pytest tests/test_chat_ui.py -v
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from core.chat_ui import _handle_command
from core.tui.app import ALMAApp
from core.tui.widgets import (
    AssistantMessage, ChatTextArea, CompletionPopup, NoticeMessage, ToolCallMessage, UserMessage,
)


@pytest.fixture(autouse=True)
def _private_history(tmp_path, monkeypatch):
    """Keep test messages out of the real input history."""
    import core.tui.app
    monkeypatch.setattr(core.tui.app, "HISTORY_FILE", str(tmp_path / "history"))


@pytest.mark.parametrize("text, is_command", [
    ("/help", True), ("help", True), ("/load abc", True), ("/whatever", True),
    ("list the reports about fires", False), ("load the report", False), ("olá", False),
])
def test_what_counts_as_a_command(text, is_command, capsys):
    session = MagicMock(current_chat_id=None)
    session.list_chats.return_value = []
    assert bool(_handle_command(text, session, {"last_prompt": None})) is is_command


def _fake_retriever():
    retriever = MagicMock()
    retriever.metadata_index = {"report.pdf": [], "report.pdf_1": []}
    retriever.session.list_chats.return_value = []
    return retriever


def _fake_query(question, retriever, stream_callback=None):
    stream_callback("tool_call_start", {"tool_name": "search_documents", "arguments": {"query": question}})
    stream_callback("tool_call_result", {"tool_name": "search_documents", "status": "success"})
    for chunk in ["**Duas** causas", " principais:\n\n- raios\n", "- negligência\n"]:
        stream_callback("text_chunk", {"text": chunk})
    return "", None


async def _type(pilot, text):
    area = pilot.app.query_one(ChatTextArea)
    area.text = text
    await pilot.press("enter")


async def _wait_idle(pilot):
    for _ in range(100):
        await pilot.pause(0.05)
        if not pilot.app.busy:
            return
    raise AssertionError("the answer never finished")


def test_question_shows_message_tool_and_markdown_answer():
    async def scenario():
        app = ALMAApp(_fake_retriever(), _fake_query)
        async with app.run_test(size=(90, 30)) as pilot:
            await _type(pilot, "quais são as causas?")
            await _wait_idle(pilot)
            users = app.query(UserMessage)
            tools = app.query(ToolCallMessage)
            answers = app.query(AssistantMessage)
            assert len(users) == 1 and len(tools) == 1 and len(answers) == 1
            assert not tools.first()._is_spinning                       # settled to ✓
            assert answers.first().get_content().startswith("**Duas** causas")
            assert app.query_one(ChatTextArea).text == ""                # input cleared
            # Resizing reflows instead of breaking the layout.
            await pilot.resize_terminal(50, 20)
            await pilot.pause(0.1)
            assert app.query_one("#input-box").size.width <= 50
    asyncio.run(scenario())


def test_slash_shows_suggestions_and_help_prints_commands():
    async def scenario():
        app = ALMAApp(_fake_retriever(), _fake_query)
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.press("/")
            await pilot.pause(0.1)
            popup = app.query_one(CompletionPopup)
            assert popup.styles.display == "block" and len(popup.children) == 8
            await pilot.press("h", "e", "l")
            await pilot.pause(0.1)
            assert len(popup.children) == 1
            await pilot.press("enter")                  # accepts the suggestion: "/help "
            await pilot.press("enter")                  # sends it
            await pilot.pause(0.1)
            notice = app.query(NoticeMessage).last()
            assert "/config" in notice._text and "/exit" in notice._text
    asyncio.run(scenario())


def test_clear_empties_the_transcript():
    async def scenario():
        app = ALMAApp(_fake_retriever(), _fake_query)
        async with app.run_test(size=(90, 30)) as pilot:
            await _type(pilot, "olá")
            await _wait_idle(pilot)
            await _type(pilot, "/clear")
            await pilot.pause(0.1)
            assert not app.query(AssistantMessage)
            app.retriever.session.clear.assert_called_once()
    asyncio.run(scenario())
