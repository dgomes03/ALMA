# The screen layout (transcript, loading line, input box, bottom bar) follows
# Mistral Vibe v2.26.1's VibeApp (vibe/cli/textual_ui/app.py), Copyright 2025
# Mistral AI, Apache License 2.0 — see THIRD_PARTY_NOTICES.md. The bridge to
# ALMA's generators and the command handling are ALMA's own.
"""ALMA's terminal chat, built on Textual."""
from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalGroup
from textual.widgets import Static

from core import cancel
from core.chat_ui import (
    COMMANDS, HISTORY_FILE, TOOL_LABELS, _BARE_COMMANDS, _document_count, _model_label,
)
from core.tui.widgets import (
    AssistantMessage, Banner, ChatInput, ChatScroll, ChatTextArea, CompletionEntry, ErrorMessage,
    InterruptMessage, LoadingWidget, NoMarkupStatic, NoticeMessage, ToolCallMessage, UserMessage,
)

HELP_TEXT = "\n".join(
    [f"{cmd:<9} {desc}" for cmd, desc in COMMANDS.items()]
    + ["", "Shift+Enter  new line        Up/Down  earlier messages",
       "Esc          stop an answer  Ctrl+C   copy selection, clear input, or twice to quit"])


def _tool_text(name: str, args) -> str:
    label = TOOL_LABELS.get(name, name.replace("_", " ").capitalize())
    detail = next((str(v) for v in args.values() if v), "") if isinstance(args, dict) else ""
    if len(detail) > 80:
        detail = detail[:77] + "..."
    return f'{label}: "{detail}"' if detail else label


class ALMAApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "ALMA"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", show=False, priority=True),
        Binding("ctrl+d", "quit", show=False, priority=True),
        Binding("escape", "interrupt", show=False),
    ]

    def __init__(self, retriever, run_query) -> None:
        super().__init__()
        self.theme = "textual-light"        # white background; app.tcss keeps the blue accent
        self.retriever = retriever
        self.run_query = run_query
        self.state = {"last_prompt": None}
        self.busy = False
        self.muted = False                  # after Esc: ignore the rest of the answer
        self._answer: AssistantMessage | None = None
        self._tool: ToolCallMessage | None = None
        self._loading: LoadingWidget | None = None
        self._last_ctrl_c = 0.0

    # ── layout ──────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        model = _model_label(self.retriever)
        with ChatScroll(id="chat"):
            yield Banner(model, _document_count(self.retriever))
            yield VerticalGroup(id="messages")
        with Horizontal(id="loading-area"):
            yield Static(id="loading-area-content")
        with Static(id="bottom-app-container"):
            yield ChatInput(HISTORY_FILE, [CompletionEntry(c, d) for c, d in COMMANDS.items()],
                            id="input-container")
        with Horizontal(id="bottom-bar"):
            yield NoMarkupStatic("/help commands · /config settings",
                                 id="bottom-hints")
            yield NoMarkupStatic(model, id="bottom-model")

    def on_mount(self) -> None:
        self.query_one("#chat", ChatScroll).anchor()
        self.query_one("#loading-area").display = False
        self.query_one(ChatTextArea).focus()

    async def _add(self, widget) -> None:
        await self.query_one("#messages").mount(widget)

    # ── input ───────────────────────────────────────────────
    async def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        area = self.query_one(ChatTextArea)
        if self.busy:
            self.notify("Wait for the answer to finish, or press Esc to stop it.", severity="warning")
            return
        text = event.value
        area.remember(text)
        area.clear_input()
        if text.startswith("/") or text.lower() in _BARE_COMMANDS:
            await self._run_command(text)
        else:
            await self._ask(text)

    async def _run_command(self, raw: str) -> None:
        await self._add(UserMessage(raw, command=True))
        session = self.retriever.session
        cmd, _, arg = raw.strip().partition(" ")
        cmd, arg = cmd.lower().lstrip("/"), arg.strip()
        if cmd in ("exit", "quit"):
            self.exit()
        elif cmd == "help":
            await self._add(NoticeMessage(HELP_TEXT))
        elif cmd == "config":
            await self._open_settings()
        elif cmd == "clear":
            session.clear()
            self.state["last_prompt"] = None
            await self.query_one("#messages").remove_children()
            await self._add(NoticeMessage("New conversation."))
        elif cmd == "save":
            chat = session.save_cli(last_prompt=self.state["last_prompt"])
            await self._add(NoticeMessage(f"Saved as {chat['id']} ({chat['title']})"))
        elif cmd == "list":
            chats = session.list_chats()
            lines = [f"{'●' if c['id'] == session.current_chat_id else ' '} {c['id']}  "
                     f"{c['title'][:60]}  ({c['message_count']} messages)" for c in chats]
            await self._add(NoticeMessage("\n".join(lines) or "No saved conversations."))
        elif cmd == "load":
            if not arg:
                await self._add(NoticeMessage("Usage: /load <id>  (see /list)"))
            else:
                session.restore(arg)
                self.state["last_prompt"] = None
                await self._add(NoticeMessage(f"Reopened {arg}."))
        elif cmd == "delete":
            if not arg:
                await self._add(NoticeMessage("Usage: /delete <id>  (see /list)"))
            else:
                done = session.delete_chat(arg)
                await self._add(NoticeMessage("Deleted." if done else "No conversation with that id."))
        else:
            await self._add(NoticeMessage(f"Unknown command /{cmd}. Type /help for the list."))

    async def _open_settings(self) -> None:
        from textual.app import SuspendNotSupported
        from core.config_menu import run_menu
        try:
            with self.suspend():
                run_menu()
        except SuspendNotSupported:
            await self._add(NoticeMessage("The settings menu can't open from here on this system. "
                                          "Quit and run settings_windows.bat instead."))
            return
        await self._add(NoticeMessage("Settings closed. Saved changes apply the next time ALMA starts."))

    # ── answering ───────────────────────────────────────────
    async def _ask(self, question: str) -> None:
        await self._add(UserMessage(question))
        self.busy, self.muted = True, False
        self._loading = LoadingWidget("Thinking")
        area = self.query_one("#loading-area")
        area.display = True
        await self.query_one("#loading-area-content", Static).mount(self._loading)
        self.run_worker(lambda: self._generate(question), thread=True, exclusive=True)

    def _generate(self, question: str) -> None:
        """Runs in a worker thread; the generator calls _on_event as it streams."""
        try:
            _, self.state["last_prompt"] = self.run_query(question, self.retriever,
                                                          stream_callback=self._on_event)
            self.call_from_thread(self._finish, None)
        except cancel.Cancelled:                    # stopped with Esc; already rolled back
            self.call_from_thread(self._finish, None)
        except Exception as e:                      # noqa: BLE001 - shown to the user
            self.call_from_thread(self._finish, str(e))

    def _on_event(self, event: str, data) -> None:
        if not self.muted:
            self.call_from_thread(self._handle_event, event, data or {})

    async def _settle_tool(self, success: bool = True) -> None:
        if self._tool is not None:
            self._tool.stop_spinning(success)
            self._tool = None

    async def _close_answer(self) -> None:
        if self._answer is not None:
            await self._answer.stop_stream()
            self._answer = None

    async def _handle_event(self, event: str, data: dict) -> None:
        if self.muted:
            return
        if event == "text_chunk":
            await self._settle_tool()
            if self._answer is None:
                self._answer = AssistantMessage()
                await self._add(self._answer)
            self._loading and self._loading.set_status("Writing")
            await self._answer.append_content(data.get("text", ""))
        elif event == "tool_call_start":
            await self._close_answer()
            await self._settle_tool()
            self._tool = ToolCallMessage(_tool_text(data.get("tool_name", ""), data.get("arguments")))
            await self._add(self._tool)
            self._loading and self._loading.set_status("Searching")
        elif event == "tool_call_result":
            failed = data.get("status", "success") == "error"
            await self._settle_tool(not failed)
            if failed:
                reason = str(data.get("result", "")).strip().splitlines()[:1]
                await self._add(NoticeMessage(reason[0][:200] if reason else "The tool failed."))
            self._loading and self._loading.set_status("Reading results")
        elif event == "reasoning_step":
            goals = data.get("goals") or []
            if goals:
                plan = "; ".join(g.get("description", "") for g in goals[:4])
                await self._add(ToolCallMessage(f"Research plan: {plan}", running=False))
            self._loading and self._loading.set_status("Researching")
        elif event in ("thinking_start", "thinking_chunk"):
            self._loading and self._loading.set_status("Reasoning")

    async def _finish(self, error: str | None) -> None:
        await self._settle_tool(error is None)
        await self._close_answer()
        if self._loading is not None:
            await self._loading.remove()
            self._loading = None
        self.query_one("#loading-area").display = False
        if error and not self.muted:
            await self._add(ErrorMessage(f"Something went wrong: {error}"))
        self.busy = False
        self.query_one(ChatTextArea).focus()

    # ── keys ────────────────────────────────────────────────
    async def action_interrupt(self) -> None:
        if not self.busy or self.muted:
            return
        self.muted = True
        cancel.request()
        await self._settle_tool(False)
        await self._close_answer()
        await self._add(InterruptMessage("Interrupted"))
        if self._loading is not None:
            self._loading.set_status("Stopping")

    async def action_interrupt_or_quit(self) -> None:
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.notify("Copied.")
            return
        if self.busy and not self.muted:
            await self.action_interrupt()
            return
        area = self.query_one(ChatTextArea)
        if area.text:
            area.clear_input()
            return
        if time.time() - self._last_ctrl_c < 2:
            self.exit()
            return
        self._last_ctrl_c = time.time()
        self.notify("Press Ctrl+C again to quit.")


def prepare_for_textual(retriever) -> None:
    """One-time set-up that must happen before the app starts.

    While Textual runs, sys.stderr is a capture object with no real file
    descriptor, so starting a helper process fails with "bad value(s) in
    fds_to_keep". tqdm's lock (taken on first use of a progress bar, e.g. when
    the embedding model encodes a query) starts multiprocessing's resource
    tracker that way. Start both now, and load the embedding and reranker
    models so the first question doesn't wait for them either.
    """
    from multiprocessing import resource_tracker
    from tqdm import tqdm
    resource_tracker.ensure_running()
    tqdm.get_lock()
    retriever.embedding_model
    retriever.reranker


def run(retriever, run_query) -> None:
    prepare_for_textual(retriever)
    ALMAApp(retriever, run_query).run()
