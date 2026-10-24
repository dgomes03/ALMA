# Portions of this file are adapted from Mistral Vibe v2.26.1
# (https://github.com/mistralai/mistral-vibe, vibe/cli/textual_ui/),
# Copyright 2025 Mistral AI, licensed under the Apache License, Version 2.0.
# See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
#
# Changes for ALMA: stripped of Vibe's app-server, voice, image, queue and
# approval features; Python 3.10 compatible (no StrEnum); ALMA colours and
# labels instead of Mistral's; a smaller chat input and banner written for ALMA.
"""Textual widgets for ALMA's terminal chat."""
from __future__ import annotations

import json
import math
import random
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from time import time
from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.message import Message
from rich.style import Style
from textual.widgets import Markdown, Static, TextArea
from textual.widgets.text_area import TextAreaTheme

# ALMA's palette for the loading gradient (Vibe used Mistral's brand colours):
# shades around the accent blue that stay readable on a white background.
ALMA_COLORS = ("#2C8C9E", "#3A96C4", "#4A9FD8", "#3F7FCC", "#5266C4")


# ── Static helpers (from widgets/no_markup_static.py) ─────────────

class NoMarkupStatic(Static):
    def __init__(self, content: Any = "", **kwargs: Any) -> None:
        super().__init__(content, markup=False, **kwargs)


class NonSelectableStatic(NoMarkupStatic):
    @property
    def text_selection(self) -> None:
        return None

    @text_selection.setter
    def text_selection(self, value: Any) -> None:
        pass

    def get_selection(self, selection: Any) -> None:
        return None


# ── Braille rendering (from widgets/braille_renderer.py) ──────────

_BRAILLE_DOT_COUNT = 8


def _braille_dot_index(x: int, y: int) -> int:
    if y < 3:
        return y + 1 + 3 * x
    return 7 + x


def _braille_char_from_dot_indices(indices: list[int]) -> str:
    if any(n < 1 or n > _BRAILLE_DOT_COUNT for n in indices):
        raise ValueError(f"Invalid braille dot indices: {indices}")
    return chr(0x2800 + sum(2 ** (d - 1) for d in indices)) if indices else " "


def render_braille(dot_coords: Iterable[complex], width: int, height: int) -> str:
    """Draw dots (complex x+yj, origin top-left) with braille characters."""
    dots_matrix: list[list[list[int]]] = [
        [[] for _ in range(math.ceil(width / 2))] for _ in range(math.ceil(height / 4))
    ]
    for coord in dot_coords:
        x, y = int(coord.real // 2), int(coord.imag // 4)
        dots_matrix[y][x].append(_braille_dot_index(int(coord.real) % 2, int(coord.imag) % 4))
    return "\n".join("".join(_braille_char_from_dot_indices(d) for d in row) for row in dots_matrix)


# ── Spinners (from widgets/spinner.py) ────────────────────────────

class Spinner(ABC):
    FRAMES: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self._position = 0

    def next_frame(self) -> str:
        frame = self.FRAMES[self._position]
        self._position = (self._position + 1) % len(self.FRAMES)
        return frame

    def current_frame(self) -> str:
        return self.FRAMES[self._position]


class BrailleSpinner(Spinner):
    FRAMES: ClassVar[tuple[str, ...]] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class PulseSpinner(Spinner):
    FRAMES: ClassVar[tuple[str, ...]] = ("■",) * 6 + ("□",) * 4


class SnakeSpinner(Spinner):
    """A three-dot snake wandering a 4×4 braille grid."""

    MAP_WIDTH: ClassVar[int] = 4
    MAP_HEIGHT: ClassVar[int] = 4
    SNAKE_LENGTH: ClassVar[int] = 3

    def __init__(self) -> None:
        self._positions: list[complex] = [1, 0, 1j]
        super().__init__()

    @property
    def current_direction(self) -> complex:
        return self._positions[0] - self._positions[1]

    def _is_in_bounds(self, position: complex) -> bool:
        return 0 <= position.real < self.MAP_WIDTH and 0 <= position.imag < self.MAP_HEIGHT

    def _get_direction(self) -> complex:
        if (
            len({z.real for z in self._positions}) > 1
            and len({z.imag for z in self._positions}) > 1
            and self._is_in_bounds(self._positions[0] + self.current_direction)
        ):
            return self.current_direction
        valid = []
        for rotation in (1, 1j, -1j):
            offset = rotation * self.current_direction
            new_position = self._positions[0] + offset
            if self._is_in_bounds(new_position) and new_position not in self._positions:
                valid.append(offset)
        return random.choice(valid)

    def _next_positions(self) -> list[complex]:
        if len(self._positions) > self.SNAKE_LENGTH:
            return self._positions[: self.SNAKE_LENGTH]
        head = self._positions[0]
        direction = self._get_direction()
        if self.current_direction != direction:
            return [head + direction] + self._positions
        return [head + direction] + self._positions[:-1]

    def current_frame(self) -> str:
        return render_braille(self._positions, self.MAP_WIDTH, self.MAP_HEIGHT)

    def next_frame(self) -> str:
        self._positions = self._next_positions()
        return self.current_frame()


class SpinnerType(Enum):
    BRAILLE = auto()
    PULSE = auto()
    SNAKE = auto()


_SPINNER_CLASSES = {SpinnerType.BRAILLE: BrailleSpinner, SpinnerType.PULSE: PulseSpinner,
                    SpinnerType.SNAKE: SnakeSpinner}


class SpinnerMixin:
    SPINNER_TYPE: ClassVar[SpinnerType] = SpinnerType.BRAILLE

    def init_spinner(self) -> None:
        self._spinner = _SPINNER_CLASSES[self.SPINNER_TYPE]()
        self._spinner_timer = None
        self._is_spinning = True
        self._indicator_widget: Static | None = None

    def start_spinner_timer(self) -> None:
        self._spinner_timer = self.set_interval(0.1, self._update_spinner_frame)  # type: ignore[attr-defined]

    def _update_spinner_frame(self) -> None:
        if not self._is_spinning or not self._indicator_widget:
            return
        # Frames are all the same size, so skip the relayout.
        self._indicator_widget.update(self._spinner.next_frame(), layout=False)

    def stop_spinning(self, success: bool = True) -> None:
        self._is_spinning = False
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None
        if self._indicator_widget:
            self._indicator_widget.update("✓" if success else "✕")
            self._indicator_widget.add_class("success" if success else "error")

    def on_unmount(self) -> None:
        if self._spinner_timer:
            self._spinner_timer.stop()
            self._spinner_timer = None


# ── Transcript pieces (from widgets/messages.py) ──────────────────

class ExpandingBorder(NonSelectableStatic):
    """A one-column left border that grows with the message next to it."""

    def get_content_width(self, container: Any, viewport: Any) -> int:
        return 1

    def render(self) -> str:
        height = self.size.height
        return "\n".join(["⎢"] * (height - 1) + ["⎣"])

    def on_resize(self) -> None:
        self.refresh()


class ExpandingSeparator(NonSelectableStatic):
    def render(self) -> str:
        return "─" * max(self.size.width, 1)

    def on_resize(self) -> None:
        self.refresh()


class UserMessage(Static):
    PROMPT_CHAR: ClassVar[str] = ">"

    def __init__(self, content: str, *, command: bool = False) -> None:
        super().__init__()
        self.add_class("user-message")
        if command:
            self.add_class("slash-command-message")
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical(classes="user-message-wrapper"):
            with Horizontal(classes="user-message-container"):
                yield NonSelectableStatic(f"{self.PROMPT_CHAR} ", classes="user-message-prompt")
                yield NoMarkupStatic(self._content, classes="user-message-content")
            yield ExpandingSeparator(classes="user-message-separator")


class AssistantMessage(Static):
    """An answer rendered as Markdown while it streams in.

    Text received while the user has scrolled up is buffered and written when
    they return to the bottom, so reading earlier messages isn't disturbed.
    """

    def __init__(self, content: str = "") -> None:
        super().__init__()
        self.add_class("assistant-message")
        self._content = content
        self._markdown: Markdown | None = None
        self._stream = None
        self._to_write_buffer = ""

    def compose(self) -> ComposeResult:
        self._markdown = Markdown("")
        yield self._markdown

    def _ensure_stream(self):
        if self._stream is None:
            self._stream = Markdown.get_stream(self._markdown)
        return self._stream

    def _is_chat_at_bottom(self) -> bool:
        try:
            return self.app.query_one("#chat", ChatScroll).is_at_bottom
        except Exception:
            return True

    async def append_content(self, content: str) -> None:
        if not content:
            return
        self._content += content
        if self._is_chat_at_bottom():
            to_write, self._to_write_buffer = self._to_write_buffer + content, ""
            await self._ensure_stream().write(to_write)
        else:
            self._to_write_buffer += content

    async def stop_stream(self) -> None:
        if self._to_write_buffer:
            await self._ensure_stream().write(self._to_write_buffer)
        self._to_write_buffer = ""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None

    def get_content(self) -> str:
        return self._content


class _BorderedLine(Static):
    """Interrupt / error / notice line with a growing left border."""

    KIND: ClassVar[str] = "notice"

    def __init__(self, text: str) -> None:
        super().__init__()
        self.add_class(f"{self.KIND}-message")
        self._text = text

    def compose(self) -> ComposeResult:
        with Horizontal(classes=f"{self.KIND}-container"):
            yield ExpandingBorder(classes=f"{self.KIND}-border")
            yield NoMarkupStatic(self._text, classes=f"{self.KIND}-content")


class InterruptMessage(_BorderedLine):
    KIND = "interrupt"


class ErrorMessage(_BorderedLine):
    KIND = "error"


class NoticeMessage(_BorderedLine):
    KIND = "notice"


# ── Tool calls (after widgets/status_message.py) ──────────────────

class ToolCallMessage(SpinnerMixin, Static):
    """`■ Searching documents: "…"` while running, `✓ …` when done."""

    SPINNER_TYPE = SpinnerType.PULSE

    def __init__(self, text: str, *, running: bool = True) -> None:
        super().__init__()
        self.add_class("tool-call")
        self._text = text
        self.init_spinner()
        self._is_spinning = running

    def compose(self) -> ComposeResult:
        with Horizontal():
            self._indicator_widget = NonSelectableStatic(
                self._spinner.current_frame() if self._is_spinning else "✓",
                classes="status-indicator-icon" + ("" if self._is_spinning else " success"))
            yield self._indicator_widget
            yield NoMarkupStatic(self._text, classes="status-indicator-text")

    def on_mount(self) -> None:
        if self._is_spinning:
            self.start_spinner_timer()


# ── Loading line (from widgets/loading.py) ────────────────────────

def _format_elapsed(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins}m{secs}s"


class LoadingWidget(SpinnerMixin, Static):
    """Snake spinner and a status word whose colour sweeps through a gradient."""

    TARGET_COLORS = ALMA_COLORS
    SPINNER_TYPE = SpinnerType.SNAKE

    def __init__(self, status: str = "Thinking") -> None:
        super().__init__(classes="loading-widget")
        self.init_spinner()
        self.status = status
        self.current_color_index = 0
        self._color_direction = 1
        self.transition_progress = 0
        self._status_widget: Static | None = None
        self.hint_widget: Static | None = None
        self.start_time: float | None = None
        self._last_elapsed = -1

    def set_status(self, status: str) -> None:
        if status == self.status:
            return
        self.status = status
        if self._status_widget:
            self._status_widget.update(self._build_status_text())

    def compose(self) -> ComposeResult:
        with Horizontal(classes="loading-container"):
            self._indicator_widget = Static(self._spinner.current_frame(), classes="loading-indicator")
            yield self._indicator_widget
            self._status_widget = Static(self._build_status_text(), classes="loading-status")
            yield self._status_widget
            self.hint_widget = NoMarkupStatic("(0s · Esc to interrupt)", classes="loading-hint")
            yield self.hint_widget

    def on_mount(self) -> None:
        self.start_time = time()
        self._update_animation()
        self.start_spinner_timer()

    def _update_spinner_frame(self) -> None:
        if self._is_spinning:
            self._update_animation()

    def _next_color_index(self) -> int:
        return self.current_color_index + self._color_direction

    def _get_color_for_position(self, position: int) -> str:
        if position < self.transition_progress:
            return self.TARGET_COLORS[self._next_color_index()]
        return self.TARGET_COLORS[self.current_color_index]

    def _build_status_text(self) -> str:
        parts = [f"[{self._get_color_for_position(1 + i)}]{ch}[/]" for i, ch in enumerate(self.status)]
        parts.append(f"[{self._get_color_for_position(1 + len(self.status))}]… [/]")
        return "".join(parts)

    def _update_animation(self) -> None:
        total_elements = 1 + len(self.status) + 1
        if self._indicator_widget:
            frame = self._spinner.next_frame()
            self._indicator_widget.update(f"[{self._get_color_for_position(0)}]{frame}[/]", layout=False)
        if self._status_widget:
            self._status_widget.update(self._build_status_text(), layout=False)
        self.transition_progress += 1
        if self.transition_progress > total_elements:
            self.current_color_index = self._next_color_index()
            if not 0 < self.current_color_index < len(self.TARGET_COLORS) - 1:
                self._color_direction *= -1
            self.transition_progress = 0
        if self.hint_widget and self.start_time is not None:
            elapsed = int(time() - self.start_time)
            if elapsed != self._last_elapsed:
                self._last_elapsed = elapsed
                self.hint_widget.update(f"({_format_elapsed(elapsed)} · Esc to interrupt)")


# ── Chat scroll (after app.py: ChatScroll) ────────────────────────

class ChatScroll(VerticalScroll):
    @property
    def is_at_bottom(self) -> bool:
        return self.scroll_target_y >= self.max_scroll_y


# ── Command suggestions (from widgets/chat_input/completion_popup.py) ─

@dataclass
class CompletionEntry:
    label: str
    description: str = ""


class _CompletionItem(Static):
    pass


class _CompletionRow(Horizontal):
    pass


class CompletionPopup(VerticalScroll):
    MAX_HEIGHT = 12

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="completion-popup", **kwargs)
        self.styles.display = "none"
        self.styles.max_height = self.MAX_HEIGHT
        self.styles.padding = (0, 1)
        self.can_focus = False
        self._suggestions: list[CompletionEntry] = []

    def update_suggestions(self, suggestions: list[CompletionEntry], selected: int) -> None:
        if not suggestions:
            self.hide()
            return
        if suggestions != self._suggestions:
            self.remove_children()
            self._suggestions = suggestions
            width = max(len(e.label) for e in suggestions)
            rows = []
            for entry in suggestions:
                command = _CompletionItem(entry.label, classes="completion-command")
                command.styles.width = width
                rows.append(_CompletionRow(command, _CompletionItem(entry.description,
                                                                    classes="completion-description")))
            self.mount_all(rows)
        for idx, row in enumerate(self.query(_CompletionRow)):
            row.set_class(idx == selected, "completion-selected")
        self.styles.display = "block"

    def hide(self) -> None:
        self.remove_children()
        self._suggestions = []
        self.styles.display = "none"


# ── Chat input (written for ALMA, after Vibe's chat_input package) ─

class ChatTextArea(TextArea):
    """Multi-line input: Enter sends, Shift+Enter or Ctrl+J adds a line,
    up/down on the first/last line walk through earlier messages."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    # No highlighted cursor line or background: the input reads as plain text (as in Vibe).
    _CHAT_THEME: ClassVar[TextAreaTheme] = TextAreaTheme(
        name="alma-chat", base_style=None, selection_style=Style(reverse=True), cursor_line_style=None)

    def __init__(self, history_file: str, **kwargs: Any) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False, compact=True, **kwargs)
        self.register_theme(self._CHAT_THEME)
        self.theme = self._CHAT_THEME.name
        self._history_file = history_file
        self._history = self._load_history()
        self._history_index: int | None = None
        self._draft = ""
        self.popup: CompletionPopup | None = None
        self.commands: list[CompletionEntry] = []
        self._suggestions: list[CompletionEntry] = []
        self._selected = 0

    # history ─────────────────────────────────────────────
    def _load_history(self) -> list[str]:
        """One JSON string per line. Lines written by the earlier prompt_toolkit
        interface (`+text`) are read too; anything else is skipped."""
        entries = []
        try:
            with open(self._history_file, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line.startswith("+"):
                        entries.append(line[1:])
                    elif line.startswith('"'):
                        try:
                            entries.append(json.loads(line))
                        except ValueError:
                            pass
        except OSError:
            return []
        return entries[-500:]

    def remember(self, text: str) -> None:
        if not text or (self._history and self._history[-1] == text):
            return
        self._history.append(text)
        try:
            with open(self._history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _show_history(self, index: int | None) -> None:
        self._history_index = index
        self.text = self._draft if index is None else self._history[index]
        last = self.text.split("\n")
        self.move_cursor((len(last) - 1, len(last[-1])))

    # suggestions ─────────────────────────────────────────
    def _refresh_suggestions(self) -> None:
        text = self.text
        if text.startswith("/") and " " not in text and "\n" not in text:
            self._suggestions = [c for c in self.commands if c.label.startswith(text)]
        else:
            self._suggestions = []
        self._selected = min(self._selected, max(len(self._suggestions) - 1, 0))
        if self.popup:
            self.popup.update_suggestions(self._suggestions, self._selected)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._refresh_suggestions()

    def _accept_suggestion(self) -> None:
        self.text = self._suggestions[self._selected].label + " "
        self.move_cursor((0, len(self.text)))
        self._suggestions = []
        if self.popup:
            self.popup.hide()

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if self._suggestions and key == "enter" and self.text.strip() == self._suggestions[self._selected].label:
            # Already typed in full: send it rather than "accepting" it again.
            self._suggestions = []
            if self.popup:
                self.popup.hide()
        if self._suggestions:
            if key in ("down", "tab"):
                self._selected = (self._selected + 1) % len(self._suggestions)
            elif key in ("up", "shift+tab"):
                self._selected = (self._selected - 1) % len(self._suggestions)
            elif key == "escape":
                self._suggestions = []
                if self.popup:
                    self.popup.hide()
            elif key == "enter":
                self._accept_suggestion()
            else:
                return await super()._on_key(event)
            if self._suggestions and self.popup:
                self.popup.update_suggestions(self._suggestions, self._selected)
            event.prevent_default()
            event.stop()
            return

        if key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
            return
        if key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        row, _ = self.cursor_location
        if key == "up" and row == 0 and self._history:
            event.prevent_default()
            event.stop()
            if self._history_index is None:
                self._draft = self.text
                self._show_history(len(self._history) - 1)
            elif self._history_index > 0:
                self._show_history(self._history_index - 1)
            return
        if key == "down" and row == self.document.line_count - 1 and self._history_index is not None:
            event.prevent_default()
            event.stop()
            nxt = self._history_index + 1
            self._show_history(nxt if nxt < len(self._history) else None)
            return
        await super()._on_key(event)

    def clear_input(self) -> None:
        self._history_index = None
        self._draft = ""
        self.text = ""


class ChatInput(Vertical):
    """Suggestion popup above a box with a `>` prompt and the text area."""

    def __init__(self, history_file: str, commands: list[CompletionEntry], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history_file = history_file
        self._commands = commands

    def compose(self) -> ComposeResult:
        popup = CompletionPopup()
        yield popup
        with Vertical(id="input-box"):
            with Horizontal(id="input-body"):
                yield NonSelectableStatic(">", id="prompt")
                area = ChatTextArea(self._history_file, id="input")
                area.popup = popup
                area.commands = self._commands
                yield area


# ── Banner (written for ALMA, after widgets/banner/banner.py) ─────

class Banner(Static):
    def __init__(self, model: str, documents: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.can_focus = False
        self._model = model
        self._documents = documents

    def compose(self) -> ComposeResult:
        docs = (f"{self._documents} document{'s' if self._documents != 1 else ''} indexed"
                if self._documents else "no documents indexed yet: add files to the documents folder")
        with VerticalGroup(id="banner-container"):
            with Horizontal(classes="banner-line"):
                yield NoMarkupStatic("ALMA", id="banner-brand")
                yield NoMarkupStatic(f"  {self._model}", classes="banner-meta")
            with Horizontal(classes="banner-line"):
                yield NoMarkupStatic(docs, classes="banner-meta-muted")
            with Horizontal(classes="banner-line"):
                yield NoMarkupStatic("Type ", classes="banner-meta-muted")
                yield NoMarkupStatic("/help", classes="banner-cmd")
                yield NoMarkupStatic(" for commands, ", classes="banner-meta-muted")
                yield NoMarkupStatic("/config", classes="banner-cmd")
                yield NoMarkupStatic(" for settings", classes="banner-meta-muted")
