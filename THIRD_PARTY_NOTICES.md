# Third-party notices

## Mistral Vibe

ALMA's terminal chat interface (`core/tui/`) adapts code from **Mistral Vibe** v2.26.1, the last release with a Python (Textual) interface.

- Source: https://github.com/mistralai/mistral-vibe/tree/v2.26.1/vibe/cli/textual_ui
- Copyright 2025 Mistral AI
- License: Apache License, Version 2.0. The full text is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

| ALMA file | Adapted from (`vibe/cli/textual_ui/…`) |
|---|---|
| `core/tui/widgets.py` | `widgets/no_markup_static.py`, `widgets/braille_renderer.py`, `widgets/spinner.py`, `widgets/messages.py`, `widgets/status_message.py`, `widgets/loading.py`, `widgets/chat_input/completion_popup.py`, `ChatScroll` in `app.py` |
| `core/tui/app.tcss` | `app.tcss` |
| `core/tui/app.py` | screen layout of `VibeApp` in `app.py` |

Changes made for ALMA:

- Removed everything tied to Vibe's coding agent: the app server, approvals, diffs, voice, images, message queue, sessions and plugins.
- Made the code run on Python 3.10 (no `StrEnum`).
- Replaced Mistral's colours, logo and status texts with ALMA's own.
- Wrote a smaller chat input, banner and app logic for ALMA (`ChatTextArea`, `ChatInput`, `Banner`, `ALMAApp`). These follow the structure of Vibe's widgets but are new code.

Each adapted file starts with a header naming its source.

## Textual

The interface runs on **Textual** (https://github.com/Textualize/textual), Copyright (c) 2021 Will McGugan, MIT License. Textual is installed as a dependency from PyPI and is not included in this repository.
