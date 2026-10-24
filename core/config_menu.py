"""Interactive settings menu for core/config.py.

Open it with `config` in the terminal chat, or before startup with
`python main.py --config` (run.sh, run_mac.command and run_windows.bat pass
the flag through).

In a real terminal the menu is driven with the arrow keys and runs on the
alternate screen, so the chat history is untouched when it closes. When stdin
or stdout is not a terminal (piped input, a console without ANSI support) it
falls back to numbered prompts.

Edits are written back into core/config.py in place, so comments and layout
survive. Modules read the config once at import, so saved changes apply the
next time the app starts. Settings whose value spans several lines (such as
SQL_DATABASE_CONFIGS) are shown but must be edited in the file itself.
"""
import ast
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import core.config as cfg

CONFIG_PATH = os.path.join(cfg.PROJECT_ROOT, "core", "config.py")


@dataclass
class Setting:
    key: str
    kind: str                       # bool | int | float | str | choice | multi | list | intlist
    help: str
    choices: tuple = ()
    empty: Optional[str] = None     # if set, the value may be left empty (None); shown as this label
    auto: Optional[str] = None      # expression written back when 'auto' is picked
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Section:
    title: str
    settings: List[Setting]
    summary: tuple = ()             # keys shown next to the title in the main menu
    note: str = ""
    info: Optional[Callable[["ConfigState"], List[str]]] = None
    actions: list = field(default_factory=list)   # (label, description, handler(state))


def _benchmark_datasets():
    try:
        from benchmark.beir_data import KNOWN_DATASETS
        return tuple(KNOWN_DATASETS)
    except Exception:
        return ()


# ── Terminal ──────────────────────────────────────────────────────

_ALT_ON, _ALT_OFF = "\033[?1049h", "\033[?1049l"
_HIDE_CURSOR, _SHOW_CURSOR = "\033[?25l", "\033[?25h"
_CLEAR = "\033[H\033[2J"

UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE, INTERRUPT = (
    "up", "down", "left", "right", "enter", "esc", "space", "interrupt")
HOME, END, BACKSPACE, DELETE, CLEAR_LINE = "home", "end", "backspace", "delete", "clear-line"

INTERACTIVE = False     # arrow-key mode; set by _init_terminal()
_COLOR = False
_status = []            # messages shown once under the next screen
_final = []             # messages printed after the menu closes


def _enable_windows_ansi():
    """Turn on escape-code support in the Windows console (Windows 10+)."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


def _init_terminal():
    global INTERACTIVE, _COLOR
    tty = sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    INTERACTIVE = bool(tty and _enable_windows_ansi())
    _COLOR = INTERACTIVE and not os.environ.get("NO_COLOR")


def _write(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def _bold(s):
    return f"\033[1m{s}\033[0m" if _COLOR else s


def _dim(s):
    return f"\033[2m{s}\033[0m" if _COLOR else s


def _reverse(s):
    return f"\033[7m{s}\033[0m" if _COLOR else ">" + s[1:]


def _read_key(timeout=None):
    """Wait for one key press and return a name (UP, ENTER, ...) or the character
    typed. With `timeout` (seconds), return None if no key arrives in time."""
    if os.name == "nt":
        import msvcrt
        if timeout is not None:
            import time
            deadline = time.monotonic() + timeout
            while not msvcrt.kbhit():
                if time.monotonic() >= deadline:
                    return None
                time.sleep(0.02)
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                   # special keys arrive as a two-key sequence
            return {"H": UP, "P": DOWN, "K": LEFT, "M": RIGHT,
                    "G": HOME, "O": END, "S": DELETE}.get(msvcrt.getwch(), "")
    else:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # TCSANOW: the default (TCSAFLUSH) discards keys typed ahead, so
            # holding an arrow key down would only move one row.
            tty.setraw(fd, termios.TCSANOW)
            if timeout is not None and not select.select([fd], [], [], timeout)[0]:
                return None
            raw = os.read(fd, 1)
            if raw == b"\x1b":
                # A lone Esc has nothing behind it; arrows send ESC [ A and friends.
                seq = ""
                while select.select([fd], [], [], 0.03)[0]:
                    seq += os.read(fd, 1).decode(errors="ignore")
                    if len(seq) >= 2 and (seq[-1].isalpha() or seq[-1] == "~"):
                        break
                if not seq:
                    return ESC
                return {"[A": UP, "[B": DOWN, "[C": RIGHT, "[D": LEFT,
                        "OA": UP, "OB": DOWN, "OC": RIGHT, "OD": LEFT,
                        "[H": HOME, "OH": HOME, "[1~": HOME, "[F": END, "OF": END, "[4~": END,
                        "[3~": DELETE}.get(seq, "")
            if raw and raw[0] >= 0xC0:
                # First byte of a multi-byte UTF-8 character (é, ç, ...): read the rest.
                extra = 1 if raw[0] < 0xE0 else 2 if raw[0] < 0xF0 else 3
                raw += os.read(fd, extra)
            ch = raw.decode(errors="ignore")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\r", "\n"):
        return ENTER
    if ch == "\x1b":
        return ESC
    if ch == " ":
        return SPACE
    if ch in ("\x7f", "\x08"):
        return BACKSPACE
    if ch == "\x15":                                   # Ctrl+U
        return CLEAR_LINE
    if ch in ("\x03", "\x04"):
        return INTERRUPT
    return ch


def _ask(prompt):
    """Line input that treats Ctrl+C / Ctrl+D as 'q'."""
    if INTERACTIVE:
        _write(_SHOW_CURSOR)
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return "q"
    finally:
        if INTERACTIVE:
            _write(_HIDE_CURSOR)


def confirm(prompt, default=False):
    answer = _ask(f"{prompt} [{'Y/n' if default else 'y/N'}] ").lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _notify(message, final=False):
    if INTERACTIVE:
        _status.append(message)
    else:
        print(f"  {message}")
    if final:
        _final.append(message)


def _width():
    return shutil.get_terminal_size((100, 24)).columns


def _height():
    return shutil.get_terminal_size((100, 24)).lines


def _shorten(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    keep = max(limit - 3, 1)
    return text[: keep // 2] + "..." + text[len(text) - (keep - keep // 2):]


def _row_text(row, label_width, width):
    label, value = row
    if not value:
        return _shorten(label, width)
    return label.ljust(label_width) + _shorten(value, max(width - label_width, 8))


def _text_field(text, cursor, width):
    """The text being edited, scrolled to keep the cursor visible, with the
    cursor drawn as a reversed character."""
    start = max(0, cursor - width + 1)
    visible = text[start:start + width]
    pos = cursor - start
    under = visible[pos] if pos < len(visible) else " "
    if _COLOR:
        return f"{visible[:pos]}\033[7m{under}\033[0m{visible[pos + 1:]}"
    return f"{visible[:pos]}|{visible[pos:]}"


def _wrap(text, width):
    return textwrap.wrap(text, max(width, 10)) or [""]


def _render(title, rows, index, header, details, hints, editing=None, details_below=False):
    width = _width() - 4
    label_width = min(max((len(r[0]) for r in rows), default=0) + 3, width // 2)
    top = [_CLEAR, f"\n  {_bold(title)}\n"]
    for line in header:
        top += [f"  {_dim(part)}\n" for part in _wrap(line, width)]
    top.append("\n")

    bottom = ["\n"]
    if details and details[index] and not details_below:
        bottom += [f"  {part}\n" for part in _wrap(details[index], width)]
    bottom += [f"  {_bold(message)}\n" for message in _status]
    _status.clear()
    bottom.append(f"\n  {_dim(hints)}\n")

    def block(i):
        """The lines one row takes: the row itself, plus its wrapped description."""
        text = _row_text(rows[i], label_width, width - 2)
        if i == index and editing is not None:
            lines = [f"  > {rows[i][0].ljust(label_width)}"
                     f"{_text_field(*editing, max(width - 4 - label_width, 10))}\n"]
        elif i == index:
            lines = [f"  {_reverse(' ' + text.ljust(width - 2) + ' ')}\n"]
        else:
            label, value = rows[i]
            line = label.ljust(label_width) + _dim(_shorten(value, max(width - 2 - label_width, 8))) \
                if value else _shorten(label, width - 2)
            lines = [f"   {line}\n"]
        if details_below and details and details[i]:
            lines += [f"     {_dim(part)}\n" for part in _wrap(details[i], width - 4)]
        return lines

    # Show as many rows around the selected one as fit, counting wrapped lines.
    room = max(_height() - len(top) - len(bottom) - 2, 3)
    blocks = {index: block(index)}
    start = end = index
    used = len(blocks[index])
    grown = True
    while grown:
        grown = False
        for candidate in (end + 1, start - 1):
            if 0 <= candidate < len(rows) and candidate not in blocks:
                lines = block(candidate)
                if used + len(lines) <= room:
                    blocks[candidate] = lines
                    used += len(lines)
                    start, end = min(start, candidate), max(end, candidate)
                    grown = True
    middle = [_dim("    ...\n")] if start > 0 else []
    for i in range(start, end + 1):
        middle += blocks[i]
    if end < len(rows) - 1:
        middle.append(_dim("    ...\n"))
    _write("".join(top + middle + bottom))


def _next_key():
    """The next key press, or None when the window was resized meanwhile."""
    size = shutil.get_terminal_size()
    while True:
        key = _read_key(timeout=0.2)
        if key is not None:
            return key
        if shutil.get_terminal_size() != size:
            return None


def choose(title, rows, index=0, header=(), details=None, hints="", keys="", fallback_hints="",
           details_below=False):
    """Let the user pick one of `rows` ((label, value) pairs).

    Returns (action, index): ("enter", i) when a row is picked, (key, i) for a
    hotkey listed in `keys`, or ("back", i) for Esc / left arrow / blank input.
    """
    index = min(index, max(len(rows) - 1, 0))
    if INTERACTIVE:
        while True:
            _render(title, rows, index, header, details, hints, details_below=details_below)
            key = _next_key()
            if key is None:                     # window resized: draw again
                continue
            if len(key) == 1:
                key = key.lower()
            if key == UP and rows:
                index = (index - 1) % len(rows)
            elif key == DOWN and rows:
                index = (index + 1) % len(rows)
            elif key == SPACE and " " in keys:
                return " ", index
            elif key in (ENTER, RIGHT, SPACE) and rows:
                return "enter", index
            elif len(key) == 1 and key in keys:
                return key, index
            elif key in (ESC, LEFT, INTERRUPT, "b", "q"):
                return "back", index
            elif key.isdigit() and 1 <= int(key) <= min(len(rows), 9):
                return "enter", int(key) - 1

    while True:
        print(f"\n{_bold(title)}")
        for line in header:
            print(_dim(f"  {line}"))
        print()
        label_width = max((len(r[0]) for r in rows), default=0) + 3
        for i, row in enumerate(rows, 1):
            print(f"  {i:>2}  {_row_text(row, label_width, _width() - 8)}")
            if details and details[i - 1]:
                print(_dim(f"        {_shorten(details[i - 1], _width() - 10)}"))
        print(f"\n  {fallback_hints or 'number = pick   blank = back'}")
        raw = _ask("\n> ").lower()
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            return "enter", int(raw) - 1
        if len(raw) == 1 and raw in keys and raw != " ":
            return raw, index
        if raw in ("", "b", "q"):
            return "back", index
        print("  Type a number or one of the letters shown.")


def pick(title, rows, details=None, header=(), index=0):
    """One arrow-key choice on its own screen, with each row's description
    under it. Returns the chosen index, or None if the user pressed Esc."""
    _init_terminal()
    if INTERACTIVE:
        _write(_ALT_ON + _HIDE_CURSOR)
    try:
        action, index = choose(title, rows, index, header=header, details=details, details_below=True,
                               hints="up/down move   Enter choose   Esc cancel",
                               fallback_hints="number = choose   blank = cancel")
    finally:
        if INTERACTIVE:
            _write(_SHOW_CURSOR + _ALT_OFF)
    return index if action == "enter" else None


# ── Reading and writing config.py ─────────────────────────────────

def _assignments(source):
    """{name: value node} for every top-level `NAME = value` in the source."""
    found = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            found[node.targets[0].id] = node.value
    return found


def write_changes(changes, path=CONFIG_PATH):
    """Replace the value expression of each `KEY = ...` line, keeping its comment.

    `changes` maps a key to the new expression as source text. The file is
    checked to still compile before it replaces the original.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()
    nodes = _assignments(source)
    lines = source.splitlines(keepends=True)
    for key, expr in sorted(changes.items(), key=lambda kv: nodes[kv[0]].lineno, reverse=True):
        node = nodes[key]
        if node.lineno != node.end_lineno:
            raise ValueError(f"{key} spans several lines; edit it in core/config.py")
        # ast offsets are UTF-8 byte offsets, so splice the encoded line.
        raw = lines[node.lineno - 1].encode("utf-8")
        lines[node.lineno - 1] = (raw[:node.col_offset] + expr.encode("utf-8")
                                  + raw[node.end_col_offset:]).decode("utf-8")
    new_source = "".join(lines)
    compile(new_source, path, "exec")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_source)
    os.replace(tmp, path)


class ConfigState:
    """Saved expressions from config.py plus the user's unsaved edits."""

    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.pending = {}           # key -> expression source
        self.reload()

    def reload(self):
        with open(self.path, encoding="utf-8") as f:
            source = f.read()
        self.exprs = {k: ast.get_source_segment(source, v) for k, v in _assignments(source).items()}

    def expr(self, key):
        return self.pending.get(key, self.exprs.get(key))

    def value(self, key):
        expr = self.expr(key)
        if expr is None:
            return getattr(cfg, key, None)
        try:
            return ast.literal_eval(expr)
        except (ValueError, SyntaxError):
            # Expressions such as the platform-dependent backend default.
            try:
                return eval(expr, dict(vars(cfg)))
            except Exception:
                return getattr(cfg, key, None)

    def set(self, key, value=None, expr=None):
        expr = expr if expr is not None else repr(value)
        if expr == self.exprs.get(key):
            self.pending.pop(key, None)
        else:
            self.pending[key] = expr

    def save(self):
        write_changes(self.pending, self.path)
        saved = set(self.pending)
        self.pending.clear()
        self.reload()
        return saved


# ── Formatting ────────────────────────────────────────────────────

def _fmt(setting, state):
    value = state.value(setting.key)
    if setting.auto and state.expr(setting.key) == setting.auto:
        return f"auto ({value})"
    if value is None:
        return setting.empty or "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


# ── Editing one setting ───────────────────────────────────────────

def _parse_number(raw, setting):
    caster = int if setting.kind in ("int", "intlist") else float
    try:
        number = caster(raw)
    except ValueError:
        _notify(f"'{raw}' is not a valid {'whole number' if caster is int else 'number'}.")
        return None
    if setting.minimum is not None and number < setting.minimum:
        _notify(f"{setting.key} must be at least {setting.minimum}.")
        return None
    if setting.maximum is not None and number > setting.maximum:
        _notify(f"{setting.key} must be at most {setting.maximum}.")
        return None
    return number


def _edit_choice(setting, state):
    options = list(setting.choices) + (["auto"] if setting.auto else [])
    current = "auto" if setting.auto and state.expr(setting.key) == setting.auto else state.value(setting.key)
    rows = [(option, "current" if option == current else "") for option in options]
    index = options.index(current) if current in options else 0
    action, index = choose(setting.key, rows, index, header=(setting.help,),
                           hints="up/down move   Enter select   Esc cancel")
    if action != "enter":
        return
    if options[index] == "auto":
        state.set(setting.key, expr=setting.auto)
    else:
        state.set(setting.key, options[index])


def _edit_multi(setting, state):
    selected = list(state.value(setting.key) or [])
    index = 0
    while True:
        rows = [(f"[{'x' if option in selected else ' '}] {option}", "") for option in setting.choices]
        action, index = choose(setting.key, rows, index, header=(setting.help,), keys=" ",
                               hints="up/down move   Space tick   Enter done   Esc cancel",
                               fallback_hints="number = tick/untick   blank = done")
        toggle = action == " " or (action == "enter" and not INTERACTIVE)
        if toggle:
            option = setting.choices[index]
            if option in selected:
                selected.remove(option)
            else:
                selected.append(option)
            continue
        if action == "back" and INTERACTIVE:
            return                                    # Esc cancels in arrow mode
        break
    if not selected:
        _notify("Pick at least one; kept the previous value.")
        return
    state.set(setting.key, [o for o in setting.choices if o in selected])


def _line_editor(initial, redraw):
    """Edit `initial` in place; redraw(text, cursor) draws the screen.
    Returns the new text, or None if the user pressed Esc."""
    text, cursor = initial, len(initial)
    _write(_HIDE_CURSOR)
    while True:
        redraw(text, cursor)
        key = _next_key()
        if key is None:                         # window resized: draw again
            continue
        if key == ENTER:
            return text
        if key in (ESC, INTERRUPT):
            return None
        if key == LEFT:
            cursor = max(cursor - 1, 0)
        elif key == RIGHT:
            cursor = min(cursor + 1, len(text))
        elif key == HOME:
            cursor = 0
        elif key == END:
            cursor = len(text)
        elif key == BACKSPACE and cursor:
            text, cursor = text[:cursor - 1] + text[cursor:], cursor - 1
        elif key == DELETE:
            text = text[:cursor] + text[cursor + 1:]
        elif key == CLEAR_LINE:
            text, cursor = "", 0
        elif key == SPACE or (len(key) == 1 and key.isprintable()):
            char = " " if key == SPACE else key
            text, cursor = text[:cursor] + char + text[cursor:], cursor + 1


def _edit_typed(setting, state, redraw=None):
    key, kind = setting.key, setting.kind
    value = state.value(key)
    if value is None:
        current = ""
    elif isinstance(value, (list, tuple)):
        current = ", ".join(str(v) for v in value)
    else:
        current = str(value)

    if INTERACTIVE and redraw:
        raw = _line_editor(current, redraw)
        if raw is None or raw.strip() == current:
            return
    else:
        raw = _ask(f"\n  New value for {key} (Enter = keep): ")
        if not raw or raw == "q":
            return

    raw = raw.strip()
    if not raw:
        if setting.empty:
            state.set(key, None)
        else:
            _notify(f"{key} can't be empty; kept {current}.")
    elif kind == "str":
        state.set(key, raw)
    elif kind == "list":
        state.set(key, [part.strip() for part in raw.split(",") if part.strip()])
    elif kind == "intlist":
        numbers = [_parse_number(part.strip(), setting) for part in raw.split(",") if part.strip()]
        if numbers and None not in numbers:
            state.set(key, sorted(set(numbers)))
    else:
        number = _parse_number(raw, setting)
        if number is not None:
            state.set(key, number)


def _edit(setting, state, redraw=None):
    if setting.kind == "bool":
        state.set(setting.key, not state.value(setting.key))
    elif setting.kind == "choice":
        _edit_choice(setting, state)
    elif setting.kind == "multi":
        _edit_multi(setting, state)
    else:
        _edit_typed(setting, state, redraw)


# ── Section-specific extras ───────────────────────────────────────

def _model_info(state):
    if state.value("LLM_BACKEND") != "api":
        return []
    from core.llm_backend import API_PROVIDERS
    provider = state.value("API_PROVIDER")
    if provider not in API_PROVIDERS:
        return [f"Unknown API_PROVIDER '{provider}'."]
    _, default_url, default_model, key_env = API_PROVIDERS[provider]
    model = state.value("API_MODEL") or default_model or "(set API_MODEL)"
    url = state.value("API_BASE_URL") or default_url or "(set API_BASE_URL)"
    key = os.environ.get(key_env) or (os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None)
    status = f"set (...{key[-4:]})" if key else "not set"
    return [f"API backend: {provider}, model {model}, {url}",
            f"Key {key_env}: {status}. Questions and retrieved passages are sent to this provider."]


def _write_env_value(path, key, value):
    """Set KEY=value in a .env file, replacing an existing line for KEY."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    entry = f"{key}={value}"
    for i, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if name == key:
            lines[i] = entry
            break
    else:
        lines.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o600)       # owner-only; a no-op beyond read-only on Windows
    except OSError:
        pass


def _set_api_key(state):
    import getpass
    from core.llm_backend import API_PROVIDERS
    provider = state.value("API_PROVIDER")
    if provider not in API_PROVIDERS:
        _notify(f"Pick a valid API_PROVIDER first (now '{provider}').")
        return
    key_env = API_PROVIDERS[provider][3]
    if INTERACTIVE:
        _write(_SHOW_CURSOR)
    try:
        key = getpass.getpass(f"\n  {key_env} for {provider} (typing is hidden; blank = cancel): ").strip()
    except (KeyboardInterrupt, EOFError):
        key = ""
    finally:
        if INTERACTIVE:
            _write(_HIDE_CURSOR)
    if not key:
        return
    _write_env_value(cfg.ENV_FILE, key_env, key)
    os.environ[key_env] = key
    _notify(f"Saved {key_env} to .env, which git ignores. It is never written to config.py.")


def _sql_info(state):
    lines = []
    for name, conf in cfg.SQL_DATABASE_CONFIGS.items():
        target = conf.get("connection_string", "")
        status = ""
        if conf.get("db_type") == "sqlite":
            status = "ok" if os.path.exists(target) else "file not found"
            target = os.path.relpath(target, cfg.PROJECT_ROOT)
        lines.append(f"{name:<16} {conf.get('db_type', ''):<10} {target}  {status}")
    lines.append("Add or remove databases in SQL_DATABASE_CONFIGS in core/config.py.")
    return lines


def _browse_datasets(state):
    try:
        from benchmark.beir_data import KNOWN_DATASETS, DEFAULT_DATA_DIR
    except Exception as e:
        _notify(f"Could not read the dataset list: {e}")
        return
    names = list(KNOWN_DATASETS)
    rows = []
    for name in names:
        size, domain = KNOWN_DATASETS[name]
        downloaded = "  [downloaded]" if os.path.isdir(os.path.join(DEFAULT_DATA_DIR, name)) else ""
        rows.append((name, f"{size}, {domain}{downloaded}"))
    current = state.value("BENCHMARK_DATASET")
    action, index = choose(
        "Benchmark datasets", rows, names.index(current) if current in names else 0,
        header=(f"Downloaded to {os.path.relpath(DEFAULT_DATA_DIR, cfg.PROJECT_ROOT)}/ "
                "the first time a benchmark uses them.",),
        hints="up/down move   Enter use this dataset   Esc back")
    if action == "enter":
        state.set("BENCHMARK_DATASET", names[index])
        _notify(f"BENCHMARK_DATASET = {names[index]} (not saved yet)")


def _run_module(module, *args):
    command = [sys.executable, "-m", module, *args]
    if INTERACTIVE:
        _write(_ALT_OFF + _SHOW_CURSOR)
    print(_dim(f"\n  $ python -m {module} {' '.join(args)}".rstrip()))
    print(_dim("  (Ctrl+C stops it and returns to the menu)\n"))
    try:
        code = subprocess.call(command, cwd=cfg.PROJECT_ROOT)
    except KeyboardInterrupt:
        code = 130
    if code not in (0, 130):
        print(f"\n  Exited with code {code}. If a package is missing, install the extras:")
        print("      pip install -r requirements-benchmark.txt")
    _ask("\n  Enter to go back to the menu ")
    if INTERACTIVE:
        _write(_ALT_ON + _HIDE_CURSOR)


def _run_eval(state):
    rows = [("benchmark", "RAGAS on EVAL_DATASET (loads the LLM)"),
            ("beir-retrieval", "retrieval metrics on SciFact, no LLM")]
    action, index = choose("RAGAS evaluation mode", rows,
                           hints="up/down move   Enter run   Esc cancel")
    if action == "enter":
        _run_module("evaluation.run_evaluation", "--mode", rows[index][0])


# ── Catalogue ─────────────────────────────────────────────────────

def build_sections():
    return [
        Section("Model & backend", [
            Setting("LLM_BACKEND", "choice",
                    "mlx = Apple Silicon; ollama = local Ollama app; api = hosted provider (data leaves this machine)",
                    choices=("mlx", "ollama", "api"), auto="'mlx' if IS_APPLE_SILICON else 'ollama'"),
            Setting("MODEL_PATH", "str", "MLX model: a folder or a Hugging Face repo id",
                    empty="the model in the model folder"),
            Setting("OLLAMA_MODEL", "str", "Ollama model name; run `ollama pull <name>` first"),
            Setting("OLLAMA_HOST", "str", "Address of the Ollama server"),
            Setting("OLLAMA_NUM_CTX", "int", "Ollama context window in tokens", minimum=2048),
            Setting("OLLAMA_KEEP_ALIVE", "str", "How long Ollama keeps the model loaded, e.g. 30m"),
            Setting("API_PROVIDER", "choice", "Provider for the api backend; custom = any OpenAI-compatible server",
                    choices=("openai", "anthropic", "gemini", "mistral", "openrouter", "deepseek", "custom")),
            Setting("API_MODEL", "str", "Model name at the API provider", empty="provider default"),
            Setting("API_BASE_URL", "str", "API server address; needed for the custom provider",
                    empty="provider default"),
            Setting("API_TIMEOUT", "int", "Seconds to wait for the provider to start answering", minimum=5),
            Setting("REASONING_MODEL", "bool", "Turn on for reasoning models that think before answering"),
        ], summary=("LLM_BACKEND",), info=_model_info, actions=[
            ("Set API key", "Stored in .env (git-ignored) under the selected provider's variable",
             _set_api_key),
        ]),

        Section("Generation", [
            Setting("PROMPT_LANGUAGE", "choice",
                    "Language of the prompts sent to the model; answers still follow the question's language",
                    choices=("pt", "en")),
            Setting("LLM_TEMPERATURE", "float", "Randomness of answers; lower is more literal", minimum=0, maximum=2),
            Setting("LLM_TOP_P", "float", "Nucleus sampling cut-off", minimum=0, maximum=1),
            Setting("LLM_REPETITION_PENALTY", "float", "1.0 = off; higher discourages repeating", minimum=0),
            Setting("LLM_REPETITION_CONTEXT_SIZE", "int", "Tokens the repetition penalty looks back over", minimum=1),
            Setting("MAX_RESPONSE_TOKENS", "int", "Longest answer the model may write", minimum=1),
            Setting("MAX_TOOL_ROUNDS", "int", "Tool calls (searches, reads) allowed per answer before it must reply",
                    minimum=1, maximum=20),
        ], summary=("PROMPT_LANGUAGE", "LLM_TEMPERATURE")),

        Section("Documents & indexing", [
            Setting("CHUNK_SIZE", "int", "Chunk length in characters", minimum=100),
            Setting("CHUNK_OVERLAP", "int", "Characters shared by neighbouring chunks", minimum=0),
            Setting("MIN_CHUNK_CHARS", "int", "Shorter chunks are dropped as noise", minimum=0),
            Setting("HIERARCHICAL_INDEXING", "bool", "Index parent + child chunks (flat is usually better)"),
            Setting("PARENT_CHUNK_SIZE", "int", "Parent chunk length (hierarchical only)", minimum=100),
            Setting("PDF_TEXT_BACKEND", "choice", "PDF text extractor", choices=("pymupdf", "unstructured")),
            Setting("PDF_CROSS_PAGE_CHUNKS", "bool", "Let a chunk span a PDF page break"),
            Setting("PDF_SECTION_PATH", "bool", "Prefix chunks with their heading path"),
            Setting("BM25_ENABLE_STEMMING", "bool", "Stem words for keyword search"),
            Setting("BM25_ENABLE_STOPWORDS", "bool", "Drop stop words for keyword search"),
            Setting("BM25_LANGUAGES", "list", "Languages for stemming and stop words, e.g. english, portuguese"),
            Setting("EMBEDDING_MODEL_NAME", "str", "Sentence-transformers embedding model"),
            Setting("EMBEDDING_USE_PREFIX", "bool", "Add E5 query:/passage: prefixes"),
            Setting("FAISS_INDEX_TYPE", "choice", "Vector index; flat is exact",
                    choices=("flat", "ivf", "hnsw", "auto")),
            Setting("CHECK_NEW_DOCUMENTS_ON_START", "bool", "Index new or changed documents at startup"),
        ], summary=("CHUNK_SIZE",),
            note="Except the last one, these only take effect after the index is rebuilt."),

        Section("Retrieval", [
            Setting("RERANK", "bool", "Rerank results with the cross-encoder"),
            Setting("RERANKER_MODEL_NAME", "str", "Cross-encoder reranker model"),
            Setting("RETRIEVAL_TOP_K", "int", "Candidates taken from BM25 and FAISS each", minimum=1),
            Setting("RERANK_CANDIDATES", "int", "Candidates the reranker scores",
                    empty="all", minimum=1),
            Setting("RERANKER_TOP_N", "int", "Chunks kept after reranking", minimum=1),
            Setting("RERANKER_CONFIDENCE_THRESHOLD", "float", "Drop chunks scoring below this (0 to 1)",
                    empty="off", minimum=0, maximum=1),
            Setting("RRF_K", "int", "Rank fusion constant; higher flattens rank differences", minimum=1),
            Setting("FRONT_MATTER_PENALTY", "float", "Score multiplier for cover and contents pages (0 to 1)",
                    empty="off", minimum=0, maximum=1),
            Setting("MAX_DOCUMENT_CHARS", "int", "Longest document opened in full; longer ones get an overview",
                    minimum=2000),
            Setting("QUERY_EXPANSION_MIN_WORDS", "int", "Shorter queries skip LLM expansion; 0 = always expand",
                    minimum=0),
            Setting("PARENT_TOP_N", "int", "Parent chunks returned (hierarchical only)", minimum=1),
        ], summary=("RERANK", "RETRIEVAL_TOP_K")),

        Section("Research agent & web", [
            Setting("ADVANCED_RESEARCH", "bool", "Let the LLM call the multi-step research pipeline"),
            Setting("MAX_RESEARCH_STEPS", "int", "Research loop iterations", minimum=1),
            Setting("MAX_RESEARCH_GOALS", "int", "Sub-goals the planner may create", minimum=1),
            Setting("RESEARCH_SATURATION_ROUNDS", "int", "Stop after this many rounds with no new sources",
                    empty="off", minimum=1),
            Setting("AGENTIC_USE_DOCUMENT_SEARCH", "bool", "Research may search your documents"),
            Setting("AGENTIC_USE_WEB_SEARCH", "bool", "Research may search the web"),
            Setting("AGENTIC_USE_DOC_RETRIEVAL", "bool", "Research may open whole documents"),
            Setting("DUCKDUCKGO_RESULTS_COUNT", "int", "Results per web search", minimum=1),
            Setting("MAX_URLS_PER_SEARCH_LOOP", "int", "Pages fetched per search", minimum=1),
            Setting("MAX_FETCH_URL_CHARS", "int", "Characters kept from each fetched page", minimum=100),
        ], summary=("ADVANCED_RESEARCH",)),

        Section("Interface & logging", [
            Setting("ENABLE_WEB_INTERFACE", "bool", "on = chat in the browser, off = chat in the terminal"),
            Setting("WEB_INTERFACE_HOST", "str", "127.0.0.1 = this computer only; 0.0.0.0 = whole network, no login"),
            Setting("WEB_INTERFACE_PORT", "int", "Port of the web interface", minimum=1, maximum=65535),
            Setting("SHOW_CLI_SPINNER", "bool", "Animated indicator while the model reads the prompt"),
            Setting("DEBUG_MESSAGES", "bool", "Print the full chat template on every turn"),
            Setting("ENABLE_LOGGING", "bool", "Show log output"),
            Setting("LOG_LEVEL", "choice", "Minimum level shown", choices=("DEBUG", "INFO", "WARNING", "ERROR")),
            Setting("LOG_FILE", "str", "Also write logs to this file", empty="off"),
        ], summary=("ENABLE_WEB_INTERFACE",)),

        Section("SQL databases", [
            Setting("ENABLE_SQL_DATABASES", "bool", "Let the LLM query the databases listed above"),
        ], summary=("ENABLE_SQL_DATABASES",), info=_sql_info),

        Section("Benchmarks", [
            Setting("BENCHMARK_DATASET", "choice" if _benchmark_datasets() else "str",
                    "BEIR dataset for the retrieval benchmark and RRF sweep", choices=_benchmark_datasets()),
            Setting("BENCHMARK_METHODS", "multi", "Retrieval methods to compare",
                    choices=("bm25", "dense", "hybrid", "hybrid_rerank")),
            Setting("BENCHMARK_K_VALUES", "intlist", "Metric cut-offs, e.g. 1, 3, 5, 10, 100", minimum=1),
            Setting("BENCHMARK_MAX_QUERIES", "int", "Limit queries for a quick run",
                    empty="all", minimum=1),
            Setting("BENCHMARK_MAX_CORPUS", "int", "Subsample the corpus to this many documents; 0 = full corpus",
                    empty="automatic", minimum=0),
            Setting("EVAL_DATASET", "choice", "Dataset for the RAGAS benchmark mode",
                    choices=("amnesty_qa", "hotpotqa", "scifact", "ms_marco")),
            Setting("EVAL_RAGAS_JUDGE", "choice", "RAGAS judge: mlx (local) or claude (ANTHROPIC_API_KEY)",
                    choices=("mlx", "claude"), auto="'mlx' if IS_APPLE_SILICON else 'claude'"),
            Setting("EVAL_MAX_QUESTIONS", "int", "Limit the number of RAGAS questions", empty="all", minimum=1),
            Setting("EVAL_LLM_MAX_TOKENS", "int", "Token limit for the local RAGAS judge", minimum=1),
        ], summary=("BENCHMARK_DATASET",), actions=[
            ("Browse datasets", "See every BEIR dataset, its size and whether it is downloaded",
             _browse_datasets),
            ("Run the retrieval benchmark", "python -m benchmark.run_benchmark with the settings above",
             lambda state: _run_module("benchmark.run_benchmark")),
            ("Run the RRF sweep", "Grid over RRF_K and candidate depth on BENCHMARK_DATASET",
             lambda state: _run_module("benchmark.sweep_rrf")),
            ("Plot benchmark results", "Figures from benchmark/results/ into benchmark/figures/",
             lambda state: _run_module("benchmark.plot_results")),
            ("Run a RAGAS evaluation", "Generation quality with RAGAS", _run_eval),
        ]),
    ]


# ── Screens ───────────────────────────────────────────────────────

def _section_screen(section, state):
    index = 0
    n_settings = len(section.settings)
    while True:
        rows = [(f"{s.key}{' *' if s.key in state.pending else ''}", _fmt(s, state))
                for s in section.settings]
        rows += [(f"> {label}", "") for label, _, _ in section.actions]
        details = [s.help for s in section.settings] + [desc for _, desc, _ in section.actions]
        header = ([section.note] if section.note else []) + (section.info(state) if section.info else [])
        action, index = choose(
            section.title, rows, index, header=header, details=details,
            hints="up/down move   Enter change   Esc back   (* = not saved yet)")
        if action == "back":
            return
        if index < n_settings:
            setting = section.settings[index]
            edit_hints = "Enter save   Esc cancel"
            if setting.empty:
                edit_hints += f"   empty = {setting.empty}"
            elif setting.kind in ("list", "intlist"):
                edit_hints += "   separate with commas"

            def redraw(text, cursor, at=index, hints=edit_hints):
                _render(section.title, rows, at, header, details, hints, editing=(text, cursor))

            _edit(setting, state, redraw)
            continue
        _, _, handler = section.actions[index - n_settings]
        runs_something = handler not in (_browse_datasets, _set_api_key)
        if runs_something and state.pending:
            if confirm("\n  The run reads the saved config. Save your changes first?", default=True):
                _save(state)
        handler(state)


def _save(state):
    if not state.pending:
        _notify("Nothing to save.")
        return
    indexing = next(s for s in build_sections() if s.title == "Documents & indexing")
    indexing_keys = {s.key for s in indexing.settings} - {"CHECK_NEW_DOCUMENTS_ON_START"}
    try:
        saved = state.save()
    except Exception as e:
        _notify(f"Could not save: {e}")
        return
    _notify(f"Saved {len(saved)} change(s) to core/config.py. Restart ALMA to apply them.", final=True)
    index_dir = os.path.dirname(cfg.FAISS_INDEX_PATH)
    if saved & indexing_keys and os.path.isdir(index_dir):
        rel = os.path.relpath(index_dir, cfg.PROJECT_ROOT)
        if confirm(f"\n  Indexing settings changed. Delete {rel}/ so it is rebuilt on the next start?"):
            shutil.rmtree(index_dir)
            _notify(f"Deleted {rel}/. The index will be rebuilt on the next start.", final=True)


def _main_screen(state, sections):
    index = 0
    while True:
        rows = []
        for section in sections:
            summary = "   ".join(f"{s.key.lower()} {_fmt(s, state)}"
                                 for s in section.settings if s.key in section.summary)
            changed = any(s.key in state.pending for s in section.settings)
            rows.append((f"{section.title}{' *' if changed else ''}", summary))
        unsaved = f"   {len(state.pending)} unsaved change(s)" if state.pending else ""
        action, index = choose(
            "ALMA settings", rows, index, header=(f"core/config.py{unsaved}",), keys="sq",
            hints="up/down move   Enter open   s save   q quit",
            fallback_hints="number = open   s = save   q = quit")
        if action == "enter":
            _section_screen(sections[index], state)
        elif action == "s":
            _save(state)
        else:
            if state.pending and confirm(f"\n  Save {len(state.pending)} unsaved change(s)?", default=True):
                _save(state)
            return


def run_menu():
    """Show the settings menu until the user quits."""
    _init_terminal()
    _final.clear()
    state = ConfigState()
    sections = build_sections()
    if INTERACTIVE:
        _write(_ALT_ON + _HIDE_CURSOR)
    try:
        _main_screen(state, sections)
    finally:
        if INTERACTIVE:
            _write(_SHOW_CURSOR + _ALT_OFF)
    for message in _final if INTERACTIVE else []:
        print(message)


if __name__ == "__main__":
    run_menu()
