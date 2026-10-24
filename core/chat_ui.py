"""Terminal chat: picks the interface and holds what both versions share.

In a real terminal the chat runs as a Textual app (core/tui). Piped input, or a
missing textual package, falls back to a plain input() loop with the same
commands.
"""
import os
import sys

from core.config import LLM_BACKEND, PROJECT_ROOT

HISTORY_FILE = os.path.join(PROJECT_ROOT, ".chat_history")

COMMANDS = {
    "/help": "show the commands",
    "/config": "open the settings menu",
    "/clear": "start a new conversation",
    "/save": "save this conversation",
    "/list": "list saved conversations",
    "/load": "reopen a saved conversation: /load <id>",
    "/delete": "delete a saved conversation: /delete <id>",
    "/exit": "quit ALMA",
}

# Typed without the slash, these single words still work as commands (older habit).
_BARE_COMMANDS = {"exit", "quit", "help", "config", "clear", "save", "list"}

TOOL_LABELS = {
    "search_documents": "Searching documents",
    "retrieve_document_by_name": "Opening document",
    "list_available_documents": "Listing documents",
    "duckduckgo_search": "Searching the web",
    "fetch_url_content": "Reading web page",
    "search_wikipedia": "Searching Wikipedia",
    "list_databases": "Listing databases",
    "get_database_schema": "Reading database schema",
    "query_database": "Querying database",
    "activate_advanced_research": "Researching",
}


def _model_label(retriever):
    from core.llm_backend import ACTIVE_MODEL
    name = os.path.basename(str(ACTIVE_MODEL or "?").rstrip("/"))
    if LLM_BACKEND == "api":
        return f"{name} ({getattr(getattr(retriever, 'llm_model', None), 'provider', 'API')})"
    return f"{name} ({'MLX' if LLM_BACKEND == 'mlx' else 'Ollama'})"


def _document_count(retriever):
    try:
        names = list(retriever.metadata_index or {})
    except Exception:
        return 0
    # Whole-document entries are also stored as `<name>_<n>`; count each file once.
    def base(name):
        stem, _, suffix = name.rpartition("_")
        return stem if stem and suffix.isdigit() and stem in names else name
    return len({base(n) for n in names})


def _handle_command(raw, session, state):
    """Run a command in the plain loop. Returns True if `raw` was one,
    'exit' to quit, False for a question."""
    raw = raw.strip()
    if not raw.startswith("/") and raw.lower() not in _BARE_COMMANDS:
        return False
    cmd, _, arg = raw.partition(" ")
    cmd = cmd.lower().lstrip("/")
    arg = arg.strip()
    if cmd in ("exit", "quit"):
        return "exit"
    if cmd == "help":
        for name, desc in COMMANDS.items():
            print(f"  {name:<9} {desc}")
    elif cmd == "config":
        from core.config_menu import run_menu
        run_menu()
    elif cmd == "clear":
        session.clear()
        state["last_prompt"] = None
        print("  New conversation.")
    elif cmd == "save":
        chat = session.save_cli(last_prompt=state["last_prompt"])
        print(f"  Saved as {chat['id']} ({chat['title']})")
    elif cmd == "list":
        chats = session.list_chats()
        if not chats:
            print("  No saved conversations.")
        for c in chats:
            marker = "*" if c["id"] == session.current_chat_id else " "
            print(f"  {marker} {c['id']}  {c['title'][:60]}  ({c['message_count']} messages)")
    elif cmd == "load":
        if not arg:
            print("  Usage: /load <id>  (see /list)")
        else:
            session.restore(arg)
            state["last_prompt"] = None
            print(f"  Reopened {arg}.")
    elif cmd == "delete":
        if not arg:
            print("  Usage: /delete <id>  (see /list)")
        else:
            print("  Deleted." if session.delete_chat(arg) else "  No conversation with that id.")
    else:
        print(f"  Unknown command /{cmd}. Type /help for the list.")
    return True


def _plain_loop(retriever, run_query):
    session = retriever.session
    state = {"last_prompt": None}
    print("\nALMA is ready. Type a question, /config for settings, or /exit to quit.")
    while True:
        try:
            raw = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not raw:
            continue
        handled = _handle_command(raw, session, state)
        if handled == "exit":
            return
        if not handled:
            from core import cancel
            try:
                _, state["last_prompt"] = run_query(raw, retriever)
            except cancel.Cancelled:
                print("\n  Stopped.")


def run_chat(retriever, run_query):
    """Chat in the terminal until the user quits."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from core.tui.app import run
        except ImportError:
            pass
        else:
            return run(retriever, run_query)
    return _plain_loop(retriever, run_query)
