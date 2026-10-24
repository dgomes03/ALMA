"""Stopping an answer part-way.

The interface calls request(). The generation path calls check() after every
generated token and after every prompt-reading (prefill) step, see
core/llm_backend.py, so a request takes effect within a second or two even
while the model is reading a long prompt. check() raises Cancelled.

run_cancellable() runs one question on the MLX thread. If it is cancelled, the
turn stays in the conversation: the question, the tool calls that finished, and
the partial answer marked as interrupted, so a follow-up can refer to it. The
KV-cache is kept as it is: CacheManager knows which tokens it holds, and the next
prompt reuses everything up to the first token that differs from what is stored
(usually just the added note), so nothing is read twice.
"""
import threading

_requested = threading.Event()


class Cancelled(BaseException):
    """Raised inside the generation when the user stops it.

    A BaseException (like KeyboardInterrupt) so the `except Exception` blocks
    around tools and retrieval let it through instead of reporting an error."""


def request():
    _requested.set()


def check():
    if _requested.is_set():
        raise Cancelled()


def run_cancellable(session, fn, *args, **kwargs):
    from core.mlx_runtime import submit_on_mlx

    length = len(session.conversation_manager.get_conversation())
    _requested.clear()
    future = submit_on_mlx(fn, *args, **kwargs)
    while True:
        try:
            return future.result()
        except KeyboardInterrupt:           # Ctrl+C in the plain terminal chat
            request()                       # then wait for the worker to stop
        except Cancelled as stop:
            _keep_interrupted_turn(session.conversation_manager, length, getattr(stop, "partial_text", ""))
            raise


def _keep_interrupted_turn(manager, length, partial):
    """Close the interrupted turn so the conversation stays valid for the model."""
    from core import prompts
    conversation = manager.get_conversation()
    if len(conversation) <= length:             # stopped before the question was added
        return
    # A tool call whose results did not all arrive can't be sent back to the model.
    for i in range(length, len(conversation)):
        message = conversation[i]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            asked = {call.get("id") for call in message["tool_calls"]}
            answered = {m.get("tool_call_id") for m in conversation[i + 1:] if m.get("role") == "tool"}
            if not asked <= answered:
                del conversation[i:]
                break
    note = prompts.get("interrupted")
    partial = (partial or "").strip()
    manager.add_assistant_message(f"{partial} … {note}" if partial else note)
