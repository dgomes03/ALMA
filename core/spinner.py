"""Animated one-line progress indicator for long CLI waits (e.g. LLM prefill).

Prefill gives no output until the first token is sampled, which on a long RAG
prompt looks like a hang. Spinner fills that gap on a background thread and
clears its line the moment generation starts streaming.

Silent unless it is both enabled and writing to a TTY, so the server path,
piped output and benchmark runs stay clean.
"""

import itertools
import sys
import threading

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_CLEAR_LINE = "\r\033[K"


class Spinner:

    def __init__(self, stream=None, interval=0.1, enabled=True):
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self.enabled = bool(enabled) and getattr(self.stream, "isatty", lambda: False)()
        self._stop_event = threading.Event()
        self._thread = None
        # Serializes frame writes against the final line-clear, so a frame can
        # never land after stop() has wiped the line.
        self._lock = threading.Lock()

    def start(self):
        if not self.enabled or self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(_FRAMES):
            if self._stop_event.is_set():
                return
            with self._lock:
                if self._stop_event.is_set():
                    return
                self.stream.write(f"\r{frame}")
                self.stream.flush()
            self._stop_event.wait(self.interval)

    def stop(self):
        # Idempotent and cheap on repeat calls: safe to call once per token.
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        with self._lock:
            self.stream.write(_CLEAR_LINE)
            self.stream.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()
        return False


def with_prefill_spinner(token_stream, enabled=True):
    """Wrap a token stream so a spinner runs until its first item arrives.

    The first yielded token means prefill finished and text is about to hit the
    console, so the spinner is torn down before anything is printed.
    """
    spinner = Spinner(enabled=enabled).start()
    try:
        for item in token_stream:
            spinner.stop()
            yield item
    finally:
        spinner.stop()
