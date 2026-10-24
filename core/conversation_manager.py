"""
ConversationManager: Manages persistent conversation history for multi-turn chat.

This enables proper KV-cache utilization by maintaining a growing conversation
that the cache can prefix-match against, rather than rebuilding from scratch
each query.
"""

from core import prompts


class ConversationManager:
    """Manages persistent conversation history for multi-turn chat."""

    # Language follows PROMPT_LANGUAGE in core/config.py (see core/prompts.py).
    SYSTEM_PROMPT = prompts.get("system")

    def __init__(self, system_prompt: str = None, reasoning_model: bool = False):
        """
        Initialize the conversation manager.

        Args:
            system_prompt: Optional custom system prompt. Uses default if None.
            reasoning_model: If True, use reasoning system prompt with [THINK] support.
        """
        self._system_prompt = system_prompt if system_prompt else self.SYSTEM_PROMPT 

        # Inject current date/time at the start of system prompt
        self._system_prompt = f"{prompts.current_datetime()}\n\n{self._system_prompt}"
        self.conversation = [{"role": "system", "content": self._system_prompt}]

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.conversation.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant response to the conversation."""
        self.conversation.append({"role": "assistant", "content": content})

    def add_assistant_message_with_thinking(self, thinking: str, text: str):
        """Add an assistant response with structured thinking content.

        The chat template renders these as [THINK]...[/THINK] + plain text,
        maintaining KV-cache consistency across turns.
        """
        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking})
        content.append({"type": "text", "text": text})
        self.conversation.append({"role": "assistant", "content": content})

    def load_from_messages(self, messages: list):
        """Restore conversation from saved messages (user/assistant only).
        Note: This loses tool call/result messages, which invalidates the KV-cache.
        Prefer load_from_state() when restoring alongside a saved cache.
        """
        self.clear()
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                self.add_user_message(content)
            elif role == 'assistant':
                self.add_assistant_message(content)

    def load_from_state(self, conversation: list):
        """Restore full conversation state including tool calls and results.
        This preserves the exact token sequence the KV-cache was built from.
        """
        self.conversation = conversation

    def get_conversation(self) -> list:
        """Return the full conversation history."""
        return self.conversation

    def clear(self):
        """Reset conversation to just the system prompt."""
        self.conversation = [{"role": "system", "content": self._system_prompt}]

    def get_turn_count(self) -> int:
        """Return the number of user messages in the conversation."""
        return sum(1 for msg in self.conversation if msg.get("role") == "user")
