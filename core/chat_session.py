"""Per-conversation session state: prompt cache, conversation history, disk persistence.

Used identically by the CLI (main.py) and the WebSocket server (server/app.py). All
KV-cache and chat-persistence logic lives here so neither transport has to reimplement it.

Persistent prompt caching is an MLX-only feature: Ollama and API providers manage
their KV-cache server-side, so on those backends `prompt_cache` stays None, save/load become no-ops
and conversations are restored from the chat JSON instead of cache metadata.
"""
import gc
import json

from core.llm_backend import supports_prompt_cache
from core.conversation_manager import ConversationManager
from core.chat_storage import ChatStorageManager
from core.cache_manager import CacheManager
from core.mlx_runtime import run_on_mlx  # cache arrays must live on the MLX thread
from core.logging_config import get_logger

logger = get_logger(__name__)


class ChatSession:
    def __init__(self, llm_model, llm_tokenizer, chats_dir=None, reasoning_model=False):
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.prompt_cache = self._new_prompt_cache()
        self.conversation_manager = ConversationManager(reasoning_model=reasoning_model)
        self.current_chat_id = None
        self.storage = ChatStorageManager(chats_dir)

    def _new_prompt_cache(self):
        if not supports_prompt_cache():
            return None
        from mlx_lm.models.cache import make_prompt_cache
        return run_on_mlx(make_prompt_cache, self.llm_model)

    # ── Lifecycle ──────────────────────────────────────────────────

    def clear(self):
        """Reset prompt cache + conversation; forget the current chat."""
        old_cache = self.prompt_cache
        self.prompt_cache = self._new_prompt_cache()
        del old_cache
        self.conversation_manager.clear()
        self.current_chat_id = None
        gc.collect()

    # ── Disk persistence ───────────────────────────────────────────

    def save_cache(self, last_prompt: str = None) -> bool:
        """Persist KV-cache + conversation state for the current chat.

        Saves `prompt_token_count` in metadata so generated tokens can be trimmed on
        restore — only the prompt prefix forms a valid cache for future prompts.
        """
        if not self.current_chat_id or not self.prompt_cache:
            return False
        cache_path = str(self.storage.cache_path(self.current_chat_id))
        metadata = {
            "conversation": json.dumps(
                self.conversation_manager.get_conversation(), ensure_ascii=False
            )
        }
        tokens = CacheManager.tokens(self.prompt_cache)
        if tokens is not None:
            # The exact tokens in the cache: on restore, the next prompt reuses
            # whatever part of them still matches.
            metadata["tokens"] = json.dumps(tokens)
        elif last_prompt is not None:
            prompt_token_count = len(
                self.llm_tokenizer.encode(last_prompt, add_special_tokens=False)
            )
            metadata["prompt_token_count"] = str(prompt_token_count)
        try:
            from mlx_lm.models.cache import save_prompt_cache
            run_on_mlx(save_prompt_cache, cache_path, self.prompt_cache, metadata)  # reads array state
            logger.info(f"KV-cache saved to {cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save KV-cache: {e}")
            return False

    def save_messages(self, messages: list, title: str = None) -> dict:
        """Update the chat's JSON file (UI-facing representation).

        Creates the chat first if no current_chat_id is set. Server uses this when the
        frontend pushes UI messages; CLI derives messages from the conversation manager
        via save_cli().
        """
        if not self.current_chat_id:
            chat = self.storage.create_chat(title=title)
            self.current_chat_id = chat["id"]
        return self.storage.update_chat(self.current_chat_id, messages, title)

    def save_cli(self, last_prompt: str = None) -> dict:
        """CLI-side save: persist cache + write UI messages derived from conversation."""
        if not self.current_chat_id:
            chat = self.storage.create_chat()
            self.current_chat_id = chat["id"]
        self.storage.update_chat(self.current_chat_id, self._conversation_to_messages())
        self.save_cache(last_prompt)
        return self.storage.load_chat(self.current_chat_id)

    def restore(self, chat_id: str, fallback_messages: list = None) -> bool:
        """Load KV-cache + conversation for `chat_id`. Returns True if cache was loaded.

        If the .safetensors file is missing or load fails, creates a fresh cache and
        rebuilds conversation from `fallback_messages` (loses tool-call entries — cache
        will be a miss on next query).
        """
        self.current_chat_id = chat_id
        cache_path = self.storage.cache_path(chat_id)
        saved_conversation = None
        cache_loaded = False

        if supports_prompt_cache() and cache_path.exists():
            try:
                from mlx_lm.models.cache import load_prompt_cache
                old_cache = self.prompt_cache
                loaded_cache, metadata = run_on_mlx(load_prompt_cache, str(cache_path), return_metadata=True)
                self.prompt_cache = loaded_cache
                del old_cache

                conv_json = metadata.get("conversation")
                if conv_json:
                    saved_conversation = json.loads(conv_json)

                cache_offset = self.prompt_cache[0].offset if self.prompt_cache else 0
                logger.info(f"KV-cache loaded from {cache_path} (offset={cache_offset})")
                tokens_json = metadata.get("tokens")
                CacheManager.set_tokens(self.prompt_cache, json.loads(tokens_json) if tokens_json else None)

                # Trim generated tokens — only the prompt prefix is reusable.
                prompt_token_count_str = metadata.get("prompt_token_count")
                if prompt_token_count_str:
                    prompt_token_count = int(prompt_token_count_str)
                    trim_amount = cache_offset - prompt_token_count
                    if trim_amount > 0:
                        # trim() slices array state, so it must run on the MLX thread
                        run_on_mlx(lambda: [layer.trim(trim_amount) for layer in self.prompt_cache])
                        logger.info(f"Trimmed {trim_amount} generated tokens "
                              f"({cache_offset} → {prompt_token_count})")
                cache_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load cache (incompatible?): {e}")
                old_cache = self.prompt_cache
                self.prompt_cache = self._new_prompt_cache()
                del old_cache
        else:
            old_cache = self.prompt_cache
            self.prompt_cache = self._new_prompt_cache()
            del old_cache
            if supports_prompt_cache():
                logger.info(f"No cache file at {cache_path}; created fresh prompt cache")

        gc.collect()

        if saved_conversation:
            self.conversation_manager.load_from_state(saved_conversation)
            logger.info(f"Conversation restored from cache metadata ({len(saved_conversation)} entries)")
        elif fallback_messages:
            self.conversation_manager.load_from_messages(fallback_messages)
            logger.info(f"Conversation restored from UI messages ({len(fallback_messages)} entries, no cache metadata)")
        else:
            stored = self.storage.load_chat(chat_id)
            if stored and stored.get("messages"):
                self.conversation_manager.load_from_messages(stored["messages"])
                logger.info(f"Conversation restored from JSON ({len(stored['messages'])} entries)")

        return cache_loaded

    # ── Convenience passthroughs ───────────────────────────────────

    def list_chats(self) -> list:
        return self.storage.list_chats()

    def delete_chat(self, chat_id: str) -> bool:
        if chat_id == self.current_chat_id:
            self.current_chat_id = None
        return self.storage.delete_chat(chat_id)

    # ── Helpers ────────────────────────────────────────────────────

    def _conversation_to_messages(self) -> list:
        """Flatten the internal conversation into simple user/assistant messages for UI storage."""
        out = []
        for m in self.conversation_manager.get_conversation():
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if content:
                out.append({"role": role, "content": content})
        return out
