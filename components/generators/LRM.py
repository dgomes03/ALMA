"""
LRM Generator: Language Reasoning Model generator with [THINK] marker support.

Uses stream_generate() from mlx_lm for generation. Decodes tokens by converting
IDs to raw BPE strings (convert_ids_to_tokens) and applying the GPT-2
bytes_to_unicode inverse mapping, bypassing the broken tokenizer.decode().
Handles [THINK]...[/THINK] reasoning blocks during streaming.
"""

import re

from core.llm_backend import (
    stream_generate, make_sampler, make_logits_processors
)

from core.config import MAX_RESPONSE_TOKENS, MAX_TOOL_ROUNDS, SHOW_CLI_SPINNER
from core.cache_manager import CacheManager
from core.conversation_manager import ConversationManager
from core import cancel, prompts
from core.parser import parse_tool_calls
from core.utils import strip_emoji
from core.logging_config import get_logger
from core.spinner import with_prefill_spinner

logger = get_logger(__name__)


# State machine states for streaming
_STATE_NORMAL = "NORMAL"
_STATE_THINKING = "THINKING"

# Markers
_THINK_START = "[THINK]"
_THINK_END = "[/THINK]"
_TOOL_MARKER = "[TOOL_CALLS]"
# Buffer size must cover the longest marker
_BUFFER_TAIL = max(len(_THINK_START), len(_THINK_END), len(_TOOL_MARKER))


class LRMGenerator:
    """Generator for reasoning models with [THINK] block support."""

    @staticmethod
    def _extract_thinking(text):
        """Strip [THINK]...[/THINK] from raw response.

        If the model placed everything inside [THINK] (nothing outside),
        try to split reasoning from the actual response by looking for
        common boundary markers like "Response:", "Answer:", etc.

        Returns:
            (thinking_content, clean_text) tuple.
        """
        pattern = re.compile(r'\[THINK\](.*?)\[/THINK\]', re.DOTALL)
        thinking_parts = pattern.findall(text)
        thinking = "\n".join(thinking_parts).strip() if thinking_parts else ""
        clean = pattern.sub("", text).strip()

        # Handle unclosed [THINK] block (model didn't emit [/THINK])
        if not thinking and '[THINK]' in clean:
            idx = clean.find('[THINK]')
            rest = clean[idx + len('[THINK]'):]
            # Don't swallow [TOOL_CALLS] content into thinking
            tc_idx = rest.find('[TOOL_CALLS]')
            if tc_idx >= 0:
                thinking = rest[:tc_idx].strip()
                clean = clean[:idx].strip() + rest[tc_idx:]
            else:
                thinking = rest.strip()
                clean = clean[:idx].strip()

        return thinking, clean

    @staticmethod
    def _stream_with_thinking(model, tokenizer, prompt_tokens, max_tokens,
                              stream_callback, prompt_cache=None,
                              sampler=None, logits_processors=None,
                              show_spinner=False):
        """
        Stream generate with [THINK] marker detection.

        Uses a simple index-based approach: accumulate the full decoded text
        and track how far we've emitted.  A ``while`` loop inside each token
        step handles multiple state transitions (e.g. [THINK]...[/THINK] in
        one chunk) without waiting for the next token.

        Emits via *stream_callback*:
          thinking_start → thinking_chunk* → thinking_complete
          text_chunk (regular response text)
        """
        state = _STATE_NORMAL
        buf = ""
        emitted = 0           # index into decoded text already sent to UI
        text_emitted = False
        stop_streaming = False

        kwargs = {}
        if prompt_cache is not None:
            kwargs["prompt_cache"] = prompt_cache
        if sampler is not None:
            kwargs["sampler"] = sampler
        if logits_processors is not None:
            kwargs["logits_processors"] = logits_processors

        def _emit(event, data=None):
            if stream_callback:
                data = data or {}
                if event == 'text_chunk':
                    data = {**data, 'text': strip_emoji(data.get('text', ''))}
                stream_callback(event, data)

        try:
            for response in with_prefill_spinner(
                stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_tokens,
                    max_tokens=max_tokens,
                    **kwargs
                ),
                enabled=show_spinner,
            ):
                buf += response.text
                if stop_streaming:
                    continue

                # Process all transitions possible with current decoded text
                while emitted < len(buf):
                    remaining = buf[emitted:]

                    if state == _STATE_NORMAL:
                        ti = remaining.find(_THINK_START)
                        mi = remaining.find(_TOOL_MARKER)

                        # Pick whichever marker comes first
                        if ti >= 0 and (mi < 0 or ti <= mi):
                            before = remaining[:ti]
                            if before.strip():
                                _emit('text_chunk', {'text': before})
                                text_emitted = True
                            emitted += ti + len(_THINK_START)
                            state = _STATE_THINKING
                            _emit('thinking_start')
                            continue  # re-check for [/THINK] immediately

                        if mi >= 0 and (ti < 0 or mi < ti):
                            before = remaining[:mi]
                            if before.strip():
                                _emit('text_chunk', {'text': before})
                                text_emitted = True
                            stop_streaming = True
                            break

                        # No marker yet — emit text safely outside the marker zone
                        safe_len = len(remaining) - _BUFFER_TAIL
                        if safe_len > 0:
                            _emit('text_chunk', {'text': remaining[:safe_len]})
                            text_emitted = True
                            emitted += safe_len
                        break  # wait for more tokens

                    elif state == _STATE_THINKING:
                        ei = remaining.find(_THINK_END)
                        if ei >= 0:
                            if remaining[:ei]:
                                _emit('thinking_chunk', {'text': remaining[:ei]})
                            _emit('thinking_complete')
                            emitted += ei + len(_THINK_END)
                            state = _STATE_NORMAL
                            continue  # re-check for text / new [THINK]

                        # No end marker yet — stream safe portion
                        safe_len = len(remaining) - _BUFFER_TAIL
                        if safe_len > 0:
                            _emit('thinking_chunk', {'text': remaining[:safe_len]})
                            emitted += safe_len
                        break  # wait for more tokens

            # --- Flush after generation ends ---
        except cancel.Cancelled as stop:
            # Stopped by the user: hand the answer text so far (without the
            # thinking) to run_cancellable, which keeps it in the conversation.
            stop.partial_text = LRMGenerator._extract_thinking(buf)[1]
            raise

        full_text = buf
        remaining = full_text[emitted:]

        if remaining and state == _STATE_THINKING:
            ei = remaining.find(_THINK_END)
            if ei >= 0:
                if remaining[:ei]:
                    _emit('thinking_chunk', {'text': remaining[:ei]})
                _emit('thinking_complete')
                rest = remaining[ei + len(_THINK_END):]
                if rest.strip():
                    _emit('text_chunk', {'text': rest})
                    text_emitted = True
            else:
                _emit('thinking_chunk', {'text': remaining})
                _emit('thinking_complete')
        elif remaining:
            mi = remaining.find(_TOOL_MARKER)
            to_emit = remaining[:mi] if mi >= 0 else remaining
            if to_emit.strip():
                _emit('text_chunk', {'text': to_emit})
                text_emitted = True

        return full_text, text_emitted

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever,
                              prompt_cache=None, stream_callback=None,
                              verbose=True, conversation_manager=None):
        from tools.tools import get_tools_for_standard_generator
        from components.generators.standard import Generator  # for _execute_tool

        tools = get_tools_for_standard_generator()

        sampler = make_sampler(temp=0.6, top_k=40, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)

        if conversation_manager is None:
            conversation_manager = ConversationManager(reasoning_model=True)

        retriever.current_user_query = query

        conversation_manager.add_user_message(query)
        conversation = conversation_manager.get_conversation()

        pre_query_checkpoint = CacheManager.get_checkpoint(prompt_cache)
        is_tool_call_continuation = False
        tool_rounds = 0

        while True:
            if is_tool_call_continuation:
                CacheManager.restore_checkpoint(prompt_cache, pre_query_checkpoint)

            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                # After MAX_TOOL_ROUNDS the tools are withheld so the model has
                # to answer from what it has gathered (no endless search loops).
                tools=tools if tool_rounds < MAX_TOOL_ROUNDS else None,
                tokenize=False
            )

            prompt_tokens = CacheManager.prepare_prompt_tokens(
                llm_tokenizer, prompt, prompt_cache)

            # Offline mode: use a console callback for real-time verbose output
            def _console_callback(event, data):
                if event == 'thinking_start':
                    print("\n--- Thinking ---", flush=True)
                elif event == 'thinking_chunk':
                    print(data.get('text', ''), end='', flush=True)
                elif event == 'thinking_complete':
                    print("\n--- End Thinking ---", flush=True)
                elif event == 'text_chunk':
                    print(data.get('text', ''), end='', flush=True)

            callback = stream_callback or (_console_callback if verbose else None)
            response_text, text_emitted = LRMGenerator._stream_with_thinking(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt_tokens=prompt_tokens,
                max_tokens=MAX_RESPONSE_TOKENS,
                stream_callback=callback,
                prompt_cache=prompt_cache,
                sampler=sampler,
                logits_processors=logits_processors,
                # `callback` is non-None even on the CLI, so key off the caller's
                # stream_callback to tell CLI apart from the WebSocket path.
                show_spinner=SHOW_CLI_SPINNER and stream_callback is None,
            )
            if stream_callback is None and verbose:
                print(flush=True)  # Final newline after streaming

            response_text = response_text.replace('</s>', '').strip()

            # Tool call handling
            if "[TOOL_CALLS]" in response_text and tool_rounds >= MAX_TOOL_ROUNDS:
                # Asked for yet another tool with none offered: keep what it wrote
                # before the call, or say the step limit was reached.
                answer = response_text[:response_text.find("[TOOL_CALLS]")]
                answer = answer.split("[/THINK]")[-1].strip()
                if not answer:
                    answer = prompts.get("tool_limit")
                    (stream_callback or Generator._stdout_callback)('text_chunk', {'text': answer})
                conversation_manager.add_assistant_message(answer)
                return answer, prompt

            if "[TOOL_CALLS]" in response_text:
                logger.info("Model requested to use tools. Processing tool calls...")
                try:
                    # Extract text after [TOOL_CALLS], strip stray [/THINK]
                    tc_idx = response_text.find('[TOOL_CALLS]')
                    after_tc = response_text[tc_idx + len('[TOOL_CALLS]'):].strip()
                    after_tc = after_tc.replace('[/THINK]', '').strip()
                    logger.debug("Tool call text after marker: %r", after_tc[:300])

                    tool_calls = parse_tool_calls(after_tc)

                    if not tool_calls:
                        logger.warning("No valid tool calls found")
                        break

                    logger.info("Found %d tool call(s): %s", len(tool_calls),
                                [call.get('name', 'unknown') for call in tool_calls])

                    Generator._run_tool_calls(
                        tool_calls, conversation, retriever,
                        llm_model, llm_tokenizer, stream_callback)

                    logger.info("All tool executions completed. Preparing final response...")
                    is_tool_call_continuation = True
                    tool_rounds += 1
                    continue
                except Exception as e:
                    logger.exception("Error processing tool calls: %s", e)
                    break
            else:
                # No tool calls — extract thinking and store in conversation
                logger.info("No tool calls detected, returning response")

                thinking, clean_text = LRMGenerator._extract_thinking(response_text)

                # If streaming didn't emit any text_chunk to the message area,
                # emit the extracted response now (not the full thinking content).
                if not text_emitted and stream_callback and clean_text:
                    logger.debug("Response not streamed, emitting clean_text (%d chars)", len(clean_text))
                    stream_callback('text_chunk', {'text': clean_text})

                response_for_history = clean_text if clean_text else thinking
                if thinking:
                    conversation_manager.add_assistant_message_with_thinking(thinking, response_for_history)
                else:
                    conversation_manager.add_assistant_message(response_for_history)

                if prompt_cache is not None:
                    CacheManager.log_cache_stats(prompt_cache, f"After query (turn {conversation_manager.get_turn_count()})")

                return response_for_history, prompt

        # Reached only via `break` (tool-call parse failure or execution error)
        logger.error("Tool-call processing failed; no final response generated.")
        return None, prompt
