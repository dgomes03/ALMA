import json
import random
import string
import sys
from core.llm_backend import (
    stream_generate, make_sampler, make_logits_processors
)

from core.config import (
    MAX_RESPONSE_TOKENS, MAX_TOOL_ROUNDS, ADVANCED_RESEARCH, SHOW_CLI_SPINNER,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_REPETITION_PENALTY, LLM_REPETITION_CONTEXT_SIZE, DEBUG_MESSAGES
)
from core.cache_manager import CacheManager
from core.logging_config import get_logger
from core.spinner import with_prefill_spinner
from core.conversation_manager import ConversationManager
from core import cancel, prompts
from core.parser import parse_tool_calls
from core.utils import strip_emoji
from tools.wikipedia import search_wikipedia
from tools.web_search import duckduckgo_search
from tools.fetch_url import fetch_url_content

logger = get_logger(__name__)


class Generator:
    @staticmethod
    def _execute_tool(tool_name, tool_args, retriever, llm_model=None, llm_tokenizer=None):
        """Execute a single tool by name and return the result string.

        Returns the tool result (always a string).
        """
        if tool_name == "search_documents":
            tool_result = retriever.search_documents_tool(tool_args.get("query", ""))
        elif tool_name == "retrieve_document_by_name":
            tool_result = retriever.retrieve_document_by_name_tool(tool_args.get("document_name", ""))
        elif tool_name == "list_available_documents":
            tool_result = retriever.list_available_documents_tool(tool_args.get("filter_keyword", ""))
        elif tool_name == "search_wikipedia":
            tool_result = search_wikipedia(tool_args.get("query", ""))
        elif tool_name == "duckduckgo_search":
            tool_result = duckduckgo_search(tool_args.get("query", ""), tool_args.get("max_results"))
        elif tool_name == "fetch_url_content":
            tool_result = fetch_url_content(tool_args.get("url", ""), tool_args.get("max_chars"))
        elif tool_name == "query_database":
            from tools.SQL_database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.execute_query(
                    tool_args.get("db_name", ""), tool_args.get("sql_query", ""))
        elif tool_name == "list_databases":
            from tools.SQL_database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.list_databases()
        elif tool_name == "get_database_schema":
            from tools.SQL_database import get_sql_connector
            sql_connector = get_sql_connector()
            if sql_connector is None:
                tool_result = {"success": False, "error": "SQL databases are not configured"}
            else:
                tool_result = sql_connector.get_schema(tool_args.get("db_name", ""))
        else:
            tool_result = f"Error: Unknown tool: {tool_name}"

        if not isinstance(tool_result, str):
            tool_result = json.dumps(tool_result, ensure_ascii=False)

        return tool_result

    @staticmethod
    def _format_tool_calls(tool_calls):
        """Normalize parsed tool calls into chat-template entries
        (random call id + JSON-string arguments)."""
        formatted = []
        for call in tool_calls:
            call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
            args = call.get("arguments")
            if isinstance(args, dict):
                arguments_str = json.dumps(args)
            elif isinstance(args, str):
                try:
                    json.loads(args)
                    arguments_str = args
                except json.JSONDecodeError:
                    # Model emitted a bare string — wrap it in the parameter
                    # the tool expects.
                    tool_name = call.get("name", "")
                    if tool_name in ["search_documents", "search_wikipedia"]:
                        arguments_str = json.dumps({"query": args})
                    elif tool_name == "retrieve_document_by_name":
                        arguments_str = json.dumps({"document_name": args})
                    else:
                        arguments_str = json.dumps({"query": args})
            else:
                arguments_str = "{}"

            formatted.append({
                "id": call_id,
                "function": {
                    "name": call["name"],
                    "arguments": arguments_str
                },
                "type": "function"
            })
        return formatted

    @staticmethod
    def _run_tool_calls(tool_calls, conversation, retriever, llm_model,
                        llm_tokenizer, stream_callback):
        """Format parsed tool calls, execute them sequentially and append the
        assistant/tool entries to `conversation`. Returns the last tool result.

        Shared by the standard and LRM generators.
        """
        formatted_tool_calls = Generator._format_tool_calls(tool_calls)

        conversation.append({
            "role": "assistant",
            "content": "",
            "tool_calls": formatted_tool_calls
        })

        tool_result = ""
        for i, tool_call in enumerate(formatted_tool_calls):
            tool_name = tool_call["function"]["name"]
            tool_id = tool_call["id"]
            args_str = tool_call["function"]["arguments"]
            try:
                tool_args = json.loads(args_str)
            except json.JSONDecodeError as e:
                tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
            else:
                logger.info("Executing tool %d/%d: %s with args: %s",
                            i + 1, len(formatted_tool_calls), tool_name, tool_args)

                if stream_callback:
                    stream_callback('tool_call_start', {
                        'tool_id': tool_id,
                        'tool_name': tool_name,
                        'arguments': tool_args
                    })

                tool_result = Generator._execute_tool(
                    tool_name, tool_args, retriever, llm_model, llm_tokenizer)

            if stream_callback:
                stream_callback('tool_call_result', {
                    'tool_id': tool_id,
                    'tool_name': tool_name,
                    'result': tool_result,
                    'status': 'success' if not tool_result.startswith('Error') else 'error'
                })

            conversation.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
                "tool_call_id": tool_id
            })

        return tool_result

    @staticmethod
    def _stdout_callback(event_type, data):
        """Default text-chunk sink for CLI mode: write to stdout, flush immediately."""
        if event_type == 'text_chunk':
            sys.stdout.write(data.get('text', ''))
            sys.stdout.flush()

    @staticmethod
    def _generate_with_streaming(model, tokenizer, prompt, max_tokens, sampler, logits_processors, prompt_cache, stream_callback):
        """Stream-generate text, emitting chunks via `stream_callback`. With no callback,
        falls back to stdout — the CLI gets the same live streaming as the WebSocket.

        A small tail buffer (matching `[TOOL_CALLS]` length) keeps the marker from being
        split across emits. Once detected, streaming stops but the full text is still
        returned so the caller can parse the tool-call block.
        """
        TOOL_MARKER = '[TOOL_CALLS]'
        TOOL_MARKER_LEN = len(TOOL_MARKER)

        callback = stream_callback if stream_callback is not None else Generator._stdout_callback
        buf = ""
        emitted = 0
        stop_streaming = False

        kwargs = {}
        if prompt_cache is not None:
            kwargs['prompt_cache'] = prompt_cache
        if sampler is not None:
            kwargs['sampler'] = sampler
        if logits_processors is not None:
            kwargs['logits_processors'] = logits_processors
        try:
            for response in with_prefill_spinner(
                stream_generate(
                    model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_tokens, **kwargs
                ),
                enabled=SHOW_CLI_SPINNER and stream_callback is None,
            ):
                buf += response.text
                if stop_streaming:
                    continue

                remaining = buf[emitted:]
                mi = remaining.find(TOOL_MARKER)
                if mi >= 0:
                    to_emit = remaining[:mi]
                    if to_emit.strip():
                        callback('text_chunk', {'text': strip_emoji(to_emit)})
                    stop_streaming = True
                    continue

                safe_len = len(remaining) - TOOL_MARKER_LEN
                if safe_len > 0:
                    callback('text_chunk', {'text': strip_emoji(remaining[:safe_len])})
                    emitted += safe_len
        except cancel.Cancelled as stop:
            # Stopped by the user: hand the text so far to run_cancellable,
            # which keeps it in the conversation.
            stop.partial_text = buf.split(TOOL_MARKER)[0]
            raise

        remaining = buf[emitted:]
        if not stop_streaming and remaining:
            mi = remaining.find(TOOL_MARKER)
            to_emit = remaining[:mi] if mi >= 0 else remaining
            if to_emit.strip():
                callback('text_chunk', {'text': strip_emoji(to_emit)})

        if stream_callback is None:
            sys.stdout.write('\n')
            sys.stdout.flush()
        return buf

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=None, stream_callback=None, conversation_manager=None):

        from tools.tools import get_tools_for_standard_generator

        tools = get_tools_for_standard_generator()

        # Temperature and top sampling
        sampler = make_sampler(temp=LLM_TEMPERATURE, top_p=LLM_TOP_P)
        logits_processors = make_logits_processors(
            repetition_penalty=LLM_REPETITION_PENALTY,
            repetition_context_size=LLM_REPETITION_CONTEXT_SIZE,
        )

        if conversation_manager is None:
            conversation_manager = ConversationManager()

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
                tools=tools if tool_rounds < MAX_TOOL_ROUNDS else None,
                tokenize=False
            )
            
            if DEBUG_MESSAGES:
                print("\nFORMATTED PROMPT:")
                print(prompt)

            prompt_tokens = CacheManager.prepare_prompt_tokens(
                llm_tokenizer, prompt, prompt_cache)

            response = Generator._generate_with_streaming(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt_tokens,
                max_tokens=MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                stream_callback=stream_callback,
            )

            response_text = response.strip()

            if "[TOOL_CALLS]" in response_text and tool_rounds >= MAX_TOOL_ROUNDS:
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
                    tool_calls = parse_tool_calls(response_text)

                    if not tool_calls:
                        logger.warning("No valid tool calls found")
                        break

                    logger.info("Found %d tool call(s): %s", len(tool_calls),
                                [call.get('name', 'unknown') for call in tool_calls])
                    
                    if ADVANCED_RESEARCH:
                        adv_call = next(
                            (call for call in tool_calls if call.get("name") == "activate_advanced_research"),
                            None
                        )
                        if adv_call is not None:
                            reasoning_query = adv_call.get("arguments", {}).get("query", query)
                            call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                            logger.info("[ADVANCED RESEARCH] Delegating to AgenticGenerator for: %s", reasoning_query)

                            if stream_callback:
                                stream_callback('tool_call_start', {
                                    'tool_id': call_id,
                                    'tool_name': 'activate_advanced_research',
                                    'arguments': {'query': reasoning_query}
                                })

                            from components.generators.research import AgenticGenerator
                            response_text = AgenticGenerator.agentic_answer_query(
                                query=reasoning_query,
                                llm_model=llm_model,
                                llm_tokenizer=llm_tokenizer,
                                retriever=retriever,
                                prompt_cache=prompt_cache,
                                conversation_manager=None,
                                stream_callback=stream_callback
                            )

                            conversation_manager.add_assistant_message(response_text)

                            return response_text, prompt

                    tool_result = Generator._run_tool_calls(
                        tool_calls, conversation, retriever,
                        llm_model, llm_tokenizer, stream_callback)

                    # DEBUG
                    logger.debug("====== RETRIEVED RESULTS: ======\n%s", tool_result)
                    logger.info("All tool executions completed. Preparing final response...")
                    is_tool_call_continuation = True
                    tool_rounds += 1
                    continue
                except Exception as e:
                    logger.exception("Error processing tool calls: %s", e)
                    break
            else:
                logger.info("No tool calls detected, returning response")

                conversation_manager.add_assistant_message(response_text)

                if prompt_cache is not None:
                    CacheManager.log_cache_stats(prompt_cache, f"After query (turn {conversation_manager.get_turn_count()})")

                return response_text, prompt

        logger.error("Tool-call processing failed; no final response generated.")
        return None, prompt
