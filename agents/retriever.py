import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import (
    DUCKDUCKGO_RESULTS_COUNT, MAX_URLS_PER_SEARCH_LOOP, MAX_SEARCH_LOOPS, MAX_FETCH_URL_CHARS
)
from core.logging_config import get_logger

logger = get_logger(__name__)

"""Per-goal multi-tool retriever for advanced research mode.

Dispatches each goal to its assigned tool: search_documents,
duckduckgo_search (+ fetch_url_content second stage), list_available_documents,
or retrieve_document_by_name.
"""


class AgenticRetriever:
    """Handles retrieval for reasoning goals, dispatching per-goal by tool name."""

    @staticmethod
    def batch_retrieve_for_goals(goals, llm_model, llm_tokenizer, retriever,
                                  stream_callback=None, goal_indices=None):
        """Retrieve information for every goal using its assigned tool.

        Returns a list of (retrieved_context: str, tool_name: str) tuples
        in the same order as *goals*.
        """
        if goal_indices is None:
            goal_indices = list(range(len(goals)))

        goal_all_results = [[] for _ in goals]

        # Partition goals by tool
        web_goal_idxs = [i for i, g in enumerate(goals) if g.tool == "duckduckgo_search"]
        doc_goal_idxs = [i for i, g in enumerate(goals) if g.tool != "duckduckgo_search"]

        # ── Document-tool goals (single-call, no second stage) ───────────
        if doc_goal_idxs:
            AgenticRetriever._run_document_tool_goals(
                goals=goals,
                goal_idxs=doc_goal_idxs,
                goal_indices=goal_indices,
                goal_all_results=goal_all_results,
                retriever=retriever,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                stream_callback=stream_callback,
            )

        # ── Web-search goals (DDG → fetch) ───────────────────────────────
        if web_goal_idxs:
            AgenticRetriever._run_web_search_goals(
                goals=goals,
                goal_idxs=web_goal_idxs,
                goal_indices=goal_indices,
                goal_all_results=goal_all_results,
                retriever=retriever,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                stream_callback=stream_callback,
            )

        return [
            ("\n---\n".join(goal_all_results[i]), goals[i].tool)
            for i in range(len(goals))
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Document tool execution (search_documents, list_available_documents,
    # retrieve_document_by_name)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _run_document_tool_goals(goals, goal_idxs, goal_indices, goal_all_results,
                                  retriever, llm_model, llm_tokenizer, stream_callback):
        from components.generators import Generator

        logger.info(f"\n\n ######################## Started document-tool dispatch ({len(goal_idxs)} goal(s)) ########################\n\n")

        # Run document tools SERIALLY. FAISS, BM25, the reranker and the
        # lazy index loader are not safe to call concurrently from multiple
        # Python threads — parallelizing here caused a malloc double-free.
        # Each call is local and fast, so serializing 1-4 of them is fine.

        # Temporarily clear retriever.current_user_query so expand_query uses
        # each goal's tight keyword description instead of the original
        # multi-part user query stashed by the standard generator.
        saved_user_query = getattr(retriever, "current_user_query", None)

        try:
            for i in goal_idxs:
                goal = goals[i]
                tool = goal.tool
                if tool == "search_documents":
                    args = {"query": goal.description}
                elif tool == "list_available_documents":
                    args = {"filter_keyword": goal.description or ""}
                elif tool == "retrieve_document_by_name":
                    args = {"document_name": goal.description}
                else:
                    continue

                logger.info(f"[{tool.upper()}] Goal {i}: {goal.description}")
                try:
                    retriever.current_user_query = goal.description
                    result = Generator._execute_tool(
                        tool, args, retriever, llm_model, llm_tokenizer
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    logger.info(f"[{tool.upper()}] Goal {i}: {len(result)} chars")
                except Exception as e:
                    logger.warning(f"[{tool.upper()}] Goal {i} error: {e}")
                    result = ""

                if result:
                    goal_all_results[i].append(result)
                    if stream_callback:
                        stream_callback('reasoning_retrieval', {
                            'goal': goals[i].description,
                            'goal_index': goal_indices[i],
                            'tool_name': goals[i].tool,
                            'chars_retrieved': len(result),
                            'preview': result[:500],
                            'sources': [{'tool_name': goals[i].tool,
                                         'label': goals[i].description,
                                         'chars': len(result), 'result': result}]
                        })
        finally:
            retriever.current_user_query = saved_user_query

    # ─────────────────────────────────────────────────────────────────────
    # Web search execution (DDG → fetch)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _run_web_search_goals(goals, goal_idxs, goal_indices, goal_all_results,
                               retriever, llm_model, llm_tokenizer, stream_callback):
        from components.generators import Generator

        for search_loop in range(MAX_SEARCH_LOOPS):
            logger.info(f"\n[SEARCH_LOOP {search_loop+1}/{MAX_SEARCH_LOOPS}] {len(goal_idxs)} web goal(s)")

            # ── Step A: DDG search for all web goals in parallel ───────────
            logger.info("\n\n ######################## Started DDG search for web goals ########################\n\n")
            def _ddg_search(i):
                goal = goals[i]
                logger.info(f"[DDG] Goal {i}: {goal.description}")
                try:
                    result = Generator._execute_tool(
                        "duckduckgo_search",
                        {"query": goal.description, "max_results": DUCKDUCKGO_RESULTS_COUNT},
                        retriever, llm_model, llm_tokenizer
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    logger.info(f"[DDG] Goal {i}: {len(result)} chars")
                    return i, result
                except Exception as e:
                    logger.warning(f"[DDG] Goal {i} error: {e}")
                    return i, ""

            ddg_results = {}
            with ThreadPoolExecutor(max_workers=min(len(goal_idxs), 4)) as ex:
                futures = {ex.submit(_ddg_search, i): i for i in goal_idxs}
                for f in as_completed(futures):
                    i, result = f.result()
                    ddg_results[i] = result
                    if result:
                        goal_all_results[i].append(result)
                        if stream_callback:
                            stream_callback('reasoning_retrieval', {
                                'goal': goals[i].description,
                                'goal_index': goal_indices[i],
                                'tool_name': 'duckduckgo_search',
                                'chars_retrieved': len(result),
                                'preview': result[:500],
                                'sources': [{'tool_name': 'duckduckgo_search',
                                             'label': goals[i].description,
                                             'chars': len(result), 'result': result}]
                            })

            # ── Step B: Extract URLs and fetch in parallel ─────────────────
            logger.info("\n\n ############## Reading DDG results and deciding what to fech #############\n\n")
            def _extract_urls(ddg_result):
                urls = []
                try:
                    data = json.loads(ddg_result)
                    for item in data.get("results", []):
                        url = item.get("url", "")
                        if url and not url.lower().endswith(".pdf") and "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
                            urls.append(url)
                            if len(urls) >= MAX_URLS_PER_SEARCH_LOOP:
                                break
                except (json.JSONDecodeError, AttributeError):
                    pass
                return urls

            fetch_tasks = []
            for i in goal_idxs:
                for url in _extract_urls(ddg_results.get(i, ""))[:MAX_URLS_PER_SEARCH_LOOP]:
                    fetch_tasks.append((i, url))

            def _fetch_url(task):
                goal_idx, url = task
                logger.info(f"[FETCH] Goal {goal_idx}: {url}")
                try:
                    result = Generator._execute_tool(
                        "fetch_url_content", {"url": url},
                        retriever, llm_model, llm_tokenizer
                    )
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                    result = result[:MAX_FETCH_URL_CHARS]
                    logger.info(f"[FETCH] Goal {goal_idx} {url[:60]}: {len(result)} chars")
                    return goal_idx, url, result
                except Exception as e:
                    logger.warning(f"[FETCH] Goal {goal_idx} {url[:60]} error: {e}")
                    return goal_idx, url, ""

            if fetch_tasks:
                with ThreadPoolExecutor(max_workers=min(len(fetch_tasks), 6)) as ex:
                    futures = {ex.submit(_fetch_url, task): task for task in fetch_tasks}
                    for f in as_completed(futures):
                        goal_idx, url, result = f.result()
                        if result:
                            goal_all_results[goal_idx].append(result)
                            if stream_callback:
                                stream_callback('reasoning_retrieval', {
                                    'goal': goals[goal_idx].description,
                                    'goal_index': goal_indices[goal_idx],
                                    'tool_name': 'fetch_url_content',
                                    'chars_retrieved': len(result),
                                    'preview': result[:500],
                                    'sources': [{'tool_name': 'fetch_url_content',
                                                 'label': url, 'chars': len(result),
                                                 'result': result}]
                                })
