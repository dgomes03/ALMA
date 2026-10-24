from core import prompts
from core.llm_backend import make_sampler, make_logits_processors

from core.config import (
    MAX_RESPONSE_TOKENS, MAX_RESEARCH_STEPS, MAX_REPLANS_PER_GOAL,
    MAX_TOTAL_REPLANS, RESEARCH_SATURATION_ROUNDS,
)
from core.conversation_manager import ConversationManager
from core.logging_config import get_logger

logger = get_logger(__name__)


class AgenticGenerator:
    """Agentic reasoning coordinator.

    Orchestrates multi-step reasoning, aka advanced research: planning → batch retrieval →
    batch evaluation → dynamic replanning → streaming synthesis.

    All heavy logic lives in the agents/ package:
      - agents.planner       — plan creation and goal replanning
      - agents.retriever     — DDG→fetch retrieval loops
      - agents.evaluator     — batch goal evaluation
      - agents.batch_generator — BatchGenerator wrapper
    """

    @staticmethod
    def agentic_answer_query(
        query: str,
        llm_model,
        llm_tokenizer,
        retriever,
        prompt_cache=None,
        conversation_manager=None,
        stream_callback=None
    ) -> str:
        """Main agentic loop: plan → retrieve → evaluate → replan → synthesize."""
        from agents import AgenticPlanner
        from agents.evaluator import AgenticEvaluator
        from agents.retriever import AgenticRetriever

        logger.info(f"\n{'='*60}")
        logger.info(f"[AGENTIC] REASONING: {query}")
        logger.info(f"{'='*60}\n")

        # ── Step 1: Create initial plan ───────────────────────────────
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)

        if stream_callback:
            stream_callback('reasoning_step', {
                'step': 0,
                'total_steps': MAX_RESEARCH_STEPS,
                'goal': 'Planning complete',
                'strategy': 'planning',
                'goals': [{'description': g.description} for g in plan.goals]
            })

        all_previous_contexts = []
        total_replans = 0
        main_loop_iter = 0
        # Track which (document, page) pairs the loop has already seen.
        retriever.seen_sources = set()
        barren_rounds = 0

        logger.info("\n\n ######################## Plan created! ########################\n")

        # ── Step 2: Flat batch execution loop ─────────────────────────
        while not plan.is_complete() and main_loop_iter < MAX_RESEARCH_STEPS:
            pending_goals = [g for g in plan.goals if g.status == "pending"]
            if not pending_goals:
                logger.info("[AGENTIC] No pending goals — done.")
                break

            main_loop_iter += 1
            logger.info(f"\n{'─'*60}")
            logger.info(f"[AGENTIC] Loop iter {main_loop_iter} — {len(pending_goals)} pending goal(s) - ")
            for i, g in enumerate(pending_goals, 1):
                logger.info(f"  {i}. {g.description}")
            logger.info(f"{'─'*60}")

            for goal in pending_goals:
                goal.status = "in_progress"

            if stream_callback:
                stream_callback('reasoning_step', {
                    'step': main_loop_iter,
                    'total_steps': MAX_RESEARCH_STEPS,
                    'goal': f'{len(pending_goals)} goal(s)',
                    'goals': [{'description': g.description} for g in pending_goals]
                })

            goal_indices = [plan.goals.index(g) for g in pending_goals]

            # ── RETRIEVE ──────────────────────────────────────────────

            logger.info("\n\n ######################## Initiated retriever ########################\n")
            sources_before = len(retriever.seen_sources)
            try:
                batch_results = AgenticRetriever.batch_retrieve_for_goals(
                    goals=pending_goals,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    retriever=retriever,
                    stream_callback=stream_callback,
                    goal_indices=goal_indices
                )
            except Exception as e:
                logger.warning(f"[AGENTIC] Retrieval failed: {e}")
                import traceback; traceback.print_exc()
                for goal in pending_goals:
                    goal.status = "failed"
                continue

            retrieved_contexts = [ctx for ctx, _ in batch_results]

            # Source-saturation check: if a whole iteration surfaced no
            # (document, page) we had not already read, more iterations of the
            # same plan will not either.
            new_sources = len(retriever.seen_sources) - sources_before
            logger.info(f"[AGENTIC] Iter {main_loop_iter}: {new_sources} new source(s); "
                  f"{len(retriever.seen_sources)} distinct so far")
            if RESEARCH_SATURATION_ROUNDS:
                barren_rounds = 0 if new_sources else barren_rounds + 1
                if barren_rounds >= RESEARCH_SATURATION_ROUNDS:
                    logger.info(f"[AGENTIC] Source saturation after {barren_rounds} "
                          f"barren iteration(s) — stopping early.")
                    for goal in pending_goals:
                        if goal.status == "in_progress":
                            goal.status = "cancelled"
                    break

            for goal, retrieved_context in zip(pending_goals, retrieved_contexts):
                if retrieved_context:
                    goal.retrieved_info.append(retrieved_context)

            # ── EVALUATE ──────────────────────────────────────────────
            evaluations = AgenticEvaluator.batch_evaluate_goals(
                goals=pending_goals,
                retrieved_contexts=retrieved_contexts,
                all_previous_contexts=all_previous_contexts,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer,
                stream_callback=stream_callback,
                goal_indices=goal_indices
            )

            # ── PROCESS each evaluation ────────────────────────────────
            for goal, retrieved_context, evaluation in zip(
                    pending_goals, retrieved_contexts, evaluations):
                gidx = plan.goals.index(goal)
                goal.confidence = evaluation.get("confidence", 0.5)
                sparse = evaluation.get("sparse_results", False)

                logger.info(f"[EVAL] Goal {gidx} '{goal.description[:60]}':")
                logger.info(f"  sparse={sparse}, confidence={goal.confidence:.2f}, "
                      f"gain={evaluation.get('information_gain', 0.5):.2f}")
                logger.info(f"  reasoning: {evaluation.get('reasoning', 'N/A')}")

                if not sparse:
                    goal.status = "completed"
                    if retrieved_context:
                        all_previous_contexts.append(retrieved_context)
                    logger.info(f"[EVAL] Goal {gidx} → completed")
                else:
                    if (goal.replan_count >= MAX_REPLANS_PER_GOAL
                            or total_replans >= MAX_TOTAL_REPLANS):
                        goal.status = "cancelled"
                        logger.info(f"[EVAL] Goal {gidx} → cancelled "
                              f"(replan_count={goal.replan_count}, total_replans={total_replans})")
                        if stream_callback:
                            stream_callback('reasoning_evaluation', {
                                'confidence': goal.confidence,
                                'information_gain': evaluation.get('information_gain', 0.5),
                                'is_complete': False,
                                'sparse_results': True,
                                'contradictory_info': evaluation.get('contradictory_info', False),
                                'reasoning': evaluation.get('reasoning', ''),
                                'goal_index': gidx,
                                'cancelled': True
                            })
                    else:
                        goal.status = "failed"
                        logger.info(f"[REPLAN] Reformulating goal {gidx}: {goal.description}")
                        new_goal = AgenticPlanner.replan_goal(
                            goal, evaluation, llm_model, llm_tokenizer
                        )
                        plan.add_goal(new_goal)
                        total_replans += 1
                        logger.info(f"[REPLAN] New goal {len(plan.goals)-1}: {new_goal.description} "
                              f"(replan_count={new_goal.replan_count}, total_replans={total_replans})")
                        if stream_callback:
                            stream_callback('reasoning_replan', {
                                'action': 'reformulate',
                                'new_goals': [new_goal.description],
                                'replaced_goal_index': gidx,
                                'reasoning': evaluation.get(
                                    'reasoning', 'Sparse results — reformulating')
                            })

        # ── Step 3: Synthesize final answer ──────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"[SYNTHESIS] Synthesizing final answer")
        logger.info(f"{'='*60}")

        return AgenticGenerator._synthesize_answer(
            query=query,
            plan=plan,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            conversation_manager=conversation_manager,
            prompt_cache=prompt_cache,
            stream_callback=stream_callback
        )

    @staticmethod
    def _synthesize_answer(query, plan, llm_model, llm_tokenizer,
                            conversation_manager, prompt_cache, stream_callback):
        """Build a synthesis prompt from all accumulated context and generate
        the final answer with streaming."""
        from components.generators import Generator
        from core.cache_manager import CacheManager

        # Blocks are "\n---\n"-joined chunks, each tagged "[Source: file, p. N]".
        # Two goals routinely retrieve the same chunk, and duplicates used to
        # consume the 8000-char budget twice, pushing real evidence past the
        # cut. Dedup on exact block text, keeping first-seen order.
        blocks, seen_blocks = [], set()
        duplicates = 0
        for goal in plan.goals:
            for info in goal.retrieved_info:
                text = info[0] if isinstance(info, tuple) else info
                for block in str(text).split("\n---\n"):
                    key = block.strip()
                    if not key:
                        continue
                    if key in seen_blocks:
                        duplicates += 1
                        continue
                    seen_blocks.add(key)
                    blocks.append(key)
        synthesis_context = "\n---\n".join(blocks)
        if duplicates:
            logger.info(f"[SYNTHESIS] dropped {duplicates} duplicate block(s)")

        if len(synthesis_context) > 8000:
            synthesis_context = synthesis_context[:8000] + "\n...(truncated)"

        logger.info(f"\n[SYNTHESIS] Total context: {len(synthesis_context)} chars "
              f"from {sum(len(g.retrieved_info) for g in plan.goals)} retrievals")

        user_msg = prompts.get("synthesis_user", context=synthesis_context, query=query)

        if conversation_manager is None:
            conversation_manager = ConversationManager()

        system_msgs = [m for m in conversation_manager.get_conversation()
                       if m["role"] == "system"]
        synthesis_conv = system_msgs + [{"role": "user", "content": user_msg}]

        prompt = llm_tokenizer.apply_chat_template(
            synthesis_conv, add_generation_prompt=True, tokenize=False
        )

        sampler = make_sampler(temp=0.3, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.0)

        # Token-level cache handling is MLX-only; the Ollama and API backends take
        # the prompt object directly (no prompt cache).
        if prompt_cache is not None:
            gen_prompt = llm_tokenizer.encode(prompt, add_special_tokens=False)
            if CacheManager.get_cache_size(prompt_cache) > 0:
                logger.info(f"[SYNTHESIS] Resetting KV-cache for fresh synthesis conversation")
                CacheManager.reset(prompt_cache)
            logger.info(f"[SYNTHESIS] Prompt tokens: {len(gen_prompt)}")
        else:
            gen_prompt = prompt

        response = Generator._generate_with_streaming(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=gen_prompt,
            max_tokens=MAX_RESPONSE_TOKENS,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=prompt_cache,
            stream_callback=stream_callback
        )

        response_text = response.strip()
        conversation_manager.add_user_message(query)
        conversation_manager.add_assistant_message(response_text)

        if prompt_cache is not None:
            CacheManager.log_cache_stats(
                prompt_cache,
                f"After agentic synthesis (turn {conversation_manager.get_turn_count()})")

        return response_text
