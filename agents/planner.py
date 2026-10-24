from dataclasses import dataclass, field
from typing import List, Dict
from core import prompts
from core.llm_backend import generate, make_sampler

from core.parser import parse_json_safely
from core.config import (
    MAX_RESEARCH_GOALS,
    AGENTIC_USE_DOCUMENT_SEARCH,
    AGENTIC_USE_WEB_SEARCH,
    AGENTIC_USE_DOC_RETRIEVAL,
)
from core.logging_config import get_logger

logger = get_logger(__name__)

_PLANNER_SAMPLER = make_sampler(temp=0.2)


def _allowed_tools() -> List[str]:
    """Tools the planner is allowed to choose from, gated by config flags."""
    tools: List[str] = []
    if AGENTIC_USE_DOCUMENT_SEARCH:
        tools.append("search_documents")
    if AGENTIC_USE_WEB_SEARCH:
        tools.append("duckduckgo_search")
    if AGENTIC_USE_DOC_RETRIEVAL:
        tools.append("list_available_documents")
        tools.append("retrieve_document_by_name")
    if not tools:
        tools.append("duckduckgo_search")
    return tools


# One line per tool for the planner prompts, in PROMPT_LANGUAGE.
_TOOL_DESCRIPTIONS = {
    name: prompts.get(f"tool_{name}")
    for name in ("search_documents", "duckduckgo_search",
                 "list_available_documents", "retrieve_document_by_name")
}


@dataclass
class ReasoningGoal:
    """Represents a sub-goal in multi-step reasoning"""
    description: str
    tool: str = "duckduckgo_search"  # one of _TOOL_DESCRIPTIONS keys
    status: str = "pending"  # pending, in_progress, completed, failed, cancelled
    retrieved_info: List[str] = field(default_factory=list)
    confidence: float = 0.0
    replan_count: int = 0


@dataclass
class ReasoningPlan:
    """Complete reasoning plan with multiple goals"""
    goals: List[ReasoningGoal] = field(default_factory=list)

    def add_goal(self, goal: ReasoningGoal):
        self.goals.append(goal)

    def is_complete(self) -> bool:
        return all(g.status in ("completed", "failed", "cancelled") for g in self.goals)


class AgenticPlanner:

    @staticmethod
    def create_initial_plan(query: str, llm_model, llm_tokenizer) -> ReasoningPlan:
        """Decompose query into a set of tool-annotated sub-goals."""
        allowed = _allowed_tools()
        tools_block = "\n".join(
            f'- "{t}": {_TOOL_DESCRIPTIONS[t]}' for t in allowed
        )
        default_tool = "search_documents" if "search_documents" in allowed else allowed[0]

        system_prompt = prompts.get("planner_system", datetime=prompts.current_datetime(),
                                    tools_block=tools_block, max_goals=MAX_RESEARCH_GOALS)

        user_prompt = prompts.get("planner_user", query=query, max_goals=MAX_RESEARCH_GOALS,
                                  allowed=allowed)

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        logger.info(f"\n[PLANNER] Formatted prompt:\n{prompt}")

        response = generate(
            llm_model, llm_tokenizer, prompt=prompt, max_tokens=500, verbose=False,
            sampler=_PLANNER_SAMPLER
        )

        logger.info(f"\n[PLANNER] Raw LLM response:\n{repr(response)}")

        plan = ReasoningPlan()
        plan_data = parse_json_safely(response)
        logger.info(f"[PLANNER] Parsed JSON: {plan_data}")

        if plan_data and "goals" in plan_data:
            for goal_data in plan_data.get("goals", [])[:MAX_RESEARCH_GOALS]:
                if isinstance(goal_data, dict) and "description" in goal_data:
                    tool = goal_data.get("tool", default_tool)
                    if tool not in allowed:
                        logger.info(f"[PLANNER] Tool '{tool}' not allowed; falling back to {default_tool}")
                        tool = default_tool
                    plan.add_goal(ReasoningGoal(
                        description=goal_data["description"],
                        tool=tool,
                    ))

            if plan.goals:
                logger.info(f"\n[PLANNER] Created plan with {len(plan.goals)} goals:")
                for i, g in enumerate(plan.goals, 1):
                    logger.info(f"  {i}. [{g.tool}] {g.description}")
            else:
                logger.info(f"[PLANNER] No valid goals found, falling back to single goal")
                plan.add_goal(ReasoningGoal(description=query, tool=default_tool))
        else:
            logger.warning(f"[PLANNER] JSON parse failed, falling back to single goal")
            plan.add_goal(ReasoningGoal(description=query, tool=default_tool))

        return plan

    @staticmethod
    def replan_goal(goal: ReasoningGoal, evaluation: Dict, llm_model, llm_tokenizer) -> ReasoningGoal:
        """Reformulate one sparse goal into a new ReasoningGoal. The replan may
        also switch tool — e.g. from search_documents to duckduckgo_search."""
        allowed = _allowed_tools()
        tools_block = "\n".join(
            f'- "{t}": {_TOOL_DESCRIPTIONS[t]}' for t in allowed
        )

        system_prompt = prompts.get("replanner_system", datetime=prompts.current_datetime(),
                                    tools_block=tools_block, allowed=allowed)

        user_prompt = prompts.get("replanner_user", description=goal.description, tool=goal.tool,
                                  reasoning=evaluation.get("reasoning", prompts.get("sparse_results")))

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        logger.info(f"\n[REPLAN] Formatted prompt:\n{prompt}")

        response = generate(
            llm_model, llm_tokenizer, prompt=prompt, max_tokens=200, verbose=False,
            sampler=_PLANNER_SAMPLER
        )

        logger.info(f"\n[REPLAN] Raw LLM response:\n{repr(response)}")

        replan_data = parse_json_safely(response)
        logger.info(f"[REPLAN] Parsed JSON: {replan_data}")

        if replan_data and "new_description" in replan_data:
            new_desc = replan_data["new_description"]
            new_tool = replan_data.get("new_tool", goal.tool)
            if new_tool not in allowed:
                logger.info(f"[REPLAN] Tool '{new_tool}' not allowed; keeping {goal.tool}")
                new_tool = goal.tool
            logger.info(f"[REPLAN] New goal: [{new_tool}] {new_desc}")
        else:
            logger.warning(f"[REPLAN] Parse failed — falling back to original description/tool")
            new_desc = goal.description
            new_tool = goal.tool

        return ReasoningGoal(
            description=new_desc,
            tool=new_tool,
            replan_count=goal.replan_count + 1
        )
