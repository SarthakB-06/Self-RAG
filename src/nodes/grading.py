"""
Grading nodes for document relevance and generation quality.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage

from src.state import AgentState
from src.prompts import RELEVANCE_GRADER_PROMPT, GENERATION_QUALITY_GRADER_PROMPT, format_document_preview
from src.config import Config, logger
from src.llm_factory import create_llm

llm = create_llm()


def grade_document_relevance(state: AgentState) -> Dict[str, Any]:
    """
    Grade whether the retrieved document is relevant to the incident.

    Args:
        state: Current agent state

    Returns:
        Updated state with relevance_score
    """
    logger.info("Grading document relevance")

    if not state["documents"]:
        logger.warning("No documents to grade for relevance")
        return {
            "relevance_score": "irrelevant",
            "execution_path": state["execution_path"] + ["grade_no_docs"]
        }

    document = state["documents"][0]

    prompt = RELEVANCE_GRADER_PROMPT.format(
        incident=state["initial_incident"],
        document_title="Retrieved Runbook",
        document_preview=format_document_preview(document)
    )

    try:
        logger.info(
            f"Invoking LLM for relevance grading (model: {Config.LLM_MODEL})...")
        response = llm.invoke([HumanMessage(content=prompt)])
        relevance = response.content.strip().lower()

        if relevance not in ["relevant", "irrelevant"]:
            logger.warning(
                f"Unexpected relevance grading response: {response.content}")
            relevance = "irrelevant"
        logger.info(f"Document relevance graded as: {relevance}")

        return {
            "relevance_score": relevance,
            "selected_document_index": 0,
            "total_llm_calls": state["total_llm_calls"] + 1,
            "execution_path": state["execution_path"] + [f"grade_{relevance}"]
        }
    except Exception as e:
        logger.error(f"Error during relevance grading: {e}")
        return {
            "relevance_score": "irrelevant",
            "errors": state["errors"] + [f"Relevance grading error: {str(e)}"],
            "execution_path": state["execution_path"] + ["grade_error"]

        }


def grade_generation_quality(state: AgentState) -> Dict[str, Any]:
    """
    Grade the quality of the generated mitigation plan.

    Args:
        state: Current agent state

    Returns:
        Updated state with generation_quality
    """
    logger.info("Grading generation quality")

    prompt = GENERATION_QUALITY_GRADER_PROMPT.format(
        incident=state["initial_incident"],
        plan_explanation=state["plan_explanation"],
        commands="\n".join(state["executable_commands"])
        # generated_plan = state["generated_plan"]
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        quality = response.content.strip().lower()

        if quality not in ["high", "medium", "low"]:
            logger.warning(
                f"Unexpected generation quality grading response: {response.content}")
            quality = "low"
        logger.info(f"Generation quality graded as: {quality}")

        return {
            "generation_quality": quality,
            "total_llm_calls": state["total_llm_calls"] + 1,
            "execution_path": state["execution_path"] + [f"grade_quality_{quality}"]
        }
    except Exception as e:
        logger.error(f"Error during generation quality grading: {e}")
        return {
            "generation_quality": "low",
            "errors": state["errors"] + [f"Generation quality grading error: {str(e)}"],
            "execution_path": state["execution_path"] + ["grade_quality_error"]

        }
