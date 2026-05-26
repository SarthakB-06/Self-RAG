"""
Validation nodes for hallucination detection and resolution checking.
"""


import time
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from src.state import AgentState
from src.prompts import HALLUCINATION_CHECKER_PROMPT, RESOLUTION_CHECKER_PROMPT
from src.config import Config, logger
from src.llm_factory import create_llm

llm = create_llm()


def check_hallucination(state: AgentState) -> Dict[str, Any]:
    """
    Check for hallucinated commands not present in the source document.
    This uses a direct string search instead of a slow LLM call.
    """
    start_time = time.time()
    logger.info("Performing fast hallucination check (string search)...")

    executable_commands = state.get("executable_commands", [])
    doc_index = state.get("selected_document_index")

    if doc_index is None or not executable_commands:
        # Nothing to check, pass through
        return {
            "hallucination_detected": False,
            "execution_path": state["execution_path"] + ["hallucination_check_skipped"],
            "hallucination_check_time": time.time() - start_time,
        }

    document = state["documents"][doc_index]

    # Handle both Document objects and plain strings
    if hasattr(document, 'page_content'):
        document_content = ' '.join(document.page_content.split()).lower()
    else:
        document_content = ' '.join(str(document).split()).lower()

    hallucinated_commands = []
    for command in executable_commands:
        # Normalize command and check if it's in the document content
        normalized_command = ' '.join(command.split()).lower()
        if normalized_command not in document_content:
            hallucinated_commands.append(command)

    if hallucinated_commands:
        logger.warning(
            f"Hallucination detected! The following commands were not found in the runbook: {hallucinated_commands}")
        error_message = f"Hallucination check failed. Commands not in source: {hallucinated_commands}"
        return {
            "hallucination_detected": True,
            "errors": state["errors"] + [error_message],
            "execution_path": state["execution_path"] + ["hallucination_check_failed"],
            "hallucination_check_time": time.time() - start_time,
        }
    else:
        logger.info(
            "Hallucination check passed. All commands are from the runbook.")
        return {
            "hallucination_detected": False,
            # This step no longer uses an LLM call, so we don't increment the counter
            "execution_path": state["execution_path"] + ["hallucination_check_passed"],
            "hallucination_check_time": time.time() - start_time,
        }


def check_resolution(state: AgentState) -> Dict[str, Any]:
    """
    Check if the mitigation plan is actionable and likely to resolve the incident
    using a rule-based approach instead of a slow LLM call.
    """
    start_time = time.time()
    logger.info("Checking if mitigation plan is actionable (rule-based)...")

    commands = state.get("executable_commands", [])

    if not commands:
        logger.warning("Plan has no commands, marking as non-resolving.")
        return {
            "resolves_incident": False,
            "execution_path": state["execution_path"] + ["does_not_resolve"],
            "resolution_check_time": time.time() - start_time,
        }

    # Define keywords for diagnostic and mitigation commands
    DIAGNOSTIC_KEYWORDS = [
        "get", "list", "describe", "ps", "top", "grep", "logs", "check",
        "status", "df", "netstat", "iostat", "vmstat", "curl", "ping"
    ]
    MITIGATION_KEYWORDS = [
        "reboot", "restart", "stop", "start", "kill", "delete", "update",
        "apply", "scale", "terminate", "rm", "systemctl"
    ]

    has_diagnostic = False
    has_mitigation = False

    for command in commands:
        cmd_lower = command.lower()
        if any(keyword in cmd_lower for keyword in DIAGNOSTIC_KEYWORDS):
            has_diagnostic = True
        if any(keyword in cmd_lower for keyword in MITIGATION_KEYWORDS):
            has_mitigation = True

    # A plan is considered resolving if it contains at least one diagnostic or mitigation command.
    if has_diagnostic or has_mitigation:
        logger.info(
            "Plan is considered actionable and likely to resolve the incident.")
        return {
            "resolves_incident": True,
            "execution_path": state["execution_path"] + ["resolves"],
            "resolution_check_time": time.time() - start_time,
        }
    else:
        logger.warning(
            "Plan contains no recognizable diagnostic or mitigation commands.")
        return {
            "resolves_incident": False,
            "execution_path": state["execution_path"] + ["does_not_resolve"],
            "resolution_check_time": time.time() - start_time,
        }


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    """
    Rewrite the query to find better documents when retrieval fails.

    Args:
        state: Current agent state with failed query

    Returns:
        Updated state with new current_query
    """

    from src.prompts import QUERY_REWRITER_PROMPT
    logger.info("Rewriting query to improve retrieval results")

    prompt = QUERY_REWRITER_PROMPT.format(
        incident=state["initial_incident"],
        previous_query=state["current_query"],
        failure_reason="Retrieved documents were not relevant to the incident"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip()
        logger.info(f"Rewritten query: {new_query}")

        return {
            "current_query": new_query,
            "query_history": state["query_history"] + [new_query],
            "total_llm_calls": state["total_llm_calls"] + 1,
            "execution_path": state["execution_path"] + ["query_rewrite"]
        }
    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        return {
            "current_query": state["current_query"],
            "errors": state["errors"] + [f"Query rewrite error: {str(e)}"],
            "execution_path": state["execution_path"] + ["query_rewrite_error"]
        }


def aggregate_validation_time(state: AgentState) -> Dict[str, Any]:
    """
    Aggregates the time spent in all validation steps.
    """
    validation_time = state.get("hallucination_check_time", 0.0) + \
        state.get("resolution_check_time", 0.0)
    logger.info(f"Total validation time: {validation_time:.4f}s")
    return {"validation_time": validation_time}
