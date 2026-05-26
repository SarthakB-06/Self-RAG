"""
Conditional edge routing logic for Self-RAG graph.

These functions determine which node to execute next based on the current state.
"""
from typing import Literal
from src.state import AgentState
from src.config import Config, logger


def decide_after_grading(state: AgentState) -> Literal["generate", "rewrite", "escalate"]:
    """Route after document relevance grading."""
    relevance = state.get("relevance_score", "irrelevant")
    loop_count = state.get("retrieval_loop_count", 0)
    
    logger.info(f"Routing after grading: relevance={relevance}, loop={loop_count}/{Config.MAX_RETRIEVAL_LOOPS}")
    
    if relevance == "relevant":
        logger.info("Route to: generate_mitigation")  # Removed →
        return "generate"
    
    elif loop_count < Config.MAX_RETRIEVAL_LOOPS:
        logger.info("Route to: rewrite_query (retry retrieval)")  # Removed →
        return "rewrite"
    
    else:
        logger.warning("Route to: escalate (max retrieval loops reached)")  # Removed →
        return "escalate"


def decide_after_generation(state: AgentState) -> Literal["check_hallucination", "escalate"]:
    """Route after mitigation plan generation."""
    commands = state.get("executable_commands", [])
    
    logger.info(f"Routing after generation: {len(commands)} command(s) generated")
    
    if len(commands) > 0:
        logger.info("Route to: check_hallucination")  # Removed →
        return "check_hallucination"
    else:
        logger.warning("Route to: escalate (no commands generated)")  # Removed →
        return "escalate"


def decide_after_hallucination_check(state: AgentState) -> Literal["check_resolution", "escalate"]:
    """Route after hallucination detection."""
    hallucination = state.get("hallucination_detected", True)
    
    logger.info(f"Routing after hallucination check: detected={hallucination}")
    
    if not hallucination:
        logger.info("Route to: check_resolution")  # Removed →
        return "check_resolution"
    else:
        logger.warning("Route to: escalate (hallucination detected)")  # Removed →
        return "escalate"


def decide_after_resolution_check(state: AgentState) -> Literal["__end__", "escalate"]:
    """Final routing decision after resolution validation."""
    resolves = state.get("resolves_incident", False)
    
    logger.info(f"Final routing: resolves_incident={resolves}")
    
    if resolves:
        logger.info("Route to: END (Success!)")  # Removed →
        return "__end__"
    else:
        logger.warning("Route to: escalate (plan insufficient)")  # Removed →
        return "escalate"


def should_continue(state: AgentState) -> bool:
    """
    Global check if the agent should continue processing.
    
    Called at various checkpoints to enforce safety limits:
    - Maximum retrieval loops
    - Maximum generation retries
    - Maximum total LLM calls
    - Timeout
    
    Args:
        state: Current agent state
    
    Returns:
        True if agent should continue, False to stop
    """
    if state["retrieval_loop_count"] >= Config.MAX_RETRIEVAL_LOOPS:
        logger.warning(f"Max retrieval loops reached: {state['retrieval_loop_count']}")
        return False
    
    # Check generation retries
    if state["generation_retry_count"] >= Config.MAX_GENERATION_RETRIES:
        logger.warning(f"Max generation retries reached: {state['generation_retry_count']}")
        return False
    
    # Check total LLM calls
    max_llm_calls = Config.MAX_RETRIEVAL_LOOPS * 5  
    if state["total_llm_calls"] >= max_llm_calls:
        logger.warning(f"Max LLM calls reached: {state['total_llm_calls']}")
        return False
    
    # Check for critical errors
    if len(state["errors"]) > 5:
        logger.error(f"Too many errors encountered: {len(state['errors'])}")
        return False
    
    # Check if already escalated
    if state.get("requires_escalation", False):
        logger.info("Escalation flag set, stopping")
        return False
    
    return True