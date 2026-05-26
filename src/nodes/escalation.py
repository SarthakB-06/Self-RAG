"""
Escalation node for handling cases where the agent cannot safely proceed.
"""

from typing import Dict, Any
from src.state import AgentState
from src.config import Config, logger


def escalate_to_human(state: AgentState) -> Dict[str, Any]:
    """
    Escalate to human when agent reaches limits or encounters unsafe conditions.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state with escalation message
    """
    logger.warning("ESCALATING TO HUMAN OPERATOR")
    
    # Determine escalation reason
    reasons = []
    
    if state["retrieval_loop_count"] >= Config.MAX_RETRIEVAL_LOOPS:
        reasons.append(f"Maximum retrieval attempts reached ({Config.MAX_RETRIEVAL_LOOPS})")
    
    if state["generation_retry_count"] >= Config.MAX_GENERATION_RETRIES:
        reasons.append(f"Maximum generation retries reached ({Config.MAX_GENERATION_RETRIES})")
    
    if state["hallucination_detected"]:
        reasons.append("Hallucinated commands detected in generated plan")
    
    if state["relevance_score"] == "irrelevant" and state["retrieval_loop_count"] >= Config.MAX_RETRIEVAL_LOOPS:
        reasons.append("Could not find relevant runbooks after multiple attempts")
    
    if not state["resolves_incident"]:
        reasons.append("Generated plan does not fully resolve the incident")
    
    if state["errors"]:
        reasons.append(f"System errors encountered: {len(state['errors'])} error(s)")
    
    if not reasons:
        reasons.append("Unknown escalation trigger")
    
    # Build escalation message
    escalation_message = "ESCALATION REQUIRED\n\n"
    escalation_message += "This incident requires manual investigation by L2/L3 Support.\n\n"
    escalation_message += "Escalation Reasons:\n"
    for i, reason in enumerate(reasons, 1):
        escalation_message += f"{i}. {reason}\n"
    
    escalation_message += f"\nOriginal Incident: {state['initial_incident']}\n"
    escalation_message += f"Queries Attempted: {', '.join(state['query_history'])}\n"
    escalation_message += f"Documents Retrieved: {len(state['documents'])}\n"
    escalation_message += f"Total LLM Calls: {state['total_llm_calls']}\n"
    
    if state["errors"]:
        escalation_message += f"\nErrors Encountered:\n"
        for error in state["errors"]:
            escalation_message += f"  - {error}\n"
    
    logger.info(f"Escalation reasons: {', '.join(reasons)}")
    
    return {
        "plan_explanation": escalation_message,
        "executable_commands": [],
        "requires_escalation": True,
        "execution_path": state["execution_path"] + ["escalate"]
    }