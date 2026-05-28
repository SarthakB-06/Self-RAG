"""
State definition for the Self-RAG agent graph.

This module defines the AgentState TypedDict that flows through
the LangGraph workflow, tracking all data and control flow.
"""


from typing import TypedDict, List, Optional, Literal

RelevanceScore = Literal["relevant", "irrelevant"]
GenerationQuality = Literal["low", "medium", "high"]
ValidationResult = Literal["fail", "pass"]


class AgentState(TypedDict):
    """
    The state object that flows through the self rag Graph

    This state tracks the entire lifecycle of an incident resolution attempt,
    including query refinement, document retrieval, mitigation generation,
    and multiple quality gates.

    Flow Stages:
    1. Initial query → retrieval
    2. Document grading → relevance check
    3. Generation → quality check
    4. Validation → hallucination + resolution checks
    5. Output or escalation
    """

    initial_incident: str
    current_query: str
    query_history: List[str]
    documents: List[str]
    document_scores: List[float]
    selected_document_index: int
    plan_explanation: str
    executable_commands: List[str]
    executable_metadata: dict
    retrieval_loop_count: int
    generation_retry_count: int
    total_llm_calls: int
    requires_escalation: bool
    relevance_score: Optional[RelevanceScore]
    generation_quality: Optional[GenerationQuality]
    hallucination_detected: Optional[bool]
    resolves_incident: Optional[bool]
    start_time: float
    end_time: Optional[float]
    execution_path: List[str]
    errors: List[str]
    escalation_reason: Optional[str]

    # Metrics
    retrieval_time: float
    generation_time: float
    relevance_grading_time: float
    hallucination_check_time: float
    resolution_check_time: float
    validation_time: float
    retrieval_score: float
    hallucination_score: float
    total_duration_ms: float
    is_resolved: bool


def create_initial_state(incident: str, initial_query: Optional[str] = None) -> AgentState:
    """
    Factory function to create an initial state for the agent.

    Args:
        incident: The incident description
        initial_query: Optional initial query (defaults to incident description)

    Returns:
        AgentState initialized with default values
    """
    import time
    return AgentState(
        initial_incident=incident,
        current_query=initial_query or incident,
        query_history=[],
        documents=[],
        document_scores=[],
        selected_document_index=0,
        plan_explanation="",
        executable_commands=[],
        executable_metadata={},
        retrieval_loop_count=0,
        generation_retry_count=0,
        total_llm_calls=0,
        relevance_score=None,
        generation_quality=None,
        hallucination_detected=None,
        resolves_incident=None,
        start_time=time.time(),
        end_time=None,
        execution_path=[],
        errors=[],
        escalation_reason=None
    )
