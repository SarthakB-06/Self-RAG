"""
LangGraph workflow definition for Self-RAG agent.

This file builds the state graph connecting all nodes with conditional edges.
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes.retrieval import retrieve_documents
from src.nodes.grading import grade_document_relevance
from src.nodes.generation import generate_mitigation_plan
from src.nodes.validation import (
    check_hallucination,
    check_resolution,
    rewrite_query,
    aggregate_validation_time
)
from src.nodes.escalation import escalate_to_human
from src.edges.routing import (
    decide_after_grading,
    decide_after_generation,
    decide_after_hallucination_check,
    decide_after_resolution_check
)
from src.config import logger


def build_self_rag_graph() -> StateGraph:
    """
    Build the Self-RAG LangGraph workflow.

    Graph Structure:

    START
      ↓
    retrieve_documents
      ↓
    grade_document_relevance
      ↓ (conditional)
      ├─ relevant → generate_mitigation_plan
      ├─ irrelevant (retries left) → rewrite_query → retrieve_documents (loop)
      └─ irrelevant (max retries) → escalate_to_human → END

    generate_mitigation_plan
      ↓ (conditional)
      ├─ has commands → check_hallucination
      └─ no commands → escalate_to_human → END

    check_hallucination
      ↓ (conditional)
      ├─ no hallucination → check_resolution
      └─ hallucination detected → escalate_to_human → END

    check_resolution
      ↓ (conditional)
      ├─ resolves incident → END (SUCCESS)
      └─ doesn't resolve → escalate_to_human → END

    Returns:
        Compiled StateGraph ready for execution
    """

    logger.info("Building Self-RAG workflow graph")
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_document_relevance)
    workflow.add_node("generate", generate_mitigation_plan)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("check_resolution", check_resolution)
    workflow.add_node("aggregate_validation_time", aggregate_validation_time)
    workflow.add_node("escalate", escalate_to_human)

    logger.debug("Adding 8 nodes to the graph")

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "escalate": "escalate"
        }
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "check_hallucination": "check_hallucination",
            "escalate": "escalate"
        }
    )

    workflow.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            "check_resolution": "check_resolution",
            "escalate": "escalate"
        }
    )

    workflow.add_conditional_edges(
        "check_resolution",
        decide_after_resolution_check,
        {
            "__end__": END,
            "escalate": "escalate"
        }
    )

    workflow.add_edge("escalate", END)

    logger.debug("Added edges between nodes with conditional routing")

    app = workflow.compile()
    logger.info("Workflow graph compiled successfully")

    return app


_graph_instance = None


def get_graph() -> StateGraph:
    """
    Get the compiled Self-RAG graph (singleton).

    Returns:
        Compiled StateGraph
    """
    global _graph_instance

    if _graph_instance is None:
        _graph_instance = build_self_rag_graph()

    return _graph_instance
