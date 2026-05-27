from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time

from src.agent import SelfRAGAgent
from src.config import logger

app = FastAPI(
    title="Enterprise DevOps SRE Copilot API",
    description="REST API for the Self-RAG runbook-based incident resolution agent.",
    version="1.0.0"
)

# Initialize agent globally
agent_instance = None


@app.on_event("startup")
async def startup_event():
    global agent_instance
    logger.info("Initializing Self-RAG Agent for FastAPI...")
    agent_instance = SelfRAGAgent()
    logger.info("Agent initialized successfully.")

# --- Pydantic Models for Validation ---


class IncidentRequest(BaseModel):
    incident_description: str
    config_override: Optional[Dict[str, Any]] = None


class IncidentResponse(BaseModel):
    incident: str
    plan_explanation: str
    executable_commands: List[str]
    execution_path: List[str]
    total_llm_calls: int
    requires_escalation: bool
    is_resolved: bool
    total_duration_ms: float
    retrieval_time: float
    relevance_grading_time: float
    generation_time: float
    hallucination_check_time: float
    resolution_check_time: float
    retrieval_score: float
    errors: List[str]

# --- Endpoints ---


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/diagnose", response_model=IncidentResponse)
async def diagnose_incident(request: IncidentRequest):
    """
    Receive an incident description, run the Self-RAG graph, and return a mitigation plan.
    """
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        start_time = time.time()

        # Call the LangGraph agent
        result = agent_instance.resolve_incident(
            request.incident_description,
            config_override=request.config_override,
            display_results=False
        )

        return IncidentResponse(
            incident=request.incident_description,
            plan_explanation=result.get("plan_explanation", ""),
            executable_commands=result.get("executable_commands", []),
            execution_path=result.get("execution_path", []),
            total_llm_calls=result.get("total_llm_calls", 0),
            requires_escalation=result.get("requires_escalation", False),
            is_resolved=result.get("is_resolved", False),
            total_duration_ms=result.get("total_duration_ms", 0.0),
            retrieval_time=result.get("retrieval_time", 0.0),
            relevance_grading_time=result.get("relevance_grading_time", 0.0),
            generation_time=result.get("generation_time", 0.0),
            hallucination_check_time=result.get(
                "hallucination_check_time", 0.0),
            resolution_check_time=result.get("resolution_check_time", 0.0),
            retrieval_score=result.get("retrieval_score", 0.0),
            errors=result.get("errors", [])
        )

    except Exception as e:
        logger.error(f"Error during incident resolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
