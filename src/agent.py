"""
Self-RAG Agent - Main interface for runbook-based incident resolution.

This module provides a high-level API for using the Self-RAG system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from src.state import AgentState
from src.graph import get_graph
from src.config import Config, logger
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import box

console = Console()


class SelfRAGAgent:
    """
    Self-RAG Agent for automated incident resolution using enterprise runbooks.

    Features:
    - Retrieval-Augmented Generation with self-correction
    - Document relevance grading
    - Hallucination detection
    - Resolution validation
    - Query rewriting with retry logic
    - Human escalation when needed

    Example:
        >>> agent = SelfRAGAgent()
        >>> result = agent.resolve_incident("PostgreSQL CPU at 99%")
        >>> print(result["plan_explanation"])
        >>> for cmd in result["executable_commands"]:
        ...     print(f"  $ {cmd}")
    """

    def __init__(self):
        """Initialize the Self-RAG agent."""
        self.graph = get_graph()
        logger.info("Self-RAG Agent initialized")

    def resolve_incident(
        self,
        incident_description: str,
        config_override: Optional[Dict[str, Any]] = None,
        display_results: bool = True
    ) -> Dict[str, Any]:
        """
        Resolve an incident using the Self-RAG workflow.

        Args:
            incident_description: Description of the incident/problem
            config_override: Optional runtime configuration overrides

        Returns:
            Dictionary containing:
            - plan_explanation: Human-readable mitigation plan
            - executable_commands: List of commands to execute
            - execution_path: List of nodes executed
            - total_llm_calls: Number of LLM API calls made
            - requires_escalation: Whether human intervention is needed
            - errors: List of errors encountered (if any)
        """
        logger.info(f"Resolving incident: '{incident_description}'")

        # Initialize state
        initial_state: AgentState = {
            "initial_incident": incident_description,
            "current_query": incident_description,
            "documents": [],
            "document_scores": [],
            "selected_document_index": -1,
            "relevance_score": "",
            "plan_explanation": "",
            "executable_commands": [],
            "hallucination_detected": False,
            "resolves_incident": False,
            "generation_retry_count": 0,
            "retrieval_loop_count": 0,
            "total_llm_calls": 0,
            "query_history": [incident_description],
            "execution_path": ["start"],
            "requires_escalation": False,
            "errors": [],

            # Metrics
            "start_time": time.time(),
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "relevance_grading_time": 0.0,
            "hallucination_check_time": 0.0,
            "resolution_check_time": 0.0,
            "retrieval_score": 0.0,
            "hallucination_score": 0.0,
            "total_duration_ms": 0.0,
            "validation_time": 0.0,
            "is_resolved": False,
        }

        # Apply config overrides
        if config_override:
            logger.debug(f"Applying config overrides: {config_override}")
            # Store in state for nodes to access
            initial_state["config_override"] = config_override

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Calculate total time and add to state
        total_time_ms = (time.time() - final_state['start_time']) * 1000
        final_state['total_duration_ms'] = total_time_ms
        final_state['is_resolved'] = final_state.get(
            'resolves_incident', False)

        # Display results if requested
        if display_results:
            self.display_results(final_state)

        return final_state

    def display_results(self, state: AgentState):
        """Display the final results and metrics in a structured format."""

        print("\n" + "="*80)
        if state.get("requires_escalation", True) and not state.get('is_resolved'):
            print(" ESCALATION REQUIRED".center(80))
        else:
            print(" MITIGATION PLAN GENERATED".center(80))
        print("="*80)

        # --- Metadata Table ---
        metadata_table = Table(title="Execution Metadata", show_header=False,
                               box=box.ROUNDED, title_style="bold magenta")
        metadata_table.add_column("Key", style="cyan")
        metadata_table.add_column("Value", style="white")

        metadata_table.add_row(
            "Total Duration", f"{state.get('total_duration_ms', 0.0):.2f}ms")
        metadata_table.add_row(
            "Resolution Status", "Success" if state['is_resolved'] else "Escalated")
        metadata_table.add_row("LLM Calls", str(state['total_llm_calls']))
        metadata_table.add_row("Retrieval Loops", str(
            state['retrieval_loop_count']))
        metadata_table.add_row(
            "Execution Path", ' → '.join(state['execution_path']))

        # --- Performance Metrics Table ---
        perf_table = Table(title="Performance Metrics", show_header=False,
                           box=box.ROUNDED, title_style="bold magenta")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")

        perf_table.add_row("Retrieval Score",
                           f"{state.get('retrieval_score', 0.0):.4f}")
        perf_table.add_row(
            "Retrieval Time", f"{state.get('retrieval_time', 0.0):.2f}s")
        perf_table.add_row("Relevance Grading Time",
                           f"{state.get('relevance_grading_time', 0.0):.2f}s")
        perf_table.add_row("Generation Time",
                           f"{state.get('generation_time', 0.0):.2f}s")
        perf_table.add_row("Hallucination Check Time",
                           f"{state.get('hallucination_check_time', 0.0):.2f}s")
        perf_table.add_row("Resolution Check Time",
                           f"{state.get('resolution_check_time', 0.0):.2f}s")

        # --- Plan Panel ---
        plan_panel = Panel(
            state.get("plan_explanation", "No explanation generated."),
            title="Mitigation Plan",
            border_style="green" if state['is_resolved'] else "red",
            expand=True
        )

        # --- Commands Panel ---
        commands_str = "\n".join(state.get("executable_commands", []))
        commands_panel = Panel(
            Syntax(commands_str, "bash", theme="monokai",
                   line_numbers=True) if commands_str else "No commands generated.",
            title="Executable Commands",
            border_style="blue",
            expand=True
        )

        # Print layout
        console.print(metadata_table)
        console.print(perf_table)
        console.print(plan_panel)
        if state.get("executable_commands"):
            console.print(commands_panel)

        if state.get("errors"):
            error_panel = Panel(
                "\n".join(f"• {e}" for e in state["errors"]),
                title="Errors Encountered",
                border_style="bold red",
                expand=True
            )
            console.print(error_panel)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics and configuration.

        Returns:
            Dictionary with agent stats
        """
        return {
            "config": {
                "llm_model": Config.LLM_MODEL,
                "max_retrieval_loops": Config.MAX_RETRIEVAL_LOOPS,
                "max_generation_retries": Config.MAX_GENERATION_RETRIES,
                "top_k_documents": Config.TOP_K_DOCUMENTS,
                "min_relevance_score": Config.MIN_RELEVANCE_SCORE,
            },
            "graph": {
                "nodes": 7,
                "entry_point": "retrieve",
                "exit_points": ["escalate", "check_resolution"]
            }
        }


def create_agent() -> SelfRAGAgent:
    """
    Factory function to create a Self-RAG agent instance.

    Returns:
        Configured SelfRAGAgent
    """
    return SelfRAGAgent()
