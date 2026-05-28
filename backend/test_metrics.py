"""
Dedicated script for running a suite of tests to gather performance metrics.

This script runs the agent against a predefined set of incidents and
calculates aggregate metrics to evaluate the agent's performance,
making it easy to generate resume-ready statistics.
"""

from src.config import logger, Config
from src.agent import create_agent
import asyncio
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time
import numpy as np

# Ensure the script can find the 'src' module
import sys
sys.path.append(str(Path(__file__).parent))


console = Console()


# --- Test Configuration ---

# A diverse set of incidents to test the agent's capabilities
TEST_INCIDENTS = [
    # Core Infrastructure
    "SSH connection to EC2 instance is timing out",
    "High CPU utilization on EC2 instance",

    # Serverless
    "My Lambda function exceeds its memory limit",
    "An S3 event is not triggering my Lambda function",

    # Containers
    "My EKS pod is stuck in CrashLoopBackOff",
    "Image pull is failing in ECS with Docker errors",

    # Database & Storage
    "There is a connection timeout from my Lambda to my RDS instance",

    # K8s General
    "A Kubernetes node is not joining the cluster",

    # Generic/Ambiguous (to test robustness)
    "The application is slow",
    "Database is unresponsive",
]

# Reduce the number of incidents for a quick test run if needed
IS_QUICK_TEST = False
QUICK_TEST_SIZE = 5


def run_all_tests():
    """
    Executes the agent against all defined test incidents and gathers metrics.
    """
    print_header()

    incidents_to_run = TEST_INCIDENTS
    if IS_QUICK_TEST:
        console.print(
            f"[yellow]Performing a quick test with {QUICK_TEST_SIZE} incidents.[/yellow]\n")
        incidents_to_run = incidents_to_run[:QUICK_TEST_SIZE]

    console.print(f"Found {len(incidents_to_run)} incidents to process.")
    console.print(
        f"LLM Model: [cyan]{Config.LLM_MODEL}[/cyan], Environment: [cyan]{Config.ENV}[/cyan]")
    console.print("-" * 80)

    # Suppress detailed agent logging for clean output
    logger.setLevel("ERROR")
    agent = create_agent()
    results = []

    start_time = time.time()

    for i, incident in enumerate(incidents_to_run, 1):
        console.print(
            f"[{i}/{len(incidents_to_run)}] Processing: '[bold blue]{incident}[/bold blue]'...")
        try:
            # The agent now returns the final state, which contains all metrics
            final_state = agent.resolve_incident(
                incident, display_results=False)
            results.append(final_state)
        except Exception as e:
            console.print(
                f"  [bold red]Error processing incident: {e}[/bold red]")
            # Record a failure but continue the test run
            results.append({"error": str(e), "requires_escalation": True})

    total_duration = time.time() - start_time

    console.print("\n" + "=" * 80)
    console.print("[bold green]Test Run Complete[/bold green]")
    console.print(f"Total execution time: {total_duration:.2f} seconds")
    console.print("=" * 80 + "\n")

    calculate_and_display_metrics(results, total_duration)


def calculate_and_display_metrics(results: list, total_duration: float):
    """
    Calculates and displays aggregate performance metrics from the test results.
    """
    if not results:
        console.print("[bold red]No results to analyze.[/bold red]")
        return

    num_total = len(results)
    num_errors = sum(1 for r in results if "error" in r)
    valid_results = [r for r in results if "error" not in r]
    num_valid = len(valid_results)

    if not num_valid:
        console.print(
            f"[bold red]All {num_total} runs failed. Cannot calculate metrics.[/bold red]")
        return

    # --- Calculate Metrics ---
    num_successful = sum(1 for r in valid_results if not r.get(
        "requires_escalation", True))
    num_escalated = num_valid - num_successful
    success_rate = (num_successful / num_valid) * 100 if num_valid > 0 else 0

    # Timing metrics (ms)
    durations = [r["total_duration_ms"] for r in valid_results]
    retrieval_times = [r["retrieval_time"] for r in valid_results]
    generation_times = [r["generation_time"] for r in valid_results]
    validation_times = [r["validation_time"] for r in valid_results]

    # Score metrics
    retrieval_scores = [r["retrieval_score"]
                        for r in valid_results if r.get("retrieval_score") is not None]
    hallucination_scores = [r["hallucination_score"]
                            for r in valid_results if r.get("hallucination_score") is not None]

    # --- Display Summary ---
    summary_panel = Panel(
        f"[bold green]Success Rate: {success_rate:.2f}%[/bold green]\n"
        f"Total Incidents: {num_total}\n"
        f"Successful Resolutions: {num_successful}\n"
        f"Required Escalation: {num_escalated}\n"
        f"Failed Runs (Errors): {num_errors}",
        title="[bold cyan]Overall Performance[/bold cyan]",
        border_style="cyan"
    )
    console.print(summary_panel)

    # --- Display Tables ---
    timing_table = Table(title="⏱️ Performance Timing (milliseconds)")
    timing_table.add_column("Metric", style="cyan", no_wrap=True)
    timing_table.add_column("Average", style="magenta")
    timing_table.add_column("Median", style="green")
    timing_table.add_column("Min", style="blue")
    timing_table.add_column("Max", style="yellow")
    timing_table.add_column("Std Dev", style="red")

    timing_table.add_row("Total Duration", f"{np.mean(durations):.2f}", f"{np.median(durations):.2f}",
                         f"{np.min(durations):.2f}", f"{np.max(durations):.2f}", f"{np.std(durations):.2f}")
    timing_table.add_row("  - Retrieval", f"{np.mean(retrieval_times):.2f}", f"{np.median(retrieval_times):.2f}",
                         f"{np.min(retrieval_times):.2f}", f"{np.max(retrieval_times):.2f}", f"{np.std(retrieval_times):.2f}")
    timing_table.add_row("  - Generation", f"{np.mean(generation_times):.2f}", f"{np.median(generation_times):.2f}",
                         f"{np.min(generation_times):.2f}", f"{np.max(generation_times):.2f}", f"{np.std(generation_times):.2f}")
    timing_table.add_row("  - Validation", f"{np.mean(validation_times):.2f}", f"{np.median(validation_times):.2f}",
                         f"{np.min(validation_times):.2f}", f"{np.max(validation_times):.2f}", f"{np.std(validation_times):.2f}")

    quality_table = Table(title="✅ Quality & Efficiency")
    quality_table.add_column("Metric", style="cyan", no_wrap=True)
    quality_table.add_column("Average", style="magenta")
    quality_table.add_column("Median", style="green")
    quality_table.add_column("Min", style="blue")
    quality_table.add_column("Max", style="yellow")

    if retrieval_scores:
        quality_table.add_row("Retrieval Score (1-5)", f"{np.mean(retrieval_scores):.2f}",
                              f"{np.median(retrieval_scores):.2f}", f"{np.min(retrieval_scores):.2f}", f"{np.max(retrieval_scores):.2f}")
    if hallucination_scores:
        quality_table.add_row("Hallucination Score (0-1)", f"{np.mean(hallucination_scores):.2f}",
                              f"{np.median(hallucination_scores):.2f}", f"{np.min(hallucination_scores):.2f}", f"{np.max(hallucination_scores):.2f}")

    llm_calls = [r["total_llm_calls"] for r in valid_results]
    quality_table.add_row("LLM Calls per Incident", f"{np.mean(llm_calls):.2f}",
                          f"{np.median(llm_calls):.2f}", f"{np.min(llm_calls):.2f}", f"{np.max(llm_calls):.2f}")

    retrieval_loops = [r["retrieval_loops"] for r in valid_results]
    quality_table.add_row("Retrieval Loops", f"{np.mean(retrieval_loops):.2f}",
                          f"{np.median(retrieval_loops):.2f}", f"{np.min(retrieval_loops):.2f}", f"{np.max(retrieval_loops):.2f}")

    console.print(timing_table)
    console.print(quality_table)


def print_header():
    """Prints the script header."""
    console.print("=" * 80, style="bold cyan")
    console.print("📊 Self-RAG Agent Performance & Metrics Test Suite 📊",
                  justify="center", style="bold cyan")
    console.print("=" * 80, style="bold cyan")
    console.print()


if __name__ == "__main__":
    # Note: The agent's async nature might cause issues if run directly in a
    # non-async context on some platforms. This structure is simplified for clarity.
    # For robust async handling, consider using asyncio.run(main_async_function()).
    run_all_tests()
