"""
Main entry point for Self-RAG Agent CLI.

Usage:
    python main.py "PostgreSQL CPU is at 99%"
    python main.py --interactive
    python main.py --file incidents.txt
"""

from src.config import Config, logger
from src.agent import create_agent
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.console import Console
from typing import List
from pathlib import Path
import argparse
import sys
print("Starting main.py...", flush=True)


print("Importing agent modules...", flush=True)
print("Agent modules imported.", flush=True)

console = Console()


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║              🤖 Self-RAG Incident Resolution Agent        ║
║                                                           ║
║  Automated runbook-based mitigation with self-correction  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print(f"Environment: {Config.ENV}", style="dim")
    console.print(f"LLM Model: {Config.LLM_MODEL}", style="dim")
    console.print()


def resolve_single_incident(agent, incident: str):
    """Resolve a single incident."""
    console.print(f"\n Analyzing: {incident}", style="bold cyan")
    console.print("=" * 80)
    # The agent's display_results method now handles all printing
    agent.resolve_incident(incident)


def resolve_from_file(agent, filepath: Path):
    """Resolve incidents from a file (one per line)."""
    console.print(f"\n Reading incidents from: {filepath}", style="bold cyan")

    with open(filepath, 'r', encoding='utf-8') as f:
        incidents = [line.strip() for line in f if line.strip()
                     and not line.startswith('#')]

    console.print(f"Found {len(incidents)} incident(s)\n")

    results = []
    for i, incident in enumerate(incidents, 1):
        console.print(
            f"\n[{i}/{len(incidents)}] Processing: {incident[:60]}...", style="bold")
        result = agent.resolve_incident(incident)
        results.append(result)

        if i < len(incidents):
            console.print("\n" + "─" * 80 + "\n")

    # Summary - Note: This summary part will need adjustment if you want to aggregate
    # the new rich metrics from each run. For now, it's disabled to prevent errors.
    console.print("\n" + "=" * 80)
    console.print("BATCH PROCESSING COMPLETE", style="bold cyan")
    console.print(f"Processed {len(results)} incidents.", style="cyan")
    console.print("=" * 80)


def interactive_mode(agent):
    """Run agent in interactive mode."""
    console.print("\n Interactive Mode", style="bold cyan")
    console.print("Enter incidents one at a time (or 'quit' to exit)\n")

    while True:
        try:
            incident = console.input(
                "[bold green]Incident:[/bold green] ").strip()

            if not incident:
                continue

            if incident.lower() in ['quit', 'exit', 'q']:
                console.print("\nGoodbye! 👋", style="bold cyan")
                break

            resolve_single_incident(agent, incident)

        except KeyboardInterrupt:
            console.print("\n\nInterrupted. Goodbye! 👋", style="bold cyan")
            break
        except Exception as e:
            console.print(f"\n Error: {e}", style="bold red")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Self-RAG Agent for automated incident resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resolve single incident
  python main.py "PostgreSQL CPU is at 99%"
  
  # Interactive mode
  python main.py --interactive
  
  # Batch processing from file
  python main.py --file incidents.txt
  
  # With debug logging
  python main.py "EC2 instance not responding" --debug
        """
    )

    parser.add_argument(
        "incident",
        nargs="?",
        help="Incident description (if not using --interactive or --file)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Read incidents from file (one per line)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Enable debug logging
    if args.debug:
        logger.setLevel("DEBUG")
        logger.debug("Debug logging enabled")

    # Print banner
    print_banner()

    # Create agent
    console.print(" Initializing agent...", style="bold yellow")
    agent = create_agent()
    console.print(" Agent ready!\n", style="bold green")

    # Determine mode
    if args.interactive:
        interactive_mode(agent)

    elif args.file:
        if not args.file.exists():
            console.print(f" File not found: {args.file}", style="bold red")
            sys.exit(1)
        resolve_from_file(agent, args.file)

    elif args.incident:
        resolve_single_incident(agent, args.incident)

    else:
        parser.print_help()
        console.print(
            "\n Tip: Try 'python main.py --interactive' to start", style="dim")
        sys.exit(1)


if __name__ == "__main__":
    main()
