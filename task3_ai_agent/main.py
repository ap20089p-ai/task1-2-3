"""Command-Line Interface for Autonomous Multi-Tool AI Agent.

Provides an interactive and command-line execution interface with
telemetry logs, built-in query examples, and robust error handling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure proper Unicode handling in Windows terminals
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent import AutonomousAgent, ToolExecutionStep

# Example queries specified in project requirements
EXAMPLE_QUERIES = [
    (
        "A. Calculator",
        "Calculate 1250 multiplied by 18 and then add 500.",
    ),
    (
        "B. Web Search",
        "Search the web for the latest Python release and provide the official source URL.",
    ),
    (
        "C. File Reader",
        "Read notes.txt and summarize the project.",
    ),
    (
        "D. Multi-Tool Task",
        "Read notes.txt, summarize the project, and count the listed features using the calculator when needed.",
    ),
    (
        "E. Research Task",
        "Search for two relevant sources about Python, compare their findings, and summarize the differences.",
    ),
]


def print_banner(model_name: str, workspace_dir: Path, show_logs: bool) -> None:
    """Displays the terminal application banner."""
    divider = "=" * 74
    print("\n" + divider)
    print("   AUTONOMOUS MULTI-TOOL AI AGENT (LANGCHAIN + GOOGLE GEMINI)")
    print("   AI Engineering Internship Submission | Safe AST Math & DDG Search")
    print(divider)
    print(f" Model           : {model_name}")
    print(f" Workspace       : {workspace_dir}")
    print(f" Tool Logs       : {'[ENABLED]' if show_logs else '[DISABLED]'} (Type /logs to toggle)")
    print(" Commands        : 'exit'/'quit' to leave | '/examples' for demo queries | '/help'")
    print(divider + "\n")


def print_help() -> None:
    """Displays command help and usage guidelines."""
    print("\n--- AVAILABLE COMMANDS ---")
    print("  exit, quit, q    : Terminate the application.")
    print("  /logs, logs      : Toggle live tool execution logging on or off.")
    print("  /examples        : Display the 5 canonical project benchmark queries.")
    print("  /workspace       : List files currently available in the isolated workspace.")
    print("  /help, help, ?   : Show this help message.")
    print("--------------------------\n")


def print_examples() -> None:
    """Displays the standard example benchmark queries."""
    print("\n--- CANONICAL EXAMPLE QUERIES ---")
    for category, query in EXAMPLE_QUERIES:
        print(f"\n[{category}]")
        print(f"  Query: \"{query}\"")
    print("\nTip: You can copy and paste any of the above queries into the prompt.\n")


def list_workspace_files(workspace_dir: Path) -> None:
    """Lists files located in the dedicated workspace directory."""
    print(f"\n--- WORKSPACE DIRECTORY: {workspace_dir} ---")
    if not workspace_dir.exists():
        print("  Workspace directory does not exist.")
        return

    files = [f for f in workspace_dir.iterdir() if f.is_file()]
    if not files:
        print("  No files currently found in workspace.")
    else:
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  * {f.name:<25} ({size_kb:.2f} KB)")
    print("------------------------------------------\n")


def display_step_log(step: ToolExecutionStep) -> None:
    """Prints a single tool execution step to the console."""
    print(f"  --> [TOOL CALL] {step.tool_name}", flush=True)
    print(f"      Input  : {step.tool_input}", flush=True)
    print(f"      Result : {step.tool_output_summary}", flush=True)


def execute_query(agent: AutonomousAgent, query: str, show_logs: bool) -> None:
    """Executes a query through the autonomous agent and prints formatted output."""
    if not query.strip():
        print("Warning: Please provide a non-empty query.\n", flush=True)
        return

    print("\n" + "-" * 74, flush=True)
    print(f"USER QUERY: {query}", flush=True)
    print("-" * 74, flush=True)

    if show_logs:
        print("AGENT EXECUTION TELEMETRY:", flush=True)

    response = agent.run(
        query=query,
        log_callback=display_step_log if show_logs else None,
    )

    if not show_logs and response.steps:
        print(f"Execution completed in {len(response.steps)} tool step(s). (Use /logs to view telemetry)", flush=True)

    print("\n" + "=" * 74, flush=True)
    print("AGENT FINAL ANSWER:", flush=True)
    print("=" * 74, flush=True)
    print(response.final_answer, flush=True)
    print("=" * 74 + "\n", flush=True)


def main() -> None:
    """Entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Tool AI Agent with LangChain and Google Gemini"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Run a single query non-interactively and exit.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Specify Gemini model override (e.g. gemini-3.8-flash).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable tool execution logs by default.",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Path to custom workspace directory.",
    )

    args = parser.parse_args()

    # Determine logging preference
    env_logs = os.getenv("SHOW_TOOL_LOGS", "true").lower() in ("true", "1", "yes")
    show_logs = args.verbose or env_logs

    # Initialize Autonomous Agent
    try:
        agent = AutonomousAgent(
            model_name=args.model,
            workspace_dir=args.workspace,
        )
    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
        print("Please check your .env file or refer to .env.example.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nInitialization Error: {e}\n")
        sys.exit(1)

    # Non-interactive mode
    if args.query:
        execute_query(agent, args.query, show_logs=show_logs)
        return

    # Interactive CLI Mode
    print_banner(
        model_name=agent.model_name,
        workspace_dir=agent.workspace_dir,
        show_logs=show_logs,
    )

    while True:
        try:
            prompt_label = f"Agent [{agent.model_name}] > "
            user_input = input(prompt_label).strip()

            if not user_input:
                continue

            # Command routing
            lower_input = user_input.lower()
            if lower_input in ("exit", "quit", "q"):
                print("\nExiting Autonomous Agent. Have a productive day!\n")
                break

            if lower_input in ("/help", "help", "?"):
                print_help()
                continue

            if lower_input in ("/examples", "examples"):
                print_examples()
                continue

            if lower_input in ("/workspace", "workspace"):
                list_workspace_files(agent.workspace_dir)
                continue

            if lower_input in ("/logs", "logs"):
                show_logs = not show_logs
                status = "ENABLED" if show_logs else "DISABLED"
                print(f"\n[Configuration] Live tool execution telemetry is now: {status}\n")
                continue

            # Run query through agent
            execute_query(agent, user_input, show_logs=show_logs)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nUnexpected error during execution: {e}\n")


if __name__ == "__main__":
    main()
