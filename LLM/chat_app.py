#!/usr/bin/env python3
"""
FinSight LLM - Command-Line Chat App
====================================

A lightweight local chat interface to interact with the FinSight LLM client.
Uses the Phi-3 Mini model through Ollama for one-shot conversational responses.

Author: FinSight AI Team
Date: October 2025
"""

from llm_client import LLMClient
from rich.console import Console

# Create a styled console for better output readability
console = Console()

def main():
    """Start the interactive chat loop with FinSight AI."""
    console.print("\n[bold cyan]====================================[/bold cyan]")
    console.print("[bold yellow]  FinSight AI – Local Chat Assistant[/bold yellow]")
    console.print("[bold cyan]====================================[/bold cyan]\n")

    console.print("Type your message below (or 'exit' to quit):\n", style="bold white")

    # Initialize the LLM client
    client = LLMClient()

    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]User:[/bold green] ").strip()

            # Exit condition
            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[bold magenta]Goodbye! 👋[/bold magenta]")
                break

            # Generate model response
            response = client.generate_response(user_input)

            # Display FinSight's answer
            console.print(f"[bold blue]FinSight:[/bold blue] {response}\n")

        except KeyboardInterrupt:
            console.print("\n\n[bold magenta]Chat ended by user. Goodbye! 👋[/bold magenta]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            break


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()