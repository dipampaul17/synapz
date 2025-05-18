#!/usr/bin/env python3
"""
CLI utility to view and explore algebra concepts.

This module provides a command-line interface for viewing and exploring
the algebra concepts used in the teaching experiments.
"""

import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from synapz.models.concepts.algebra_concepts import (
    get_all_concepts,
    load_concept,
    get_concepts_by_difficulty,
    get_concept_sequence
)

console = Console()

def display_concept_list(concepts=None, sort_by="difficulty"):
    """
    Display a list of concepts.
    
    Args:
        concepts: List of concepts to display (default: all concepts)
        sort_by: Field to sort concepts by (default: difficulty)
    """
    if concepts is None:
        concepts = get_all_concepts()
    
    # Sort concepts
    if sort_by == "difficulty":
        sorted_concepts = sorted(concepts, key=lambda c: c["difficulty"])
    elif sort_by == "title":
        sorted_concepts = sorted(concepts, key=lambda c: c["title"])
    else:
        sorted_concepts = concepts
    
    table = Table(title="Algebra Concepts")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Description", style="white", no_wrap=False)
    
    for concept in sorted_concepts:
        # Truncate description for display
        description = concept["description"]
        if len(description) > 60:
            description = description[:57] + "..."
            
        table.add_row(
            concept["id"],
            concept["title"],
            str(concept["difficulty"]),
            description
        )
    
    console.print(table)

def display_concept_details(concept_id):
    """
    Display detailed information for a specific concept.
    
    Args:
        concept_id: ID of the concept to display
    """
    try:
        concept = load_concept(concept_id)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return
    
    # Display basic info
    console.print(Panel(
        f"[bold cyan]{concept['title']}[/] ([italic]{concept_id}[/]) - Difficulty: {concept['difficulty']}/5"
    ))
    
    # Display description
    console.print("[bold]Description:[/]")
    console.print(concept["description"])
    console.print()
    
    # Display examples
    console.print("[bold]Examples:[/]")
    for i, example in enumerate(concept["examples"], 1):
        console.print(f"{i}. [green]{example}[/]")
    
    # If we have related concepts, display them
    if "related_concepts" in concept:
        console.print("\n[bold]Related Concepts:[/]")
        for related in concept["related_concepts"]:
            console.print(f"• [cyan]{related}[/]")

def display_difficulty_levels():
    """Display concept counts by difficulty level."""
    table = Table(title="Algebra Concepts by Difficulty Level")
    table.add_column("Difficulty", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Count", style="yellow")
    table.add_column("Concepts", style="white")
    
    difficulty_descriptions = {
        1: "Introductory - Basic building blocks",
        2: "Elementary - Simple applications",
        3: "Intermediate - Combined concepts",
        4: "Advanced - Complex applications", 
        5: "Expert - Specialized topics"
    }
    
    for level in range(1, 6):
        concepts = get_concepts_by_difficulty(level)
        concept_names = ", ".join([c["title"] for c in concepts])
        
        table.add_row(
            str(level),
            difficulty_descriptions.get(level, ""),
            str(len(concepts)),
            concept_names
        )
    
    console.print(table)

def display_learning_path(start_level=1, max_level=5):
    """
    Display a recommended learning path through the concepts.
    
    Args:
        start_level: Starting difficulty level
        max_level: Maximum difficulty level
    """
    sequence = get_concept_sequence(start_level, max_level)
    
    console.print(Panel(
        f"[bold]Learning Path (Difficulty {start_level}-{max_level})[/]"
    ))
    
    for i, concept in enumerate(sequence, 1):
        console.print(
            f"{i}. [bold cyan]{concept['title']}[/] " +
            f"([yellow]Level {concept['difficulty']}[/]): " +
            f"{concept['description'][:80]}..."
        )

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="View algebra concepts")
    parser.add_argument("--concept", help="Specific concept ID to view")
    parser.add_argument("--difficulty", type=int, help="Show concepts at a specific difficulty level")
    parser.add_argument("--list", action="store_true", help="List all concepts")
    parser.add_argument("--sort", choices=["difficulty", "title"], default="difficulty", 
                        help="Sort concepts by this field")
    parser.add_argument("--path", action="store_true", help="Show recommended learning path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--levels", action="store_true", help="Show difficulty level overview")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.concept, args.difficulty, args.list, args.path, args.levels]):
        parser.print_help()
        return
    
    # Handle json output
    if args.json:
        if args.concept:
            try:
                concept = load_concept(args.concept)
                print(json.dumps(concept, indent=2))
            except ValueError as e:
                print(json.dumps({"error": str(e)}, indent=2))
        elif args.difficulty:
            concepts = get_concepts_by_difficulty(args.difficulty)
            print(json.dumps(concepts, indent=2))
        else:
            concepts = get_all_concepts()
            print(json.dumps(concepts, indent=2))
        return
    
    # Handle various display modes
    if args.concept:
        display_concept_details(args.concept)
    elif args.difficulty:
        concepts = get_concepts_by_difficulty(args.difficulty)
        display_concept_list(concepts, args.sort)
    elif args.list:
        display_concept_list(sort_by=args.sort)
    elif args.path:
        display_learning_path()
    elif args.levels:
        display_difficulty_levels()

if __name__ == "__main__":
    main() 