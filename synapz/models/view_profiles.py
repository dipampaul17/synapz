#!/usr/bin/env python3
"""
CLI utility to view detailed cognitive profiles.

This module provides a command-line interface for viewing and examining
the detailed cognitive profiles used for adaptation.
"""

import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from synapz.models.learner_profiles import (
    get_all_profiles,
    load_profile,
    get_profile_for_adaptation
)
from synapz.core.profiles import CognitiveProfile

console = Console()

def display_profile_list():
    """Display a list of all available profiles."""
    profiles = get_all_profiles()
    
    table = Table(title="Available Cognitive Profiles")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Primary Modalities", style="yellow")
    table.add_column("Processing Style", style="magenta")
    
    for profile in profiles:
        profile_id = profile["id"]
        name = profile["name"]
        primary = ", ".join(profile["modality_preferences"]["primary"])
        processing = profile["cognitive_traits"]["processing_style"]
        
        table.add_row(profile_id, name, primary, processing)
    
    console.print(table)

def display_profile_details(profile_id):
    """Display detailed information for a specific profile."""
    try:
        profile = get_profile_for_adaptation(profile_id)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return
    
    # Display basic info
    console.print(Panel(f"[bold cyan]{profile['name']}[/] ([italic]{profile_id}[/])"))
    
    # Display cognitive traits
    traits_table = Table(title="Cognitive Traits")
    traits_table.add_column("Trait", style="cyan")
    traits_table.add_column("Value", style="green")
    
    traits = profile["cognitive_traits"]
    for key, value in traits.items():
        if key not in ["strengths", "challenges"]:
            traits_table.add_row(key.replace("_", " ").title(), value)
    
    console.print(traits_table)
    
    # Display strengths and challenges
    console.print("[bold cyan]Strengths:[/]")
    for item in traits.get("strengths", []):
        console.print(f"• [green]{item}[/]")
    
    console.print("\n[bold cyan]Challenges:[/]")
    for item in traits.get("challenges", []):
        console.print(f"• [yellow]{item}[/]")
    
    # Display modality preferences
    mod_table = Table(title="Modality Preferences")
    mod_table.add_column("Type", style="cyan")
    mod_table.add_column("Modalities", style="green")
    
    modality = profile["modality_preferences"]
    mod_table.add_row("Primary", ", ".join(modality.get("primary", [])))
    mod_table.add_row("Secondary", ", ".join(modality.get("secondary", [])))
    mod_table.add_row("Avoid", ", ".join(modality.get("avoid", [])))
    
    console.print(mod_table)
    
    # Display pedagogical needs
    ped_table = Table(title="Pedagogical Needs")
    ped_table.add_column("Need", style="cyan")
    ped_table.add_column("Recommendation", style="green")
    
    pedagogy = profile["pedagogical_needs"]
    for key, value in pedagogy.items():
        if key != "example_types":
            ped_table.add_row(key.replace("_", " ").title(), value)
    
    console.print(ped_table)
    
    # Display example types
    console.print("[bold cyan]Recommended Example Types:[/]")
    for item in pedagogy.get("example_types", []):
        console.print(f"• [green]{item}[/]")
    
    # Display adaptation parameters
    console.print("\n[bold cyan]Adaptation Parameters:[/]")
    for key, value in profile.get("adaptation", {}).items():
        console.print(f"• [yellow]{key.replace('_', ' ').title()}:[/] {value}")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="View detailed cognitive profiles")
    parser.add_argument("--profile", help="Specific profile ID to view")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.profile:
        if args.json:
            # Output as JSON
            try:
                profile = get_profile_for_adaptation(args.profile)
                print(json.dumps(profile, indent=2))
            except ValueError as e:
                print(json.dumps({"error": str(e)}, indent=2))
        else:
            # Output as rich formatted text
            display_profile_details(args.profile)
    else:
        # List all profiles
        if args.json:
            profiles = get_all_profiles()
            print(json.dumps(profiles, indent=2))
        else:
            display_profile_list()

if __name__ == "__main__":
    main() 