#!/usr/bin/env python3
"""
Synapz CLI - Run adaptive learning experiments and generate evidence.

This script provides a command-line interface for running experiments
with different cognitive profiles and topics.
"""

import os
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from synapz import PACKAGE_ROOT, DATA_DIR, PROMPTS_DIR
from synapz.core import (
    BudgetTracker, 
    APIClient, 
    CognitiveProfile,
    ProfileManager,
    ContentAdapter,
    Database
)
from synapz.data import ExperimentStorage, AdaptationAnalyzer

# Import profiles if available
try:
    from synapz.models.view_profiles import display_profile_list, display_profile_details
    from synapz.models.learner_profiles import get_all_profiles
    PROFILES_AVAILABLE = True
except ImportError:
    PROFILES_AVAILABLE = False

# Import concepts if available
try:
    from synapz.models.concepts.view_concepts import (
        display_concept_list, 
        display_concept_details,
        display_difficulty_levels,
        display_learning_path
    )
    from synapz.models.concepts.algebra_concepts import (
        get_all_concepts,
        get_concepts_by_difficulty
    )
    CONCEPTS_AVAILABLE = True
except ImportError:
    CONCEPTS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Setup rich console
console = Console()

def setup_environment(
    db_path: str, 
    budget_limit: float = 20.0,
    api_key: Optional[str] = None,
    use_new_db: bool = False
) -> Dict[str, Any]:
    """
    Set up the environment for running experiments.
    
    Args:
        db_path: Path to SQLite database
        budget_limit: Maximum budget in dollars
        api_key: OpenAI API key (defaults to env var)
        use_new_db: Whether to use the new Database class
        
    Returns:
        Dictionary of initialized components
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize components
    budget_tracker = BudgetTracker(db_path, max_budget=budget_limit)
    api_client = APIClient(budget_tracker, api_key=api_key)
    profile_manager = ProfileManager()
    
    # Choose database implementation
    if use_new_db:
        # Use the new Database class
        database = Database(db_path=db_path)
        # Create ExperimentStorage with the same path for compatibility
        experiment_storage = ExperimentStorage(db_path)
    else:
        # Use the original ExperimentStorage
        database = None
        experiment_storage = ExperimentStorage(db_path)
    
    analysis = AdaptationAnalyzer(experiment_storage)
    
    # Create content adapter
    content_adapter = ContentAdapter(
        api_client=api_client,
        profile_manager=profile_manager,
        prompts_dir=PROMPTS_DIR
    )
    
    return {
        "budget_tracker": budget_tracker,
        "api_client": api_client,
        "profile_manager": profile_manager,
        "experiment_storage": experiment_storage,
        "content_adapter": content_adapter,
        "analysis": analysis,
        "database": database
    }

def run_experiment(
    components: Dict[str, Any],
    topic: str,
    profile_name: str,
    learning_objective: str,
    background_knowledge: Optional[str] = None,
    run_control: bool = True
) -> tuple:
    """
    Run an experiment with a specific profile and topic.
    
    Args:
        components: Dictionary of components from setup_environment
        topic: Educational topic to explain
        profile_name: Name of the cognitive profile
        learning_objective: What the student should learn
        background_knowledge: Optional prior knowledge
        run_control: Whether to also run control experiment
        
    Returns:
        Tuple of (adapted_id, control_id) experiment IDs
    """
    content_adapter = components["content_adapter"]
    storage = components["experiment_storage"]
    
    # Convert profile name to enum
    profile = CognitiveProfile.from_string(profile_name)
    
    # Run adapted experiment
    console.print(f"[bold green]Running experiment for {profile.value} profile on topic: {topic}[/]")
    adapted_result = content_adapter.generate_adapted_content(
        topic=topic,
        profile=profile,
        learning_objective=learning_objective,
        background_knowledge=background_knowledge
    )
    
    # Store adapted experiment
    adapted_id = storage.store_experiment(
        profile=profile.value,
        topic=topic,
        learning_objective=learning_objective,
        is_control=False,
        content=adapted_result["content"],
        tokens_in=adapted_result["usage"]["tokens_in"],
        tokens_out=adapted_result["usage"]["tokens_out"],
        cost=adapted_result["usage"]["cost"]
    )
    
    console.print(f"[green]Adapted content generated and stored with ID: {adapted_id}[/]")
    
    # Run control experiment if requested
    control_id = None
    if run_control:
        console.print(f"[bold yellow]Running control experiment for topic: {topic}[/]")
        control_result = content_adapter.generate_adapted_content(
            topic=topic,
            profile=CognitiveProfile.CONTROL,
            learning_objective=learning_objective,
            background_knowledge=background_knowledge
        )
        
        # Store control experiment
        control_id = storage.store_experiment(
            profile=CognitiveProfile.CONTROL.value,
            topic=topic,
            learning_objective=learning_objective,
            is_control=True,
            content=control_result["content"],
            tokens_in=control_result["usage"]["tokens_in"],
            tokens_out=control_result["usage"]["tokens_out"],
            cost=control_result["usage"]["cost"]
        )
        
        console.print(f"[yellow]Control content generated and stored with ID: {control_id}[/]")
    
    return adapted_id, control_id

def display_budget_status(budget_tracker: BudgetTracker) -> None:
    """Display current budget status."""
    current_spend = budget_tracker.get_current_spend()
    remaining = budget_tracker.max_budget - current_spend
    
    table = Table(title="Budget Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Current Spend", f"${current_spend:.4f}")
    table.add_row("Budget Limit", f"${budget_tracker.max_budget:.2f}")
    table.add_row("Remaining", f"${remaining:.4f}")
    table.add_row("Percent Used", f"{(current_spend / budget_tracker.max_budget) * 100:.1f}%")
    
    console.print(table)

def display_experiment_comparison(
    components: Dict[str, Any],
    adapted_id: int,
    control_id: int
) -> None:
    """Display comparison between adapted and control experiments with scientific metrics."""
    analysis = components["analysis"]
    comparison = analysis.compare_experiments(adapted_id, control_id)
    
    # Display key metrics
    table = Table(title=f"Experiment Comparison - {comparison['topic']} ({comparison['adapted_profile']})")
    table.add_column("Metric", style="cyan")
    table.add_column("Adapted", style="green")
    table.add_column("Control", style="yellow")
    table.add_column("Difference %", style="magenta")
    table.add_column("Significance", style="blue")
    
    # Show readability metrics with statistical significance
    for metric in ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]:
        # Get significance if available
        sig_info = comparison.get('statistical_significance', {}).get(metric, {})
        p_value = sig_info.get('p_value', 1.0)
        is_sig = sig_info.get('is_significant', False)
        
        sig_text = f"p={p_value:.3f}"
        if is_sig:
            sig_text += " *"
            
        table.add_row(
            metric.replace("_", " ").title(),
            f"{comparison['adapted_metrics'][metric]:.2f}",
            f"{comparison['control_metrics'][metric]:.2f}",
            f"{comparison['readability_diff_percent'][metric]:.1f}%",
            sig_text
        )
    
    # Show similarity
    table.add_row(
        "Content Similarity",
        f"{comparison['similarity'] * 100:.1f}%",
        "100.0%",
        f"{(comparison['similarity'] - 1.0) * 100:.1f}%",
        ""
    )
    
    # Show effectiveness score
    effectiveness = comparison.get('effectiveness_score', 0.0)
    effectiveness_color = "green" if effectiveness > 0 else "red"
    console.print(f"[bold]Overall Effectiveness Score:[/bold] [{effectiveness_color}]{effectiveness:.2f}[/]")
    
    if effectiveness > 0.5:
        console.print("[bold green]Strong evidence[/] of adaptation effectiveness")
    elif effectiveness > 0.2:
        console.print("[bold yellow]Moderate evidence[/] of adaptation effectiveness")
    elif effectiveness > 0:
        console.print("[bold blue]Suggestive evidence[/] of adaptation effectiveness")
    else:
        console.print("[bold red]No evidence[/] of adaptation effectiveness")
    
    console.print(table)

def generate_evidence_report(components: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive evidence report with scientific metrics."""
    analysis = components["analysis"]
    
    console.print("[bold]Generating evidence report...[/]")
    report = analysis.generate_evidence_report(output_path)
    
    # Print summary statistics
    table = Table(title="Evidence Summary by Profile")
    table.add_column("Profile", style="cyan")
    table.add_column("Sample Size", style="green")
    table.add_column("Effectiveness", style="magenta")
    table.add_column("Significance", style="blue")
    table.add_column("Effect Size", style="yellow")
    
    for profile, stats in report["aggregate_results"].items():
        # Get p-value for statistical significance
        p_value = stats.get("overall_p_value", 1.0)
        is_significant = p_value < 0.05
        
        # Find largest effect size
        largest_effect = 0.0
        for metric, result in stats.get("statistical_significance", {}).items():
            effect = result.get("effect_size", 0)
            if effect > largest_effect:
                largest_effect = effect
        
        # Determine evidence level
        evidence_level = "None"
        if stats.get("effectiveness_score", 0) > 0.5:
            evidence_level = "[bold green]Strong[/]"
        elif stats.get("effectiveness_score", 0) > 0.2:
            evidence_level = "[yellow]Moderate[/]"
        elif stats.get("effectiveness_score", 0) > 0:
            evidence_level = "[blue]Suggestive[/]"
        else:
            evidence_level = "[red]None[/]"
            
        table.add_row(
            profile,
            str(stats["sample_count"]),
            f"{stats.get('effectiveness_score', 0):.2f} ({evidence_level})",
            f"p={p_value:.3f}" + (" *" if is_significant else ""),
            f"{largest_effect:.2f}" + (" (Large)" if largest_effect > 0.8 else 
                                     " (Medium)" if largest_effect > 0.5 else 
                                     " (Small)" if largest_effect > 0.2 else 
                                     " (Minimal)")
        )
    
    console.print(table)
    
    # Print overall evidence summary
    if "evidence_summary" in report:
        summary = report["evidence_summary"]
        
        console.print("\n[bold]Evidence Summary:[/]")
        console.print(summary.get("summary", "No summary available"))
        
        if summary.get("profiles_with_significant_improvement"):
            console.print(
                f"\n[bold green]Profiles with statistically significant improvement:[/] "
                f"{', '.join(summary['profiles_with_significant_improvement'])}"
            )
    
    console.print(f"\n[green]Full report saved to: {output_path}[/]")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Synapz - Adaptive Learning Experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run an experiment")
    exp_parser.add_argument("--topic", required=True, help="Educational topic")
    exp_parser.add_argument(
        "--profile", 
        required=True, 
        choices=["adhd", "dyslexic", "visual", "control"],
        help="Cognitive profile"
    )
    exp_parser.add_argument(
        "--objective", 
        required=True, 
        help="Learning objective"
    )
    exp_parser.add_argument("--background", help="Background knowledge")
    exp_parser.add_argument(
        "--no-control", 
        action="store_true", 
        help="Skip control experiment"
    )
    exp_parser.add_argument(
        "--use-new-db",
        action="store_true",
        help="Use the new Database implementation"
    )
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate evidence report")
    report_parser.add_argument(
        "--output", 
        default="evidence_report.json",
        help="Output file path"
    )
    report_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations from the report"
    )
    report_parser.add_argument(
        "--viz-dir",
        default="./visualizations",
        help="Directory for saving visualizations"
    )
    
    # Budget command
    subparsers.add_parser("budget", help="Show budget status")
    
    # Profile commands (only if available)
    if PROFILES_AVAILABLE:
        profile_parser = subparsers.add_parser("profiles", help="View detailed cognitive profiles")
        profile_parser.add_argument("--id", help="Specific profile ID to view")
        profile_parser.add_argument("--json", action="store_true", help="Output profiles as JSON")
    
    # Concept commands (only if available)
    if CONCEPTS_AVAILABLE:
        concept_parser = subparsers.add_parser("concepts", help="View algebra concepts")
        concept_parser.add_argument("--id", help="Specific concept ID to view")
        concept_parser.add_argument("--difficulty", type=int, help="Show concepts at a specific difficulty level")
        concept_parser.add_argument("--list", action="store_true", help="List all concepts")
        concept_parser.add_argument("--path", action="store_true", help="Show recommended learning path")
        concept_parser.add_argument("--levels", action="store_true", help="Show difficulty level overview")
        concept_parser.add_argument("--json", action="store_true", help="Output concepts as JSON")
    
    # Global arguments
    parser.add_argument(
        "--db-path", 
        default=str(DATA_DIR / "synapz.db"),
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--budget", 
        type=float, 
        default=20.0,
        help="Budget limit in dollars"
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for command
    if not args.command:
        parser.print_help()
        return
    
    # Handle profile command separately
    if args.command == "profiles" and PROFILES_AVAILABLE:
        if args.id:
            if args.json:
                # This functionality is handled by the imported module
                from synapz.models.learner_profiles import get_profile_for_adaptation
                try:
                    profile = get_profile_for_adaptation(args.id)
                    print(json.dumps(profile, indent=2))
                except ValueError as e:
                    print(json.dumps({"error": str(e)}, indent=2))
            else:
                display_profile_details(args.id)
        else:
            if args.json:
                profiles = get_all_profiles()
                print(json.dumps(profiles, indent=2))
            else:
                display_profile_list()
        return
    
    # Handle concept command separately
    if args.command == "concepts" and CONCEPTS_AVAILABLE:
        if args.id:
            if args.json:
                from synapz.models.concepts.algebra_concepts import load_concept
                try:
                    concept = load_concept(args.id)
                    print(json.dumps(concept, indent=2))
                except ValueError as e:
                    print(json.dumps({"error": str(e)}, indent=2))
            else:
                display_concept_details(args.id)
        elif args.difficulty is not None:
            concepts = get_concepts_by_difficulty(args.difficulty)
            if args.json:
                print(json.dumps(concepts, indent=2))
            else:
                display_concept_list(concepts)
        elif args.list:
            if args.json:
                concepts = get_all_concepts()
                print(json.dumps(concepts, indent=2))
            else:
                display_concept_list()
        elif args.path:
            display_learning_path()
        elif args.levels:
            display_difficulty_levels()
        else:
            # Default to list all concepts
            display_concept_list()
        return
    
    # Setup environment for other commands
    components = setup_environment(
        db_path=args.db_path,
        budget_limit=args.budget,
        api_key=args.api_key,
        use_new_db=getattr(args, 'use_new_db', False)
    )
    
    # Run appropriate command
    if args.command == "experiment":
        adapted_id, control_id = run_experiment(
            components=components,
            topic=args.topic,
            profile_name=args.profile,
            learning_objective=args.objective,
            background_knowledge=args.background,
            run_control=not args.no_control
        )
        
        # Display comparison if control was run
        if control_id:
            display_experiment_comparison(components, adapted_id, control_id)
            
        # Show budget status
        display_budget_status(components["budget_tracker"])
        
    elif args.command == "report":
        generate_evidence_report(components, Path(args.output))
        
        # Generate visualizations if requested
        if args.visualize:
            from synapz.data.visualization import create_visualization_from_report
            create_visualization_from_report(
                Path(args.output),
                Path(args.viz_dir)
            )
            console.print(f"[green]Visualizations generated in {args.viz_dir}[/]")
        
    elif args.command == "budget":
        display_budget_status(components["budget_tracker"])
        
if __name__ == "__main__":
    main() 