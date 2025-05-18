"""Command-line interface for Synapz teaching sessions."""

import argparse
import os
import json
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from synapz import DATA_DIR
from synapz.core.budget import BudgetTracker
from synapz.core.llm_client import LLMClient
from synapz.core.models import Database
from synapz.core.teacher import TeacherAgent
from synapz.core.simulator import StudentSimulator

# OpenAI API key
API_KEY = os.environ.get("OPENAI_API_KEY")

def load_profiles() -> List[Dict[str, Any]]:
    """Load all available learner profiles."""
    profiles_dir = DATA_DIR / "profiles"
    profiles = []
    
    if not profiles_dir.exists():
        return profiles
    
    for filename in os.listdir(profiles_dir):
        if filename.endswith(".json"):
            with open(profiles_dir / filename, "r") as f:
                profiles.append(json.load(f))
    return profiles

def load_concepts() -> List[Dict[str, Any]]:
    """Load all available concepts."""
    concepts_dir = DATA_DIR / "concepts"
    concepts = []
    
    if not concepts_dir.exists():
        return concepts
    
    for filename in os.listdir(concepts_dir):
        if filename.endswith(".json"):
            with open(concepts_dir / filename, "r") as f:
                concepts.append(json.load(f))
    return concepts

def display_budget_info(budget_tracker: BudgetTracker, console: Console) -> None:
    """Display current budget usage."""
    current_spend = budget_tracker.get_current_spend()
    budget_pct = (current_spend / budget_tracker.max_budget) * 100
    
    table = Table(title="Budget Status", box=box.ROUNDED)
    table.add_column("Current Spend")
    table.add_column("Budget")
    table.add_column("% Used")
    
    table.add_row(
        f"${current_spend:.4f}", 
        f"${budget_tracker.max_budget:.2f}",
        f"{budget_pct:.1f}%"
    )
    
    console.print(table)

def interactive_session(
    teacher: TeacherAgent, 
    learner_id: str, 
    concept_id: str, 
    is_adaptive: bool,
    console: Console,
    budget_tracker: BudgetTracker,
    max_turns: int = 5
) -> None:
    """Run an interactive teaching session with the user providing feedback."""
    # Create session
    session_id = teacher.create_session(learner_id, concept_id, is_adaptive)
    
    # Get profiles and concepts for display
    profiles = load_profiles()
    learner_name = next((p.get("name", p["id"]) for p in profiles if p["id"] == learner_id), learner_id)
    
    concepts = load_concepts()
    concept_title = next((c.get("title", c["id"]) for c in concepts if c["id"] == concept_id), concept_id)
    
    console.print(Panel(
        f"[bold]New Teaching Session[/bold]\n"
        f"Learner: {learner_name}\n"
        f"Concept: {concept_title}\n"
        f"Mode: {'Adaptive' if is_adaptive else 'Control (static)'}\n"
        f"Session ID: {session_id}"
    ))
    
    # Teaching loop
    turn = 1
    while turn <= max_turns:
        console.print(f"\n[bold cyan]Turn {turn}[/bold cyan]")
        
        # Generate explanation
        try:
            result = teacher.generate_explanation(session_id)
            
            # Display explanation
            console.print(Panel(
                Markdown(result["explanation"]),
                title=f"Explanation (Turn {turn})",
                subtitle=f"Tags: {', '.join(result['pedagogy_tags'])}"
            ))
            
            # Display follow-up
            console.print(f"\n{result['follow_up']}")
            
            # Get clarity rating from user
            while True:
                try:
                    clarity = int(console.input("[bold yellow]Enter clarity rating (1-5): [/bold yellow]"))
                    if 1 <= clarity <= 5:
                        break
                    console.print("[red]Please enter a number between 1 and 5.[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number.[/red]")
            
            # Record feedback
            teacher.record_feedback(result["interaction_id"], clarity)
            
            # Show budget after each turn
            display_budget_info(budget_tracker, console)
            
            turn += 1
            
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    console.print("\n[bold green]Session completed successfully![/bold green]")
    console.print(f"Data saved for session {session_id}")

def automated_session(
    teacher: TeacherAgent, 
    simulator: StudentSimulator,
    learner_id: str, 
    concept_id: str, 
    is_adaptive: bool,
    console: Console,
    budget_tracker: BudgetTracker,
    use_llm_simulator: bool = False,
    max_turns: int = 5
) -> None:
    """Run an automated teaching session with simulated student feedback."""
    # Create session
    session_id = teacher.create_session(learner_id, concept_id, is_adaptive)
    
    # Get profiles and concepts for display
    profiles = load_profiles()
    learner_profile = next((p for p in profiles if p["id"] == learner_id), {"id": learner_id, "name": learner_id, "cognitive_style": "unknown"})
    learner_name = learner_profile.get("name", learner_id)
    
    concepts = load_concepts()
    concept_title = next((c.get("title", c["id"]) for c in concepts if c["id"] == concept_id), concept_id)
    
    console.print(Panel(
        f"[bold]New Automated Teaching Session[/bold]\n"
        f"Learner: {learner_name} (Style: {learner_profile.get('cognitive_style', 'N/A')})\n"
        f"Concept: {concept_title}\n"
        f"Mode: {'[bold green]Adaptive[/bold green]' if is_adaptive else '[bold yellow]Control (static)[/bold yellow]'}\n"
        f"Simulator: {'[bold magenta]LLM-based[/bold magenta]' if use_llm_simulator else '[bold blue]Heuristic-only[/bold blue]'}\n"
        f"Session ID: {session_id}"
    ))
    
    # Teaching loop
    turn = 1
    session_feedback_metrics = [] # To store dicts of feedback for better summary
    
    while turn <= max_turns:
        console.print(f"\n[bold cyan]🔄 Turn {turn}/{max_turns} 🔄[/bold cyan]")
        
        # Generate explanation
        try:
            result = teacher.generate_explanation(session_id)
            
            # Display explanation
            console.print(Panel(
                Markdown(result["explanation"]),
                title=f"👨‍🏫 Teacher's Explanation (Turn {turn})",
                subtitle=f"💡 Tags: {', '.join(result.get('pedagogy_tags', ['N/A']))}",
                border_style="blue",
                padding=(1,2)
            ))
            
            # Simulate student feedback
            sim_result = simulator.generate_feedback(
                explanation=result["explanation"],
                learner_profile=learner_profile, # Pass the full profile
                pedagogy_tags=result.get("pedagogy_tags", []),
                use_llm=use_llm_simulator
            )
            
            session_feedback_metrics.append(sim_result) # Store the whole dict
            
            # Display simulated feedback using a more structured layout
            scores_table = Table(box=None, show_header=False, padding=(0,1,0,0), show_lines=False)
            scores_table.add_column("Metric", style="dim")
            scores_table.add_column("Value")
            
            scores_table.add_row("🧠 LLM Clarity Rating:", f"{sim_result.get('clarity_rating', 'N/A')}/5")
            scores_table.add_row("🎯 Engagement Rating:", f"{sim_result.get('engagement_rating', 'N/A')}/5")
            if use_llm_simulator: # Only show blended score if LLM was used
                scores_table.add_row("⚖️ Final (Blended) Clarity:", f"{sim_result.get('final_clarity_score', 'N/A'):.2f}/5" if isinstance(sim_result.get('final_clarity_score'), (int, float)) else "N/A")

            heuristic_scores_table = Table(title="[dim]🔧 Heuristic Assessment Details[/dim]", box=box.MINIMAL, show_header=True, padding=(0,1,0,0))
            heuristic_scores_table.add_column("Heuristic Metric", style="italic")
            heuristic_scores_table.add_column("Score", style="bold")

            heuristic_scores_table.add_row("Profile Match Score", f"{sim_result.get('heuristic_profile_match_score', 0.0):.2f}")
            heuristic_scores_table.add_row("Readability (FKG)", f"{sim_result.get('heuristic_readability_score', 0.0):.1f}")
            heuristic_scores_table.add_row("Text Complexity", f"{sim_result.get('heuristic_complexity_score', 0.0):.2f}")
            heuristic_scores_table.add_row("Term Density", f"{sim_result.get('heuristic_term_density_score', 0.0):.2f}")
            heuristic_scores_table.add_row("Sentiment Score", f"{sim_result.get('heuristic_sentiment_score', 0.0):.2f}")
            heuristic_scores_table.add_row("Base Clarity (from heuristics)", f"{sim_result.get('base_clarity_heuristic', 0.0):.2f}/5")


            qualitative_feedback_md = f"""
[bold underline]📝 Detailed Feedback:[/bold underline]
{sim_result.get('detailed_feedback', 'N/A')}

[bold underline]👍 Helpful Elements:[/bold underline]
"""
            helpful_elements = sim_result.get("helpful_elements", [])
            if helpful_elements:
                for element in helpful_elements:
                    qualitative_feedback_md += f"- {element}\n"
            else:
                qualitative_feedback_md += "_None reported._\n"
            
            qualitative_feedback_md += """
[bold underline]❓ Confusion Points:[/bold underline]
"""
            confusion_points = sim_result.get("confusion_points", [])
            if confusion_points:
                for point in confusion_points:
                    qualitative_feedback_md += f"- {point}\n"
            else:
                qualitative_feedback_md += "_None reported._\n"

            qualitative_feedback_md += """
[bold underline]💡 Improvement Suggestions:[/bold underline]
"""
            improvement_suggestions = sim_result.get("improvement_suggestions", [])
            if improvement_suggestions:
                for suggestion in improvement_suggestions:
                    qualitative_feedback_md += f"- {suggestion}\n"
            else:
                qualitative_feedback_md += "_None reported._\n"

            console.print(Panel(
                scores_table,
                title="🎓 Simulated Student Feedback Scores",
                border_style="yellow",
                padding=(1,2)
            ))
            if use_llm_simulator or sim_result.get('base_clarity_heuristic') is not None: # Show heuristics if LLM or heuristic only
                 console.print(Panel(
                    heuristic_scores_table,
                    title="🛠️ Student Heuristic Analysis",
                    border_style="dim yellow",
                    padding=(1,2)
                ))

            if use_llm_simulator: # Only show qualitative feedback if LLM was used
                console.print(Panel(
                    Markdown(qualitative_feedback_md),
                    title="🗣️ Student Qualitative Feedback (LLM)",
                    border_style="green",
                    padding=(1,2)
                ))
            
            # Record feedback using the final_clarity_score for consistency
            # The DB schema expects an int, so we round it.
            # If not present (e.g. heuristic only), use the direct clarity_rating.
            clarity_to_record = sim_result.get('final_clarity_score')
            if clarity_to_record is None: # Fallback for heuristic mode or if missing
                clarity_to_record = sim_result.get('clarity_rating', 0)
            teacher.record_feedback(result["interaction_id"], int(round(clarity_to_record)))
            
            # Show budget after each turn
            display_budget_info(budget_tracker, console)
            
            turn += 1
            
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            # sys.exit(1) # Avoid exiting for one turn error, log and continue if possible
            console.print("[yellow]Attempting to continue to next turn or end session.[/yellow]")
            # Optionally break or implement more robust error handling for session continuation
            break # For now, break the loop on error to avoid cascading issues
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred:[/bold red] {str(e)}")
            console.print_exception(show_locals=True) # Rich traceback
            break

    if not session_feedback_metrics: # If loop didn't run or broke early
        console.print("[yellow]Session ended prematurely, no summary metrics to display.[/yellow]")
        return

    # Calculate statistics using final_clarity_score if available, else clarity_rating
    clarity_values_for_summary = []
    for sfm in session_feedback_metrics:
        val = sfm.get('final_clarity_score')
        if val is None:
            val = sfm.get('clarity_rating', 0) # Fallback to LLM direct or heuristic
        clarity_values_for_summary.append(val)
        
    avg_clarity = sum(clarity_values_for_summary) / len(clarity_values_for_summary) if clarity_values_for_summary else 0
    initial_clarity = clarity_values_for_summary[0] if clarity_values_for_summary else 0
    final_clarity = clarity_values_for_summary[-1] if clarity_values_for_summary else 0
    clarity_trend = final_clarity - initial_clarity if len(clarity_values_for_summary) > 1 else 0
    
    # Display summary
    summary_table = Table(title="Session Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")
    
    summary_table.add_row("Average Final Clarity", f"{avg_clarity:.2f}/5")
    summary_table.add_row("Initial Final Clarity", f"{initial_clarity:.2f}/5") # Keep as float for consistency
    summary_table.add_row("Final Final Clarity", f"{final_clarity:.2f}/5")
    summary_table.add_row("Clarity Trend", f"{clarity_trend:+.2f} ({'improved' if clarity_trend > 0.1 else 'declined' if clarity_trend < -0.1 else 'stable'})")
    summary_table.add_row("Total Turns", str(len(session_feedback_metrics)))
    
    console.print(summary_table)
    console.print("\n[bold green]✅ Automated Session Completed Successfully![/bold green]")
    console.print(f"Data saved for session {session_id}")

def main():
    """Main CLI entrypoint."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Synapz - Adaptive Learning Agent")
    parser.add_argument("--learner", type=str, help="Learner profile ID to use")
    parser.add_argument("--concept", type=str, help="Concept ID to teach")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive mode (default)")
    parser.add_argument("--control", action="store_true", help="Use control (static) mode")
    parser.add_argument("--budget", type=float, default=1.0, help="Max budget in USD")
    parser.add_argument("--compare", action="store_true", 
                       help="Run both adaptive and control for comparison")
    parser.add_argument("--turns", type=int, default=3, help="Maximum number of turns in a session")
    parser.add_argument("--simulate", action="store_true", 
                       help="Use simulated student instead of interactive mode")
    parser.add_argument("--llm-simulator", action="store_true", 
                       help="Use LLM-based simulator (more realistic but uses API budget)")
    parser.add_argument("--teacher-model", type=str, default="gpt-4o",
                        help="Specify the OpenAI model for the TeacherAgent (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--simulator-model", type=str, default="gpt-4o",
                        help="Specify the OpenAI model for the LLM-based StudentSimulator (e.g., gpt-4o, gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Setup rich console
    console = Console()
    
    if not API_KEY:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set.[/bold red]")
        console.print("Please set your OpenAI API key to run the CLI.")
        sys.exit(1)

    # Make sure data directories exist
    os.makedirs(DATA_DIR / "profiles", exist_ok=True)
    os.makedirs(DATA_DIR / "concepts", exist_ok=True)
    
    # Setup components
    db_path = str(DATA_DIR / "cli_sessions.db")
    budget_tracker = BudgetTracker(db_path=db_path, max_budget=args.budget)
    db = Database(db_path=db_path)
    llm_client = LLMClient(budget_tracker=budget_tracker, api_key=API_KEY)
    teacher = TeacherAgent(llm_client=llm_client, db=db, teacher_model_name=args.teacher_model)
    simulator = StudentSimulator(llm_client=llm_client, simulator_model_name=args.simulator_model)
    
    # Create sample data if needed
    profiles = load_profiles()
    concepts = load_concepts()
    
    if not profiles:
        console.print("[yellow]No learner profiles found. Creating sample profiles...[/yellow]")
        create_sample_profiles()
        profiles = load_profiles()
    
    if not concepts:
        console.print("[yellow]No concepts found. Creating sample concepts...[/yellow]")
        create_sample_concepts()
        concepts = load_concepts()
    
    # List available options if none specified
    if not args.learner or not args.concept:
        console.print("[bold]Available Learner Profiles:[/bold]")
        for profile in profiles:
            console.print(f"  - {profile['id']}: {profile.get('name', 'Unknown')} ({profile.get('cognitive_style', 'Unknown')})")
        
        console.print("\n[bold]Available Concepts:[/bold]")
        for concept in concepts:
            console.print(f"  - {concept['id']}: {concept.get('title', 'Unknown')} (Difficulty: {concept.get('difficulty', 'Unknown')})")
        
        console.print("\nUse --learner and --concept to specify options.")
        sys.exit(0)
    
    # Verify learner and concept exist
    if args.learner and args.learner not in [p["id"] for p in profiles]:
        console.print(f"[bold red]Error:[/bold red] Learner '{args.learner}' not found.")
        sys.exit(1)
        
    if args.concept and args.concept not in [c["id"] for c in concepts]:
        console.print(f"[bold red]Error:[/bold red] Concept '{args.concept}' not found.")
        sys.exit(1)
    
    # Determine mode
    is_adaptive = not args.control
    
    # Run comparison if requested
    if args.compare:
        console.print("[bold]Running comparison: Adaptive vs. Control[/bold]")
        
        # Choose session type based on simulate flag
        if args.simulate:
            # Run automated comparison
            console.print("\n[bold green]Starting Automated Adaptive Session...[/bold green]")
            automated_session(
                teacher, simulator, args.learner, args.concept, True, 
                console, budget_tracker, args.llm_simulator, args.turns
            )
            
            console.print("\n[bold red]Starting Automated Control Session...[/bold red]")
            automated_session(
                teacher, simulator, args.learner, args.concept, False, 
                console, budget_tracker, args.llm_simulator, args.turns
            )
        else:
            # Run interactive comparison
            console.print("\n[bold green]Starting Adaptive Session...[/bold green]")
            interactive_session(teacher, args.learner, args.concept, True, console, budget_tracker, args.turns)
            
            console.print("\n[bold red]Starting Control Session...[/bold red]")
            interactive_session(teacher, args.learner, args.concept, False, console, budget_tracker, args.turns)
        
        # Show final budget
        display_budget_info(budget_tracker, console)
    else:
        # Run single session
        if args.simulate:
            # Run automated session
            automated_session(
                teacher, simulator, args.learner, args.concept, is_adaptive, 
                console, budget_tracker, args.llm_simulator, args.turns
            )
        else:
            # Run interactive session
            interactive_session(teacher, args.learner, args.concept, is_adaptive, console, budget_tracker, args.turns)

def create_sample_profiles():
    """Create sample learner profiles if none exist."""
    profiles_dir = DATA_DIR / "profiles"
    os.makedirs(profiles_dir, exist_ok=True)
    
    # ADHD profile
    adhd_profile = {
        "id": "adhd_learner",
        "name": "Sam",
        "cognitive_style": "adhd",
        "attention_span": "short",
        "reading_level": "intermediate",
        "learning_preferences": ["visual", "interactive", "chunked"],
        "strengths": ["creativity", "pattern recognition"],
        "challenges": ["sustained attention", "sequential tasks"],
        "strategies": ["frequent breaks", "visual aids", "concrete examples"]
    }
    
    # Dyslexic profile
    dyslexic_profile = {
        "id": "dyslexic_learner",
        "name": "Taylor",
        "cognitive_style": "dyslexic",
        "attention_span": "medium",
        "reading_level": "basic",
        "learning_preferences": ["audio", "examples", "spaced repetition"],
        "strengths": ["verbal reasoning", "big picture thinking"],
        "challenges": ["text processing", "spelling", "working memory"],
        "strategies": ["audio support", "simplified language", "structured organization"]
    }
    
    # Visual learner profile
    visual_profile = {
        "id": "visual_learner",
        "name": "Alex",
        "cognitive_style": "visual",
        "attention_span": "long",
        "reading_level": "advanced",
        "learning_preferences": ["diagrams", "spatial organization", "mindmaps"],
        "strengths": ["spatial reasoning", "memory recall with visuals"],
        "challenges": ["auditory processing without visuals"],
        "strategies": ["diagrams", "color coding", "visual metaphors"]
    }
    
    # Save profiles
    with open(profiles_dir / "adhd_learner.json", "w") as f:
        json.dump(adhd_profile, f, indent=2)
    
    with open(profiles_dir / "dyslexic_learner.json", "w") as f:
        json.dump(dyslexic_profile, f, indent=2)
        
    with open(profiles_dir / "visual_learner.json", "w") as f:
        json.dump(visual_profile, f, indent=2)

def create_sample_concepts():
    """Create sample concepts if none exist."""
    concepts_dir = DATA_DIR / "concepts"
    os.makedirs(concepts_dir, exist_ok=True)
    
    # Binary search concept
    binary_search = {
        "id": "binary_search",
        "title": "Binary Search Algorithm",
        "difficulty": 3,
        "description": "A divide and conquer search algorithm that finds the position of a target value within a sorted array.",
        "keywords": ["algorithm", "search", "divide and conquer", "sorted array"],
        "prerequisites": ["basic array operations", "logarithmic complexity"]
    }
    
    # Variables concept
    variables = {
        "id": "variables",
        "title": "Variables in Programming",
        "difficulty": 1,
        "description": "Fundamental storage locations that hold values in computer programs.",
        "keywords": ["variables", "programming", "data types", "assignment"],
        "prerequisites": ["basic computer literacy"]
    }
    
    # Neural networks concept
    neural_networks = {
        "id": "neural_networks",
        "title": "Introduction to Neural Networks",
        "difficulty": 4,
        "description": "Computational systems inspired by the human brain that can learn from data.",
        "keywords": ["machine learning", "neural networks", "AI", "deep learning"],
        "prerequisites": ["basic algebra", "statistics", "programming basics"]
    }
    
    # Save concepts
    with open(concepts_dir / "binary_search.json", "w") as f:
        json.dump(binary_search, f, indent=2)
    
    with open(concepts_dir / "variables.json", "w") as f:
        json.dump(variables, f, indent=2)
        
    with open(concepts_dir / "neural_networks.json", "w") as f:
        json.dump(neural_networks, f, indent=2)

if __name__ == "__main__":
    main() 