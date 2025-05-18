#!/usr/bin/env python3
"""Simplified batch evaluation for scientific validation of adaptive learning effectiveness."""

import os
import json
import time
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import csv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import box

from synapz.core.budget import BudgetTracker
from synapz.core.llm_client import LLMClient
from synapz.core.teacher import TeacherAgent
from synapz.core.simulator import StudentSimulator
from synapz.data.metrics import calculate_readability_metrics, calculate_text_similarity

class DirectEvaluator:
    """Evaluates adaptive vs. control teaching without database dependencies."""
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        budget_tracker: BudgetTracker,
        console: Console
    ):
        """Initialize with API client."""
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker
        self.console = console
        self.simulator = StudentSimulator(llm_client)
        
    def evaluate_concept(
        self, 
        learner_profile: Dict[str, Any],
        concept: Dict[str, Any],
        turns: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate teaching a concept to a learner profile using both adaptive and control approaches.
        
        Args:
            learner_profile: The learner profile data
            concept: The concept data
            turns: Number of teaching turns
            
        Returns:
            Evaluation results dictionary
        """
        # Prepare prompt templates
        # Adjust path to be relative to project root or use synapz.PROMPTS_DIR
        from synapz import PROMPTS_DIR
        # prompts_dir = Path(__file__).parent / "synapz" / "prompts" # Old path
        prompts_dir = PROMPTS_DIR

        if not prompts_dir.exists():
            self.console.print(f"[red]Error: Prompts directory not found at {prompts_dir}[/red]")
            raise FileNotFoundError(f"Prompts directory not found at {prompts_dir}")

        adaptive_prompt_path = prompts_dir / "adaptive_system.txt"
        control_prompt_path = prompts_dir / "control_system.txt"

        if not adaptive_prompt_path.exists():
            self.console.print(f"[red]Error: Adaptive prompt not found at {adaptive_prompt_path}[/red]")
            raise FileNotFoundError(f"Adaptive prompt not found at {adaptive_prompt_path}")
        if not control_prompt_path.exists():
            self.console.print(f"[red]Error: Control prompt not found at {control_prompt_path}[/red]")
            raise FileNotFoundError(f"Control prompt not found at {control_prompt_path}")

        with open(adaptive_prompt_path, "r") as f:
            adaptive_template = f.read()
            
        with open(control_prompt_path, "r") as f:
            control_template = f.read()
        
        # Track results
        adaptive_explanations = []
        control_explanations = []
        adaptive_clarity = []
        control_clarity = []
        adaptive_tags = []
        control_tags = []
        
        # Run experiment turns
        for turn in range(1, turns + 1):
            self.console.print(f"Running turn {turn}/{turns}...")
            
            # Generate adaptive explanation
            adaptive_result = self._generate_explanation(
                adaptive_template, 
                learner_profile, 
                concept, 
                turn, 
                adaptive_clarity[-1] if adaptive_clarity else None,
                adaptive_explanations
            )
            
            adaptive_explanations.append(adaptive_result["explanation"])
            adaptive_tags.append(adaptive_result["pedagogy_tags"])
            
            # Simulate student feedback for adaptive
            adaptive_feedback = self.simulator.generate_feedback(
                explanation=adaptive_result["explanation"],
                learner_profile=learner_profile,
                pedagogy_tags=adaptive_result["pedagogy_tags"],
                use_llm=False
            )
            
            adaptive_clarity.append(adaptive_feedback["clarity_rating"])
            
            # Generate control explanation
            control_result = self._generate_explanation(
                control_template, 
                None,  # No learner profile for control
                concept, 
                turn, 
                control_clarity[-1] if control_clarity else None,
                control_explanations
            )
            
            control_explanations.append(control_result["explanation"])
            control_tags.append(control_result["pedagogy_tags"])
            
            # Simulate student feedback for control
            control_feedback = self.simulator.generate_feedback(
                explanation=control_result["explanation"],
                learner_profile=learner_profile,  # Same learner for fair comparison
                pedagogy_tags=control_result["pedagogy_tags"],
                use_llm=False
            )
            
            control_clarity.append(control_feedback["clarity_rating"])
            
            # Display progress
            self.console.print(f"Adaptive clarity: {adaptive_clarity[-1]}/5, Control clarity: {control_clarity[-1]}/5")
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            adaptive_explanations, control_explanations,
            adaptive_clarity, control_clarity,
            adaptive_tags, control_tags
        )
        
        return {
            "learner_id": learner_profile["id"],
            "concept_id": concept["id"],
            "turns": turns,
            "adaptive_clarity": adaptive_clarity,
            "control_clarity": control_clarity,
            "adaptive_tags": adaptive_tags,
            "control_tags": control_tags,
            "metrics": metrics
        }
    
    def _generate_explanation(
        self, 
        template: str, 
        learner_profile: Optional[Dict[str, Any]],
        concept: Dict[str, Any],
        turn_number: int,
        previous_clarity: Optional[int],
        previous_explanations: List[str]
    ) -> Dict[str, Any]:
        """Generate a teaching explanation using the specified system prompt template."""
        # Format interaction history for context
        interaction_history = ""
        # Check if previous_clarity is a list or a single value, handle accordingly
        current_previous_clarity_rating = None
        if previous_clarity is not None:
            if isinstance(previous_clarity, list) and previous_clarity:
                current_previous_clarity_rating = previous_clarity[-1]
            elif isinstance(previous_clarity, (int, float)):
                current_previous_clarity_rating = previous_clarity
            # If it's something else, it remains None

        if previous_explanations:
            history_log = []
            for i, explanation_text in enumerate(previous_explanations[-3:]):
                turn_log = f"Turn {len(previous_explanations) - len(previous_explanations[-3:]) + i + 1}:\n"
                turn_log += f"Explanation: {explanation_text[:100]}...\n"
                # Assuming previous_clarity is a list of all past clarities for the session
                # This part might need adjustment based on how previous_clarity is actually populated
                # For _generate_explanation, previous_clarity is just the *last* one.
                # The interaction_history for the prompt usually takes a list of dicts.
                # This DirectEvaluator builds it simpler for this example script.
                # For now, we will pass only the most recent clarity to the prompt.
                # The history string is mostly for the LLM to see past explanations.
                history_log.append(turn_log)
            interaction_history = "\n".join(history_log)

        # Fill in template placeholders
        system_prompt = template
        if learner_profile:
            system_prompt = system_prompt.replace("{learner_profile_json}", json.dumps(learner_profile, indent=2))
        system_prompt = system_prompt.replace("{concept_json}", json.dumps(concept, indent=2))
        system_prompt = system_prompt.replace("{turn_number}", str(turn_number))
        system_prompt = system_prompt.replace("{previous_clarity}", str(current_previous_clarity_rating) if current_previous_clarity_rating is not None else "None")
        system_prompt = system_prompt.replace("{interaction_history}", interaction_history) # This history is just text, not the JSON list of dicts the main TeacherAgent uses.
        
        # Simple user prompt - the system prompt has all the context
        user_prompt = "Please teach this concept based on the learner's profile and history."
        
        # Get completion with structured output
        result = self.llm_client.get_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Parse response
        response = result["content"]
        
        # Validate response structure
        required_fields = ["teaching_strategy", "explanation", "pedagogy_tags", "follow_up"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing field in response: {field}. Full response: {response}")
        
        return response
    
    def _calculate_metrics(
        self,
        adaptive_explanations: List[str],
        control_explanations: List[str],
        adaptive_clarity: List[int],
        control_clarity: List[int],
        adaptive_tags: List[List[str]],
        control_tags: List[List[str]]
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics based on explanations and clarity scores."""
        # Get final explanations
        adaptive_final = adaptive_explanations[-1]
        control_final = control_explanations[-1]
        
        # Calculate text similarity
        text_similarity_metrics = calculate_text_similarity(adaptive_final, control_final)
        text_difference = 1.0 - text_similarity_metrics["levenshtein_similarity"]
        
        # Normalize tags into flat sets (in case they're not the expected type)
        adaptive_tag_set = set()
        for tags_list_item in adaptive_tags:
            if isinstance(tags_list_item, list):
                adaptive_tag_set.update(map(str, tags_list_item)) # Ensure tags are strings
            elif isinstance(tags_list_item, dict):
                adaptive_tag_set.update(map(str, tags_list_item.values()))
            else:
                adaptive_tag_set.add(str(tags_list_item))
        
        control_tag_set = set()
        for tags_list_item in control_tags:
            if isinstance(tags_list_item, list):
                control_tag_set.update(map(str, tags_list_item)) # Ensure tags are strings
            elif isinstance(tags_list_item, dict):
                control_tag_set.update(map(str, tags_list_item.values()))
            else:
                control_tag_set.add(str(tags_list_item))
        
        # Calculate tag difference (Jaccard distance)
        tag_intersection = len(adaptive_tag_set.intersection(control_tag_set))
        tag_union = len(adaptive_tag_set.union(control_tag_set))
        tag_difference = 1.0 - (tag_intersection / tag_union if tag_union > 0 else 0.0)
        
        # Calculate readability metrics
        adaptive_readability = calculate_readability_metrics(adaptive_final)
        control_readability = calculate_readability_metrics(control_final)
        
        # Use Flesch Reading Ease as primary readability metric (higher is better)
        readability_difference = adaptive_readability["flesch_reading_ease"] - control_readability["flesch_reading_ease"]
        
        # Calculate clarity improvements
        adaptive_start = adaptive_clarity[0] if adaptive_clarity else 0
        adaptive_end = adaptive_clarity[-1] if adaptive_clarity else 0
        adaptive_improvement = adaptive_end - adaptive_start
        
        control_start = control_clarity[0] if control_clarity else 0
        control_end = control_clarity[-1] if control_clarity else 0
        control_improvement = control_end - control_start
        
        # Determine advantage
        improvement_difference = adaptive_improvement - control_improvement
        advantage = "adaptive" if improvement_difference > 0 else "control" if improvement_difference < 0 else "tie"
        
        return {
            "text_difference": text_difference,
            "tag_difference": tag_difference,
            "readability_difference": readability_difference,
            "adaptive_improvements": {
                "starting_clarity": adaptive_start,
                "final_clarity": adaptive_end,
                "absolute_improvement": adaptive_improvement
            },
            "control_improvements": {
                "starting_clarity": control_start,
                "final_clarity": control_end,
                "absolute_improvement": control_improvement
            },
            "improvement_difference": improvement_difference,
            "advantage": advantage,
            "adaptive_readability": {
                "flesch_reading_ease": adaptive_readability.get("flesch_reading_ease", 0.0),
                "flesch_kincaid_grade": adaptive_readability.get("flesch_kincaid_grade", 0.0)
            },
            "control_readability": {
                "flesch_reading_ease": control_readability.get("flesch_reading_ease", 0.0),
                "flesch_kincaid_grade": control_readability.get("flesch_kincaid_grade", 0.0)
            }
        }

def run_batch_evaluation(
    evaluator: DirectEvaluator,
    console: Console,
    experiment_size: int = 3,
    turns_per_session: int = 3
) -> Dict[str, Any]:
    """Run a batch of evaluations across different concepts and learner profiles."""
    # Load profiles and concepts
    # Adjust path to be relative to project root or use synapz.DATA_DIR
    from synapz import DATA_DIR
    profiles_dir = DATA_DIR / "profiles"
    concepts_dir = DATA_DIR / "concepts"

    if not profiles_dir.exists() or not concepts_dir.exists():
        console.print(f"[red]Error: Profiles ({profiles_dir}) or Concepts ({concepts_dir}) directory not found.[/red]")
        # Create sample data if directories are missing, as per cli.py logic
        # This requires importing functions from cli.py or replicating logic, which might be too complex here.
        # For an example script, it's better to state the dependency.
        console.print(f"[yellow]Please ensure sample profiles and concepts exist. You can generate them using synapz/cli.py.[/yellow]")
        return { "experiments": [], "summary": {}, "error": "Data directories not found."}
    
    profiles = []
    for filename in os.listdir(profiles_dir):
        if filename.endswith(".json"):
            with open(profiles_dir / filename, "r") as f:
                profile = json.load(f)
                profiles.append(profile)
    
    concepts = []
    for filename in os.listdir(concepts_dir):
        if filename.endswith(".json"):
            with open(concepts_dir / filename, "r") as f:
                concept = json.load(f)
                concepts.append(concept)
    
    if not profiles or not concepts:
        console.print(f"[red]Error: No profiles or concepts found in {profiles_dir} / {concepts_dir}.[/red]")
        console.print(f"[yellow]Please ensure sample profiles and concepts exist. You can generate them using synapz/cli.py.[/yellow]")
        return { "experiments": [], "summary": {}, "error": "No profiles or concepts found."}

    # Generate combinations
    combinations = [(profile, concept) for profile in profiles for concept in concepts]
    
    # Select combinations to test
    if len(combinations) > experiment_size:
        combinations_to_test = random.sample(combinations, experiment_size)
    else:
        combinations_to_test = combinations
    
    if not combinations_to_test:
        console.print(f"[yellow]No profile/concept combinations to test.[/yellow]")
        return { "experiments": [], "summary": {}, "error": "No combinations to test."}

    console.print(f"[bold]Running {len(combinations_to_test)} experiments...[/bold]")
    
    # Track results
    results = {
        "experiments": [],
        "summary": {
            "avg_text_difference": 0.0,
            "avg_readability_difference": 0.0,
            "avg_tag_difference": 0.0,
            "avg_adaptive_improvement": 0.0,
            "avg_control_improvement": 0.0,
            "avg_improvement_difference": 0.0,
            "adaptive_win_rate": 0.0
        }
    }
    
    successful_experiments = 0
    adaptive_wins = 0
    
    # Setup metrics
    total_text_diff = 0.0
    total_readability_diff = 0.0
    total_tag_diff = 0.0
    total_adaptive_improvement = 0.0
    total_control_improvement = 0.0
    total_improvement_diff = 0.0
    
    # Run experiments
    with Progress() as progress_bar: # Renamed to avoid conflict with Progress class from rich.progress
        task = progress_bar.add_task("[cyan]Running experiments...", total=len(combinations_to_test))
        
        for profile, concept in combinations_to_test:
            progress_bar.update(task, description=f"Testing {profile.get('id', 'unknown_profile')} + {concept.get('id','unknown_concept')}")
            
            try:
                # Run evaluation
                result = evaluator.evaluate_concept(profile, concept, turns_per_session)
                
                # Save result
                results["experiments"].append(result)
                
                # Update counters
                successful_experiments += 1
                if result.get("metrics", {}).get("advantage") == "adaptive":
                    adaptive_wins += 1
                
                # Update totals
                total_text_diff += result.get("metrics", {}).get("text_difference", 0.0)
                total_readability_diff += result.get("metrics", {}).get("readability_difference", 0.0)
                total_tag_diff += result.get("metrics", {}).get("tag_difference", 0.0)
                total_adaptive_improvement += result.get("metrics", {}).get("adaptive_improvements", {}).get("absolute_improvement", 0.0)
                total_control_improvement += result.get("metrics", {}).get("control_improvements", {}).get("absolute_improvement", 0.0)
                total_improvement_diff += result.get("metrics", {}).get("improvement_difference", 0.0)
                
            except Exception as e:
                console.print(f"[red]Error evaluating {profile.get('id', 'unknown_profile')} + {concept.get('id','unknown_concept')}: {str(e)}[/red]")
            
            progress_bar.update(task, advance=1)
    
    # Calculate summary
    if successful_experiments > 0:
        results["summary"]["avg_text_difference"] = total_text_diff / successful_experiments
        results["summary"]["avg_readability_difference"] = total_readability_diff / successful_experiments
        results["summary"]["avg_tag_difference"] = total_tag_diff / successful_experiments
        results["summary"]["avg_adaptive_improvement"] = total_adaptive_improvement / successful_experiments
        results["summary"]["avg_control_improvement"] = total_control_improvement / successful_experiments
        results["summary"]["avg_improvement_difference"] = total_improvement_diff / successful_experiments
        results["summary"]["adaptive_win_rate"] = adaptive_wins / successful_experiments
    
    # Display summary table
    summary_table = Table(title="Batch Evaluation Results", box=box.ROUNDED) # Renamed to avoid conflict with Table class from rich.table
    summary_table.add_column("Metric")
    summary_table.add_column("Value")
    
    if successful_experiments > 0:
        summary_table.add_row("Experiments run", str(successful_experiments))
        summary_table.add_row("Avg text difference", f"{results['summary']['avg_text_difference']:.2%}")
        summary_table.add_row("Avg readability improvement", f"{results['summary']['avg_readability_difference']:.2f} points")
        summary_table.add_row("Avg tag difference", f"{results['summary']['avg_tag_difference']:.2%}")
        summary_table.add_row("Avg adaptive clarity improvement", f"{results['summary']['avg_adaptive_improvement']:.2f} points")
        summary_table.add_row("Avg control clarity improvement", f"{results['summary']['avg_control_improvement']:.2f} points")
        summary_table.add_row("Avg clarity advantage", f"{results['summary']['avg_improvement_difference']:.2f} points")
        summary_table.add_row("Adaptive win rate", f"{results['summary']['adaptive_win_rate']:.2%}")
    else:
        summary_table.add_row("Error", "No successful experiments")
    
    console.print(summary_table)
    
    # Save results to CSV
    results_dir = Path("results") # Ensure results dir is at project root
    os.makedirs(results_dir, exist_ok=True)
    save_results_to_csv(results, results_dir / "simple_batch_results.csv")
    
    return results

def save_results_to_csv(results: Dict[str, Any], filename: Path) -> None: # Changed filename to Path for consistency
    """Save experiment results to CSV file."""
    # Ensure directory exists (already done in run_batch_evaluation)
    # os.makedirs(os.path.dirname(filename), exist_ok=True) # Not strictly needed if parent func does it
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Learner", "Concept", 
            "Text Difference", "Readability Difference", "Tag Difference",
            "Adaptive Improvement", "Control Improvement", "Advantage"
        ])
        
        # Write data rows
        for exp in results.get("experiments", []):
            metrics = exp.get("metrics", {})
            adaptive_improvements = metrics.get("adaptive_improvements", {})
            control_improvements = metrics.get("control_improvements", {})
            writer.writerow([
                exp.get("learner_id", "N/A"),
                exp.get("concept_id", "N/A"),
                f"{metrics.get('text_difference', 0.0):.4f}",
                f"{metrics.get('readability_difference', 0.0):.4f}",
                f"{metrics.get('tag_difference', 0.0):.4f}",
                f"{adaptive_improvements.get('absolute_improvement', 0.0):.1f}",
                f"{control_improvements.get('absolute_improvement', 0.0):.1f}",
                metrics.get('advantage', "N/A")
            ])
    
    # Also save summary
    summary_filename = filename.with_name(filename.stem + "_summary.csv")
    with open(summary_filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header and data
        for key, value in results.get("summary", {}).items():
            writer.writerow([key, f"{value:.4f}"])

def main():
    """Main entry point for simplified batch evaluation."""
    parser = argparse.ArgumentParser(description="Synapz Simplified Batch Evaluation - Example Script") # Updated description
    parser.add_argument("--size", type=int, default=2, help="Number of experiments to run (e.g., 2 for quick test)") # Reduced default
    parser.add_argument("--turns", type=int, default=2, help="Turns per teaching session (e.g., 2 for quick test)") # Reduced default
    parser.add_argument("--budget", type=float, default=1.0, help="Max budget in USD (e.g., $1.00)") # Reduced default
    parser.add_argument("--api-key", type=str, help="OpenAI API key (defaults to env var OPENAI_API_KEY)")
    parser.add_argument("--data-root", type=str, default=".", help="Path to project root if script is run from elsewhere.")
    
    args = parser.parse_args()

    # Determine project root for data loading if needed, assuming script is in synapz/examples
    # This allows DATA_DIR and PROMPTS_DIR from synapz package to work correctly.
    # If running this script directly, ensure PYTHONPATH includes the parent of the 'synapz' package directory.
    # For simplicity, this example will assume it can import from `synapz` directly.

    # Create required directories relative to where script is run or data-root
    # For an example script, it's often better to rely on existing structure
    # or output to a clearly marked example_results directory.
    project_root_path = Path(args.data_root).resolve()
    data_dir_for_budget = project_root_path / "data"
    results_dir_for_output = project_root_path / "results" / "simple_eval_example"

    os.makedirs(data_dir_for_budget, exist_ok=True)
    os.makedirs(results_dir_for_output, exist_ok=True)
    
    # Setup components
    console = Console()
    # Use a distinct DB path for this example to avoid conflicts
    budget_db_path = data_dir_for_budget / "simple_eval_budget.db"
    budget_tracker = BudgetTracker(str(budget_db_path), max_budget=args.budget)
    
    # Check for API Key
    api_key_to_use = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        console.print("[red]Error: OpenAI API key not found.[/red]")
        console.print("Please provide it via --api-key argument or set OPENAI_API_KEY environment variable.")
        return

    llm_client = LLMClient(budget_tracker=budget_tracker, api_key=api_key_to_use)
    evaluator = DirectEvaluator(llm_client, budget_tracker, console)
    
    # Display budget before starting
    current_spend = budget_tracker.get_current_spend()
    console.print(f"[bold]Starting budget for simple_eval_example:[/bold] ${current_spend:.4f} / ${args.budget:.2f}")
    console.print(f"Budget DB: {budget_db_path}")
    console.print(f"Results will be saved to: {results_dir_for_output}")
    
    # Run batch evaluation
    start_time = time.time()
    results = run_batch_evaluation(evaluator, console, args.size, args.turns)
    end_time = time.time()
    
    # Display final budget
    final_spend = budget_tracker.get_current_spend()
    console.print(f"\n[bold]Final budget for simple_eval_example:[/bold] ${final_spend:.4f} / ${args.budget:.2f}")
    console.print(f"[bold]Cost of this batch evaluation:[/bold] ${final_spend - current_spend:.4f}")
    console.print(f"[bold]Time taken:[/bold] {end_time - start_time:.1f} seconds")
    
    # Save results to file
    if "error" not in results:
        with open(results_dir_for_output / "batch_results.json", "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[bold green]Results saved to {results_dir_for_output / 'batch_results.json'} and related CSVs.[/bold green]")
    else:
        console.print(f"[yellow]Batch evaluation did not complete successfully. Error: {results['error']}[/yellow]")

if __name__ == "__main__":
    main() 