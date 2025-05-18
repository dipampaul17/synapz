#!/usr/bin/env python3
"""Batch evaluation system for running experiments."""

import os
import json
import time
import argparse
import random
import csv
from typing import Dict, List, Any, Optional, Set, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, track
from rich import box
from pathlib import Path
import uuid
import logging
import numpy as np
from collections import defaultdict
from rich.markdown import Markdown

from synapz.core.budget import BudgetTracker
from synapz.core.llm_client import LLMClient
from synapz.core.models import Database
from synapz.core.teacher import TeacherAgent
from synapz.core.simulator import StudentSimulator
from synapz.data.metrics import (
    calculate_readability_metrics, 
    calculate_text_similarity, 
    extract_pedagogy_tags, 
    calculate_statistical_significance,
    NumpyEncoder, # Ensure NumpyEncoder is imported
    MetricsCalculator # Import MetricsCalculator
)
from synapz import DATA_DIR

#from synapz.test_harness import run_simulated_test_session # Assuming test_harness.py is not ready or its functionality is integrated

logger = logging.getLogger(__name__)

def get_available_combinations() -> List[Tuple[str, str]]:
    """Get all available learner/concept combinations."""
    from synapz import DATA_DIR # Import DATA_DIR from synapz package
    profiles_dir = DATA_DIR / "profiles"
    concepts_dir = DATA_DIR / "concepts"
    
    learners = [f[:-5] for f in os.listdir(profiles_dir) if f.endswith(".json")]
    concepts = [f[:-5] for f in os.listdir(concepts_dir) if f.endswith(".json")]
    
    return [(l, c) for l in learners for c in concepts]

def _run_one_session(
    session_id: str,
    teacher: TeacherAgent,
    simulator: StudentSimulator,
    learner_profile: Dict[str, Any],
    console: Console, 
    use_llm_simulator: bool,
    turns_per_session: int,
    is_adaptive_session: bool, 
    concept_id: str 
) -> Tuple[List[Dict[str, Any]], float]: 
    """
    Run a single teaching session (either adaptive or control) and log details.
    Returns a list of feedback dicts from the simulator and the total cost of the session.
    """
    session_type_str = "[bold green]Adaptive[/bold green]" if is_adaptive_session else "[bold yellow]Control[/bold yellow]"
    learner_name = learner_profile.get("name", learner_profile.get("id", "Unknown Learner"))
    concept_title = concept_id # In future, could load concept title here
    
    console.print(Panel(
        f"▶️ Starting {session_type_str} session for learner [u]{learner_name}[/u] on concept [u]{concept_title}[/u]. Session ID: {session_id}",
        title="Session Start",
        border_style="bright_blue" if is_adaptive_session else "bright_yellow",
        padding=(0,1)
    ))

    current_session_feedback_log = []
    turn = 1
    
    while turn <= turns_per_session:
        console.print(f"\n[cyan]🔄 Turn {turn}/{turns_per_session} ({session_type_str.lower()}) 🔄[/cyan]")
        try:
            explanation_result = teacher.generate_explanation(session_id)
            
            console.print(Panel(
                Markdown(explanation_result["explanation"]),
                title=f"👨‍🏫 Teacher's Explanation (Turn {turn})",
                subtitle=f"💡 Tags: {', '.join(explanation_result.get('pedagogy_tags', ['N/A']))}",
                border_style="blue",
                padding=(1,1)
            ))

            sim_result = simulator.generate_feedback(
                explanation=explanation_result["explanation"],
                learner_profile=learner_profile,
                pedagogy_tags=explanation_result.get("pedagogy_tags", []),
                use_llm=use_llm_simulator
            )
            current_session_feedback_log.append(sim_result)

            scores_table = Table(box=None, show_header=False, padding=(0,1,0,0), show_lines=False)
            scores_table.add_column("Metric", style="dim")
            scores_table.add_column("Value")
            scores_table.add_row("🧠 LLM Clarity:", f"{sim_result.get('clarity_rating', 'N/A')}/5")
            scores_table.add_row("🎯 Engagement:", f"{sim_result.get('engagement_rating', 'N/A')}/5")
            if use_llm_simulator:
                final_clarity_val = sim_result.get('final_clarity_score')
                scores_table.add_row("⚖️ Final Clarity (Blended):", f"{final_clarity_val:.2f}/5" if isinstance(final_clarity_val, (int, float)) else "N/A")
            
            console.print(Panel(
                scores_table,
                title="🎓 Simulated Student Scores",
                border_style="yellow",
                padding=(1,1)
            ))

            if sim_result.get('base_clarity_heuristic') is not None or not use_llm_simulator:
                heuristic_scores_table = Table(title="[dim]🔧 Heuristic Assessment[/dim]", box=box.MINIMAL, show_header=True, padding=(0,1,0,0))
                heuristic_scores_table.add_column("Heuristic Metric", style="italic")
                heuristic_scores_table.add_column("Score", style="bold")
                heuristic_scores_table.add_row("Profile Match", f"{sim_result.get('heuristic_profile_match_score', 0.0):.2f}")
                heuristic_scores_table.add_row("Readability (FKG)", f"{sim_result.get('heuristic_readability_score', 0.0):.1f}")
                heuristic_scores_table.add_row("Base Clarity (Heuristic)", f"{sim_result.get('base_clarity_heuristic', 0.0):.2f}/5")
                console.print(Panel(heuristic_scores_table, title="🛠️ Student Heuristic Analysis", border_style="dim yellow", padding=(1,1)))
            
            if use_llm_simulator and sim_result.get('detailed_feedback'):
                qualitative_feedback_md = f"""
[bold u]📝 Detailed Feedback:[/bold u] {sim_result.get('detailed_feedback', '_No detailed feedback provided._')}

[bold u]👍 Helpful Elements:[/bold u]
{'- ' + '\n- '.join(sim_result.get('helpful_elements', [])) if sim_result.get('helpful_elements') else '_None reported._'}

[bold u]❓ Confusion Points:[/bold u]
{'- ' + '\n- '.join(sim_result.get('confusion_points', [])) if sim_result.get('confusion_points') else '_None reported._'}

[bold u]💡 Improvement Suggestions:[/bold u]
{'- ' + '\n- '.join(sim_result.get('improvement_suggestions', [])) if sim_result.get('improvement_suggestions') else '_None reported._'}
"""
                console.print(Panel(
                    Markdown(qualitative_feedback_md),
                    title="🗣️ Student Qualitative Feedback (LLM)",
                    border_style="green",
                    padding=(1,1)
                ))

            clarity_to_record = sim_result.get('final_clarity_score')
            if clarity_to_record is None: # Fallback
                clarity_to_record = sim_result.get('clarity_rating') 
            if clarity_to_record is None: # Further fallback if clarity_rating also missing
                 clarity_to_record = sim_result.get('base_clarity_heuristic', 0) # Heuristic base clarity
            
            if clarity_to_record is None: # Absolute fallback
                logger.error(f"Clarity score is None for interaction {explanation_result['interaction_id']} in session {session_id}, even after all fallbacks. Recording as 0.")
                clarity_to_record = 0.0

            teacher.record_feedback(explanation_result["interaction_id"], int(round(float(clarity_to_record))))

        except Exception as e:
            logger.error(f"Error during turn {turn} in session {session_id} ({session_type_str.lower()}): {e}", exc_info=True)
            console.print(f"[bold red]Error in turn {turn} ({session_type_str.lower()}): {e}[/bold red]")
        
        turn += 1
    
    session_cost = 0.0
    try:
        session_history_for_cost = teacher.db.get_session_history(session_id) 
        for interaction_cost_item in session_history_for_cost:
            session_cost += float(interaction_cost_item.get('cost', 0.0) or 0.0)
    except Exception as e:
        logger.error(f"Could not calculate cost for session {session_id}: {e}")

    console.print(Panel(
        f"⏹️ Finished {session_type_str} session for [u]{learner_name}[/u] on [u]{concept_title}[/u]. Session Cost: ${session_cost:.4f}",
        title="Session End",
        border_style="dim bright_blue" if is_adaptive_session else "dim bright_yellow",
        padding=(0,1)
    ))
    return current_session_feedback_log, session_cost

def run_batch_experiment(
    teacher: TeacherAgent,
    simulator: StudentSimulator,
    metrics_calculator: MetricsCalculator,
    console: Console,
    budget_tracker: BudgetTracker, # Moved before args with defaults
    experiment_size: int,
    turns_per_session: int,
    specific_combinations: Optional[List[Tuple[str, str]]] = None,
    use_llm_simulator: bool = True
) -> Dict[str, Any]:
    """Run a batch of experiments, comparing adaptive and control teaching."""
    
    console.print(Panel(
        f"[bold]🔬 Synapz Batch Experiment Run 🔬[/bold]\n"
        f"Experiment Pairs to Run: {experiment_size}\n"
        f"Turns per Session: {turns_per_session}\n"
        f"Student Simulator Mode: {'[bold magenta]LLM-based[/bold magenta]' if use_llm_simulator else '[bold blue]Heuristic-only[/bold blue]'}\n"
        f"Teacher Model: [i]{teacher.teacher_model_name}[/i]\n"
        f"Simulator Model: [i]{simulator.simulator_model_name}[/i]",
        title="Batch Configuration",
        expand=False,
        border_style="bold white"
    ))

    available_combinations = get_available_combinations()
    if not available_combinations:
        console.print("[bold red]Error: No learner profiles or concepts found in data directories. Cannot run experiments.[/bold red]")
        return {"error": "No data for experiments.", "experiments": [], "aggregate_results": {}, "profile_results": {}}

    if specific_combinations:
        valid_specific_combinations = []
        all_learners_ids = {combo[0] for combo in available_combinations}
        all_concept_ids = {combo[1] for combo in available_combinations}
        for l_id, c_id in specific_combinations:
            if l_id in all_learners_ids and c_id in all_concept_ids:
                valid_specific_combinations.append((l_id, c_id))
            else:
                console.print(f"[yellow]Warning: Skipping specified combination ('{l_id}', '{c_id}') as it's not found in available data files.[/yellow]")
        combinations_to_run = valid_specific_combinations
        actual_experiment_size = len(combinations_to_run) 
        if not combinations_to_run:
            console.print("[bold red]No valid specific combinations to run based on available data. Exiting.[/bold red]")
            return {"error": "No valid specific combinations.", "experiments": [], "aggregate_results": {}, "profile_results": {}}
    else:
        actual_experiment_size = min(experiment_size, len(available_combinations))
        if experiment_size > len(available_combinations):
            console.print(f"[yellow]Warning: Requested experiment size ({experiment_size}) is larger than available unique combinations ({len(available_combinations)}). Running with all {len(available_combinations)} available pairs.[/yellow]")
        
        random.shuffle(available_combinations)
        combinations_to_run = available_combinations[:actual_experiment_size]

    if not combinations_to_run:
        console.print("[bold red]Error: No experiment combinations to run after filtering. Exiting.[/bold red]")
        return {"error": "No combinations to run.", "experiments": [], "aggregate_results": {}, "profile_results": {}}

    console.print(f"[cyan]Preparing to run [b]{actual_experiment_size}[/b] experiment pair(s).[/cyan]")

    experiment_pair_results_for_compilation = [] 

    profiles_dir = DATA_DIR / "profiles"
    concepts_dir = DATA_DIR / "concepts"
    all_profiles_data = {}
    if profiles_dir.exists():
        for filename in os.listdir(profiles_dir):
            if filename.endswith(".json"):
                try:
                    with open(profiles_dir / filename, "r") as f:
                        profile = json.load(f)
                        all_profiles_data[profile["id"]] = profile
                except Exception as e:
                    console.print(f"[red]Error loading profile {filename}: {e}[/red]")
    
    all_concepts_data = {}
    if concepts_dir.exists():
        for filename in os.listdir(concepts_dir):
            if filename.endswith(".json"):
                try:
                    with open(concepts_dir / filename, "r") as f:
                        concept = json.load(f)
                        all_concepts_data[concept["id"]] = concept
                except Exception as e:
                    console.print(f"[red]Error loading concept {filename}: {e}[/red]")
    
    if not all_profiles_data or not all_concepts_data:
        console.print("[bold red]Error: Missing profile or concept data files from `synapz/data/profiles` or `synapz/data/concepts`. Cannot proceed.[/bold red]")
        return {"error": "Missing profile or concept data files.", "experiments": [], "aggregate_results": {}, "profile_results": {}}

    with Progress(console=console, transient=True) as progress:
        overall_task = progress.add_task("[cyan]Batch Progress...", total=len(combinations_to_run))

        for i, (learner_id, concept_id) in enumerate(combinations_to_run):
            progress.update(overall_task, description=f"[cyan]Pair {i+1}/{len(combinations_to_run)}: [b]{learner_id}[/b] on [b]{concept_id}[/b][/cyan]")
            
            learner_profile = all_profiles_data.get(learner_id)
            concept_info = all_concepts_data.get(concept_id)

            if not learner_profile or not concept_info:
                error_msg = f"Missing data for learner '{learner_id}' or concept '{concept_id}'. Skipping pair."
                console.print(f"[bold red]Error: {error_msg}[/bold red]")
                experiment_pair_results_for_compilation.append({
                    "learner_id": learner_id, "concept_id": concept_id, "error": error_msg,
                    "adaptive_session_id": None, "control_session_id": None,
                    "metrics_file_path": None 
                })
                progress.advance(overall_task)
                continue

            learner_name = learner_profile.get('name', learner_id)
            concept_title = concept_info.get('title', concept_id)
            console.print(Panel(f"[bold]🧪 Experiment Pair {i+1}/{len(combinations_to_run)}: Learner [u]{learner_name}[/u] | Concept [u]{concept_title}[/u][/bold]", border_style="magenta", expand=False, padding=(0,1)))

            adaptive_session_id = teacher.create_session(learner_id, concept_id, is_adaptive=True)
            _, adaptive_cost = _run_one_session(
                session_id=adaptive_session_id, teacher=teacher, simulator=simulator, 
                learner_profile=learner_profile, console=console, use_llm_simulator=use_llm_simulator, 
                turns_per_session=turns_per_session, is_adaptive_session=True, concept_id=concept_title
            )

            control_session_id = teacher.create_session(learner_id, concept_id, is_adaptive=False)
            _, control_cost = _run_one_session(
                session_id=control_session_id, teacher=teacher, simulator=simulator, 
                learner_profile=learner_profile, console=console, use_llm_simulator=use_llm_simulator, 
                turns_per_session=turns_per_session, is_adaptive_session=False, concept_id=concept_title
            )
            
            metrics_pair_id = "N/A"
            try:
                metrics_pair_id = metrics_calculator.save_comparison_metrics(
                    adaptive_session_id, control_session_id, learner_id, concept_id, turns_per_session
                )
                # The returned value is experiment_pair_id, can be used as a reference
                console.print(f"Saved comparison metrics for pair {i+1} (DB Pair ID: {metrics_pair_id})")
            except Exception as e_metrics_save:
                 console.print(f"[bold red]Error saving metrics for pair {i+1} ({learner_id}, {concept_id}): {e_metrics_save}[/bold red]")
                 logger.error(f"Failed to save metrics for pair {adaptive_session_id}/{control_session_id}: {e_metrics_save}", exc_info=True)


            # Display per-pair summary
            adaptive_clarity_stats = metrics_calculator._calculate_comprehensive_clarity_stats(adaptive_session_id)
            control_clarity_stats = metrics_calculator._calculate_comprehensive_clarity_stats(control_session_id)

            pair_summary_table = Table(title=f"Pair {i+1} Results: {learner_name} & {concept_title}", box=box.ROUNDED, show_header=True, row_styles=["","dim"])
            pair_summary_table.add_column("Metric", style="bold")
            pair_summary_table.add_column("Adaptive Session", style="green")
            pair_summary_table.add_column("Control Session", style="yellow")
            
            pair_summary_table.add_row("Final Clarity", 
                                       f"{adaptive_clarity_stats.get('final_score', 'N/A')}/5", 
                                       f"{control_clarity_stats.get('final_score', 'N/A')}/5")
            pair_summary_table.add_row("Avg Clarity", 
                                       f"{adaptive_clarity_stats.get('average_score', 'N/A'):.2f}/5" if adaptive_clarity_stats.get('average_score') is not None else "N/A", 
                                       f"{control_clarity_stats.get('average_score', 'N/A'):.2f}/5" if control_clarity_stats.get('average_score') is not None else "N/A")
            pair_summary_table.add_row("Clarity Δ (End-Start)", 
                                       f"{adaptive_clarity_stats.get('absolute_improvement', 'N/A')}", 
                                       f"{control_clarity_stats.get('absolute_improvement', 'N/A')}")
            pair_summary_table.add_row("Session Cost", f"${adaptive_cost:.4f}", f"${control_cost:.4f}")
            
            ped_diff = metrics_calculator.compare_pedagogical_difference(adaptive_session_id, control_session_id)
            if "error" not in ped_diff:
                 pair_summary_table.add_row("Text Diff (vs Control)", f"{ped_diff.get('text_difference', 0.0)*100:.1f}%", "---")
                 pair_summary_table.add_row("Tag Diff (vs Control)", f"{ped_diff.get('tag_difference', 0.0)*100:.1f}%", "---")
            else:
                 pair_summary_table.add_row("Text Diff (vs Control)", ped_diff.get("error", "N/A"), "---")
                 pair_summary_table.add_row("Tag Diff (vs Control)", ped_diff.get("error", "N/A"), "---")
            console.print(pair_summary_table)

            experiment_pair_results_for_compilation.append({
                "experiment_pair_id": metrics_pair_id, # From save_comparison_metrics
                "learner_id": learner_id, "concept_id": concept_id,
                "adaptive_session_id": adaptive_session_id, "control_session_id": control_session_id,
                "adaptive_final_clarity": adaptive_clarity_stats.get('final_score'),
                "control_final_clarity": control_clarity_stats.get('final_score'),
                "adaptive_avg_clarity": adaptive_clarity_stats.get('average_score'),
                "control_avg_clarity": control_clarity_stats.get('average_score'),
                "adaptive_clarity_improvement": adaptive_clarity_stats.get('absolute_improvement'),
                "control_clarity_improvement": control_clarity_stats.get('absolute_improvement'),
                "adaptive_cost": adaptive_cost, "control_cost": control_cost,
                "pedagogical_text_difference": ped_diff.get('text_difference') if "error" not in ped_diff else None,
                "pedagogical_tag_difference": ped_diff.get('tag_difference') if "error" not in ped_diff else None,
                "error": ped_diff.get("error") # Store error from ped_diff if any
            })
            
            current_spend = budget_tracker.get_current_spend()
            budget_pct = (current_spend / budget_tracker.max_budget) * 100 if budget_tracker.max_budget > 0 else 0
            console.print(f"💰 [bold]Budget Status:[/bold] ${current_spend:.4f} / ${budget_tracker.max_budget:.2f} ({budget_pct:.1f}%)")
            if budget_tracker.is_exceeded():
                console.print("[bold red]🚨 BUDGET EXCEEDED! Halting batch evaluation. 🚨[/bold red]")
                break 
            
            progress.advance(overall_task)

    compiled_results = _compile_batch_results(experiment_pair_results_for_compilation)
    
    console.print(Panel("[bold underline]📊 Overall Batch Experiment Summary 📊[/bold underline]", style="bold blue", expand=False, padding=(1,2)))
    
    summary_table = Table(title="Overall Performance Metrics", box=box.DOUBLE_EDGE, show_lines=True)
    summary_table.add_column("Metric Key", style="bold cyan")
    summary_table.add_column("Adaptive Avg.", style="bold green")
    summary_table.add_column("Control Avg.", style="bold yellow")
    summary_table.add_column("Difference (A-C)", style="bold magenta")
    summary_table.add_column("P-Value", style="dim") # Removed "(if applicable)"

    # Simplified mapping for average_metrics which is now expected to be flat
    direct_metrics_map = {
        "avg_adaptive_final_clarity": "Final Clarity",
        "avg_control_final_clarity": "Final Clarity", # Will be handled by adaptive/control columns
        "avg_adaptive_clarity_improvement": "Clarity Improvement",
        "avg_control_clarity_improvement": "Clarity Improvement", # Handled
        "avg_text_difference_vs_control": "Text Difference (Adaptive vs Control)",
        "avg_tag_difference_vs_control": "Tag Difference (Adaptive vs Control)",
        "avg_adaptive_cost": "Session Cost",
        "avg_control_cost": "Session Cost", # Handled
    }
    
    # Metrics to display with adaptive, control, diff, p-value
    metrics_to_display = {
        "Final Clarity": ("avg_adaptive_final_clarity", "avg_control_final_clarity", "p_value_final_clarity"),
        "Clarity Improvement (Abs)": ("avg_adaptive_clarity_improvement", "avg_control_clarity_improvement", "p_value_clarity_improvement"),
        "Session Cost": ("avg_adaptive_cost", "avg_control_cost", "p_value_cost"),
    }
    
    avg_metrics_data = compiled_results.get("aggregate_results", {}).get("average_metrics", {})

    for display_name, (adapt_key, ctrl_key, pval_key) in metrics_to_display.items():
        adapt_val = avg_metrics_data.get(adapt_key)
        ctrl_val = avg_metrics_data.get(ctrl_key)
        pval = avg_metrics_data.get(pval_key)

        adapt_str = f"{adapt_val:.2f}" if isinstance(adapt_val, float) else str(adapt_val if adapt_val is not None else 'N/A')
        ctrl_str = f"{ctrl_val:.2f}" if isinstance(ctrl_val, float) else str(ctrl_val if ctrl_val is not None else 'N/A')
        
        diff_str = "N/A"
        if isinstance(adapt_val, (int, float)) and isinstance(ctrl_val, (int, float)):
            diff = adapt_val - ctrl_val
            diff_str = f"{diff:+.2f}" # Show sign
        
        pval_str = f"{pval:.3g}" if isinstance(pval, float) else str(pval if pval is not None else 'N/A')
        
        summary_table.add_row(display_name, adapt_str, ctrl_str, diff_str, pval_str)

    # Display other aggregate metrics that don't fit the A/C/Diff structure
    other_aggregates = {
        "Avg Text Difference (A vs C)": avg_metrics_data.get("avg_text_difference_vs_control"),
        "Avg Tag Difference (A vs C)": avg_metrics_data.get("avg_tag_difference_vs_control"),
        "Overall Adaptive Win Rate (%)": avg_metrics_data.get("overall_adaptive_win_rate"),
    }
    for name, val in other_aggregates.items():
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val if val is not None else 'N/A')
        summary_table.add_row(name, val_str, "---", "---", "---")


    summary_table.add_row("[hr_heavy]", "", "", "", "") # Separator
    summary_table.add_row("Total Pairs Run Successfully", str(compiled_results.get("aggregate_results", {}).get("total_pairs_run_successfully", "N/A")), "", "", "")
    summary_table.add_row("Total Adaptive Cost", f"${compiled_results.get('aggregate_results', {}).get('total_adaptive_cost', 0.0):.2f}", "", "", "")
    summary_table.add_row("Total Control Cost", f"${compiled_results.get('aggregate_results', {}).get('total_control_cost', 0.0):.2f}", "", "", "")
    summary_table.add_row("[b]Overall Total Cost[/b]", f"[b]${compiled_results.get('aggregate_results', {}).get('overall_total_cost', 0.0):.2f}[/b]", "", "", "")
    
    errors_encountered = compiled_results.get("aggregate_results", {}).get("errors_encountered", 0)
    if errors_encountered > 0:
        summary_table.add_row("[bold red]Errors Encountered[/bold red]", str(errors_encountered), "", "", "")

    console.print(summary_table)
    
    evidence_summary = compiled_results.get("evidence_summary", {})
    console.print(Panel(
        f"[bold u]Evidence Summary:[/bold u]\n"
        f"{evidence_summary.get('summary_text', 'No summary text generated.')}\n"
        f"Profiles with Adaptive Advantage: {', '.join(evidence_summary.get('profiles_with_adaptive_advantage', ['None'])) if evidence_summary.get('profiles_with_adaptive_advantage') else 'None'}\n"
        f"Overall Evidence Strength: [b]{evidence_summary.get('evidence_strength_overall', 'N/A').upper()}[/b]",
        title="Scientific Conclusion", border_style="bold green"
    ))
    
    results_dir = Path("results") / f"batch_run_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    json_results_path = results_dir / "compiled_batch_results.json"
    with open(json_results_path, "w") as f:
        json.dump(compiled_results, f, indent=2, cls=NumpyEncoder) # Use NumpyEncoder
    console.print(f"Full compiled JSON results saved to: [link=file://{json_results_path.resolve()}]{json_results_path.resolve()}[/link]")

    # Save detailed experiment pair data to CSV
    csv_detail_path = results_dir / "experiment_pair_details.csv"
    if experiment_pair_results_for_compilation:
        try:
            # Define headers based on keys in the first non-error result, or a fixed list
            # This assumes experiment_pair_results_for_compilation contains dicts
            # and we want to write these dicts to CSV
            # Filter out error-only entries for determining headers
            sample_result_for_header = next((item for item in experiment_pair_results_for_compilation if not item.get("error") or ("adaptive_session_id" in item and item["adaptive_session_id"] is not None)), None)
            if sample_result_for_header:
                 headers = list(sample_result_for_header.keys())
            else: # Fallback headers if no successful results
                 headers = ["learner_id", "concept_id", "adaptive_session_id", "control_session_id", "error"]


            with open(csv_detail_path, "w", newline="") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=headers)
                writer.writeheader()
                for pair_result in experiment_pair_results_for_compilation:
                    # Ensure all keys in headers are present in pair_result, fill with "N/A" if missing
                    row_to_write = {header: pair_result.get(header, "N/A") for header in headers}
                    writer.writerow(row_to_write)
            console.print(f"Detailed pair-by-pair CSV results saved to: [link=file://{csv_detail_path.resolve()}]{csv_detail_path.resolve()}[/link]")
        except Exception as e_csv:
            console.print(f"[red]Error saving detailed CSV results: {e_csv}[/red]")


    return compiled_results

def _compile_batch_results(experiment_pair_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile results from all experiment pairs into a structured report."""
    if not experiment_pair_results:
        return {
            "experiments": [],
            "profile_results": {},
            "aggregate_results": {"error": "No experiment results to compile"},
            "evidence_summary": {"summary_text": "No experiments run."}
        }

    # Initialize structures for aggregated data
    profile_metrics = defaultdict(lambda: {
        "total_pairs": 0,
        "adaptive_wins_clarity_improvement": 0, # Be specific about win condition
        "control_wins_clarity_improvement": 0,
        "ties_clarity_improvement": 0,
        "clarity_improvements_adaptive": [],
        "clarity_improvements_control": [],
        "final_clarity_adaptive": [],
        "final_clarity_control": [],
        "costs_adaptive": [],
        "costs_control": [],
        "text_differences": [],
        "tag_differences": []
        # Add other metrics you want to track per profile, e.g., readability
    })

    all_adaptive_clarity_improvements = []
    all_control_clarity_improvements = []
    all_final_adaptive_clarity = []
    all_final_control_clarity = []
    all_adaptive_costs = []
    all_control_costs = []
    all_text_differences = []
    all_tag_differences = []
    
    errors_encountered = 0
    total_adaptive_cost_overall = 0.0
    total_control_cost_overall = 0.0


    for res in experiment_pair_results:
        profile_id = res.get("learner_id", "unknown_profile")
        if res.get("error"):
            errors_encountered +=1
            logger.warning(f"Skipping result for profile {profile_id} due to error: {res.get('error')}")
            # Still count it as a pair for this profile if learner_id is present
            if profile_id != "unknown_profile":
                 profile_metrics[profile_id]["total_pairs"] += 1
            continue # Skip processing this errored result further for metric calculations

        profile_metrics[profile_id]["total_pairs"] += 1

        # Directly access metrics from 'res'
        adaptive_imp = res.get("adaptive_clarity_improvement")
        control_imp = res.get("control_clarity_improvement")
        
        # Handle None values for improvements, default to 0 for comparison
        adaptive_imp = adaptive_imp if adaptive_imp is not None else 0.0
        control_imp = control_imp if control_imp is not None else 0.0

        profile_metrics[profile_id]["clarity_improvements_adaptive"].append(adaptive_imp)
        profile_metrics[profile_id]["clarity_improvements_control"].append(control_imp)
        all_adaptive_clarity_improvements.append(adaptive_imp)
        all_control_clarity_improvements.append(control_imp)

        if adaptive_imp > control_imp:
            profile_metrics[profile_id]["adaptive_wins_clarity_improvement"] += 1
        elif control_imp > adaptive_imp:
            profile_metrics[profile_id]["control_wins_clarity_improvement"] += 1
        else:
            profile_metrics[profile_id]["ties_clarity_improvement"] += 1
            
        # Final Clarity
        final_adaptive_clar = res.get("adaptive_final_clarity")
        final_control_clar = res.get("control_final_clarity")
        if final_adaptive_clar is not None:
            profile_metrics[profile_id]["final_clarity_adaptive"].append(final_adaptive_clar)
            all_final_adaptive_clarity.append(final_adaptive_clar)
        if final_control_clar is not None:
            profile_metrics[profile_id]["final_clarity_control"].append(final_control_clar)
            all_final_control_clarity.append(final_control_clar)

        # Costs
        adaptive_cost = res.get("adaptive_cost", 0.0)
        control_cost = res.get("control_cost", 0.0)
        profile_metrics[profile_id]["costs_adaptive"].append(adaptive_cost)
        profile_metrics[profile_id]["costs_control"].append(control_cost)
        all_adaptive_costs.append(adaptive_cost)
        all_control_costs.append(control_cost)
        total_adaptive_cost_overall += adaptive_cost
        total_control_cost_overall += control_cost
        
        # Pedagogical differences
        text_diff = res.get("pedagogical_text_difference", 0.0) # Default to 0.0 if None
        tag_diff = res.get("pedagogical_tag_difference", 0.0)   # Default to 0.0 if None
        
        # Ensure text_diff and tag_diff are numbers before appending
        text_diff = text_diff if isinstance(text_diff, (int, float)) else 0.0
        tag_diff = tag_diff if isinstance(tag_diff, (int, float)) else 0.0

        profile_metrics[profile_id]["text_differences"].append(text_diff)
        profile_metrics[profile_id]["tag_differences"].append(tag_diff)
        all_text_differences.append(text_diff)
        all_tag_differences.append(tag_diff)


    # Calculate averages for each profile
    profile_summary_results = {}
    for profile, data in profile_metrics.items():
        num_pairs = data["total_pairs"]
        if num_pairs == 0: # Should not happen if profile_id was valid, but as a safeguard
            profile_summary_results[profile] = {
                "num_experiments": 0, "adaptive_win_rate": 0,
                "avg_adaptive_clarity_improvement": 0, "avg_control_clarity_improvement": 0,
                "avg_final_adaptive_clarity": 0, "avg_final_control_clarity": 0,
                "avg_adaptive_cost": 0, "avg_control_cost": 0,
                "avg_text_difference_vs_control": 0, "avg_tag_difference_vs_control": 0,
                "p_value_final_clarity": 1.0, "p_value_clarity_improvement": 1.0, "p_value_cost": 1.0
            }
            continue

        avg_adaptive_clarity_imp = np.mean(data["clarity_improvements_adaptive"]) if data["clarity_improvements_adaptive"] else 0.0
        avg_control_clarity_imp = np.mean(data["clarity_improvements_control"]) if data["clarity_improvements_control"] else 0.0
        
        # Significance tests for this profile
        final_clarity_p_val = calculate_statistical_significance(
            data["final_clarity_adaptive"], data["final_clarity_control"]
        ).get("p_value", 1.0) if data["final_clarity_adaptive"] and data["final_clarity_control"] else 1.0
        
        clarity_imp_p_val = calculate_statistical_significance(
            data["clarity_improvements_adaptive"], data["clarity_improvements_control"]
        ).get("p_value", 1.0) if data["clarity_improvements_adaptive"] and data["clarity_improvements_control"] else 1.0

        cost_p_val = calculate_statistical_significance(
            data["costs_adaptive"], data["costs_control"]
        ).get("p_value", 1.0) if data["costs_adaptive"] and data["costs_control"] else 1.0
            
        profile_summary_results[profile] = {
            "num_experiments": num_pairs,
            "adaptive_win_rate": (data["adaptive_wins_clarity_improvement"] / num_pairs) * 100 if num_pairs > 0 else 0,
            "avg_adaptive_clarity_improvement": avg_adaptive_clarity_imp,
            "avg_control_clarity_improvement": avg_control_clarity_imp,
            "avg_final_adaptive_clarity": np.mean(data["final_clarity_adaptive"]) if data["final_clarity_adaptive"] else 0.0,
            "avg_final_control_clarity": np.mean(data["final_clarity_control"]) if data["final_clarity_control"] else 0.0,
            "avg_adaptive_cost": np.mean(data["costs_adaptive"]) if data["costs_adaptive"] else 0.0,
            "avg_control_cost": np.mean(data["costs_control"]) if data["costs_control"] else 0.0,
            "avg_text_difference_vs_control": np.mean(data["text_differences"]) * 100 if data["text_differences"] else 0,
            "avg_tag_difference_vs_control": np.mean(data["tag_differences"]) * 100 if data["tag_differences"] else 0,
            "p_value_final_clarity": final_clarity_p_val,
            "p_value_clarity_improvement": clarity_imp_p_val,
            "p_value_cost": cost_p_val
        }
    
    num_total_experiments_processed = sum(data["total_pairs"] for data in profile_metrics.values())
    overall_adaptive_wins_clarity_imp = sum(p_data["adaptive_wins_clarity_improvement"] for p_data in profile_metrics.values())
    
    # Overall p-values
    overall_p_final_clarity = calculate_statistical_significance(
        all_final_adaptive_clarity, all_final_control_clarity
    ).get("p_value", 1.0) if all_final_adaptive_clarity and all_final_control_clarity else 1.0

    overall_p_clarity_improvement = calculate_statistical_significance(
        all_adaptive_clarity_improvements, all_control_clarity_improvements
    ).get("p_value", 1.0) if all_adaptive_clarity_improvements and all_control_clarity_improvements else 1.0
    
    overall_p_cost = calculate_statistical_significance(
        all_adaptive_costs, all_control_costs
    ).get("p_value", 1.0) if all_adaptive_costs and all_control_costs else 1.0

    # Aggregate results for the entire batch
    average_metrics = {
        "avg_adaptive_final_clarity": np.mean(all_final_adaptive_clarity) if all_final_adaptive_clarity else 0.0,
        "avg_control_final_clarity": np.mean(all_final_control_clarity) if all_final_control_clarity else 0.0,
        "p_value_final_clarity": overall_p_final_clarity,
        "avg_adaptive_clarity_improvement": np.mean(all_adaptive_clarity_improvements) if all_adaptive_clarity_improvements else 0.0,
        "avg_control_clarity_improvement": np.mean(all_control_clarity_improvements) if all_control_clarity_improvements else 0.0,
        "p_value_clarity_improvement": overall_p_clarity_improvement,
        "avg_adaptive_cost": np.mean(all_adaptive_costs) if all_adaptive_costs else 0.0,
        "avg_control_cost": np.mean(all_control_costs) if all_control_costs else 0.0,
        "p_value_cost": overall_p_cost,
        "avg_text_difference_vs_control": np.mean(all_text_differences) * 100 if all_text_differences else 0.0,
        "avg_tag_difference_vs_control": np.mean(all_tag_differences) * 100 if all_tag_differences else 0.0,
        "overall_adaptive_win_rate": (overall_adaptive_wins_clarity_imp / num_total_experiments_processed) * 100 if num_total_experiments_processed > 0 else 0.0,
    }
    
    aggregate_results_summary = {
        "average_metrics": average_metrics,
        "total_pairs_run_successfully": num_total_experiments_processed, # Number of pairs that didn't have errors
        "errors_encountered": errors_encountered,
        "total_adaptive_cost": total_adaptive_cost_overall,
        "total_control_cost": total_control_cost_overall,
        "overall_total_cost": total_adaptive_cost_overall + total_control_cost_overall
    }
    
    win_rate_for_evidence = average_metrics["overall_adaptive_win_rate"]
    evidence_strength = "weak"
    if win_rate_for_evidence > 50 and num_total_experiments_processed >= 3:
        evidence_strength = "suggestive"
    if win_rate_for_evidence > 60 and num_total_experiments_processed >= 5 and overall_p_clarity_improvement < 0.1: # Consider p-value
         evidence_strength = "moderate"
    if win_rate_for_evidence > 70 and num_total_experiments_processed >= 5 and overall_p_clarity_improvement < 0.05:
        evidence_strength = "strong"

    evidence_summary = {
        "profiles_with_adaptive_advantage": [
            profile for profile, data in profile_summary_results.items() if data["adaptive_win_rate"] > 50 and data["p_value_clarity_improvement"] < 0.1 # More stringent
        ],
        "evidence_strength_overall": evidence_strength,
        "summary_text": f"Based on {num_total_experiments_processed} successfully processed experiment pairs, "
                         f"adaptive learning shows an overall win rate (clarity improvement) of {win_rate_for_evidence:.2f}%. "
                         f"The evidence is currently considered {evidence_strength.upper()}.",
        "notes_for_interpretation": [
            "Win rate based on absolute clarity improvement (Adaptive vs Control).",
            "P-values indicate statistical significance of differences (lower is more significant).",
            f"Number of pairs with errors: {errors_encountered}."
        ]
    }

    return {
        "experiments": experiment_pair_results, 
        "profile_results": profile_summary_results, 
        "aggregate_results": aggregate_results_summary, 
        "evidence_summary": evidence_summary 
    }

def main():
    parser = argparse.ArgumentParser(description="Run Synapz batch evaluations.")
    parser.add_argument(
        "--size", 
        type=int, 
        default=3,
        help="Number of (learner, concept) pairs for batch evaluation"
    )
    parser.add_argument(
        "--turns", 
        type=int, 
        default=3, 
        help="Number of turns per teaching session"
    )
    parser.add_argument(
        "--budget", 
        type=float, 
        default=10.0, 
        help="Max budget in USD for this batch run"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DATA_DIR / "synapz_eval.db"), # Default path using DATA_DIR from synapz package
        help="Path to the SQLite database for evaluation results."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None, # Defaults to None, LLMClient will check env var
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY env variable."
    )
    parser.add_argument(
        "--no-llm-simulator",
        action="store_true",
        help="Use heuristic student simulator instead of LLM-based one (reduces API calls)."
    )
    parser.add_argument("--teacher-model", type=str, default="gpt-4o",
                        help="Specify the OpenAI model for the TeacherAgent (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--simulator-model", type=str, default="gpt-4o",
                        help="Specify the OpenAI model for the LLM-based StudentSimulator (e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--learner-id", type=str, default=None,
                        help="Specify a single learner ID to run an experiment for.")
    parser.add_argument("--concept-id", type=str, default=None,
                        help="Specify a single concept ID to run an experiment for (must be used with --learner-id).")
    # TODO: Add argument for specific_combinations if needed, loading from a JSON file perhaps

    args = parser.parse_args()
    
    # Setup components
    console = Console()
    db_path = Path(args.db_path)
    
    # Create directory for database if it doesn't exist
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    
    # Initialize components
    budget_tracker = BudgetTracker(str(db_path), max_budget=args.budget)
    db = Database(str(db_path)) # Use args.db_path for Database
    llm_client = LLMClient(budget_tracker=budget_tracker, api_key=args.api_key)
    teacher = TeacherAgent(llm_client=llm_client, db=db, teacher_model_name=args.teacher_model)
    simulator = StudentSimulator(llm_client=llm_client, simulator_model_name=args.simulator_model)
    metrics_calculator = MetricsCalculator(db) # Pass db instance
    
    # Display budget before starting
    current_spend = budget_tracker.get_current_spend()
    console.print(f"[bold]Starting budget:[/bold] ${current_spend:.4f} / ${args.budget:.2f}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Determine combinations to test
    available_combinations = get_available_combinations()
    combinations_to_run: Optional[List[Tuple[str, str]]] = None

    if args.learner_id and args.concept_id:
        if (args.learner_id, args.concept_id) in available_combinations:
            combinations_to_run = [(args.learner_id, args.concept_id)]
            console.print(f"[bold yellow]Running specific experiment for Learner: {args.learner_id}, Concept: {args.concept_id}[/bold yellow]")
        else:
            console.print(f"[bold red]Error: Specified learner/concept pair ({args.learner_id}, {args.concept_id}) not found in available combinations. Exiting.[/bold red]")
            return  # Exit if specific pair is invalid
    elif args.learner_id or args.concept_id:
        console.print(f"[bold red]Error: Both --learner-id and --concept-id must be provided together, or neither. Exiting.[/bold red]")
        return # Exit if only one is provided
    else:
        # Default behavior: use --size or all available
        experiment_size_to_run = args.size
        if len(available_combinations) > experiment_size_to_run:
            combinations_to_run = random.sample(available_combinations, experiment_size_to_run)
        else:
            combinations_to_run = available_combinations
    
    if not combinations_to_run: # Should not happen if logic above is correct, but as a safeguard
        console.print(f"[bold red]Error: No valid combinations to test. Exiting.[/bold red]")
        return

    # Run batch experiment
    start_time = time.time()
    results = run_batch_experiment(
        teacher, 
        simulator, 
        metrics_calculator, 
        console, 
        budget_tracker, # budget_tracker passed here
        len(combinations_to_run), 
        args.turns,
        specific_combinations=combinations_to_run, 
        use_llm_simulator=not args.no_llm_simulator
    )
    end_time = time.time()
    
    # Display final budget
    final_spend = budget_tracker.get_current_spend() # budget_tracker is in scope
    console.print(f"\n[bold]Final budget:[/bold] ${final_spend:.4f} / ${args.budget:.2f}")
    console.print(f"[bold]Cost of batch evaluation:[/bold] ${final_spend - (final_spend - (results.get('overall_total_cost',0.0))):.4f}") # This seems off, should be final_spend - initial_spend.
                                                                                                                                        # Let's use overall_total_cost from results
    console.print(f"[bold]Time taken:[/bold] {end_time - start_time:.1f} seconds")
    
    # Results are now saved inside run_batch_experiment with timestamped folder.
    # No need for this top-level saving anymore.
    # with open("results/batch_results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    
    console.print(f"\n[bold green]✅ Batch evaluation completed.[/bold green]")

if __name__ == "__main__":
    main() 