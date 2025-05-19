"""Analysis module for the reasoning experiment."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import re # Added for regex step counting
import logging
import yaml # For reading config

from synapz.core.models import Database
# Potentially import from synapz.analysis.visualization_reasoning for plots
from synapz.analysis.visualization_reasoning import (
    plot_metric_by_condition,
    plot_clarity_check_stats,
    plot_experiment_costs,
    OUTPUT_DIR_NAME # To create the output directory for plots
)

CONFIG_PATH = Path(__file__).parent.parent / "experiments" / "reasoning_config.yaml"


def load_experiment_name_from_config() -> str:
    """Loads the experiment_name from the reasoning_config.yaml file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("experiment_name", "reasoning_experiment_default")
    except Exception as e:
        logging.error(f"Could not load experiment name from {CONFIG_PATH}: {e}")
        return "reasoning_experiment_default"


def analyze_reasoning_experiment(db_path: str, experiment_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze the results of the reasoning experiment.

    Metrics computed:
    - Clarity improvement by condition (with Cohen's d)
    - Statistical significance tests (t-tests with correction)
    - Correlation between reasoning step count and clarity (if reasoning steps are quantifiable)
    - Token-normalized improvements
    - Qualitative analysis of reasoning patterns (summary or common themes)
    
    Args:
        db_path: Path to the SQLite database containing experiment results.
        experiment_ids: Optional list of specific experiment (session) IDs to analyze.
                        If None, analyzes all relevant data.

    Returns:
        Dict with all metrics and paths to visualizations.
    """
    db = Database(db_path)
    analysis_results = {}

    # 1. Fetch data from `interactions` and `reasoning_details` tables
    #    - Join interactions with reasoning_details on interaction_id.
    #    - Filter by condition: baseline, visible_reasoning, hidden_reasoning.
    #    - For clarity improvement, you'll need initial and final clarity scores per session/interaction sequence.
    #      The current `reasoning_details` stores clarity ratings per interaction; this might need refinement
    #      if improvement is measured over a whole session (multiple turns).
    #      Let's assume for now clarity_improvement in reasoning_details is the relevant one for that interaction.

    query = """
        SELECT 
            i.id as interaction_id,
            i.session_id,
            s.concept_id,
            s.learner_id, 
            rd.condition,
            i.explanation AS explanation_text,
            rd.reasoning_process_text,
            rd.metacognitive_supports_json,
            rd.clarity_check_text,
            rd.clarity_rating_initial, 
            rd.clarity_rating_final,
            rd.clarity_improvement,
            i.tokens_in, 
            i.tokens_out,
            i.cost
        FROM reasoning_details rd
        JOIN interactions i ON rd.interaction_id = i.id
        JOIN sessions s ON i.session_id = s.id
    """
    if experiment_ids:
        placeholders = ",".join(["?" for _ in experiment_ids])
        query += f" WHERE s.id IN ({placeholders})" # Assuming experiment_ids are session_ids
        df = pd.read_sql_query(query, db._get_connection(), params=experiment_ids)
    else:
        df = pd.read_sql_query(query, db._get_connection())

    if df.empty:
        return {"error": "No data found for analysis."}

    # --- Add this section to print some raw reasoning text ---
    console = Console() # For rich printing
    console.print("\n[bold yellow]Raw Reasoning Text Samples:[/bold yellow]")
    for cond_to_sample in ["visible_reasoning", "hidden_reasoning"]:
        sample_df = df[df['condition'] == cond_to_sample]['reasoning_process_text'].dropna().sample(min(3, len(df[df['condition'] == cond_to_sample]['reasoning_process_text'].dropna())))
        if not sample_df.empty:
            console.print(f"\n--- Condition: {cond_to_sample} ---")
            for i, text in enumerate(sample_df):
                console.print(f"Sample {i+1}:")
                console.print(Panel(text[:1000] + "..." if len(text) > 1000 else text, title="Reasoning Process Text", expand=False))
        else:
            console.print(f"No reasoning_process_text samples found for {cond_to_sample}")
    console.print("\n")
    # --- End of section ---

    # --- New section to print Explanation Text Samples for Qualitative Analysis ---
    console.print("\n[bold magenta]Explanation Text Samples for Qualitative Analysis:[/bold magenta]")

    if 'explanation_text' not in df.columns:
        console.print("[red]Error: 'explanation_text' column not found in the main DataFrame.[/red]")
    elif df.empty:
        console.print("[red]No data found in the main DataFrame for qualitative sampling.[/red]")
    else:
        concepts_to_sample = df['concept_id'].unique()
        if len(concepts_to_sample) > 3:
            # Try to get a diverse sample, ensure they exist in the dataframe
            valid_concepts = [c for c in concepts_to_sample if not df[df['concept_id'] == c].empty]
            if len(valid_concepts) >= 3:
                 concepts_to_sample = np.random.choice(valid_concepts, size=3, replace=False)
            elif valid_concepts: # if less than 3 but some are valid
                 concepts_to_sample = valid_concepts
            else:
                 concepts_to_sample = [] # no valid concepts to sample
        elif len(concepts_to_sample) == 0:
            console.print("[red]No concepts found in df for sampling.[/red]")
            # concepts_to_sample is already empty or an empty list

        if not concepts_to_sample.size > 0 and not isinstance(concepts_to_sample, list) and not concepts_to_sample:
             console.print("[yellow]No concepts selected for explanation sampling.[/yellow]")

        for concept_id_sample in concepts_to_sample:
            console.print(f"\n--- Concept: [cyan]{concept_id_sample}[/cyan] ---")
            for cond_to_sample in ['baseline', 'visible_reasoning', 'hidden_reasoning']:
                # Use the main df which should now have explanation_text
                sample_df_for_concept_condition = df[
                    (df['concept_id'] == concept_id_sample) & 
                    (df['condition'] == cond_to_sample)
                ]
                sample = sample_df_for_concept_condition['explanation_text'].dropna().head(1)
                
                console.print(f"\n  -- Condition: [yellow]{cond_to_sample}[/yellow] --")
                if not sample.empty:
                    console.print(Panel(sample.iloc[0], title=f"Explanation for {concept_id_sample} ({cond_to_sample})", expand=False, border_style="green"))
                else:
                    console.print(f"    No explanation text found for {concept_id_sample} under {cond_to_sample}")
    console.print("\n")
    # --- End of New Explanation Text Sample Section ---

    # Ensure numeric types
    numeric_cols = ['clarity_rating_initial', 'clarity_rating_final', 'clarity_improvement', 
                    'tokens_in', 'tokens_out', 'cost']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Analysis by condition
    analysis_results['by_condition'] = {}
    conditions = df['condition'].unique()

    for cond in conditions:
        cond_df = df[df['condition'] == cond].copy()
        if cond_df.empty:
            analysis_results['by_condition'][cond] = {"error": "No data for this condition"}
            continue

        avg_clarity_improvement = cond_df['clarity_improvement'].mean()
        std_clarity_improvement = cond_df['clarity_improvement'].std()
        
        # Token-normalized improvement (clarity points per 1K total tokens)
        cond_df['total_tokens'] = cond_df['tokens_in'] + cond_df['tokens_out']
        # Avoid division by zero if total_tokens is 0 or NaN
        cond_df['token_normalized_improvement'] = cond_df.apply(
            lambda row: (row['clarity_improvement'] / (row['total_tokens'] / 1000)) 
            if row['total_tokens'] and row['total_tokens'] > 0 else 0,
            axis=1
        )
        avg_token_norm_improvement = cond_df['token_normalized_improvement'].mean()

        # Reasoning step count (improved to use regex)
        if 'reasoning_process_text' in cond_df.columns and cond != 'baseline':
            def count_steps_from_json_list(json_string_or_list):
                if not pd.notna(json_string_or_list):
                    return 0
                try:
                    # If it's already a list (e.g. from direct JSON parsing if pandas did that, unlikely for string col)
                    if isinstance(json_string_or_list, list):
                        return len(json_string_or_list)
                    # Attempt to parse the JSON string
                    loaded_list = json.loads(str(json_string_or_list))
                    if isinstance(loaded_list, list):
                        return len(loaded_list)
                    else:
                        logging.warning(f"Parsed reasoning_process_text, but it was not a list. Type: {type(loaded_list)}. Text: {str(json_string_or_list)[:100]}")
                        return 0 # Or perhaps 1 if it's a non-empty string that didn't parse to list
                except json.JSONDecodeError:
                    # Fallback for old format or malformed JSON: try regex for numbered lines
                    # logger.debug(f"JSONDecodeError for reasoning_process_text: {str(json_string_or_list)[:100]}. Falling back to regex line count.")
                    return len(re.findall(r"^\s*\d+\.\s+[A-Z\s]+[:\-]", str(json_string_or_list), re.MULTILINE | re.IGNORECASE))
                except Exception as e:
                    logging.error(f"Unexpected error counting steps for: {str(json_string_or_list)[:100]}. Error: {e}")
                    return 0

            cond_df['reasoning_step_count'] = cond_df['reasoning_process_text'].apply(count_steps_from_json_list)
            avg_reasoning_steps = cond_df['reasoning_step_count'].mean()
            # Correlation: reasoning_step_count vs clarity_improvement
            if len(cond_df['reasoning_step_count'].dropna()) > 1 and len(cond_df['clarity_improvement'].dropna()) > 1:
                correlation, p_value = stats.pearsonr(
                    cond_df['reasoning_step_count'].fillna(0),
                    cond_df['clarity_improvement'].fillna(0)
                )
            else:
                correlation, p_value = np.nan, np.nan
        else:
            avg_reasoning_steps = np.nan
            correlation, p_value = np.nan, np.nan

        # --- New Metrics: Metacognitive Supports and Clarity Check ---
        avg_metacognitive_supports = np.nan
        if 'metacognitive_supports_json' in cond_df.columns:
            def count_metacognitive_supports(json_string):
                if pd.isna(json_string) or not isinstance(json_string, str):
                    return 0
                try:
                    supports_list = json.loads(json_string)
                    return len(supports_list) if isinstance(supports_list, list) else 0
                except json.JSONDecodeError:
                    return 0 # Or log warning
            cond_df['num_metacognitive_supports'] = cond_df['metacognitive_supports_json'].apply(count_metacognitive_supports)
            avg_metacognitive_supports = cond_df['num_metacognitive_supports'].mean()

        avg_clarity_check_length = np.nan
        proportion_clarity_check_is_question = np.nan
        if 'clarity_check_text' in cond_df.columns:
            cond_df['clarity_check_length'] = cond_df['clarity_check_text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            avg_clarity_check_length = cond_df['clarity_check_length'].mean()
            
            cond_df['clarity_check_has_question'] = cond_df['clarity_check_text'].apply(lambda x: isinstance(x, str) and "?" in x)
            if len(cond_df) > 0: # Avoid division by zero if cond_df is empty
                 proportion_clarity_check_is_question = cond_df['clarity_check_has_question'].mean() #.sum() / len(cond_df)
            else:
                 proportion_clarity_check_is_question = 0.0

        analysis_results['by_condition'][cond] = {
            'sample_size': len(cond_df),
            'avg_clarity_improvement': avg_clarity_improvement,
            'std_clarity_improvement': std_clarity_improvement,
            'avg_token_normalized_improvement': avg_token_norm_improvement,
            'avg_reasoning_steps': avg_reasoning_steps,
            'reasoning_clarity_correlation': {'r': correlation, 'p_value': p_value},
            'avg_metacognitive_supports': avg_metacognitive_supports,
            'avg_clarity_check_length': avg_clarity_check_length,
            'proportion_clarity_check_is_question': proportion_clarity_check_is_question
        }

    # Statistical comparisons between conditions (e.g., visible_reasoning vs baseline)
    # Example: T-test for clarity improvement
    if 'visible_reasoning' in conditions and 'baseline' in conditions:
        visible_scores = df[df['condition'] == 'visible_reasoning']['clarity_improvement'].dropna()
        baseline_scores = df[df['condition'] == 'baseline']['clarity_improvement'].dropna()
        
        if len(visible_scores) >= 2 and len(baseline_scores) >= 2:
            ttest_result = stats.ttest_ind(visible_scores, baseline_scores, equal_var=False) # Welch's t-test
            # Cohen's d
            mean_diff = visible_scores.mean() - baseline_scores.mean()
            pooled_std = np.sqrt((visible_scores.std()**2 + baseline_scores.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            analysis_results['visible_vs_baseline_clarity'] = {
                't_statistic': ttest_result.statistic,
                'p_value': ttest_result.pvalue,
                'cohens_d': cohens_d,
                'mean_visible_improvement': visible_scores.mean(),
                'mean_baseline_improvement': baseline_scores.mean()
            }

    # Add more comparisons (e.g., hidden vs baseline, visible vs hidden)

    # Qualitative Analysis (Placeholder - would involve NLP or manual review)
    # e.g., Common themes in reasoning_process_text, types of metacognitive_supports used.
    analysis_results['qualitative_summary'] = "Qualitative analysis placeholder."

    # Visualization (call functions from visualization_reasoning.py)
    # Example: plot_clarity_improvement_by_condition(analysis_results['by_condition'])
    analysis_results['visualizations'] = ["Paths to generated charts will be here."]

    return analysis_results


if __name__ == '__main__':
    import sys
    from rich.console import Console # Added for the new print section
    from rich.panel import Panel      # Added for the new print section
    # Construct the absolute path to the database
    # Assuming this script is in WORKSPACE_ROOT/synapz/analysis/
    # DB is in WORKSPACE_ROOT/synapz/results/reasoning_experiment.db
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent.parent # synapz/analysis -> synapz -> WORKSPACE_ROOT
    db_relative_path = "synapz/results/reasoning_experiment.db"
    db_file_path = (project_root / db_relative_path).resolve()

    if not db_file_path.exists():
        print(f"ERROR: Database file not found at {db_file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing database: {db_file_path}")
    analysis_output = analyze_reasoning_experiment(str(db_file_path))
    # Print JSON output, handling potential NaN values from pandas/numpy
    print(json.dumps(analysis_output, indent=2, default=lambda x: None if pd.isna(x) else x))

    # --- Generate Visualizations ---
    if analysis_output and "error" not in analysis_output:
        viz_output_dir = db_file_path.parent / OUTPUT_DIR_NAME
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nAttempting to save visualizations to: {viz_output_dir}")

        plot_metric_by_condition(
            analysis_output, 
            'avg_clarity_improvement', 
            'Avg. Clarity Improvement (Points)', 
            'Clarity Improvement by Condition', 
            viz_output_dir, 
            "clarity_improvement.png"
        )
        plot_metric_by_condition(
            analysis_output,
            'avg_token_normalized_improvement',
            'Avg. Token-Normalized Improvement (Points per 1k Tokens)',
            'Token-Normalized Clarity Improvement by Condition',
            viz_output_dir,
            "token_normalized_clarity.png"
        )
        plot_metric_by_condition(
            analysis_output,
            'avg_reasoning_steps',
            'Avg. Reasoning Steps',
            'Reasoning Steps by Condition',
            viz_output_dir,
            "reasoning_steps.png"
        )
        plot_metric_by_condition(
            analysis_output,
            'avg_metacognitive_supports',
            'Avg. Metacognitive Supports Count',
            'Metacognitive Supports by Condition',
            viz_output_dir,
            "metacognitive_supports.png"
        )
        plot_clarity_check_stats(
            analysis_output,
            viz_output_dir,
            "clarity_check_stats.png"
        )
        
        # For costs plot
        experiment_name = load_experiment_name_from_config()
        experiment_summary_json_path = db_file_path.parent / f"{experiment_name}_summary.json"
        plot_experiment_costs(
            experiment_summary_json_path,
            viz_output_dir,
            "experiment_costs.png"
        )
        print(f"Visualizations generated in {viz_output_dir}")
    else:
        print("\nSkipping visualization generation due to errors in analysis or no data.") 