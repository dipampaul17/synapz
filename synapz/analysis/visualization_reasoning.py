"""Visualization functions for the reasoning experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "reasoning_experiment_visualizations"

def _save_plot(fig, output_dir: Path, filename: str):
    """Helper to save a plot and close the figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}")
    plt.close(fig)

def plot_metric_by_condition(
    analysis_data: Dict[str, Any],
    metric_key: str,
    y_label: str,
    title: str,
    output_dir: Path,
    filename: str,
    is_percentage: bool = False
):
    """Generic function to plot a metric by condition as a bar chart."""
    conditions = []
    values = []
    errors = [] # For standard deviation if available

    for cond, data in analysis_data.get('by_condition', {}).items():
        if isinstance(data, dict) and metric_key in data:
            conditions.append(cond.replace('_', ' ').title())
            values.append(data[metric_key])
            # Add std dev if available and relevant (e.g. for clarity_improvement)
            if metric_key == 'avg_clarity_improvement' and 'std_clarity_improvement' in data:
                errors.append(data['std_clarity_improvement'])
            else:
                errors.append(0) # No error bars for other metrics for now
        else:
            logger.warning(f"Metric '{metric_key}' not found or invalid data for condition '{cond}'. Skipping.")
            conditions.append(cond.replace('_', ' ').title())
            values.append(0) # Append 0 for missing data to maintain order
            errors.append(0)


    if not conditions:
        logger.warning(f"No data to plot for metric: {metric_key}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, values, yerr=errors if any(e > 0 for e in errors) else None, capsize=5, color=sns.color_palette("viridis", len(conditions)))
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        yval = bar.get_height()
        formatted_yval = f"{yval:.1%}" if is_percentage else f"{yval:.2f}"
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, formatted_yval, va='bottom' if yval >=0 else 'top', ha='center')

    if is_percentage:
        ax.set_ylim(0, 1.1) # For percentages up to 100%

    _save_plot(fig, output_dir, filename)

def plot_clarity_check_stats(
    analysis_data: Dict[str, Any],
    output_dir: Path,
    filename: str = "clarity_check_stats.png"
):
    """Plots average clarity check length and proportion with question mark."""
    conditions = []
    avg_lengths = []
    prop_questions = []

    for cond, data in analysis_data.get('by_condition', {}).items():
        if isinstance(data, dict) and 'avg_clarity_check_length' in data and 'proportion_clarity_check_is_question' in data:
            if cond == 'baseline': # Skip baseline as it has 0 for these
                continue
            conditions.append(cond.replace('_', ' ').title())
            avg_lengths.append(data['avg_clarity_check_length'])
            prop_questions.append(data['proportion_clarity_check_is_question'])
        else:
            logger.warning(f"Clarity check stats not found or invalid data for condition '{cond}'. Skipping.")
    
    if not conditions:
        logger.warning("No data for clarity check stats plot.")
        return

    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar for average length
    bars1 = ax1.bar(x - width/2, avg_lengths, width, label='Avg. Length (chars)', color='skyblue')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Average Length (characters)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)

    # Bar for proportion with question mark on a secondary y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, prop_questions, width, label='Proportion with ?', color='lightcoral')
    ax2.set_ylabel('Proportion with Question Mark', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    ax2.set_ylim(0, 1.1) # Proportion from 0 to 1

    fig.suptitle('Clarity Check Text Statistics by Condition')
    
    # Add text labels
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.1f}", va='bottom' if yval >=0 else 'top', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2%}", va='bottom' if yval >=0 else 'top', ha='center')
        
    # Add a single legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for title and legend
    _save_plot(fig, output_dir, filename)

def plot_experiment_costs(
    summary_json_path: Path,
    output_dir: Path,
    filename: str = "experiment_costs.png"
):
    """Plots the costs per condition from the experiment summary JSON file."""
    try:
        with open(summary_json_path, 'r') as f:
            summary_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Experiment summary JSON file not found: {summary_json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from: {summary_json_path}")
        return

    conditions = []
    costs = []
    
    for cond_summary in summary_data.get('conditions_run', []):
        if isinstance(cond_summary, dict) and 'condition' in cond_summary and 'cost_usd' in cond_summary:
            conditions.append(cond_summary['condition'].replace('_', ' ').title())
            costs.append(cond_summary['cost_usd'])
        else:
            logger.warning(f"Invalid condition summary data: {cond_summary}")

    if not conditions:
        logger.warning("No cost data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, costs, color=sns.color_palette("pastel", len(conditions)))
    
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Total Cost per Experiment Condition')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"${yval:.4f}", va='bottom' if yval >=0 else 'top', ha='center')

    total_experiment_cost = summary_data.get('total_cost_usd', 0.0)
    plt.text(0.98, 0.98, f"Total Experiment Cost: ${total_experiment_cost:.4f}",
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    _save_plot(fig, output_dir, filename)

# Example of how these might be called from reasoning_analyzer.py:
# if __name__ == '__main__':
#     # This block would be in reasoning_analyzer.py
#     # ... (load analysis_output and db_file_path) ...
#     viz_output_dir = db_file_path.parent / OUTPUT_DIR_NAME
#     
#     plot_metric_by_condition(analysis_output, 'avg_clarity_improvement', 'Avg. Clarity Improvement', 'Clarity Improvement by Condition', viz_output_dir, "clarity_improvement.png")
#     # ... more plot calls
#     experiment_summary_json = db_file_path.parent / f"{analysis_output.get('experiment_name', 'reasoning_experiment')}_summary.json" # Construct path to summary
#     plot_experiment_costs(experiment_summary_json, viz_output_dir, "experiment_costs.png") 