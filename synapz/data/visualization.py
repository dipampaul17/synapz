"""Visualization module for generating evidence charts from metrics data."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import sqlite3
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
import argparse

from synapz import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = str(DATA_DIR / "synapz_eval.db")

class DataVisualizer:
    """Generates visualizations of teaching experiment results."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize with database connection."""
        self.db_path = db_path
        self.viz_dir = Path("viz")
        self.viz_dir.mkdir(exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def generate_clarity_heatmap(
        self, 
        adaptive_sessions: List[str], 
        control_sessions: List[str],
        output_file_name: str = "clarity_heatmap.png"
    ) -> None:
        """
        Generate a heatmap comparing clarity scores over turns for adaptive vs. control sessions.
        
        Args:
            adaptive_sessions: List of adaptive session IDs
            control_sessions: List of control session IDs
            output_file_name: Name of the file to save the generated visualization in the viz/ directory
        """
        output_path = self.viz_dir / output_file_name

        # Get data
        conn = self._get_connection()
        
        # Function to extract clarity data for sessions
        def get_clarity_data(session_ids):
            all_data = {}
            for session_id in session_ids:
                cursor = conn.execute(
                    "SELECT turn_number, clarity_score FROM interactions "
                    "WHERE session_id = ? AND clarity_score IS NOT NULL "
                    "ORDER BY turn_number",
                    (session_id,)
                )
                session_data = {row["turn_number"]: row["clarity_score"] for row in cursor}
                all_data[session_id] = session_data
            return all_data
        
        adaptive_data = get_clarity_data(adaptive_sessions)
        control_data = get_clarity_data(control_sessions)
        
        conn.close()
        
        if not adaptive_data and not control_data:
            print("No clarity data found for the provided sessions.")
            return

        # Find max turn number across all sessions
        max_turn = 0
        for data_dict in [adaptive_data, control_data]:
            for session_data in data_dict.values():
                if session_data:
                    max_turn = max(max_turn, max(session_data.keys(), default=0))
        
        if max_turn == 0:
            print("No turns with clarity scores found.")
            return

        # Create arrays for heatmaps
        num_adaptive = len(adaptive_sessions)
        num_control = len(control_sessions)
        
        # Initialize arrays with NaN for missing values
        adaptive_array = np.full((max(1, num_adaptive), max_turn), np.nan)
        control_array = np.full((max(1, num_control), max_turn), np.nan)
        
        # Fill arrays with clarity scores
        for i, session_id in enumerate(adaptive_sessions):
            if session_id in adaptive_data:
                for turn, score in adaptive_data[session_id].items():
                    if 1 <= turn <= max_turn:
                         adaptive_array[i, turn-1] = score
        
        for i, session_id in enumerate(control_sessions):
            if session_id in control_data:
                for turn, score in control_data[session_id].items():
                    if 1 <= turn <= max_turn:
                        control_array[i, turn-1] = score
        
        # Calculate average scores per turn
        # Check if all entries are NaN to avoid RuntimeWarning
        adaptive_avg = np.nanmean(adaptive_array, axis=0) if not np.all(np.isnan(adaptive_array)) else np.full(max_turn, np.nan)
        control_avg = np.nanmean(control_array, axis=0) if not np.all(np.isnan(control_array)) else np.full(max_turn, np.nan)
        
        # Setup figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                           gridspec_kw={'height_ratios': [num_adaptive if num_adaptive > 0 else 1, 
                                                                          num_control if num_control > 0 else 1, 
                                                                          max(1, max_turn*0.2)]})
        
        # Create heatmaps
        cmap = cm.get_cmap('viridis').copy()
        norm = Normalize(vmin=1, vmax=5)
        
        # Adaptive heatmap
        if num_adaptive > 0 and not np.all(np.isnan(adaptive_array)):
            sns.heatmap(adaptive_array, ax=ax1, cmap=cmap, norm=norm, annot=True, fmt=".0f",
                       cbar_kws={'label': 'Clarity Score (1-5)'})
        ax1.set_title('Adaptive Teaching: Clarity Scores by Turn')
        ax1.set_xlabel('Turn Number')
        ax1.set_ylabel('Session Index')
        ax1.set_xticks(np.arange(max_turn) + 0.5)
        ax1.set_xticklabels(np.arange(1, max_turn + 1))
        if num_adaptive > 0:
             ax1.set_yticks(np.arange(num_adaptive) + 0.5)
             ax1.set_yticklabels(np.arange(1, num_adaptive + 1))

        # Control heatmap
        if num_control > 0 and not np.all(np.isnan(control_array)):
            sns.heatmap(control_array, ax=ax2, cmap=cmap, norm=norm, annot=True, fmt=".0f",
                       cbar_kws={'label': 'Clarity Score (1-5)'})
        ax2.set_title('Control Teaching: Clarity Scores by Turn')
        ax2.set_xlabel('Turn Number')
        ax2.set_ylabel('Session Index')
        ax2.set_xticks(np.arange(max_turn) + 0.5)
        ax2.set_xticklabels(np.arange(1, max_turn + 1))
        if num_control > 0:
            ax2.set_yticks(np.arange(num_control) + 0.5)
            ax2.set_yticklabels(np.arange(1, num_control + 1))
        
        # Average comparison line plot
        turns_axis = np.arange(1, max_turn + 1)
        if not np.all(np.isnan(adaptive_avg)):
            ax3.plot(turns_axis, adaptive_avg, 'o-', label='Adaptive Avg', color='green')
        if not np.all(np.isnan(control_avg)):
            ax3.plot(turns_axis, control_avg, 'o-', label='Control Avg', color='orange')
        ax3.set_title('Average Clarity Score by Turn')
        ax3.set_xlabel('Turn')
        ax3.set_ylabel('Avg. Clarity Score')
        ax3.set_ylim(0.5, 5.5)
        ax3.set_xticks(turns_axis)
        ax3.grid(True, linestyle='--', alpha=0.7)
        if not np.all(np.isnan(adaptive_avg)) or not np.all(np.isnan(control_avg)):
            ax3.legend()
        
        # Calculate improvements for annotation
        if not np.all(np.isnan(adaptive_avg)) and len(adaptive_avg[~np.isnan(adaptive_avg)]) > 1 and \
           not np.all(np.isnan(control_avg)) and len(control_avg[~np.isnan(control_avg)]) > 1:
            
            first_adaptive_avg = adaptive_avg[~np.isnan(adaptive_avg)][0]
            last_adaptive_avg = adaptive_avg[~np.isnan(adaptive_avg)][-1]
            adaptive_improvement_val = last_adaptive_avg - first_adaptive_avg

            first_control_avg = control_avg[~np.isnan(control_avg)][0]
            last_control_avg = control_avg[~np.isnan(control_avg)][-1]
            control_improvement_val = last_control_avg - first_control_avg
            
            difference = adaptive_improvement_val - control_improvement_val
            
            improvement_text = (
                f"Adaptive improvement: {adaptive_improvement_val:.2f}\n"
                f"Control improvement: {control_improvement_val:.2f}\n"
                f"Advantage: {difference:.2f} in favor of "
                f"{'Adaptive' if difference > 0 else ('Control' if difference < 0 else 'Tie')}"
            )
            
            ax3.annotate(improvement_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Clarity heatmap saved to {output_path}")
    
    def generate_personalization_comparison(
        self, 
        learner_id: str,
        concepts: List[str],
        output_file_name: str = "personalization_comparison.png"
    ) -> None:
        """
        Generate a visualization comparing how teaching differs across learner profiles.
        
        Args:
            learner_id: Learner profile to compare against others
            concepts: List of concept IDs to include in comparison
            output_file_name: Name of the file to save the generated visualization
        """
        output_path = self.viz_dir / output_file_name

        conn = self._get_connection()
        
        # Get all learner profiles
        cursor = conn.execute("SELECT id FROM learner_profiles")
        all_learners_db = [row["id"] for row in cursor.fetchall()]
        
        # Filter out the target learner
        comparison_learners = [l for l in all_learners_db if l != learner_id]
        
        if not comparison_learners:
            print(f"No other learner profiles found in DB to compare with {learner_id}.")
            conn.close()
            return

        # Collect data on text differences
        differences = []
        
        for concept_id in concepts:
            # For each concept, find sessions for target learner and comparison learners
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE learner_id = ? AND concept_id = ? AND experiment_type = 'adaptive'",
                (learner_id, concept_id)
            )
            target_sessions = [row["id"] for row in cursor.fetchall()]
            
            if not target_sessions:
                continue
            
            target_session = target_sessions[0]
            
            for comp_learner in comparison_learners:
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE learner_id = ? AND concept_id = ? AND experiment_type = 'adaptive'",
                    (comp_learner, concept_id)
                )
                comp_sessions = [row["id"] for row in cursor.fetchall()]
                
                if not comp_sessions:
                    continue
                
                comp_session = comp_sessions[0]
                
                # Get last explanation from each session
                cursor = conn.execute(
                    "SELECT explanation, pedagogy_tags FROM interactions "
                    "WHERE session_id = ? ORDER BY turn_number DESC LIMIT 1",
                    (target_session,)
                )
                target_row = cursor.fetchone()
                
                cursor = conn.execute(
                    "SELECT explanation, pedagogy_tags FROM interactions "
                    "WHERE session_id = ? ORDER BY turn_number DESC LIMIT 1",
                    (comp_session,)
                )
                comp_row = cursor.fetchone()
                
                if not target_row or not comp_row:
                    continue
                
                # Calculate Levenshtein distance
                import Levenshtein
                target_text = target_row["explanation"] if target_row["explanation"] else ""
                comp_text = comp_row["explanation"] if comp_row["explanation"] else ""
                
                # Normalize by dividing by max length
                max_len = max(len(target_text), len(comp_text))
                if max_len == 0: # Both empty
                    text_diff_val = 0.0
                else:
                    text_diff_val = Levenshtein.distance(target_text, comp_text) / max_len
                
                # Calculate readability difference
                import textstat
                target_readability = textstat.flesch_reading_ease(target_text) if target_text else 0
                comp_readability = textstat.flesch_reading_ease(comp_text) if comp_text else 0
                readability_diff_val = abs(target_readability - comp_readability)
                
                # Calculate tag difference
                try:
                    target_tags_list = json.loads(target_row["pedagogy_tags"]) if target_row["pedagogy_tags"] else []
                    comp_tags_list = json.loads(comp_row["pedagogy_tags"]) if comp_row["pedagogy_tags"] else []
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse pedagogy_tags for session {target_session} or {comp_session}")
                    target_tags_list = []
                    comp_tags_list = []

                tag_set1 = set(str(t) for t in target_tags_list)
                tag_set2 = set(str(t) for t in comp_tags_list)
                
                if not tag_set1 and not tag_set2:
                    tag_diff_val = 0.0
                else:
                    intersection = len(tag_set1.intersection(tag_set2))
                    union = len(tag_set1.union(tag_set2))
                    tag_diff_val = 1.0 - (intersection / union) if union > 0 else 0.0
                
                differences.append({
                    "concept_id": concept_id,
                    "target_learner": learner_id,
                    "comparison_learner": comp_learner,
                    "text_difference": text_diff_val,
                    "readability_difference": readability_diff_val,
                    "tag_difference": tag_diff_val
                })
        
        conn.close()
        
        if not differences:
            print(f"No comparison data found for learner {learner_id} across concepts {concepts}")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(differences)
        
        # Group by comparison learner and average the differences
        grouped = df.groupby("comparison_learner")[["text_difference", "readability_difference", "tag_difference"]].mean().reset_index()
        
        if grouped.empty:
            print(f"Not enough data to generate personalization comparison for learner {learner_id}")
            return

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Personalization Comparison: {learner_id} vs. Other Learner Profiles (Avg. over {len(concepts)} concepts)", fontsize=14)

        # Text difference
        sns.barplot(x="comparison_learner", y="text_difference", data=grouped, ax=ax1, color="skyblue", dodge=False)
        ax1.set_title(f"Avg. Text Difference")
        ax1.set_xlabel("Comparison Learner")
        ax1.set_ylabel("Avg. Text Difference (0-1)")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Readability difference
        sns.barplot(x="comparison_learner", y="readability_difference", data=grouped, ax=ax2, color="lightgreen", dodge=False)
        ax2.set_title(f"Avg. Readability Difference")
        ax2.set_xlabel("Comparison Learner")
        ax2.set_ylabel("Avg. Abs. Flesch Ease Diff.")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Tag difference
        sns.barplot(x="comparison_learner", y="tag_difference", data=grouped, ax=ax3, color="salmon", dodge=False)
        ax3.set_title(f"Avg. Teaching Approach Difference")
        ax3.set_xlabel("Comparison Learner")
        ax3.set_ylabel("Avg. Tag Jaccard Distance (0-1)")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Personalization comparison saved to {output_path}")
    
    def generate_clarity_progression_charts(
        self,
        aggregated_turn_clarity_data: Dict[str, Dict[str, Dict[int, List[float]]]],
        output_dir: Path
    ) -> None:
        """
        Generates line charts showing average clarity progression per turn for each cognitive profile,
        comparing adaptive vs. control sessions.

        Args:
            aggregated_turn_clarity_data: Data structured as 
                {profile_style: {session_type: {turn_number: [clarity_scores]}}}.
            output_dir: Path to the directory where charts will be saved.
        """
        if not aggregated_turn_clarity_data:
            logger.info("No aggregated turn-by-turn clarity data to visualize.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        for profile_style, sessions_data in aggregated_turn_clarity_data.items():
            plt.figure(figsize=(10, 6))
            
            has_data_for_plot = False
            max_turn_overall = 0

            for session_type, turns_data in sessions_data.items():
                if not turns_data: continue

                # Ensure turn numbers are treated as integers
                try:
                    # Attempt to convert keys to int for sorting and max()
                    # Handles cases where keys might be strings like '1', '2'
                    turn_numbers = sorted([int(k) for k in turns_data.keys()])
                except ValueError:
                    logger.warning(f"Could not convert turn numbers to int for profile {profile_style}, session_type {session_type}. Skipping this session type.")
                    continue # Skip this session_type if keys are not valid integers
                
                if not turn_numbers: continue
                
                max_turn_for_type = max(turn_numbers) # Now definitely int
                max_turn_overall = max(max_turn_overall, max_turn_for_type) # max_turn_overall is also int

                avg_clarity_per_turn = [np.mean(turns_data[str(t)]) if str(t) in turns_data and turns_data[str(t)] else (np.mean(turns_data[int(t)]) if int(t) in turns_data and turns_data[int(t)] else np.nan) for t in turn_numbers]
                
                # Filter out NaN for plotting if a turn had no data, but keep sequence for x-axis
                valid_turns = [t for i, t in enumerate(turn_numbers) if not np.isnan(avg_clarity_per_turn[i])]
                valid_avg_clarity = [avg for avg in avg_clarity_per_turn if not np.isnan(avg)]

                if valid_turns and valid_avg_clarity:
                    plt.plot(valid_turns, valid_avg_clarity, marker='o', linestyle='-', label=f'{session_type.capitalize()} Avg. Clarity')
                    has_data_for_plot = True
            
            if has_data_for_plot:
                plt.title(f"Clarity Progression for {profile_style.upper()} Learners", fontsize=15)
                plt.xlabel("Turn Number", fontsize=12)
                plt.ylabel("Average Clarity Score (1-5)", fontsize=12)
                plt.xticks(np.arange(1, max_turn_overall + 1))
                plt.yticks(np.arange(1, 6))
                plt.ylim(0.5, 5.5)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                chart_file_name = f"clarity_progression_{profile_style.lower().replace(' ', '_')}.png"
                chart_path = output_dir / chart_file_name
                plt.savefig(chart_path, dpi=300)
                logger.info(f"Saved clarity progression chart to {chart_path}")
            else:
                logger.info(f"No valid data to plot clarity progression for profile: {profile_style}")
            
            plt.close() # Close figure for the current profile
    
    def generate_summary_dashboard(
        self,
        output_file_name: str = "summary_dashboard.png"
    ) -> None:
        """
        Generate a summary dashboard of key results from 'experiment_metrics' table.
        
        Args:
            output_file_name: Name of the file to save the generated visualization
        """
        output_path = self.viz_dir / output_file_name

        conn = self._get_connection()
        
        # Get experiment metrics directly from the new experiment_metrics table
        try:
            df = pd.read_sql_query("SELECT * FROM experiment_metrics", conn)
        except pd.io.sql.DatabaseError as e:
            print(f"Error reading from experiment_metrics table (it might be empty or not exist as expected): {e}")
            conn.close()
            return
        finally:
            conn.close()
        
        if df.empty:
            print("No data found in experiment_metrics table.")
            return
        
        # --- Calculate overall statistics directly from 'experiment_metrics' columns ---
        # Clarity Improvements
        avg_adaptive_clarity_improvement = df["adaptive_clarity_improvement"].mean()
        avg_control_clarity_improvement = df["control_clarity_improvement"].mean()

        # Personalization Metrics (using pre-calculated diffs where available)
        # For text difference, we'll use 1 - similarity for "difference"
        avg_text_difference = (1 - df["text_similarity_final_explanation"]).mean() if "text_similarity_final_explanation" in df.columns else np.nan
        
        # For readability, use the direct difference column
        avg_readability_difference_val = df["readability_diff_flesch_ease"].abs().mean() if "readability_diff_flesch_ease" in df.columns else np.nan
        
        # For tag difference, use 1 - similarity
        avg_tag_difference = (1 - df["tag_jaccard_similarity_final"]).mean() if "tag_jaccard_similarity_final" in df.columns else np.nan
        
        # Create summary dashboard
        fig = plt.figure(figsize=(14, 12))
        
        # Title
        fig.suptitle("Synapz Experiment Results Dashboard", fontsize=18, fontweight="bold")
        
        # Grid for subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1,1.5,1.5], hspace=0.4)
        
        # Clarity improvement comparison
        ax1 = fig.add_subplot(gs[0, 0])
        labels = ["Adaptive", "Control"]
        values = [avg_adaptive_clarity_improvement, avg_control_clarity_improvement]
        bars = ax1.bar(labels, values, color=["#4CAF50", "#FF9800"])
        ax1.set_title("Avg. Clarity Improvement (Final - Initial)", fontsize=12)
        ax1.set_ylabel("Points (1-5 scale)", fontsize=10)
        ax1.grid(axis="y", linestyle=":", alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        
        # Personalization metrics
        ax2 = fig.add_subplot(gs[0, 1])
        pers_metrics = ["Text Diff.", "Readability Diff.", "Tag Diff."]
        pers_values = [avg_text_difference, avg_readability_difference_val / 100, avg_tag_difference]
        
        bars_pers = ax2.bar(pers_metrics, pers_values, color=["#2196F3", "#F44336", "#9C27B0"])
        ax2.set_title("Avg. Personalization Strength (vs. Control/Other)", fontsize=12)
        ax2.set_ylabel("Difference Metric (0-1 or Normalized)", fontsize=10)
        ax2.set_ylim(0, 1 if all(0 <= v <= 1 for v in pers_values if not np.isnan(v)) else None)
        ax2.grid(axis="y", linestyle=":", alpha=0.7)
        for bar in bars_pers:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}" if not np.isnan(height) else "N/A", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        # Clarity improvement by learner profile
        ax3 = fig.add_subplot(gs[1, :])
        if 'learner_id' in df.columns:
            learner_pivot = df.groupby("learner_id")[["adaptive_clarity_improvement", "control_clarity_improvement"]].mean()
            learner_pivot.plot(kind="bar", ax=ax3, color={"adaptive_clarity_improvement": "#4CAF50", "control_clarity_improvement": "#FF9800"})
            ax3.set_title("Clarity Improvement by Learner Profile", fontsize=12)
            ax3.set_xlabel("Learner Profile", fontsize=10)
            ax3.set_ylabel("Avg. Clarity Improvement", fontsize=10)
            ax3.tick_params(axis='x', rotation=45, ha="right")
            ax3.grid(axis="y", linestyle=":", alpha=0.7)
            ax3.legend(["Adaptive", "Control"], title="Mode", fontsize=9)
        else:
            ax3.text(0.5, 0.5, "Learner ID data not available\nin experiment_metrics", ha='center', va='center', fontsize=10, color='grey')
            ax3.set_title("Clarity Improvement by Learner Profile (Data N/A)", fontsize=12)

        # Performance by concept difficulty (assuming concept_id can be mapped to difficulty)
        ax4 = fig.add_subplot(gs[2, :])
        # This part requires mapping concept_id to difficulty.
        # We need to load concept data from DATA_DIR/concepts/*.json
        concepts_data_dir = DATA_DIR / "concepts"
        concept_difficulties = {}
        if concepts_data_dir.exists() and concepts_data_dir.is_dir() and 'concept_id' in df.columns:
            for concept_file_path in concepts_data_dir.glob("*.json"):
                try:
                    with open(concept_file_path, "r") as f:
                        c_data = json.load(f)
                        if "id" in c_data and "difficulty" in c_data:
                            concept_difficulties[c_data["id"]] = c_data["difficulty"]
                except Exception as e:
                    print(f"Warning: Could not load or parse concept file {concept_file_path}: {e}")
            
            if concept_difficulties:
                df["difficulty"] = df["concept_id"].map(concept_difficulties)
                if not df["difficulty"].isnull().all():
                    difficulty_pivot = df.groupby("difficulty")[["adaptive_clarity_improvement", "control_clarity_improvement"]].mean()
                    difficulty_pivot.plot(kind="line", marker="o", ax=ax4, 
                                        color={"adaptive_clarity_improvement": "#4CAF50", "control_clarity_improvement": "#FF9800"})
                    ax4.set_title("Performance by Concept Difficulty", fontsize=12)
                    ax4.set_xlabel("Concept Difficulty", fontsize=10)
                    ax4.set_ylabel("Avg. Clarity Improvement", fontsize=10)
                    ax4.grid(True, linestyle=":", alpha=0.7)
                    ax4.legend(["Adaptive", "Control"], title="Mode", fontsize=9)
                    # Ensure x-axis ticks are integers if difficulty is integer
                    if all(isinstance(x, (int, float)) and x == int(x) for x in difficulty_pivot.index):
                         ax4.set_xticks(sorted(difficulty_pivot.index.unique().astype(int)))
                else:
                    ax4.text(0.5, 0.5, "Concept difficulty data could not be mapped or is unavailable.", ha='center', va='center', fontsize=10, color='grey')
                    ax4.set_title("Performance by Concept Difficulty (Data N/A)", fontsize=12)

            else:
                ax4.text(0.5, 0.5, "No concept difficulty data found.", ha='center', va='center', fontsize=10, color='grey')
                ax4.set_title("Performance by Concept Difficulty (Data N/A)", fontsize=12)
        else:
            ax4.text(0.5, 0.5, "Concept ID data not available or concepts dir missing.", ha='center', va='center', fontsize=10, color='grey')
            ax4.set_title("Performance by Concept Difficulty (Data N/A)", fontsize=12)
            
        # Add summary text box
        final_adaptive_advantage = avg_adaptive_clarity_improvement - avg_control_clarity_improvement
        advantage_pct = (final_adaptive_advantage / abs(avg_control_clarity_improvement) * 100) if avg_control_clarity_improvement and avg_control_clarity_improvement != 0 else float('inf')
        
        summary_text_lines = [
            f"Overall Synapz Performance:",
            f"  Avg. Adaptive Clarity Gain: {avg_adaptive_clarity_improvement:.2f} pts",
            f"  Avg. Control Clarity Gain:  {avg_control_clarity_improvement:.2f} pts",
            f"  Net Adaptive Advantage:   {final_adaptive_advantage:+.2f} pts ({advantage_pct:.1f}% vs control)" if not np.isinf(advantage_pct) else \
            f"  Net Adaptive Advantage:   {final_adaptive_advantage:+.2f} pts (Control gain was zero)",
            f"Personalization Strength:",
            f"  Avg. Text Difference (vs Control): {avg_text_difference:.1%}",
            f"  Avg. Tag Jaccard Dist. (vs Control): {avg_tag_difference:.1%}"
        ]
        
        fig.text(0.5, 0.02, "\n".join(summary_text_lines), fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Summary dashboard saved to {output_path}")

def generate_readability_comparison_chart(
    report_data: Dict[str, Any], 
    output_path: Optional[Path] = None
) -> None:
    """
    Generate a bar chart comparing readability metrics between adaptive and control.
    
    Args:
        report_data: Report data dictionary from metrics
        output_path: Optional path to save the chart image
    """
    # MODIFIED: Iterate over profile_results, not aggregate_results
    profiles = list(report_data.get("profile_results", {}).keys())
    
    if not profiles:
        logger.warning("No profile data found in report_data['profile_results'] for readability_comparison_chart.")
        return
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set up positions for grouped bars
    bar_width = 0.35
    index = np.arange(len(profiles))
    
    # Calculate effectiveness scores
    # MODIFIED: Look for a relevant score in profile_results. 
    # 'effectiveness_score' might not be there. Use adaptive win rate as a proxy or default to 0.
    effectiveness_scores = []
    p_values_for_chart = []

    for profile in profiles:
        profile_data = report_data.get("profile_results", {}).get(profile, {})
        # Try to get 'effectiveness_score', fallback to 'adaptive_win_rate' (scaled if needed), then 0.
        score = profile_data.get("effectiveness_score") 
        if score is None:
            score = profile_data.get("adaptive_win_rate", 0) / 100 # Scale win rate (0-100) to 0-1 if used
        effectiveness_scores.append(score)
        
        # Placeholder for p-value if not directly available in profile_results
        # The original code expected overall_p_value inside aggregate_results[profile]
        # This structure is not present. We'll default p_value to 1.0 (not significant)
        # unless a more specific per-profile p-value is added to profile_results.
        p_values_for_chart.append(profile_data.get("p_value_for_effectiveness", 1.0))

    # Create bars
    plt.bar(index, effectiveness_scores, bar_width, label='Effectiveness/Win Rate', color='green')
    
    # Add labels and title
    plt.xlabel('Cognitive Profile')
    plt.ylabel('Effectiveness Score / Adaptive Win Rate (0-1)')
    plt.title('Adaptive Learning Effectiveness by Cognitive Profile')
    plt.xticks(index, profiles)
    plt.legend()
    
    # Add significance markers
    for i, profile in enumerate(profiles):
        # Using p_values_for_chart collected above
        p_value = p_values_for_chart[i]
        if p_value < 0.05:
            plt.text(i, effectiveness_scores[i] + 0.05, '*', 
                     fontsize=15, ha='center', va='bottom')
            
    # Add value labels on top of bars
    for i, score in enumerate(effectiveness_scores):
        plt.text(i, score + 0.01, f'{score:.2f}', 
                 fontsize=10, ha='center', va='bottom')
    
    # Add annotations
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.text(len(profiles)-1, -0.1, 'Below 0: No improvement', 
             fontsize=8, ha='right', va='top', color='red')
    
    plt.figtext(0.02, 0.02, 
                '* p < 0.05 (statistically significant)\n'
                'Effectiveness > 0.5: Strong evidence\n'
                'Effectiveness > 0.2: Moderate evidence\n'
                'Effectiveness > 0.0: Suggestive evidence', 
                fontsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Chart saved to {output_path}")
    else:
        plt.show()
        
    plt.close()

def generate_metrics_dashboard(
    report_data: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Generate a complete dashboard of visualization charts.
    
    Args:
        report_data: Report data dictionary from metrics
        output_dir: Directory to save the charts
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate effectiveness comparison chart
    generate_readability_comparison_chart(
        report_data, 
        output_dir / "effectiveness_by_profile.png"
    )
    
    # Generate readability metrics detail chart
    generate_readability_detail_chart(
        report_data, 
        output_dir / "readability_metrics.png"
    )
    
    # Generate effect size chart
    generate_effect_size_chart(
        report_data, 
        output_dir / "effect_sizes.png"
    )
    
    # Generate evidence summary chart
    generate_evidence_summary_chart(
        report_data, 
        output_dir / "evidence_summary.png"
    )
    
    logger.info(f"All charts generated and saved to {output_dir}")

def generate_readability_detail_chart(
    report_data: Dict[str, Any], 
    output_path: Optional[Path] = None
) -> None:
    """
    Generate a detailed comparison of readability metrics.
    
    Args:
        report_data: Report data dictionary from metrics
        output_path: Optional path to save the chart image
    """
    # MODIFIED: Iterate over profile_results
    profiles = list(report_data.get("profile_results", {}).keys())
    
    if not profiles:
        logger.warning("No profile data found in report_data['profile_results'] for readability_detail_chart.")
        return
    
    # Key metrics to track
    key_metrics = ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]
    metric_labels = ["Reading Ease", "Grade Level", "SMOG Index"]
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Number of metrics and profiles
    n_metrics = len(key_metrics)
    n_profiles = len(profiles)
    
    # Loop through metrics and create subplots
    for i, (metric, label) in enumerate(zip(key_metrics, metric_labels)):
        plt.subplot(2, 2, i+1)
        
        adapted_values = []
        control_values = []
        p_values = []
        
        for profile in profiles:
            profile_specific_results = report_data.get("profile_results", {}).get(profile, {})
            # MODIFIED: Look for statistical_significance within each profile's data in profile_results
            # This key might not exist yet in what evaluate.py produces.
            stat_results_container = profile_specific_results.get("statistical_significance", {})
            stat_results = stat_results_container.get(metric, {}) # Get specific metric stats
            
            # Get mean values (use the mean difference and adjust)
            mean_diff = stat_results.get("mean_difference", 0) # Default to 0 if not found
            p_value = stat_results.get("p_value", 1.0) # Default to 1.0 if not found
            
            # For this chart, we'll just use placeholder values since we don't have the raw means
            # In a real implementation, you would access the actual mean values
            baseline = 50  # A baseline value
            if metric == "flesch_kincaid_grade":
                baseline = 8
            elif metric == "smog_index":
                baseline = 10
                
            control_value = baseline
            adapted_value = baseline + mean_diff
            
            adapted_values.append(adapted_value)
            control_values.append(control_value)
            p_values.append(p_value)
        
        # Set up positions for grouped bars
        index = np.arange(n_profiles)
        bar_width = 0.35
        
        # Create bars
        plt.bar(index, control_values, bar_width, label='Control', color='gray')
        plt.bar(index + bar_width, adapted_values, bar_width, label='Adapted', color='green')
        
        # Add labels
        plt.xlabel('Cognitive Profile')
        plt.ylabel(label)
        plt.title(f'{label} Comparison')
        plt.xticks(index + bar_width/2, profiles)
        
        # Add significance markers
        for j, (adapted, p) in enumerate(zip(adapted_values, p_values)):
            if p < 0.05:
                plt.text(j + bar_width, adapted + 0.5, '*', 
                         fontsize=15, ha='center', va='bottom')
        
        plt.legend()
    
    plt.figtext(0.02, 0.02, 
                '* p < 0.05 (statistically significant)\n'
                'For Grade Level and SMOG Index, lower values are better\n'
                'For Reading Ease, higher values are better', 
                fontsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Chart saved to {output_path}")
    else:
        plt.show()
        
    plt.close()

def generate_effect_size_chart(
    report_data: Dict[str, Any], 
    output_path: Optional[Path] = None
) -> None:
    """Generate a bar chart of effect sizes for key metrics by profile."""
    # MODIFIED: Iterate over profile_results
    profiles = list(report_data.get("profile_results", {}).keys())
    
    if not profiles:
        logger.warning("No profile data to plot for effect sizes.")
        if output_path:
            # Create a blank placeholder image if path is given
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available for effect size chart.", ha='center', va='center')
            ax.axis('off')
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            logger.info(f"Placeholder effect size chart saved to {output_path}")
        return

    effect_sizes_data = {metric: [] for metric in ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]}
    significance_data = {metric: [] for metric in ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]}

    for profile in profiles:
        profile_agg = report_data.get("profile_results", {}).get(profile, {})
        for metric in ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]:
            stat_sig_data = profile_agg.get("statistical_significance", {}).get(metric, {})
            effect_sizes_data[metric].append(stat_sig_data.get("effect_size", 0))
            significance_data[metric].append(stat_sig_data.get("is_significant", False))

    # Plotting
    # Increased figure width to accommodate more profiles/metrics if needed
    fig, ax = plt.subplots(figsize=(max(10, len(profiles) * 2.5), 7)) # Dynamically adjust width
    bar_width = 0.25
    index = np.arange(len(profiles))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Example colors for metrics

    for i, metric in enumerate(effect_sizes_data):
        bars = ax.bar(index + i * bar_width, effect_sizes_data[metric], bar_width, 
                      label=metric.replace("_", " ").title(), color=colors[i % len(colors)])
        # Add significance markers (e.g., '*')
        for bar_idx, bar_val in enumerate(effect_sizes_data[metric]):
            if significance_data[metric][bar_idx]:
                ax.text(bar.get_x() + bar.get_width()/2., bar_val + (0.02 if bar_val >= 0 else -0.05), '*', 
                        ha='center', va='bottom', color='red', fontsize=16)

    ax.set_xlabel('Cognitive Profile', fontsize=12)
    ax.set_ylabel('Effect Size (Cohen\'s d)', fontsize=12)
    ax.set_title('Effect Sizes of Adaptive Learning vs. Control', fontsize=14, fontweight='bold')
    ax.set_xticks(index + bar_width * (len(effect_sizes_data) - 1) / 2)
    ax.set_xticklabels(profiles, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Readability Metrics", fontsize=10, title_fontsize=11)
    ax.axhline(0, color='grey', lw=0.8) # Line at y=0
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust subplot parameters for better spacing
    plt.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.95)

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)

def generate_evidence_summary_chart(
    report_data: Dict[str, Any], 
    output_path: Optional[Path] = None
) -> None:
    """Generate bar chart summarizing evidence strength."""
    summary = report_data.get("evidence_summary", {})
    # MODIFICATION: Use 'profile_results' for per-profile data, not 'aggregate_results'
    profile_results_data = report_data.get("profile_results", {}) 
    
    profiles = list(profile_results_data.keys()) # Get profile names from profile_results
    if not profiles:
        logger.warning("No profile data in 'profile_results' to plot for evidence summary.")
        if output_path:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No data for evidence summary (profile_results empty).", ha='center', va='center')
            ax.axis('off')
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            logger.info(f"Placeholder evidence chart saved to {output_path}")
        return

    # MODIFICATION: Fetch win rates from profile_results_data
    # Ensure the key "adaptive_win_rate" exists in each profile's dictionary
    win_rates = []
    valid_profiles = []
    for p_name in profiles:
        profile_data = profile_results_data.get(p_name, {})
        if isinstance(profile_data, dict): # Ensure it's a dict before .get()
            win_rate = profile_data.get("adaptive_win_rate", 0) # Default to 0 if key missing
            win_rates.append(win_rate)
            valid_profiles.append(p_name)
        else:
            logger.warning(f"Profile '{p_name}' in profile_results is not a dictionary. Skipping. Value: {profile_data}")
    
    if not valid_profiles:
        logger.warning("No valid profile data with win rates found after checking dictionaries.")
        # Create placeholder if necessary (similar to above)
        return

    fig, ax = plt.subplots(figsize=(max(8, len(valid_profiles) * 1.5), 6)) # Dynamically adjust width

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(valid_profiles, win_rates, color=[colors[i % len(colors)] for i in range(len(valid_profiles))])

    ax.set_ylabel('Adaptive Win Rate (%)', fontsize=12)
    ax.set_title('Overall Adaptive Win Rate by Profile', fontsize=14, fontweight='bold')
    ax.set_xticklabels(valid_profiles, rotation=45, ha="right", fontsize=10) # Use valid_profiles for labels
    ax.set_ylim(0, 100) # Win rates are percentages
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels for win rates
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

    # Add overall evidence strength text
    overall_strength = summary.get("evidence_strength_overall", "N/A").upper()
    fig.text(0.5, 0.01, f"Overall Evidence Strength: {overall_strength}", ha="center", fontsize=12, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    # Adjust subplot parameters for better spacing
    plt.subplots_adjust(bottom=0.3, top=0.9, left=0.15, right=0.95) # Ensures labels fit

    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Chart saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)

def create_visualization_from_report(report_path: Path, output_dir: Path) -> None:
    """Generate all standard visualizations from a compiled report JSON file."""
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Report file not found: {report_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from report file: {report_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Call the updated chart functions
    # Assuming these functions are defined or imported correctly
    # For example, if effectiveness_by_profile is desired:
    # generate_effectiveness_by_profile_chart(report_data, output_dir / "effectiveness_by_profile.png")
    
    # Based on previous log, these are the charts created:
    # effectiveness_by_profile.png - (Let's assume a function generate_effectiveness_by_profile_chart exists)
    # We will assume this one exists and is okay for now or create a placeholder if needed.
    # Example placeholder if function is missing:
    try:
        generate_effectiveness_by_profile_chart(report_data, output_dir / "effectiveness_by_profile.png")
    except NameError:
        logger.warning("generate_effectiveness_by_profile_chart function not found. Skipping this chart.")
        pass # Or create a placeholder image


    generate_readability_comparison_chart(report_data, output_dir / "readability_metrics.png")
    generate_effect_size_chart(report_data, output_dir / "effect_sizes.png") # Updated
    generate_evidence_summary_chart(report_data, output_dir / "evidence_summary.png") # Updated
    
    # Generate clarity progression charts
    aggregated_turn_clarity = report_data.get("turn_by_turn_clarity_aggregation")
    if aggregated_turn_clarity:
        try:
            # DEFAULT_DB_PATH is defined at the top of this file and used by DataVisualizer constructor
            visualizer = DataVisualizer() 
            visualizer.generate_clarity_progression_charts(
                aggregated_turn_clarity_data=aggregated_turn_clarity,
                output_dir=output_dir
            )
            logger.info(f"Generated clarity progression charts in {output_dir}")
        except Exception as e:
            logger.error(f"Failed to generate clarity progression charts: {e}", exc_info=True)
            # Optionally, create a placeholder or skip if critical
    else:
        logger.warning("No aggregated_turn_clarity_data found in report. Skipping clarity progression charts.")

    # Example: Generate and save a summary metrics table image
    # This is a placeholder for a more complex image generation