"""Analysis module for measuring effectiveness of adaptive learning."""

import numpy as np
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
import textstat
import Levenshtein
from pathlib import Path
import json
import logging
import scipy.stats as stats

from .storage import ExperimentStorage
from .metrics import (
    calculate_readability_metrics, 
    calculate_text_similarity,
    calculate_statistical_significance,
    generate_comprehensive_metrics,
    save_metrics_report
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptationAnalyzer:
    """Analyze and compare adapted vs control content to measure effectiveness."""
    
    def __init__(self, experiment_storage: ExperimentStorage):
        """Initialize with experiment storage."""
        self.storage = experiment_storage
        
    def analyze_readability(self, experiment_id: int) -> Dict[str, float]:
        """
        Analyze readability metrics for an experiment.
        
        Returns:
            Dictionary of readability metrics
        """
        experiment = self.storage.get_experiment_by_id(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        content = experiment["content"]
        return calculate_readability_metrics(content)
    
    def compare_experiments(
        self, 
        adapted_id: int, 
        control_id: int
    ) -> Dict[str, Any]:
        """
        Compare adapted and control experiments.
        
        Args:
            adapted_id: ID of the adapted experiment
            control_id: ID of the control experiment
            
        Returns:
            Comparison metrics dictionary
        """
        adapted = self.storage.get_experiment_by_id(adapted_id)
        control = self.storage.get_experiment_by_id(control_id)
        
        if not adapted or not control:
            raise ValueError("Experiment not found")
            
        # Extract contents
        adapted_content = adapted["content"]
        control_content = control["content"]
        
        # Calculate similarity
        similarity_metrics = calculate_text_similarity(adapted_content, control_content)
        
        # Calculate readability metrics
        adapted_metrics = calculate_readability_metrics(adapted_content)
        control_metrics = calculate_readability_metrics(control_content)
        
        # Calculate statistical significance for key metrics
        key_metrics = [
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "smog_index",
            "gunning_fog",
            "avg_sentence_length"
        ]
        
        statistical_results = {}
        for metric in key_metrics:
            statistical_results[metric] = calculate_statistical_significance(
                [adapted_metrics], [control_metrics], metric
            )
        
        # Calculate percentage differences
        readability_diff = {
            key: ((adapted_metrics[key] - control_metrics[key]) / abs(control_metrics[key]) * 100) 
            if control_metrics[key] != 0 else 0
            for key in adapted_metrics.keys()
        }
        
        # Calculate overall effectiveness score
        weights = {
            "flesch_reading_ease": 0.3,
            "flesch_kincaid_grade": 0.2,
            "smog_index": 0.2,
            "gunning_fog": 0.2,
            "avg_sentence_length": 0.1
        }
        
        effectiveness_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in statistical_results:
                # Positive effect size and significant improvement contribute to score
                effect = statistical_results[metric]["effect_size"]
                is_sig = statistical_results[metric]["is_significant"]
                
                # For grade level metrics, lower is better (negative difference is improvement)
                if metric in ["flesch_kincaid_grade", "smog_index", "gunning_fog", "avg_sentence_length"]:
                    effect = -effect
                
                # Weight by significance and effect size
                metric_score = effect * (2 if is_sig else 1)
                effectiveness_score += metric_score * weight
                total_weight += weight
        
        if total_weight > 0:
            effectiveness_score = effectiveness_score / total_weight
        else:
            effectiveness_score = 0.0
        
        return {
            "similarity": similarity_metrics["levenshtein_similarity"],
            "text_similarity": similarity_metrics,
            "adapted_metrics": adapted_metrics,
            "control_metrics": control_metrics,
            "readability_diff_percent": readability_diff,
            "statistical_significance": statistical_results,
            "effectiveness_score": effectiveness_score,
            "adapted_profile": adapted["profile"],
            "topic": adapted["topic"]
        }
    
    def generate_evidence_report(self, output_file: Path) -> Dict[str, Any]:
        """
        Generate comprehensive evidence report comparing all experiment pairs.
        
        Args:
            output_file: Where to save the report data
            
        Returns:
            Report data dictionary
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all topics
        cursor.execute("SELECT DISTINCT topic FROM experiments")
        topics = [row["topic"] for row in cursor.fetchall()]
        
        # Analyze each topic
        topic_results = {}
        for topic in topics:
            pairs = self.storage.get_paired_experiments(topic)
            if not pairs:
                continue
                
            pair_results = []
            for adapted, control in pairs:
                comparison = self.compare_experiments(adapted["id"], control["id"])
                pair_results.append(comparison)
                
            topic_results[topic] = pair_results
            
        # Aggregate results by profile type
        profile_results = {}
        for topic, comparisons in topic_results.items():
            for comparison in comparisons:
                profile = comparison["adapted_profile"]
                if profile not in profile_results:
                    profile_results[profile] = []
                profile_results[profile].append(comparison)
        
        # Calculate aggregate statistics
        aggregates = {}
        for profile, comparisons in profile_results.items():
            # Get effectiveness scores
            effectiveness_scores = [comp["effectiveness_score"] for comp in comparisons]
            avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0
            
            # Get average similarity
            avg_similarity = np.mean([comp["similarity"] for comp in comparisons]) if comparisons else 0
            
            # Collect all readability metrics for statistical analysis
            adapted_readability_metrics = []
            control_readability_metrics = []
            for comp in comparisons:
                adapted_readability_metrics.append(comp["adapted_metrics"])
                control_readability_metrics.append(comp["control_metrics"])
            
            # Calculate key metrics statistical significance
            key_metrics = ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index"]
            statistical_results = {}
            
            for metric in key_metrics:
                statistical_results[metric] = calculate_statistical_significance(
                    adapted_readability_metrics, control_readability_metrics, metric
                )
            
            aggregates[profile] = {
                "effectiveness_score": avg_effectiveness,
                "avg_similarity": avg_similarity,
                "sample_count": len(comparisons),
                "statistical_significance": statistical_results,
                # Calculate p-value for overall effectiveness
                "overall_p_value": self._calculate_overall_p_value(effectiveness_scores)
            }
            
        # Compile report
        report = {
            "topic_results": topic_results,
            "profile_results": profile_results,
            "aggregate_results": aggregates,
            "evidence_summary": self._generate_evidence_summary(aggregates)
        }
        
        # Save report
        save_metrics_report(report, output_file)
        
        return report
    
    def _calculate_overall_p_value(self, effectiveness_scores: List[float]) -> float:
        """Calculate overall p-value using one-sample t-test against 0."""
        if not effectiveness_scores:
            return 1.0
            
        try:
            # Test if effectiveness scores are significantly greater than 0
            t_stat, p_value = stats.ttest_1samp(effectiveness_scores, 0)
            
            # One-tailed p-value (we care if it's greater than 0)
            if t_stat > 0:
                p_value = p_value / 2  # Convert to one-tailed p-value
            else:
                p_value = 1 - (p_value / 2)
                
            return p_value
        except Exception as e:
            logger.error(f"Error calculating overall p-value: {str(e)}")
            return 1.0
    
    def _generate_evidence_summary(self, aggregates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plain-language summary of the evidence."""
        profiles_with_significant_improvement = []
        profiles_with_improvement = []
        
        evidence_strength = "weak"
        largest_effect_size = 0.0
        
        # Analyze each profile type
        for profile, stats in aggregates.items():
            # Check if statistically significant
            is_significant = stats.get("overall_p_value", 1.0) < 0.05
            is_effective = stats.get("effectiveness_score", 0) > 0
            
            if is_effective:
                profiles_with_improvement.append(profile)
                
                if is_significant:
                    profiles_with_significant_improvement.append(profile)
            
            # Track largest effect size
            for metric, result in stats.get("statistical_significance", {}).items():
                effect_size = result.get("effect_size", 0)
                if effect_size > largest_effect_size:
                    largest_effect_size = effect_size
        
        # Determine strength of evidence
        if largest_effect_size > 0.8 and len(profiles_with_significant_improvement) > 0:
            evidence_strength = "strong"
        elif largest_effect_size > 0.5 and len(profiles_with_significant_improvement) > 0:
            evidence_strength = "moderate"
        elif largest_effect_size > 0.2 or len(profiles_with_improvement) > 0:
            evidence_strength = "suggestive"
        
        return {
            "profiles_with_significant_improvement": profiles_with_significant_improvement,
            "profiles_with_improvement": profiles_with_improvement,
            "evidence_strength": evidence_strength,
            "largest_effect_size": largest_effect_size,
            "summary": f"There is {evidence_strength} evidence that adaptive learning improves outcomes "
                     f"for {', '.join(profiles_with_improvement) if profiles_with_improvement else 'no'} profiles."
        } 