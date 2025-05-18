"""Scientific metrics for evaluating adaptive learning effectiveness.

This module contains functions for calculating scientific metrics to
evaluate the effectiveness of adaptive learning experiments compared
to control experiments. It provides statistical validity to the results.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import textstat
import Levenshtein
import json
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib # For Gestalt Pattern Matching
from synapz.core.models import Database # Added for MetricsCalculator
import uuid # Added for MetricsCalculator
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def calculate_readability_metrics(content: str) -> Dict[str, float]:
    """
    Calculate comprehensive readability metrics for content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Dictionary of readability metrics with scientific basis
    """
    if not content or len(content.strip()) < 10:
        logger.warning("Content too short for reliable metrics")
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog_index": 0.0,
            "gunning_fog": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "avg_syllables_per_word": 0.0,
            "avg_sentence_length": 0.0,
            "lexicon_count": 0,
            "sentence_count": 0,
            "syllable_count": 0
        }
    
    # Calculate basic metrics
    syllable_count = textstat.syllable_count(content)
    word_count = textstat.lexicon_count(content, removepunct=True)
    sentence_count = textstat.sentence_count(content)
    
    # Calculate average values
    avg_syllables = syllable_count / max(1, word_count)
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Calculate standard readability metrics
    metrics = {
        "flesch_reading_ease": float(textstat.flesch_reading_ease(content)),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(content)),
        "smog_index": float(textstat.smog_index(content)),
        "gunning_fog": float(textstat.gunning_fog(content)),
        "coleman_liau_index": float(textstat.coleman_liau_index(content)),
        "automated_readability_index": float(textstat.automated_readability_index(content)),
        "dale_chall_readability_score": float(textstat.dale_chall_readability_score(content)),
        "avg_syllables_per_word": float(avg_syllables),
        "avg_sentence_length": float(avg_sentence_length),
        "lexicon_count": int(word_count),
        "sentence_count": int(sentence_count),
        "syllable_count": int(syllable_count)
    }
    
    return metrics

def calculate_text_similarity(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate similarity between two texts using multiple measures.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary of similarity metrics
    """
    # Ensure inputs are strings to prevent errors with None or other types
    s1 = str(text1) if text1 is not None else ""
    s2 = str(text2) if text2 is not None else ""
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(s1, s2)
    
    # Normalize by maximum possible distance
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        levenshtein_similarity = 1.0  # Both empty strings
    else:
        levenshtein_similarity = 1 - (distance / max_len)
    
    # Calculate Jaccard similarity (word overlap)
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        jaccard_similarity = 1.0  # Both empty
    else:
        jaccard_similarity = intersection / union
    
    return {
        "levenshtein_similarity": float(levenshtein_similarity),
        "jaccard_similarity": float(jaccard_similarity),
        "levenshtein_distance": int(distance)
    }

def calculate_statistical_significance(
    adapted_values: List[float],
    control_values: List[float]
) -> Dict[str, float]:
    """
    Calculate statistical significance for a specific metric given lists of values.
    
    Args:
        adapted_values: List of metric values from adapted experiments
        control_values: List of metric values from control experiments
        
    Returns:
        Dictionary with p-value and effect size
    """
    # Ensure we have data and at least 2 samples in each group for t-test
    if not adapted_values or not control_values or len(adapted_values) < 2 or len(control_values) < 2:
        logger.warning(f"Insufficient data for statistical significance: adapted_n={len(adapted_values)}, control_n={len(control_values)}")
        return {
            "p_value": 1.0, # Default to 1.0 (not significant)
            "effect_size": 0.0,
            "is_significant": False,
            "mean_difference": 0.0,
            "percent_improvement": 0.0
        }
    
    # Calculate basic statistics
    adapted_mean = np.mean(adapted_values)
    control_mean = np.mean(control_values)
    mean_diff = adapted_mean - control_mean
    
    # Calculate percent improvement
    if control_mean != 0:
        # Ensure control_mean is not a list/array before abs()
        abs_control_mean = abs(float(np.mean(control_mean))) if isinstance(control_mean, (list, np.ndarray)) else abs(float(control_mean))
        if abs_control_mean > 1e-9: # Avoid division by very small number
             percent_improvement = (mean_diff / abs_control_mean) * 100
        else:
            percent_improvement = 0.0 # Or some indicator of undefined if control_mean is effectively zero
    else: # control_mean is exactly zero
        percent_improvement = 0.0 # Or np.inf if adapted_mean is positive, -np.inf if negative, or handle as special case

    # Run t-test for statistical significance
    try:
        # Ensure inputs to ttest_ind are array-like and have variance
        if np.var(adapted_values) == 0 and np.var(control_values) == 0 and adapted_mean == control_mean:
            # If both groups are identical constant values, p-value is 1 (or nan, depending on scipy version)
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = stats.ttest_ind(adapted_values, control_values, equal_var=False) # Welch's t-test by default if equal_var=False
        
        # Calculate Cohen's d for effect size
        # Using pooled standard deviation for Cohen's d
        # (n1-1)*s1^2 + (n2-1)*s2^2 / (n1+n2-2)
        # s_pooled = sqrt( ((len(A)-1)*var(A) + (len(B)-1)*var(B)) / (len(A)+len(B)-2) )
        # d = (mean(A) - mean(B)) / s_pooled
        
        var_adapted = np.var(adapted_values, ddof=1)
        var_control = np.var(control_values, ddof=1)
        n_adapted = len(adapted_values)
        n_control = len(control_values)

        # Pooled standard deviation
        pooled_std_numerator = (n_adapted - 1) * var_adapted + (n_control - 1) * var_control
        pooled_std_denominator = n_adapted + n_control - 2
        
        if pooled_std_denominator <= 0: # Avoid division by zero or negative if n_adapted + n_control <= 2
            pooled_std = 0.0
        else:
            pooled_std = np.sqrt(pooled_std_numerator / pooled_std_denominator)

        if pooled_std == 0:
            # Handle cases where pooled_std is zero (e.g., all values in both groups are identical)
            # If means are also identical, effect size is 0. Otherwise, it could be considered infinite.
            effect_size = 0.0 if adapted_mean == control_mean else np.inf 
        else:
            effect_size = abs(adapted_mean - control_mean) / pooled_std
            
        return {
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0, # Handle potential NaN p-value
            "effect_size": float(effect_size) if not np.isinf(effect_size) else 999.0, # Handle potential inf effect size
            "is_significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            "mean_difference": float(mean_diff),
            "percent_improvement": float(percent_improvement)
        }
    except Exception as e:
        logger.error(f"Error calculating significance: {str(e)} for adapted_values: {adapted_values}, control_values: {control_values}", exc_info=True)
        return {
            "p_value": 1.0,
            "effect_size": 0.0,
            "is_significant": False,
            "mean_difference": 0.0,
            "percent_improvement": 0.0,
            "error": str(e)
        }

def extract_pedagogy_tags(data: Union[List[str], str, None]) -> Set[str]:
    """Extracts a set of pedagogy tags from various input formats.

    Args:
        data: Can be a list of strings, a JSON string representing a list of strings,
              a single tag as a string, or None.

    Returns:
        A set of unique string tags.
    """
    if data is None:
        return set()
    
    if isinstance(data, list):
        # Filter out any None or empty string values from the list before converting to string
        return set(str(tag).strip() for tag in data if tag and str(tag).strip()) 

    if isinstance(data, str):
        stripped_data = data.strip()
        if not stripped_data: # Handle empty or whitespace-only string
            return set()
        try:
            # Attempt to parse as JSON list
            tags_list = json.loads(stripped_data)
            if isinstance(tags_list, list):
                # Filter out any None or empty string values from the parsed list
                return set(str(tag).strip() for tag in tags_list if tag and str(tag).strip())
            elif isinstance(tags_list, str): # Parsed to a JSON string e.g. json.loads('"tag1"')
                return {tags_list.strip()} if tags_list.strip() else set()
            else:
                # Parsed to something unexpected (e.g. number, dict). 
                # Fallback: treat original string as a single tag if it was non-empty.
                logger.warning(f"Pedagogy tags string '{stripped_data}' parsed to non-list/non-string JSON type: {type(tags_list)}. Treating original as single tag.")
                return {stripped_data}

        except json.JSONDecodeError:
            # Not a JSON string, treat the whole (stripped) string as a single tag
            return {stripped_data}
    
    logger.warning(f"extract_pedagogy_tags received unhandled type: {type(data)}. Value: {str(data)[:100]}")
    return set()

def generate_comprehensive_metrics(
    adapted_sessions: List[Dict[str, Any]],
    control_sessions: List[Dict[str, Any]],
    interactions_by_session: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Generate comprehensive metrics comparing adaptive vs control approaches.
    
    Args:
        adapted_sessions: List of adaptive teaching sessions
        control_sessions: List of control teaching sessions
        interactions_by_session: Dictionary mapping session IDs to interactions
        
    Returns:
        Comprehensive metrics report
    """
    # Extract readability metrics for final explanation in each session
    adapted_metrics = []
    control_metrics = []
    
    for session in adapted_sessions:
        session_id = session['id']
        interactions = interactions_by_session.get(session_id, [])
        
        if interactions:
            # Get the final explanation
            final_interaction = sorted(interactions, key=lambda x: x['turn_number'])[-1]
            explanation = final_interaction['explanation']
            
            # Calculate readability metrics
            metrics = calculate_readability_metrics(explanation)
            metrics['session_id'] = session_id
            adapted_metrics.append(metrics)
    
    for session in control_sessions:
        session_id = session['id']
        interactions = interactions_by_session.get(session_id, [])
        
        if interactions:
            # Get the final explanation
            final_interaction = sorted(interactions, key=lambda x: x['turn_number'])[-1]
            explanation = final_interaction['explanation']
            
            # Calculate readability metrics
            metrics = calculate_readability_metrics(explanation)
            metrics['session_id'] = session_id
            control_metrics.append(metrics)
    
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
        adapted_values = [m.get(metric, 0.0) for m in adapted_metrics]
        control_values = [m.get(metric, 0.0) for m in control_metrics]
        statistical_results[metric] = calculate_statistical_significance(
            adapted_values, control_values
        )
    
    # Calculate pedagogy tag usage
    adapted_tags = {}
    control_tags = {}
    
    for session in adapted_sessions:
        session_id = session['id']
        interactions = interactions_by_session.get(session_id, [])
        session_tags = extract_pedagogy_tags(interactions)
        
        # Merge with total counts
        for tag in session_tags:
            adapted_tags[tag] = adapted_tags.get(tag, 0) + 1
    
    for session in control_sessions:
        session_id = session['id']
        interactions = interactions_by_session.get(session_id, [])
        session_tags = extract_pedagogy_tags(interactions)
        
        # Merge with total counts
        for tag in session_tags:
            control_tags[tag] = control_tags.get(tag, 0) + 1
    
    # Generate summary
    summary = {
        "sample_size": {
            "adapted": len(adapted_metrics),
            "control": len(control_metrics)
        },
        "statistical_significance": statistical_results,
        "pedagogy_tags": {
            "adapted": adapted_tags,
            "control": control_tags
        },
        "adapted_metrics": adapted_metrics,
        "control_metrics": control_metrics
    }
    
    # Calculate overall effectiveness score (weighted average of statistical results)
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
        summary["effectiveness_score"] = float(effectiveness_score / total_weight)
    else:
        summary["effectiveness_score"] = 0.0
    
    return summary

def save_metrics_report(report: Dict[str, Any], output_path: Path) -> None:
    """
    Save metrics report to file.
    
    Args:
        report: Metrics report dictionary
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Metrics report saved to {output_path}")

# <<< COPIED MetricsCalculator class from evaluate.py >>>
class MetricsCalculator:
    """Calculate metrics comparing adaptive and control teaching sessions."""
    
    def __init__(self, db: Database):
        """Initialize with database connection."""
        self.db = db
    
    def _get_session_costs(self, session_id: str) -> float:
        """Calculate the total cost of a session."""
        interactions = self.db.get_session_history(session_id)
        costs = []
        for interaction in interactions:
            if not isinstance(interaction, dict):
                logger.error(f"Interaction is not a dict in _get_session_costs! Session ID: {session_id}, Type: {type(interaction)}, Value: {str(interaction)[:200]}")
                costs.append(0.0)
                continue
            costs.append(interaction.get('cost', 0.0) or 0.0)
        total_cost = sum(costs)
        return total_cost

    def _get_turn_by_turn_clarity_for_sessions(self, session_ids: List[str]) -> Dict[str, List[Optional[int]]]:
        """
        Fetches turn-by-turn clarity scores for a list of session IDs.

        Args:
            session_ids: A list of session IDs to process.

        Returns:
            A dictionary mapping each session ID to a list of clarity scores,
            ordered by turn number. Scores can be None if not recorded.
        """
        sessions_clarity_data: Dict[str, List[Optional[int]]] = {}
        for session_id in session_ids:
            interactions = self.db.get_session_history(session_id) # Already sorted by turn
            clarity_for_session: List[Optional[int]] = []
            if not interactions:
                logger.warning(f"No interactions found for session_id: {session_id} in _get_turn_by_turn_clarity_for_sessions")
            for i_idx, interaction in enumerate(interactions):
                if not isinstance(interaction, dict):
                    logger.error(f"Interaction item is not a dict in _get_turn_by_turn_clarity_for_sessions! Session ID: {session_id}, Index: {i_idx}, Type: {type(interaction)}")
                    clarity_for_session.append(None) # Append None if interaction data is malformed
                    continue
                
                clarity_score_raw = interaction.get("clarity_score")
                if clarity_score_raw is not None:
                    try:
                        clarity_for_session.append(int(clarity_score_raw))
                    except (ValueError, TypeError):
                        logger.error(f"Could not parse clarity_score '{clarity_score_raw}' to int for session {session_id}, turn {interaction.get('turn_number', i_idx + 1)}. Storing as None.")
                        clarity_for_session.append(None)
                else:
                    clarity_for_session.append(None) # Clarity not recorded for this turn
            sessions_clarity_data[session_id] = clarity_for_session
        return sessions_clarity_data

    def aggregate_turn_clarity_by_profile_and_type(
        self, 
        experiment_pair_results: List[Dict[str, Any]],
        all_profiles_data: Dict[str, Dict[str, Any]] # Pass all_profiles_data to get cognitive_style
    ) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
        """
        Aggregates turn-by-turn clarity scores, grouped by learner profile's cognitive_style 
        and experiment type (adaptive/control).

        Args:
            experiment_pair_results: List of dicts, where each dict contains info about an experiment pair,
                                     including 'learner_id', 'adaptive_session_id', 'control_session_id'.
            all_profiles_data: Dict mapping learner_id to their full profile data, to fetch cognitive_style.

        Returns:
            A nested dictionary:
            { 
                'profile_cognitive_style': {
                    'adaptive': {turn_number: [clarity_scores_for_this_turn_adaptive], ...},
                    'control': {turn_number: [clarity_scores_for_this_turn_control], ...}
                },
                ...
            }
            Clarity scores are floats (or will be converted). Turns are 1-indexed.
        """
        aggregated_data: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        session_ids_to_fetch = []
        session_id_to_info_map = {}

        for pair_result in experiment_pair_results:
            learner_id = pair_result.get("learner_id")
            adaptive_session_id = pair_result.get("adaptive_session_id")
            control_session_id = pair_result.get("control_session_id")

            if not learner_id:
                logger.warning(f"Skipping pair result due to missing learner_id: {pair_result}")
                continue
            
            profile_data = all_profiles_data.get(learner_id)
            if not profile_data:
                logger.warning(f"Skipping learner_id {learner_id} as profile data not found.")
                continue
            
            cognitive_style = profile_data.get("cognitive_style", "unknown_style")

            if adaptive_session_id:
                session_ids_to_fetch.append(adaptive_session_id)
                session_id_to_info_map[adaptive_session_id] = {"profile_style": cognitive_style, "type": "adaptive"}
            if control_session_id:
                session_ids_to_fetch.append(control_session_id)
                session_id_to_info_map[control_session_id] = {"profile_style": cognitive_style, "type": "control"}

        if not session_ids_to_fetch:
            return {}
            
        # Fetch all turn-by-turn clarity data in one go
        all_sessions_clarity = self._get_turn_by_turn_clarity_for_sessions(list(set(session_ids_to_fetch)))

        for session_id, clarity_scores in all_sessions_clarity.items():
            info = session_id_to_info_map.get(session_id)
            if not info:
                logger.warning(f"No info found for session_id {session_id} in map. Skipping.")
                continue

            profile_style = info["profile_style"]
            session_type = info["type"]

            for turn_idx, score in enumerate(clarity_scores):
                turn_number = turn_idx + 1 # 1-indexed turns
                if score is not None:
                    try:
                        aggregated_data[profile_style][session_type][turn_number].append(float(score))
                    except (ValueError, TypeError):
                         logger.error(f"Could not convert score '{score}' to float for session {session_id}, turn {turn_number}. Skipping score.")
        
        return aggregated_data

    def _calculate_comprehensive_clarity_stats(self, session_id: str) -> Dict[str, Any]:
        """Calculate comprehensive clarity statistics for a session."""
        history = self.db.get_session_history(session_id)
        clarity_scores = []
        for i_idx, i_val in enumerate(history):
            if not isinstance(i_val, dict):
                logger.error(f"Interaction item i_val is not a dict in _calculate_comprehensive_clarity_stats! Session ID: {session_id}, Index: {i_idx}, Type: {type(i_val)}, Value: {str(i_val)[:200]}")
                continue
            clarity_score = i_val.get("clarity_score") 
            if clarity_score is not None:
                try:
                    clarity_scores.append(int(clarity_score))
                except (ValueError, TypeError) as e_parse:
                    logger.error(f"Could not parse clarity_score '{clarity_score}' to int in _calculate_comprehensive_clarity_stats. Session ID: {session_id}, Index: {i_idx}. Error: {e_parse}")
        
        if not clarity_scores:
            return {
                "all_scores": [], "initial_score": None, "final_score": None,
                "absolute_improvement": None, "average_score": None
            }

        initial_score = clarity_scores[0] if clarity_scores else None
        final_score = clarity_scores[-1] if clarity_scores else None
        absolute_improvement = (final_score - initial_score) if initial_score is not None and final_score is not None else None
        average_score = sum(clarity_scores) / len(clarity_scores) if clarity_scores else None
        
        return {
            "all_scores": clarity_scores,
            "initial_score": initial_score,
            "final_score": final_score,
            "absolute_improvement": absolute_improvement,
            "average_score": average_score
        }

    def _calculate_comprehensive_readability_stats(self, session_id: str) -> Dict[str, Any]:
        """Calculate comprehensive readability statistics for a session."""
        history = self.db.get_session_history(session_id)
        
        all_flesch_scores = []
        final_flesch_score = None
        
        for i, turn_data in enumerate(history):
            if not isinstance(turn_data, dict):
                logger.error(f"Interaction item turn_data is not a dict in _calculate_comprehensive_readability_stats! Session ID: {session_id}, Index: {i}, Type: {type(turn_data)}, Value: {str(turn_data)[:200]}")
                continue
            explanation = turn_data.get("explanation")
            if explanation:
                # Need to ensure calculate_readability_metrics is available. It is in this file.
                readability = calculate_readability_metrics(explanation)
                all_flesch_scores.append(readability["flesch_reading_ease"])
                if i == len(history) - 1: # Last turn
                    final_flesch_score = readability["flesch_reading_ease"]
        
        average_flesch_score = sum(all_flesch_scores) / len(all_flesch_scores) if all_flesch_scores else None
        
        return {
            "all_flesch_scores": all_flesch_scores,
            "final_flesch_score": final_flesch_score,
            "average_flesch_score": average_flesch_score
        }

    def _calculate_comprehensive_content_stats(
        self, session_id: str, other_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive content statistics for a session, 
        and optionally compare with another session.
        """
        history = self.db.get_session_history(session_id)
        
        if not history:
            return {
                "final_explanation": None, "final_tags": [], 
                "text_similarity_vs_other": None, "tag_jaccard_vs_other": None,
                "error": "No history"
            }
            
        final_turn = history[-1]
        if not isinstance(final_turn, dict):
            logger.error(f"Interaction item final_turn is not a dict in _calculate_comprehensive_content_stats! Session ID: {session_id}, Type: {type(final_turn)}, Value: {str(final_turn)[:200]}")
            return { 
                "final_explanation": None, "final_tags": [], 
                "text_similarity_vs_other": None, "tag_jaccard_vs_other": None,
                "error": "final_turn was not a dict"
            }
            
        final_explanation = final_turn.get("explanation")
        raw_pedagogy_tags = final_turn.get("pedagogy_tags", [])
        pedagogy_tags_list = []
        if isinstance(raw_pedagogy_tags, list):
            pedagogy_tags_list = raw_pedagogy_tags
        elif raw_pedagogy_tags is not None:
            logger.warning(f"_calc_content_stats: pedagogy_tags from final_turn was {type(raw_pedagogy_tags)}, not list. Attempting to parse if string, else using empty. Value: {str(raw_pedagogy_tags)[:100]}")
            if isinstance(raw_pedagogy_tags, str):
                try:
                    parsed_tags = json.loads(raw_pedagogy_tags) 
                    if isinstance(parsed_tags, list):
                        pedagogy_tags_list = parsed_tags
                    else: 
                        pedagogy_tags_list = [raw_pedagogy_tags] if raw_pedagogy_tags else []
                except json.JSONDecodeError:
                    pedagogy_tags_list = [raw_pedagogy_tags] if raw_pedagogy_tags else [] 
        # extract_pedagogy_tags is available in this file.
        final_tags = set(extract_pedagogy_tags(pedagogy_tags_list)) 

        stats = {
            "final_explanation": str(final_explanation) if final_explanation is not None else None,
            "final_tags": list(final_tags),
            "text_similarity_vs_other": None,
            "tag_jaccard_vs_other": None
        }

        if other_session_id and final_explanation:
            other_history = self.db.get_session_history(other_session_id)
            if other_history:
                other_final_turn = other_history[-1]
                if not isinstance(other_final_turn, dict):
                    logger.error(f"Interaction item other_final_turn is not a dict in _calculate_comprehensive_content_stats! Session ID: {other_session_id}, Type: {type(other_final_turn)}, Value: {str(other_final_turn)[:200]}")
                    stats["text_similarity_vs_other"] = "Error: other_final_turn not a dict"
                    stats["tag_jaccard_vs_other"] = "Error: other_final_turn not a dict"
                else:
                    other_final_explanation = other_final_turn.get("explanation")
                    raw_other_pedagogy_tags = other_final_turn.get("pedagogy_tags")
                    other_pedagogy_tags_list = []
                    if isinstance(raw_other_pedagogy_tags, list):
                        other_pedagogy_tags_list = raw_other_pedagogy_tags
                    elif raw_other_pedagogy_tags is not None:
                        logger.warning(f"_calc_content_stats: pedagogy_tags from other_final_turn was {type(raw_other_pedagogy_tags)}, not list. Attempting to parse. Value: {str(raw_other_pedagogy_tags)[:100]}")
                        if isinstance(raw_other_pedagogy_tags, str):
                            try:
                                parsed_tags = json.loads(raw_other_pedagogy_tags)
                                if isinstance(parsed_tags, list):
                                    other_pedagogy_tags_list = parsed_tags
                                else:
                                    other_pedagogy_tags_list = [raw_other_pedagogy_tags] if raw_other_pedagogy_tags else []
                            except json.JSONDecodeError:
                                other_pedagogy_tags_list = [raw_other_pedagogy_tags] if raw_other_pedagogy_tags else []
                    
                    other_final_tags = set(extract_pedagogy_tags(other_pedagogy_tags_list))

                    if other_final_explanation:
                        str_final_explanation = str(final_explanation) if final_explanation is not None else ""
                        str_other_final_explanation = str(other_final_explanation) if other_final_explanation is not None else ""
                        # calculate_text_similarity is available in this file
                        similarity_metrics = calculate_text_similarity(str_final_explanation, str_other_final_explanation)
                        if isinstance(similarity_metrics, dict):
                            stats["text_similarity_vs_other"] = similarity_metrics.get("levenshtein_similarity")
                        else:
                            logger.error(f"Error: similarity_metrics was expected to be a dict but got {type(similarity_metrics)}. Value: {str(similarity_metrics)[:200]}")
                            stats["text_similarity_vs_other"] = None
                    
                    if final_tags and other_final_tags: # Ensure both are non-empty
                        intersection = len(final_tags.intersection(other_final_tags))
                        union = len(final_tags.union(other_final_tags))
                        stats["tag_jaccard_vs_other"] = intersection / union if union > 0 else 0.0
        return stats

    def compare_pedagogical_difference(self, adaptive_session_id: str, control_session_id: str) -> Dict[str, Any]:
        """Compare pedagogical differences between adaptive and control sessions."""
        try:
            adaptive_history = self.db.get_session_history(adaptive_session_id)
            control_history = self.db.get_session_history(control_session_id)
            
            if not adaptive_history or not control_history:
                logger.warning(f"Insufficient history for pedagogical comparison: A:{adaptive_session_id}, C:{control_session_id}")
                return {"error": "Insufficient history for comparison", "text_difference": None, "tag_difference": None, "abs_readability_difference": None, "signed_flesch_difference": None}

            adaptive_last = adaptive_history[-1]
            control_last = control_history[-1]
            
            adaptive_explanation = adaptive_last.get("explanation", "")
            control_explanation = control_last.get("explanation", "")
            
            adaptive_tags_raw = adaptive_last.get("pedagogy_tags", [])
            control_tags_raw = control_last.get("pedagogy_tags", [])

            adaptive_tags = set(extract_pedagogy_tags(adaptive_tags_raw))
            control_tags = set(extract_pedagogy_tags(control_tags_raw))
            
            text_similarity_metrics = calculate_text_similarity(adaptive_explanation, control_explanation)
            text_difference = 1.0 - text_similarity_metrics.get("levenshtein_similarity", 0.0) 
            
            tag_intersection = len(adaptive_tags.intersection(control_tags))
            tag_union = len(adaptive_tags.union(control_tags))
            tag_difference = 1.0 - (tag_intersection / tag_union if tag_union > 0 else 0.0)
            
            adaptive_readability = calculate_readability_metrics(adaptive_explanation)
            control_readability = calculate_readability_metrics(control_explanation)
            
            abs_readability_difference = abs(adaptive_readability.get("flesch_reading_ease", 0.0) - control_readability.get("flesch_reading_ease", 0.0))
            signed_flesch_difference = adaptive_readability.get("flesch_reading_ease", 0.0) - control_readability.get("flesch_reading_ease", 0.0)

            return {
                "text_difference": text_difference, 
                "tag_difference": tag_difference,   
                "abs_readability_difference": abs_readability_difference, 
                "signed_flesch_difference": signed_flesch_difference 
            }
        except Exception as e:
            logger.error(f"Error in compare_pedagogical_difference ({adaptive_session_id} vs {control_session_id}): {e}", exc_info=True)
            return {"error": str(e), "text_difference": None, "tag_difference": None, "abs_readability_difference": None, "signed_flesch_difference": None}

    def compare_clarity_improvements(self, adaptive_session_id: str, control_session_id: str) -> Dict[str, Any]:
        """Compare clarity improvements between adaptive and control sessions."""
        try:
            adaptive_history = self.db.get_session_history(adaptive_session_id)
            control_history = self.db.get_session_history(control_session_id)
            
            adaptive_clarity = [int(i["clarity_score"]) for i in adaptive_history if i.get("clarity_score") is not None]
            control_clarity = [int(i["clarity_score"]) for i in control_history if i.get("clarity_score") is not None]
            
            default_improvements = {
                "starting_clarity": None, "final_clarity": None,
                "absolute_improvement": None, "relative_improvement": None
            }
            default_return = {
                "adaptive_improvements": default_improvements.copy(),
                "control_improvements": default_improvements.copy(),
                "improvement_difference": None,
                "advantage": "tie",
                "error": None
            }

            if len(adaptive_clarity) < 2 or len(control_clarity) < 2:
                logger.warning(f"Not enough clarity scores for comparison ({adaptive_session_id} vs {control_session_id}). A:{len(adaptive_clarity)}, C:{len(control_clarity)}")
                default_return["error"] = "Insufficient clarity scores for comparison"
                return default_return
            
            adaptive_start = adaptive_clarity[0]
            adaptive_end = adaptive_clarity[-1]
            adaptive_abs_improvement = adaptive_end - adaptive_start
            adaptive_rel_improvement = (adaptive_abs_improvement / adaptive_start) if adaptive_start != 0 else 0
            
            control_start = control_clarity[0]
            control_end = control_clarity[-1]
            control_abs_improvement = control_end - control_start
            control_rel_improvement = (control_abs_improvement / control_start) if control_start != 0 else 0
            
            improvement_diff = adaptive_abs_improvement - control_abs_improvement
            advantage = "adaptive" if improvement_diff > 0 else "control" if improvement_diff < 0 else "tie"
            
            return {
                "adaptive_improvements": {
                    "starting_clarity": adaptive_start,
                    "final_clarity": adaptive_end,
                    "absolute_improvement": adaptive_abs_improvement,
                    "relative_improvement": adaptive_rel_improvement
                },
                "control_improvements": {
                    "starting_clarity": control_start,
                    "final_clarity": control_end,
                    "absolute_improvement": control_abs_improvement,
                    "relative_improvement": control_rel_improvement
                },
                "improvement_difference": improvement_diff,
                "advantage": advantage,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in compare_clarity_improvements ({adaptive_session_id} vs {control_session_id}): {e}", exc_info=True)
            default_return_error = default_return.copy()
            default_return_error["error"] = str(e)
            return default_return_error

    def save_comparison_metrics(
        self, 
        adaptive_session_id: str, 
        control_session_id: str,
        learner_id: str,
        concept_id: str,
        turns_per_session: int
    ) -> str:
        """Save comprehensive comparison metrics to the new experiment_metrics table."""
        experiment_pair_id = str(uuid.uuid4())

        adaptive_clarity_stats = self._calculate_comprehensive_clarity_stats(adaptive_session_id)
        adaptive_readability_stats = self._calculate_comprehensive_readability_stats(adaptive_session_id)
        adaptive_content_stats = self._calculate_comprehensive_content_stats(adaptive_session_id, control_session_id)
        adaptive_cost = self._get_session_costs(adaptive_session_id)

        control_clarity_stats = self._calculate_comprehensive_clarity_stats(control_session_id)
        control_readability_stats = self._calculate_comprehensive_readability_stats(control_session_id)
        control_content_stats = self._calculate_comprehensive_content_stats(control_session_id)
        control_cost = self._get_session_costs(control_session_id)

        key_metrics = {
            "experiment_pair_id": experiment_pair_id,
            "adaptive_session_id": adaptive_session_id,
            "control_session_id": control_session_id,
            "learner_id": learner_id,
            "concept_id": concept_id,
            "turns_per_session": turns_per_session,
            "initial_adaptive_clarity": adaptive_clarity_stats.get("initial_score"),
            "final_adaptive_clarity": adaptive_clarity_stats.get("final_score"),
            "adaptive_clarity_improvement": adaptive_clarity_stats.get("absolute_improvement"),
            "initial_control_clarity": control_clarity_stats.get("initial_score"),
            "final_control_clarity": control_clarity_stats.get("final_score"),
            "control_clarity_improvement": control_clarity_stats.get("absolute_improvement"),
            "clarity_improvement_difference": (adaptive_clarity_stats.get("absolute_improvement", 0) or 0) - 
                                              (control_clarity_stats.get("absolute_improvement", 0) or 0),
            "final_adaptive_flesch_ease": adaptive_readability_stats.get("final_flesch_score"),
            "final_control_flesch_ease": control_readability_stats.get("final_flesch_score"),
            "readability_diff_flesch_ease": (adaptive_readability_stats.get("final_flesch_score") if adaptive_readability_stats.get("final_flesch_score") is not None else 0.0) - \
                                              (control_readability_stats.get("final_flesch_score") if control_readability_stats.get("final_flesch_score") is not None else 0.0),
            "text_similarity_final_explanation": adaptive_content_stats.get("text_similarity_vs_other") if isinstance(adaptive_content_stats, dict) else None,
            "tag_jaccard_similarity_final": adaptive_content_stats.get("tag_jaccard_vs_other") if isinstance(adaptive_content_stats, dict) else None,
            "adaptive_session_cost": adaptive_cost,
            "control_session_cost": control_cost,
            "total_pair_cost": adaptive_cost + control_cost,
        }

        full_metrics_data = {
            "adaptive_session": {
                "session_id": adaptive_session_id,
                "clarity_stats": adaptive_clarity_stats,
                "readability_stats": adaptive_readability_stats,
                "content_stats": adaptive_content_stats,
                "cost": adaptive_cost
            },
            "control_session": {
                "session_id": control_session_id,
                "clarity_stats": control_clarity_stats,
                "readability_stats": control_readability_stats,
                "content_stats": control_content_stats,
                "cost": control_cost
            },
            "comparison_specific": {
                 "text_similarity_final_explanations": key_metrics.get("text_similarity_final_explanation"),
                 "tag_jaccard_similarity_final_explanations": key_metrics.get("tag_jaccard_similarity_final"),
            },
            "legacy_clarity_comparison": self.compare_clarity_improvements(adaptive_session_id, control_session_id),
            "legacy_pedagogical_difference": self.compare_pedagogical_difference(adaptive_session_id, control_session_id)
        }
        
        try:
            self.db.log_paired_experiment_metrics(
                experiment_pair_id=experiment_pair_id,
                adaptive_session_id=adaptive_session_id,
                control_session_id=control_session_id,
                learner_id=learner_id,
                concept_id=concept_id,
                turns_per_session=turns_per_session,
                key_metrics=key_metrics,
                full_metrics_data=full_metrics_data
            )
            logger.info(f"Successfully logged paired experiment metrics for pair ID: {experiment_pair_id}")
        except Exception as e_db_log:
            logger.error(f"Error logging paired experiment metrics for {adaptive_session_id} & {control_session_id}: {e_db_log}")
            raise
        return experiment_pair_id
# <<< END COPIED MetricsCalculator class >>> 