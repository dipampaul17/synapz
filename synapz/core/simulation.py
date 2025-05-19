"""Module for simulating student interactions and metrics."""

import random
import re
from typing import Dict, Any, Tuple, Optional, List

def _count_paragraphs(text: str) -> int:
    """Counts non-empty paragraphs."""
    return len([p for p in text.split('\n\n') if p.strip()])

def _count_list_items(text: str) -> int:
    """Counts simple list items (lines starting with *, -, or number.)."""
    return len(re.findall(r"^\s*([\*\-]\s+|\d+\.\s+).+", text, re.MULTILINE))

def _count_phrase_occurrences(text: str, phrases: List[str]) -> int:
    """Counts occurrences of any of the given phrases (case-insensitive)."""
    count = 0
    for phrase in phrases:
        count += len(re.findall(re.escape(phrase), text, re.IGNORECASE))
    return count

def simulate_student_clarity(
    explanation_text: str,
    simulation_config: Dict[str, Any]
) -> Tuple[int, int, int]:
    """
    Simulates a student's clarity rating based on the explanation text
    and a configuration defining heuristics.

    Args:
        explanation_text: The text of the teaching explanation.
        simulation_config: Configuration for simulation heuristics.
            Expected structure:
            {
                "base_initial_clarity": float,
                "base_improvement_points": float,
                "initial_clarity_factors": {
                    "long_paragraph_threshold_chars": int,
                    "long_paragraph_penalty_factor": float,
                    "list_item_bonus_factor": float,
                    "max_list_item_bonus": float,
                    "engagement_phrase_bonus_factor": float,
                    "max_engagement_phrase_bonus": float,
                    "engagement_phrases": List[str],
                    "overall_length_penalty_threshold_chars": int,
                    "overall_length_penalty_factor": float 
                },
                "improvement_factors": {
                    "low_initial_clarity_threshold": int,
                    "low_initial_clarity_bonus_factor": float,
                    "clarifying_phrase_bonus_factor": float,
                    "max_clarifying_phrase_bonus": float,
                    "clarifying_phrases": List[str],
                    "short_explanation_threshold_chars": int,
                    "short_explanation_improvement_penalty_factor": float,
                    "good_length_min_chars": int,
                    "good_length_max_chars": int,
                    "good_structure_improvement_bonus": float
                }
            }

    Returns:
        A tuple (initial_clarity, final_clarity, clarity_improvement),
        all as integers between 1 and 5 for clarity, and improvement
        being the difference.
    """
    text_len = len(explanation_text)
    initial_cfg = simulation_config.get("initial_clarity_factors", {})
    improve_cfg = simulation_config.get("improvement_factors", {})

    # Calculate Initial Clarity
    current_initial_clarity = float(simulation_config.get("base_initial_clarity", 3.0))

    # Penalty for long paragraphs (more relevant than overall paragraph count for ADHD)
    # Consider average length or presence of any very long paragraph.
    # For simplicity, let's check for any paragraph exceeding a threshold.
    paragraphs = [p for p in explanation_text.split('\n') if p.strip()] # Split by single newline for denser check
    num_very_long_paragraphs = 0
    for p in paragraphs:
        if len(p) > initial_cfg.get("long_paragraph_threshold_chars", 250):
            num_very_long_paragraphs +=1
            # Apply penalty per overly long paragraph or just once? Let's do per.
            current_initial_clarity += initial_cfg.get("long_paragraph_penalty_factor", -0.1)


    # Bonus for list items
    num_list_items = _count_list_items(explanation_text)
    list_bonus = num_list_items * initial_cfg.get("list_item_bonus_factor", 0.05)
    current_initial_clarity += min(list_bonus, initial_cfg.get("max_list_item_bonus", 0.5))

    # Bonus for engagement phrases
    engagement_phrases = initial_cfg.get("engagement_phrases", [])
    num_engagement_phrases = _count_phrase_occurrences(explanation_text, engagement_phrases)
    engagement_bonus = num_engagement_phrases * initial_cfg.get("engagement_phrase_bonus_factor", 0.1)
    current_initial_clarity += min(engagement_bonus, initial_cfg.get("max_engagement_phrase_bonus", 0.5))
    
    # Penalty for overall excessive length
    if text_len > initial_cfg.get("overall_length_penalty_threshold_chars", 1500):
        excess_chars = text_len - initial_cfg.get("overall_length_penalty_threshold_chars", 1500)
        current_initial_clarity += (excess_chars / 200.0) * initial_cfg.get("overall_length_penalty_factor", -0.2)


    # Cap initial clarity and convert to int for discrete rating
    initial_clarity_score = int(round(max(1.0, min(5.0, current_initial_clarity))))

    # Calculate Clarity Improvement
    current_improvement = float(simulation_config.get("base_improvement_points", 1.0))

    # Bonus if initial clarity was low
    if initial_clarity_score <= improve_cfg.get("low_initial_clarity_threshold", 2):
        current_improvement += improve_cfg.get("low_initial_clarity_bonus_factor", 0.5)

    # Bonus for clarifying phrases
    clarifying_phrases = improve_cfg.get("clarifying_phrases", [])
    num_clarifying_phrases = _count_phrase_occurrences(explanation_text, clarifying_phrases)
    clarifying_bonus = num_clarifying_phrases * improve_cfg.get("clarifying_phrase_bonus_factor", 0.2)
    current_improvement += min(clarifying_bonus, improve_cfg.get("max_clarifying_phrase_bonus", 0.4))

    # Penalty if explanation is too short
    if text_len < improve_cfg.get("short_explanation_threshold_chars", 200):
        current_improvement += improve_cfg.get("short_explanation_improvement_penalty_factor", -0.3)
    
    # Bonus for good structure (lists present) AND good length
    is_good_length = improve_cfg.get("good_length_min_chars", 500) <= text_len <= improve_cfg.get("good_length_max_chars", 2000)
    has_lists = num_list_items > 0
    if is_good_length and has_lists:
        current_improvement += improve_cfg.get("good_structure_improvement_bonus", 0.5)
    
    # Improvement should not be negative, and has some reasonable upper bound (e.g., 3 points)
    actual_improvement_points = max(0.0, min(current_improvement, 3.0)) 

    # Calculate final clarity
    final_clarity_score = int(round(max(1.0, min(5.0, float(initial_clarity_score) + actual_improvement_points))))
    
    # Ensure improvement value reflects the actual change in integer scores
    clarity_improvement_val = final_clarity_score - initial_clarity_score

    # Add a small random factor to avoid identical results for identical inputs if desired
    # This could be part of the config, e.g., "random_noise_factor": 0.1 (adds/subtracts 0 or 1 with some probability)
    # For now, keeping it deterministic for easier debugging of heuristics.

    return initial_clarity_score, final_clarity_score, clarity_improvement_val

if __name__ == '__main__':
    # Example configuration (should match reasoning_config.yaml structure)
    sample_config = {
        "base_initial_clarity": 3.0,
        "base_improvement_points": 1.0,
        "initial_clarity_factors": {
            "long_paragraph_threshold_chars": 200, # shorter for testing
            "long_paragraph_penalty_factor": -0.2,
            "list_item_bonus_factor": 0.1,
            "max_list_item_bonus": 0.5,
            "engagement_phrase_bonus_factor": 0.15,
            "max_engagement_phrase_bonus": 0.4,
            "engagement_phrases": ["what if", "can you see", "try to imagine", "for example:", "consider this:"],
            "overall_length_penalty_threshold_chars": 1000, # shorter for testing
            "overall_length_penalty_factor": -0.3 
        },
        "improvement_factors": {
            "low_initial_clarity_threshold": 2,
            "low_initial_clarity_bonus_factor": 0.5,
            "clarifying_phrase_bonus_factor": 0.25,
            "max_clarifying_phrase_bonus": 0.5,
            "clarifying_phrases": ["in other words", "to put it simply", "step-by-step", "to clarify", "essentially"],
            "short_explanation_threshold_chars": 150, # shorter for testing
            "short_explanation_improvement_penalty_factor": -0.4,
            "good_length_min_chars": 300, # shorter for testing
            "good_length_max_chars": 1200, # shorter for testing
            "good_structure_improvement_bonus": 0.6
        }
    }

    test_explanations = [
        ("Short and sweet.", "Very Short"),
        ("This is a basic explanation. It uses some clarifying phrases like 'to put it simply' and 'in other words'. For example: consider this example. It's not too long, not too short. It provides a step-by-step approach. What if we explore this more?", "Well-balanced Example"),
        ("This is an extremely long explanation designed to test the overall length penalty. It goes on and on, repeating information unnecessarily, without much structure or clear formatting. " * 20, "Very Long, Repetitive"),
        ("1. First point.\n2. Second point.\n   - Sub-point A\n   - Sub-point B\n3. Third point.\nThis explanation uses lists. Can you see how helpful they are? Try to imagine this concept without lists. It also tries to be engaging.", "List-heavy and Engaging"),
        ("This paragraph is intentionally made very very long to trigger the long paragraph penalty. It will keep going without any line breaks for more than two hundred characters, which should be enough to demonstrate the effect of the penalty on the initial clarity score if the configuration is set up appropriately. The idea is that a learner with ADHD might find such a large block of unbroken text difficult to process initially, regardless of the actual content quality within it. This is a pure structural heuristic.", "Single Long Paragraph")
    ]

    print("Testing simulate_student_clarity_v2:")
    for text, desc in test_explanations:
        init_c, final_c, improve_c = simulate_student_clarity(text, sample_config)
        print(f"\nDescription: {desc}")
        print(f"Text: '{text[:100]}...' (Length: {len(text)})")
        print(f"  Initial Clarity: {init_c}/5")
        print(f"  Final Clarity:   {final_c}/5")
        print(f"  Improvement:     {improve_c}") 