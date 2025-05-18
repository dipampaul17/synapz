"""Test script for the student simulator."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

# Add parent directory to path to import from synapz
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from synapz.core.simulator import StudentSimulator
from synapz.core.budget import BudgetTracker
from synapz.core.llm_client import LLMClient
from synapz import DATA_DIR

# Mock LLM Client for testing
@patch("synapz.core.llm_client.LLMClient")
def test_heuristic_simulation(MockLLMClient):
    """Test heuristic simulation mode for speed and basic logic."""
    mock_llm_instance = MockLLMClient()
    # We don't expect LLM to be called, but simulator requires an LLMClient instance
    simulator = StudentSimulator(llm_client=mock_llm_instance, simulator_model_name="gpt-4o") 
    
    adhd_profile = {
        "id": "adhd_learner", "name": "Sam", "cognitive_style": "adhd",
        "learning_preferences": ["visual", "interactive"], "challenges": ["sustained attention"]
    }
    
    unfriendly_explanation = """In the context of computational systems and programming paradigms, variables represent memory allocations that correspond to identifiers which maintain state throughout the execution lifecycle of application instances. These symbolic references enable preservation and manipulation of data objects within the operational scope to which they are bound, subject to type constraints, lexical boundaries, and garbage collection protocols according to the implementation semantics of the language runtime."""

    print("\n=== TESTING HEURISTIC SIMULATION (use_llm=False) ===")
    result = simulator.generate_feedback(
        explanation=unfriendly_explanation,
        learner_profile=adhd_profile,
        pedagogy_tags=["complex"],
        use_llm=False
    )

    assert "clarity_rating" in result, "clarity_rating missing in heuristic output"
    assert "engagement_rating" in result, "engagement_rating missing in heuristic output"
    assert "detailed_feedback" in result, "detailed_feedback missing in heuristic output"
    assert result["detailed_feedback"] == "Feedback generated heuristically. LLM not used.", "Incorrect heuristic feedback message"
    assert "confusion_points" in result, "confusion_points missing"
    assert "helpful_elements" in result, "helpful_elements missing"
    assert "improvement_suggestions" in result, "improvement_suggestions missing"
    assert result["improvement_suggestions"] == ["N/A - Heuristic assessment"], "Incorrect improvement_suggestions"
    
    assert "heuristic_profile_match_score" in result, "heuristic_profile_match_score missing"
    assert isinstance(result["heuristic_profile_match_score"], float), "heuristic_profile_match_score not a float"
    assert 0.0 <= result["heuristic_profile_match_score"] <= 1.0, "heuristic_profile_match_score out of range 0-1"

    assert "final_clarity_score" in result, "final_clarity_score missing"
    assert isinstance(result["final_clarity_score"], float), "final_clarity_score should be float before rounding for DB"
    
    # For heuristic mode, clarity_rating and final_clarity_score should be based on heuristic_profile_match_score
    # final_clarity_score = 1 + (heuristic_profile_match_score * 4.0)
    # clarity_rating = int(round(final_clarity_score))
    expected_final_clarity = 1.0 + (result["heuristic_profile_match_score"] * 4.0)
    assert abs(result["final_clarity_score"] - expected_final_clarity) < 1e-9, \
        f"final_clarity_score ({result['final_clarity_score']}) doesn't match expected heuristic calculation ({expected_final_clarity})"
    assert result["clarity_rating"] == int(round(expected_final_clarity)), \
        f"clarity_rating ({result['clarity_rating']}) doesn't match expected int(round(heuristic))"

    # Check engagement proxy (crude, but test it)
    expected_engagement = int(round(expected_final_clarity / 5.0 * 3.0 + 1.0))
    assert result["engagement_rating"] == expected_engagement, "Engagement proxy calculation mismatch"

    print("Heuristic simulation test passed!")

# Mock LLM Client for testing LLM path
@patch("synapz.core.llm_client.LLMClient")
def test_llm_simulation(MockLLMClient):
    """Test LLM-based simulation mode with mocked LLM responses."""
    mock_llm_instance = MockLLMClient()
    
    # Configure the mock generate_text method
    mock_llm_clarity = 4
    mock_llm_engagement = 5
    mock_response_json = json.dumps({
        "clarity_rating": mock_llm_clarity,
        "engagement_rating": mock_llm_engagement,
        "detailed_feedback": "This was very clear and engaging! Loved the examples.",
        "confusion_points": ["The third paragraph was a bit dense."],
        "helpful_elements": ["Examples were great", "Good structure"],
        "improvement_suggestions": ["Maybe add a diagram for the third paragraph concept."]
    })
    mock_llm_instance.generate_text.return_value = mock_response_json
    
    simulator = StudentSimulator(llm_client=mock_llm_instance, simulator_model_name="gpt-4o")
    
    adhd_profile = {
        "id": "adhd_learner", "name": "Sam", "cognitive_style": "adhd",
        "learning_preferences": ["visual", "interactive"], "challenges": ["sustained attention"]
    }
    friendly_explanation = "Short, sweet, and to the point with examples! 1. Step one. 2. Step two. Example: A = B."

    print("\n=== TESTING LLM SIMULATION (use_llm=True) ===")
    result = simulator.generate_feedback(
        explanation=friendly_explanation,
        learner_profile=adhd_profile,
        pedagogy_tags=["basic", "examples"],
        use_llm=True
    )

    # Assert LLM was called (optional, implicitly tested by non-default feedback)
    mock_llm_instance.generate_text.assert_called_once()
    called_prompt = mock_llm_instance.generate_text.call_args[1]['prompt']
    assert "HEURISTIC ASSESSMENT" in called_prompt, "Heuristic data not found in LLM prompt"
    assert "learner_profile_details" in called_prompt, "Learner profile not in prompt keys (check formatting)" 
    # Check if some heuristic values are in the prompt string by checking their placeholders
    assert "{readability_score}".format(readability_score=0) not in called_prompt # after formatting, placeholders should be gone
    assert "profile_match_score: " in called_prompt.lower() # check if a value was passed

    assert result["clarity_rating"] == mock_llm_clarity, "LLM clarity_rating mismatch"
    assert result["engagement_rating"] == mock_llm_engagement, "LLM engagement_rating mismatch"
    assert result["detailed_feedback"] == "This was very clear and engaging! Loved the examples.", "Detailed feedback mismatch"
    assert result["confusion_points"] == ["The third paragraph was a bit dense."], "Confusion points mismatch"
    assert result["helpful_elements"] == ["Examples were great", "Good structure"], "Helpful elements mismatch"
    assert result["improvement_suggestions"] == ["Maybe add a diagram for the third paragraph concept."], "Improvement suggestions mismatch"

    assert "heuristic_profile_match_score" in result
    heuristic_match = result["heuristic_profile_match_score"]
    assert isinstance(heuristic_match, float)

    assert "final_clarity_score" in result
    final_clarity = result["final_clarity_score"]
    assert isinstance(final_clarity, float) # It's rounded to 2 decimal places in the return dict

    # Test blending logic: 0.4 * heuristic_1_to_5 + 0.6 * llm_clarity
    heuristic_1_to_5 = 1.0 + (heuristic_match * 4.0)
    expected_final_clarity = (0.4 * heuristic_1_to_5) + (0.6 * mock_llm_clarity)
    expected_final_clarity = max(1.0, min(5.0, expected_final_clarity)) # Clamping
    
    assert abs(final_clarity - expected_final_clarity) < 1e-2, \
        f"Final clarity ({final_clarity}) doesn't match expected blended score ({expected_final_clarity}) for h_match={heuristic_match}, h_1_5={heuristic_1_to_5}, llm_clarity={mock_llm_clarity}"

    print("LLM simulation test passed!")

def main():
    """Run simulator tests."""
    # Create data directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Running heuristic simulation test...")
    test_heuristic_simulation()
    
    print("\nRunning LLM simulation test...")
    test_llm_simulation()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 