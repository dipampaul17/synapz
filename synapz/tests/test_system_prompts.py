"""Tests for the system prompts functionality."""

import unittest
import os
import json
from pathlib import Path

from synapz import PROMPTS_DIR
from synapz.core import CognitiveProfile, ContentAdapter, ProfileManager, APIClient
from synapz.models.concepts import load_concept
from synapz.models.learner_profiles import get_profile_for_adaptation

class MockAPIClient:
    """Mock API client for testing without making actual API calls."""
    
    def __init__(self):
        """Initialize mock client."""
        pass
    
    def chat_completion(self, prompt, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
        """Mock API call that returns the prompt for inspection."""
        return {
            "content": "Mock response",
            "usage": {
                "tokens_in": 100,
                "tokens_out": 50,
                "cost": 0.0
            }
        }

class TestSystemPrompts(unittest.TestCase):
    """Test suite for the system prompts."""
    
    def setUp(self):
        """Set up test environment."""
        self.adaptive_prompt_path = PROMPTS_DIR / "adaptive_system.txt"
        self.control_prompt_path = PROMPTS_DIR / "control_system.txt"
        
        # Test data
        self.concept = load_concept("variables")
        self.adhd_profile = get_profile_for_adaptation(CognitiveProfile.ADHD.value)
        
        # Mock teaching context
        self.context = {
            "turn_number": 1,
            "previous_clarity": None,
            "interaction_history": []
        }
        
        # Create content adapter
        self.mock_api = MockAPIClient()
        self.profile_manager = ProfileManager()
        self.adapter = ContentAdapter(
            api_client=self.mock_api,
            profile_manager=self.profile_manager,
            prompts_dir=PROMPTS_DIR
        )
    
    def test_prompt_files_exist(self):
        """Test that the prompt files exist."""
        self.assertTrue(self.adaptive_prompt_path.exists(), 
                       f"Adaptive prompt file does not exist: {self.adaptive_prompt_path}")
        self.assertTrue(self.control_prompt_path.exists(), 
                       f"Control prompt file does not exist: {self.control_prompt_path}")
    
    def test_load_adaptive_prompt(self):
        """Test loading and formatting the adaptive prompt."""
        with open(self.adaptive_prompt_path, "r") as f:
            prompt_template = f.read()
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            learner_profile_json=json.dumps(self.adhd_profile, indent=2),
            concept_json=json.dumps(self.concept, indent=2),
            turn_number=self.context["turn_number"],
            previous_clarity=self.context["previous_clarity"] or "None",
            interaction_history=json.dumps(self.context["interaction_history"], indent=2)
        )
        
        # Check for required sections
        self.assertIn("LEARNER PROFILE:", formatted_prompt)
        self.assertIn("CONCEPT TO TEACH:", formatted_prompt)
        self.assertIn("TEACHING CONTEXT:", formatted_prompt)
        self.assertIn("ADAPTATION RULES:", formatted_prompt)
        self.assertIn("YOUR RESPONSE MUST BE STRUCTURED AS JSON:", formatted_prompt)
        
        # Check if JSON placeholders are properly filled
        self.assertIn(self.adhd_profile["id"], formatted_prompt)
        self.assertIn(self.concept["title"], formatted_prompt)
    
    def test_load_control_prompt(self):
        """Test loading and formatting the control prompt."""
        with open(self.control_prompt_path, "r") as f:
            prompt_template = f.read()
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            concept_json=json.dumps(self.concept, indent=2),
            turn_number=self.context["turn_number"],
            previous_clarity=self.context["previous_clarity"] or "None"
        )
        
        # Check for required sections
        self.assertIn("CONCEPT TO TEACH:", formatted_prompt)
        self.assertIn("TEACHING CONTEXT:", formatted_prompt)
        self.assertIn("YOUR TASK:", formatted_prompt)
        self.assertIn("YOUR RESPONSE MUST BE STRUCTURED AS JSON:", formatted_prompt)
        
        # Check if JSON placeholders are properly filled
        self.assertIn(self.concept["title"], formatted_prompt)
        self.assertIn("turn_number", formatted_prompt)
        
        # Verify the control prompt doesn't contain adaptive elements
        self.assertNotIn("LEARNER PROFILE:", formatted_prompt)
        self.assertNotIn("ADAPTATION RULES:", formatted_prompt)
    
    def test_adapter_get_system_prompt(self):
        """Test the ContentAdapter get_system_prompt method."""
        # Test adaptive prompt
        adaptive_prompt = self.adapter.get_system_prompt(
            experiment_type="adaptive",
            concept=self.concept,
            profile=self.adhd_profile,
            context=self.context
        )
        
        # Check adaptive prompt content
        self.assertIn("LEARNER PROFILE:", adaptive_prompt)
        self.assertIn("ADAPTATION RULES:", adaptive_prompt)
        self.assertIn(self.adhd_profile["id"], adaptive_prompt)
        self.assertIn(self.concept["title"], adaptive_prompt)
        
        # Test control prompt
        control_prompt = self.adapter.get_system_prompt(
            experiment_type="control",
            concept=self.concept,
            context=self.context
        )
        
        # Check control prompt content
        self.assertIn("CONCEPT TO TEACH:", control_prompt)
        self.assertIn("TEACHING CONTEXT:", control_prompt)
        self.assertNotIn("LEARNER PROFILE:", control_prompt)
        self.assertIn(self.concept["title"], control_prompt)
        
        # Test with default context
        default_prompt = self.adapter.get_system_prompt(
            experiment_type="control",
            concept=self.concept
        )
        
        self.assertIn("CONCEPT TO TEACH:", default_prompt)
        self.assertIn("turn_number", default_prompt)
        self.assertIn("previous_clarity", default_prompt)

if __name__ == "__main__":
    unittest.main() 