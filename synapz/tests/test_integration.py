"""Integration tests for the Synapz system."""

import unittest
import tempfile
import os
from pathlib import Path

from synapz.core import (
    BudgetTracker, 
    APIClient,
    CognitiveProfile,
    ProfileManager,
    ContentAdapter
)
from synapz.models.learner_profiles import (
    ADHD_PROFILE,
    get_profile_for_adaptation
)
from synapz import PROMPTS_DIR

class MockApiClient:
    """Mock API client for testing without making actual API calls."""
    
    def __init__(self, budget_tracker=None):
        """Initialize mock client."""
        self.last_prompt = None
        self.last_model = None
        self.last_max_tokens = None
    
    def chat_completion(self, prompt, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
        """Mock API call that returns the prompt for inspection."""
        self.last_prompt = prompt
        self.last_model = model
        self.last_max_tokens = max_tokens
        
        return {
            "content": "This is mock content for testing",
            "usage": {
                "tokens_in": 100,
                "tokens_out": 50,
                "cost": 0.0
            }
        }

class TestIntegration(unittest.TestCase):
    """Integration tests for the Synapz system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        
        # Set up components
        self.budget_tracker = BudgetTracker(self.db_path, max_budget=10.0)
        self.mock_api = MockApiClient(self.budget_tracker)
        self.profile_manager = ProfileManager()
        self.adapter = ContentAdapter(
            api_client=self.mock_api,
            profile_manager=self.profile_manager,
            prompts_dir=PROMPTS_DIR
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_detailed_profile_integration(self):
        """Test that detailed profiles are integrated into the adapter."""
        # Get a detailed profile for ADHD
        detailed_profile = get_profile_for_adaptation(CognitiveProfile.ADHD.value)
        
        # Generate content
        result = self.adapter.generate_adapted_content(
            topic="Test Topic",
            profile=CognitiveProfile.ADHD,
            learning_objective="Understanding test integration"
        )
        
        # Check that the detailed profile was used
        self.assertTrue(result["detailed_profile_used"])
        
        # Inspect the prompt to ensure it contains detailed profile info
        prompt = self.mock_api.last_prompt
        
        # Check for cognitive traits
        self.assertIn("ADHD COGNITIVE PROFILE:", prompt)
        self.assertIn("attention_span", prompt)
        self.assertIn("working_memory", prompt)
        
        # Check for modality preferences
        self.assertIn("MODALITY PREFERENCES:", prompt)
        self.assertIn("Primary:", prompt)
        
        # Check for strengths from the detailed profile
        strengths = ", ".join(detailed_profile["cognitive_traits"]["strengths"])
        self.assertIn(strengths, prompt)
        
        # Check that adaptation parameters were used
        self.assertIn(str(detailed_profile["adaptation"]["chunk_size"]), prompt)
        self.assertIn(str(detailed_profile["adaptation"]["example_count"]), prompt)
        
    def test_content_generation_for_all_profiles(self):
        """Test content generation for all profile types."""
        for profile in CognitiveProfile:
            # Generate content for this profile
            result = self.adapter.generate_adapted_content(
                topic="Test Topic",
                profile=profile,
                learning_objective="Understanding test profiles"
            )
            
            # Verify basic result structure
            self.assertEqual(result["profile"], profile.value)
            self.assertEqual(result["topic"], "Test Topic")
            self.assertEqual(result["is_control"], profile == CognitiveProfile.CONTROL)
            self.assertTrue(result["detailed_profile_used"])
            
            # Verify prompt contains profile-specific elements
            prompt = self.mock_api.last_prompt
            
            if profile == CognitiveProfile.ADHD:
                self.assertIn("ADHD COGNITIVE PROFILE:", prompt)
            elif profile == CognitiveProfile.DYSLEXIC:
                self.assertIn("DYSLEXIC COGNITIVE PROFILE:", prompt)
            elif profile == CognitiveProfile.VISUAL:
                self.assertIn("VISUAL LEARNER COGNITIVE PROFILE:", prompt)
            else:  # CONTROL
                self.assertIn("STANDARD LEARNER PROFILE:", prompt)
                
if __name__ == "__main__":
    unittest.main() 