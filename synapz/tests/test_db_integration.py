"""Tests for database integration with system prompts."""

import unittest
import tempfile
import os
import json
from pathlib import Path

from synapz import PROMPTS_DIR
from synapz.core import (
    Database, 
    ContentAdapter, 
    ProfileManager, 
    CognitiveProfile,
    TeacherAgent
)
from synapz.data import (
    create_unified_experiment_from_concept,
    store_initial_interaction
)
from synapz.models.concepts import load_concept
from synapz.models.learner_profiles import get_profile_for_adaptation

class MockAPIClient:
    """Mock API client for testing."""
    
    def __init__(self):
        """Initialize mock client."""
        self.last_prompt = None
        
    def chat_completion(self, prompt, model="gpt-4o-mini", max_tokens=500, temperature=0.7):
        """Mock chat completion that returns a sample teaching response."""
        self.last_prompt = prompt
        
        # Return a standard teaching response
        return {
            "content": json.dumps({
                "teaching_strategy": "Test strategy",
                "explanation": "This is a test explanation of the concept.",
                "pedagogy_tags": ["test", "example"],
                "follow_up": "How clear was this explanation on a scale of 0-5?"
            }),
            "usage": {
                "tokens_in": 100,
                "tokens_out": 50,
                "cost": 0.0
            }
        }

class TestDatabaseIntegration(unittest.TestCase):
    """Test suite for database integration with system prompts."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(db_path=self.db_path)
        
        # Create mock components
        self.mock_api = MockAPIClient()
        self.profile_manager = ProfileManager()
        self.content_adapter = ContentAdapter(
            api_client=self.mock_api,
            profile_manager=self.profile_manager,
            prompts_dir=PROMPTS_DIR
        )
        
        # Test data
        self.concept_id = "variables"
        self.profile_id = "adhd"
        
        # TeacherAgent instance
        self.teacher = TeacherAgent(llm_client=self.mock_api, db=self.db, teacher_model_name="gpt-4o")
        
        # Create sample data if it doesn't exist
        self.mock_api.last_prompt = self.content_adapter.get_system_prompt(
            experiment_type="adaptive",
            concept=load_concept(self.concept_id),
            profile=get_profile_for_adaptation(self.profile_id),
            context={
                "turn_number": 1,
                "previous_clarity": None,
                "interaction_history": []
            }
        )
        self.mock_api.chat_completion(self.mock_api.last_prompt)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_create_session_and_generate_content(self):
        """Test creating a session and generating content with system prompts."""
        # Create a session
        session_id, control_id = create_unified_experiment_from_concept(
            db=self.db,
            concept_id=self.concept_id,
            learner_profile_id=self.profile_id,
            experiment_type="adaptive",
            with_control=True
        )
        
        # Get concept and profile data
        concept = load_concept(self.concept_id)
        profile = get_profile_for_adaptation(self.profile_id)
        
        # Generate adaptive prompt
        adaptive_prompt = self.content_adapter.get_system_prompt(
            experiment_type="adaptive",
            concept=concept,
            profile=profile,
            context={
                "turn_number": 1,
                "previous_clarity": None,
                "interaction_history": []
            }
        )
        
        # Generate control prompt
        control_prompt = self.content_adapter.get_system_prompt(
            experiment_type="control",
            concept=concept,
            context={
                "turn_number": 1,
                "previous_clarity": None
            }
        )
        
        # Use the prompt to get content (using mock API)
        self.mock_api.last_prompt = adaptive_prompt
        response = self.mock_api.chat_completion(adaptive_prompt)
        
        # Parse the response
        content_data = json.loads(response["content"])
        
        # Store the interaction
        interaction_id = store_initial_interaction(
            db=self.db,
            session_id=session_id,
            content=content_data["explanation"],
            teaching_strategy=content_data["teaching_strategy"],
            pedagogy_tags=content_data["pedagogy_tags"],
            tokens_in=response["usage"]["tokens_in"],
            tokens_out=response["usage"]["tokens_out"]
        )
        
        # Get session history
        history = self.db.get_session_history(session_id)
        
        # Verify results
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["turn_number"], 1)
        self.assertEqual(history[0]["explanation"], content_data["explanation"])
        self.assertEqual(history[0]["teaching_strategy"], content_data["teaching_strategy"])
        self.assertEqual(history[0]["pedagogy_tags"], content_data["pedagogy_tags"])
        
        # Verify the control session exists
        self.assertIsNotNone(control_id)
        
        # Complete the sessions
        self.db.complete_session(session_id)
        self.db.complete_session(control_id)
        
        # Verify we can get the control/adaptive pairs
        pairs = self.db.get_control_adaptive_pairs()
        self.assertEqual(len(pairs), 1)
        
        # Log metrics
        metrics = {
            "levenshtein_distance": 0.75,
            "readability_score": 80.0,
            "tag_similarity_score": 0.5,
            "clarity_improvement": 1.5,
            "baseline_comparison": 0.25
        }
        
        metrics_id = self.db.log_experiment_metrics(session_id, metrics)
        self.assertTrue(metrics_id.startswith("metrics_"))

if __name__ == "__main__":
    unittest.main() 