"""Tests for the TeacherAgent functionality."""

import unittest
import tempfile
import os
import json
import uuid
from unittest.mock import patch, MagicMock
from pathlib import Path

from synapz.core import BudgetTracker, LLMClient, TeacherAgent, Database


class TestTeacherAgent(unittest.TestCase):
    """Test suite for the TeacherAgent functionality."""
    
    def setUp(self):
        """Set up test environment with temporary files and mocks."""
        # Create temp directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        
        # Create a test database
        self.db = Database(db_path=self.db_path)
        
        # Create a test budget tracker
        self.budget_tracker = BudgetTracker(db_path=self.db_path, max_budget=10.0)
        
        # Create a mock LLMClient
        api_key = "test_api_key"
        self.mock_llm_client = MockLLMClient(budget_tracker=self.budget_tracker, api_key=api_key)
        
        # Create sample data directories
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.profiles_dir = self.data_dir / "profiles"
        self.concepts_dir = self.data_dir / "concepts"
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.concepts_dir, exist_ok=True)
        
        # Create a sample learner profile
        self.learner_id = "test_learner"
        self.learner_profile = {
            "id": self.learner_id,
            "name": "Test Learner",
            "cognitive_style": "visual",
            "attention_span": "short",
            "reading_level": "intermediate",
            "learning_preferences": ["diagrams", "examples", "step-by-step"]
        }
        
        with open(self.profiles_dir / f"{self.learner_id}.json", "w") as f:
            json.dump(self.learner_profile, f)
        
        # Create a sample concept
        self.concept_id = "test_concept"
        self.concept = {
            "id": self.concept_id,
            "title": "Test Concept",
            "difficulty": 3,
            "description": "A test concept for unit testing",
            "keywords": ["test", "concept", "learning"]
        }
        
        with open(self.concepts_dir / f"{self.concept_id}.json", "w") as f:
            json.dump(self.concept, f)
        
        # Create the TeacherAgent with our mocks
        with patch('pathlib.Path') as mock_path:
            # Override Path to use our temp directory
            mock_path.return_value.parent.parent = self.data_dir.parent
            
            self.teacher = TeacherAgent(llm_client=self.mock_llm_client, db=self.db, teacher_model_name="gpt-4o")
            
            # Override prompt paths
            self.teacher.adaptive_prompt = "Adaptive prompt with {learner_profile_json}, {concept_json}, {turn_number}, {previous_clarity}, {interaction_history}"
            self.teacher.control_prompt = "Control prompt with {concept_json}, {turn_number}, {interaction_history}"
        
        # Create a patch for the _build_system_prompt method
        self.build_prompt_patcher = patch.object(
            self.teacher, '_build_system_prompt', 
            return_value="Mocked system prompt"
        )
        self.mock_build_prompt = self.build_prompt_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.build_prompt_patcher.stop()
        self.temp_dir.cleanup()
    
    def test_create_session(self):
        """Test creating a new teaching session."""
        # Generate unique session IDs to avoid conflicts
        with patch.object(self.db, 'create_session') as mock_create:
            adaptive_id = f"adaptive_{uuid.uuid4()}"
            control_id = f"control_{uuid.uuid4()}"
            mock_create.side_effect = [adaptive_id, control_id]
            
            # Create adaptive session
            session_id = self.teacher.create_session(self.learner_id, self.concept_id, True)
            self.assertEqual(session_id, adaptive_id)
            
            # Create control session
            control_id = self.teacher.create_session(self.learner_id, self.concept_id, False)
            self.assertEqual(control_id, control_id)
            
            # Verify correct calls
            calls = mock_create.call_args_list
            self.assertEqual(calls[0][0][2], "adaptive")
            self.assertEqual(calls[1][0][2], "control")
    
    def test_generate_explanation(self):
        """Test generating a teaching explanation."""
        # Create a session
        session_id = f"test_session_{uuid.uuid4()}"
        
        # Mock the get_session method
        with patch.object(self.db, 'get_session') as mock_get_session:
            mock_get_session.return_value = {
                "id": session_id,
                "learner_id": self.learner_id,
                "concept_id": self.concept_id,
                "experiment_type": "adaptive"
            }
            
            # Mock the get_session_history method
            with patch.object(self.db, 'get_session_history', return_value=[]):
                # Mock the log_interaction method
                with patch.object(self.db, 'log_interaction') as mock_log:
                    interaction_id = f"interaction_{uuid.uuid4()}"
                    mock_log.return_value = interaction_id
                    
                    # Mock the LLM response
                    mock_response = {
                        "content": {
                            "teaching_strategy": "Visual learning with diagrams",
                            "explanation": "This is a test explanation",
                            "pedagogy_tags": ["visual", "simplified", "examples"],
                            "follow_up": "How clear was this explanation on a scale of 1-5?"
                        },
                        "usage": {
                            "tokens_in": 50,
                            "tokens_out": 30
                        }
                    }
                    self.mock_llm_client.get_json_completion.return_value = mock_response
                    
                    # Generate an explanation
                    result = self.teacher.generate_explanation(session_id)
                    
                    # Verify LLM was called
                    self.mock_llm_client.get_json_completion.assert_called_once()
                    
                    # Verify response was processed correctly
                    self.assertEqual(result["interaction_id"], interaction_id)
                    self.assertEqual(result["explanation"], "This is a test explanation")
                    self.assertEqual(result["teaching_strategy"], "Visual learning with diagrams")
                    self.assertEqual(result["pedagogy_tags"], ["visual", "simplified", "examples"])
    
    def test_record_feedback(self):
        """Test recording clarity feedback."""
        # Mock the update_interaction_clarity method
        with patch.object(self.db, 'update_interaction_clarity', return_value=True) as mock_update:
            # Record feedback
            interaction_id = f"interaction_{uuid.uuid4()}"
            self.teacher.record_feedback(interaction_id, 4)
            
            # Verify method was called
            mock_update.assert_called_once_with(interaction_id, 4)
            
            # Test invalid score
            with self.assertRaises(ValueError):
                self.teacher.record_feedback(interaction_id, 6)
            

if __name__ == "__main__":
    unittest.main() 