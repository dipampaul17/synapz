"""Tests for the database functionality."""

import unittest
import tempfile
import os
import json
import time
from pathlib import Path

from synapz.core import Database

class TestDatabase(unittest.TestCase):
    """Test suite for the SQLite database functionality."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(db_path=self.db_path)
        
        # Setup test data
        self.profile_id = "adhd"
        self.concept_id = "variables"
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_create_session(self):
        """Test creating a new teaching session."""
        # Create an adaptive session
        session_id = self.db.create_session(
            learner_id=self.profile_id,
            concept_id=self.concept_id,
            experiment_type="adaptive"
        )
        
        # Check session ID format
        self.assertTrue(session_id.startswith("session_"))
        self.assertIn(self.profile_id, session_id)
        self.assertIn(self.concept_id, session_id)
        
        # Create a control session
        control_id = self.db.create_session(
            learner_id=self.profile_id,
            concept_id=self.concept_id,
            experiment_type="control"
        )
        
        # Check we have unique IDs
        self.assertNotEqual(session_id, control_id)
    
    def test_log_interaction(self):
        """Test logging interactions within a session."""
        # Create a session
        session_id = self.db.create_session(
            learner_id=self.profile_id,
            concept_id=self.concept_id,
            experiment_type="adaptive"
        )
        
        # Log some interactions
        interaction1 = self.db.log_interaction(
            session_id=session_id,
            turn_number=1,
            explanation="Test explanation 1",
            clarity_score=4,
            teaching_strategy="chunking",
            pedagogy_tags=["visual", "example-based"],
            tokens_in=50,
            tokens_out=120
        )
        
        interaction2 = self.db.log_interaction(
            session_id=session_id,
            turn_number=2,
            explanation="Test explanation 2",
            clarity_score=None,  # Test None for optional field
            teaching_strategy="scaffolding",
            pedagogy_tags=["interactive", "question-based"],
            tokens_in=40,
            tokens_out=100
        )
        
        # Check interaction IDs format
        self.assertTrue(interaction1.startswith("interaction_"))
        self.assertTrue(interaction2.startswith("interaction_"))
        self.assertIn(session_id, interaction1)
        
        # Retrieve session history
        history = self.db.get_session_history(session_id)
        
        # Verify history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["turn_number"], 1)
        self.assertEqual(history[1]["turn_number"], 2)
        self.assertEqual(history[0]["explanation"], "Test explanation 1")
        self.assertEqual(history[1]["explanation"], "Test explanation 2")
        self.assertEqual(history[0]["clarity_score"], 4)
        self.assertIsNone(history[1]["clarity_score"])
        
        # Check pedagogy tags deserialization
        self.assertEqual(history[0]["pedagogy_tags"], ["visual", "example-based"])
        self.assertEqual(history[1]["pedagogy_tags"], ["interactive", "question-based"])
    
    def test_complete_session_and_metrics(self):
        """Test completing a session and logging metrics."""
        # Create two sessions (adaptive and control)
        adaptive_id = self.db.create_session(
            learner_id=self.profile_id,
            concept_id=self.concept_id,
            experiment_type="adaptive"
        )
        
        control_id = self.db.create_session(
            learner_id=self.profile_id,
            concept_id=self.concept_id,
            experiment_type="control"
        )
        
        # Log an interaction for each
        self.db.log_interaction(
            session_id=adaptive_id,
            turn_number=1,
            explanation="Adaptive explanation",
            clarity_score=5,
            teaching_strategy="visual",
            pedagogy_tags=["visual"],
            tokens_in=50,
            tokens_out=120
        )
        
        self.db.log_interaction(
            session_id=control_id,
            turn_number=1,
            explanation="Control explanation",
            clarity_score=3,
            teaching_strategy="standard",
            pedagogy_tags=["text"],
            tokens_in=50,
            tokens_out=110
        )
        
        # Complete both sessions
        self.db.complete_session(adaptive_id)
        self.db.complete_session(control_id)
        
        # Log metrics for adaptive session
        metrics = {
            "levenshtein_distance": 0.7,
            "readability_score": 85.3,
            "tag_similarity_score": 0.65,
            "clarity_improvement": 1.67,
            "baseline_comparison": 0.30,
            "additional_data": {
                "word_count": 120,
                "sentence_count": 10
            }
        }
        
        metrics_id = self.db.log_experiment_metrics(adaptive_id, metrics)
        self.assertTrue(metrics_id.startswith("metrics_"))
        
        # Test getting control/adaptive pairs
        pairs = self.db.get_control_adaptive_pairs()
        self.assertEqual(len(pairs), 1)
        
        # Check the pair matches our sessions
        adaptive, control = pairs[0]
        self.assertEqual(adaptive["id"], adaptive_id)
        self.assertEqual(control["id"], control_id)
        self.assertEqual(adaptive["experiment_type"], "adaptive")
        self.assertEqual(control["experiment_type"], "control")

if __name__ == "__main__":
    unittest.main() 