"""Tests for the budget tracking functionality."""

import os
import tempfile
import unittest
import sqlite3
from pathlib import Path

from synapz.core.budget import BudgetTracker, TokenUsage

class TestBudgetTracker(unittest.TestCase):
    """Test suite for BudgetTracker class."""
    
    def setUp(self):
        """Set up a temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.tracker = BudgetTracker(self.db_path, max_budget=10.0)
        
    def tearDown(self):
        """Clean up temporary files after tests."""
        os.unlink(self.db_path)
        
    def test_setup_db(self):
        """Test that database is properly set up with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Check journal mode is WAL
        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        self.assertEqual(journal_mode.upper(), "WAL")
        
        # Check table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='token_usage';")
        table_exists = cur.fetchone() is not None
        self.assertTrue(table_exists)
        
        conn.close()
        
    def test_project_cost(self):
        """Test cost projection calculation."""
        # Test gpt-4o-mini cost calculation
        prompt = "This is a test prompt with some tokens."
        max_tokens = 100
        model = "gpt-4o-mini"
        
        # Calculate expected cost
        # Input: ~8 tokens at $0.60/1M = ~$0.0000048
        # Output: 100 tokens at $2.40/1M = $0.00024
        # Total: ~$0.0002448
        cost = self.tracker.project_cost(prompt, max_tokens, model)
        
        # Check within reasonable range (exact token count may vary)
        self.assertGreater(cost, 0.0002)
        self.assertLess(cost, 0.0003)
        
    def test_log_usage(self):
        """Test logging usage to database."""
        model = "gpt-4o-mini"
        tokens_in = 100
        tokens_out = 50
        
        # Expected cost:
        # Input: 100 tokens at $0.60/1M = $0.00006
        # Output: 50 tokens at $2.40/1M = $0.00012
        # Total: $0.00018
        expected_cost = 0.00018
        
        # Log the usage
        self.tracker.log_usage(model, tokens_in, tokens_out)
        
        # Check database record
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT model, tokens_in, tokens_out, cost FROM token_usage")
        result = cur.fetchone()
        conn.close()
        
        self.assertEqual(result[0], model)
        self.assertEqual(result[1], tokens_in)
        self.assertEqual(result[2], tokens_out)
        self.assertAlmostEqual(result[3], expected_cost, places=6)
        
    def test_check_budget(self):
        """Test budget check functionality."""
        # Starting with max_budget of 10.0
        
        # Small cost should be allowed
        self.assertTrue(self.tracker.check_budget(1.0))
        
        # Log some usage that adds up to 9.0
        self.tracker.log_usage("gpt-4o-mini", 5000000, 2500000)  # $3.0 + $6.0 = $9.0
        
        # Small additional cost should still be allowed
        self.assertTrue(self.tracker.check_budget(0.5))
        
        # Cost that would exceed budget should be rejected
        self.assertFalse(self.tracker.check_budget(1.5))
        
if __name__ == "__main__":
    unittest.main() 