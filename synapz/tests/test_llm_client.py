"""Tests for the LLMClient functionality."""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from synapz.core import BudgetTracker, LLMClient

class MockResponse:
    """Mock OpenAI API response."""
    
    def __init__(self, content, tokens_in=100, tokens_out=50):
        """Initialize with mock data."""
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content
                )
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out
        )
        
class MockEmbeddingResponse:
    """Mock OpenAI embedding response."""
    
    def __init__(self, embedding_dim=1536):
        """Initialize with mock data."""
        self.data = [
            MagicMock(
                embedding=[0.1] * embedding_dim
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=100
        )

class TestLLMClient(unittest.TestCase):
    """Test suite for the LLMClient functionality."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.budget_tracker = BudgetTracker(db_path=self.db_path, max_budget=10.0)
        self.llm_client = LLMClient(budget_tracker=self.budget_tracker)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('openai.OpenAI')
    def test_get_completion(self, mock_openai):
        """Test getting completion with mocked API."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_completion = MockResponse("This is a test response")
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Override the actual client
        self.llm_client.client = mock_client
        
        # Call method
        result = self.llm_client.get_completion(
            system_prompt="You are a helpful assistant.",
            user_prompt="Tell me a joke.",
            model="gpt-4o-mini"
        )
        
        # Verify results
        self.assertEqual(result["content"], "This is a test response")
        self.assertEqual(result["usage"]["tokens_in"], 100)
        self.assertEqual(result["usage"]["tokens_out"], 50)
        self.assertEqual(result["model"], "gpt-4o-mini")
        
        # Verify API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o-mini")
        self.assertEqual(len(call_args["messages"]), 2)
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][1]["role"], "user")
    
    @patch('openai.OpenAI')
    def test_get_json_completion(self, mock_openai):
        """Test getting JSON completion with mocked API."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # JSON response
        json_content = json.dumps({"name": "Test", "value": 123})
        mock_completion = MockResponse(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Override the actual client
        self.llm_client.client = mock_client
        
        # Call method
        result = self.llm_client.get_json_completion(
            system_prompt="You are a helpful assistant.",
            user_prompt="Return JSON with name and value.",
            model="gpt-4o-mini"
        )
        
        # Verify results
        self.assertIsInstance(result["content"], dict)
        self.assertEqual(result["content"]["name"], "Test")
        self.assertEqual(result["content"]["value"], 123)
        
        # Verify API was called with response_format parameter
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["response_format"], {"type": "json_object"})
    
    @patch('openai.OpenAI')
    def test_get_embedding(self, mock_openai):
        """Test getting embeddings with mocked API."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_embedding = MockEmbeddingResponse()
        mock_client.embeddings.create.return_value = mock_embedding
        
        # Override the actual client
        self.llm_client.client = mock_client
        
        # Call method
        result = self.llm_client.get_embedding(
            text="This is a test.",
            model="text-embedding-3-small"
        )
        
        # Verify results
        self.assertEqual(len(result["embedding"]), 1536)
        self.assertEqual(result["usage"]["tokens"], 100)
        self.assertEqual(result["model"], "text-embedding-3-small")
        
        # Verify API was called with correct parameters
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="This is a test."
        )
    
    @patch('openai.OpenAI')
    def test_budget_enforcement(self, mock_openai):
        """Test budget enforcement with mocked API."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock a response that would cost a lot
        mock_completion = MockResponse("Expensive response", tokens_in=5000, tokens_out=5000)
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Override the actual client
        self.llm_client.client = mock_client
        
        # Set a small budget threshold
        self.budget_tracker.max_budget = 0.01
        
        # Call method - should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.llm_client.get_completion(
                system_prompt="You are a helpful assistant." * 100,  # Make it long
                user_prompt="Generate a long response." * 100,
                model="gpt-4o-mini"
            )
        
        self.assertIn("Budget limit reached!", str(context.exception))

if __name__ == "__main__":
    unittest.main() 