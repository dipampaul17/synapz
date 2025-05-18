#!/usr/bin/env python3
"""Example demonstrating how to use the LLMClient."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from synapz import DATA_DIR
from synapz.core import BudgetTracker, LLMClient

def main():
    """Run example code demonstrating LLMClient functionality."""
    # API key will be fetched from environment by LLMClient if not passed directly.
    # If OPENAI_API_KEY is not set, LLMClient will raise a ValueError.
    api_key = os.environ.get("OPENAI_API_KEY") 
    
    # Initialize components
    db_path = str(DATA_DIR / "example.db")
    budget_tracker = BudgetTracker(db_path=db_path, max_budget=1.0)  # $1 budget limit
    llm_client = LLMClient(budget_tracker=budget_tracker, api_key=api_key)
    
    # Display budget status
    print(f"Budget limit: ${budget_tracker.max_budget:.2f}")
    print(f"Current spend: ${budget_tracker.get_current_spend():.4f}")
    
    # Example 1: Basic completion
    print("\n--- Example 1: Basic Completion ---")
    try:
        result = llm_client.get_completion(
            system_prompt="You are a helpful assistant specializing in algebra.",
            user_prompt="Explain what a variable is in simple terms.",
            model="gpt-4o-mini",
            max_tokens=100
        )
        
        print(f"Response: {result['content']}")
        print(f"Tokens: {result['usage']['tokens_in']} in, {result['usage']['tokens_out']} out")
        print(f"Cost: ${result['usage']['cost']:.6f}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    # Example 2: JSON completion
    print("\n--- Example 2: JSON Completion ---")
    try:
        result = llm_client.get_json_completion(
            system_prompt="You are a helpful assistant that outputs JSON.",
            user_prompt="Create a JSON object with the following properties: name, type, and difficulty (1-5) for an algebra concept.",
            model="gpt-4o-mini",
            max_tokens=100
        )
        
        print(f"Response: {json.dumps(result['content'], indent=2)}")
        print(f"Tokens: {result['usage']['tokens_in']} in, {result['usage']['tokens_out']} out")
        print(f"Cost: ${result['usage']['cost']:.6f}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    # Example 3: Embedding
    print("\n--- Example 3: Embedding ---")
    try:
        result = llm_client.get_embedding(
            text="Variables in algebra are symbols that represent unknown values.",
            model="text-embedding-3-small"
        )
        
        # Only show first 5 dimensions to avoid cluttering the output
        print(f"Embedding (first 5 dims): {result['embedding'][:5]}...")
        print(f"Total dimensions: {len(result['embedding'])}")
        print(f"Tokens: {result['usage']['tokens']}")
        print(f"Cost: ${result['usage']['cost']:.6f}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    
    # Updated budget status
    print("\n--- Final Budget Status ---")
    print(f"Budget limit: ${budget_tracker.max_budget:.2f}")
    print(f"Current spend: ${budget_tracker.get_current_spend():.4f}")
    print(f"Remaining: ${budget_tracker.max_budget - budget_tracker.get_current_spend():.4f}")

if __name__ == "__main__":
    main() 