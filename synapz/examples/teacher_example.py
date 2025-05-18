#!/usr/bin/env python3
"""Example demonstrating how to use the TeacherAgent."""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

from synapz import DATA_DIR, PROMPTS_DIR
from synapz.core import BudgetTracker, LLMClient, Database, TeacherAgent


def create_sample_data():
    """Create sample data files for the example."""
    # Create directories
    profiles_dir = DATA_DIR / "profiles"
    concepts_dir = DATA_DIR / "concepts"
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    
    # Create a sample learner profile for a student with ADHD
    learner_id = "adhd_learner"
    adhd_profile = {
        "id": learner_id,
        "name": "Sam",
        "cognitive_style": "adhd",
        "attention_span": "short",
        "reading_level": "intermediate",
        "learning_preferences": ["visual", "interactive", "chunked"],
        "strengths": ["creativity", "pattern recognition"],
        "challenges": ["sustained attention", "sequential tasks"],
        "strategies": ["frequent breaks", "visual aids", "concrete examples"]
    }
    
    # Create a sample concept
    concept_id = "binary_search"
    binary_search = {
        "id": concept_id,
        "title": "Binary Search Algorithm",
        "difficulty": 3,
        "description": "A divide and conquer search algorithm that finds the position of a target value within a sorted array.",
        "keywords": ["algorithm", "search", "divide and conquer", "sorted array"],
        "prerequisites": ["basic array operations", "logarithmic complexity"]
    }
    
    # Create default prompts if they don't exist (copied from TeacherAgent._load_prompts for example setup)
    adaptive_prompt_path = PROMPTS_DIR / "adaptive_system.txt"
    if not adaptive_prompt_path.exists():
        with open(adaptive_prompt_path, "w") as f:
            f.write("You are Synapz, an adaptive teaching assistant specialized for neurodiverse learners.\n\n")
    
    control_prompt_path = PROMPTS_DIR / "control_system.txt"
    if not control_prompt_path.exists():
        with open(control_prompt_path, "w") as f:
            f.write("You are Synapz, a teaching assistant specialized for learners.\n\n")
    
    # Save to files
    with open(profiles_dir / f"{learner_id}.json", "w") as f:
        json.dump(adhd_profile, f, indent=2)
    
    with open(concepts_dir / f"{concept_id}.json", "w") as f:
        json.dump(binary_search, f, indent=2)
    
    return learner_id, concept_id


def main():
    """Run example code demonstrating TeacherAgent functionality."""
    # Load environment variables (which should include OPENAI_API_KEY)
    load_dotenv() # Ensure .env is loaded if present in project root
    
    # API key should be loaded by LLMClient from environment variables
    # or a .env file at the project root.
    # Remove hardcoded API key and writing to .env from this example.

    api_key_from_env = os.getenv("OPENAI_API_KEY")
    if not api_key_from_env:
        print("[ERROR] OPENAI_API_KEY not found in environment variables or .env file.")
        print("Please ensure your API key is set in a .env file at the project root or as an environment variable.")
        return
    
    # Create sample data
    learner_id, concept_id = create_sample_data()
    print(f"Created sample learner profile '{learner_id}' and concept '{concept_id}'")
    
    # Setup components
    db_path = DATA_DIR / "teacher_example.db"
    budget_tracker = BudgetTracker(db_path=str(db_path), max_budget=1.0)
    db = Database(db_path=str(db_path))
    # Ensure API_KEY is loaded from environment or defined
    api_key = os.environ.get("OPENAI_API_KEY") 
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        return
    llm_client = LLMClient(budget_tracker=budget_tracker, api_key=api_key)
    teacher = TeacherAgent(llm_client=llm_client, db=db, teacher_model_name="gpt-4o") # Explicitly set model
    
    # Create both adaptive and control sessions
    adaptive_session = teacher.create_session(learner_id, concept_id, is_adaptive=True)
    control_session = teacher.create_session(learner_id, concept_id, is_adaptive=False)
    
    print(f"Created adaptive session: {adaptive_session}")
    print(f"Created control session: {control_session}")
    
    # Example teaching flow for adaptive session
    print("\n--- ADAPTIVE TEACHING EXAMPLE ---")
    
    # First turn
    print("\nGenerating first explanation...\n")
    result1 = teacher.generate_explanation(adaptive_session)
    
    print(f"Teaching Strategy: {result1['teaching_strategy']}")
    print(f"Explanation: {result1['explanation']}")
    print(f"Pedagogy Tags: {', '.join(result1['pedagogy_tags'])}")
    print(f"Follow-up: {result1['follow_up']}")
    
    # Simulate clarity feedback (medium clarity)
    teacher.record_feedback(result1['interaction_id'], 3)
    print("\nRecorded clarity feedback: 3/5")
    
    # Second turn
    print("\nGenerating second explanation based on feedback...\n")
    result2 = teacher.generate_explanation(adaptive_session)
    
    print(f"Teaching Strategy: {result2['teaching_strategy']}")
    print(f"Explanation: {result2['explanation']}")
    print(f"Pedagogy Tags: {', '.join(result2['pedagogy_tags'])}")
    print(f"Follow-up: {result2['follow_up']}")
    
    # Control session example
    print("\n\n--- CONTROL TEACHING EXAMPLE ---")
    
    # First turn for control
    print("\nGenerating control explanation...\n")
    control_result = teacher.generate_explanation(control_session)
    
    print(f"Teaching Strategy: {control_result['teaching_strategy']}")
    print(f"Explanation: {control_result['explanation']}")
    print(f"Pedagogy Tags: {', '.join(control_result['pedagogy_tags'])}")
    
    # Show budget status
    print("\n--- Budget Status ---")
    current_spend = budget_tracker.get_current_spend()
    print(f"Budget limit: ${budget_tracker.max_budget:.2f}")
    print(f"Current spend: ${current_spend:.4f}")
    print(f"Remaining: ${budget_tracker.max_budget - current_spend:.4f}")


if __name__ == "__main__":
    main() 