"""Core teaching agent with adaptive and control modes."""

import json
import uuid
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
import logging

# Use PROMPTS_DIR and DATA_DIR from synapz package
from synapz import PROMPTS_DIR, DATA_DIR 
from .llm_client import LLMClient
from .models import Database
from .budget import BudgetTracker

# Define prompt paths - Now using imported PROMPTS_DIR
# PROMPTS_DIR = Path(__file__).parent.parent / "prompts" # Old way

# Setup logger for this module
logger = logging.getLogger(__name__)


class TeachingResponse(BaseModel):
    """Structured output for teaching responses."""
    teaching_strategy: str = Field(..., description="Explanation of teaching approach")
    explanation: str = Field(..., description="The actual teaching content")
    pedagogy_tags: List[str] = Field(..., description="Tags describing pedagogy used")
    follow_up: str = Field(..., description="Question asking for clarity rating")


class TeacherAgent:
    """Adaptive teaching agent that tailors explanations to cognitive profiles."""
    
    def __init__(self, llm_client: LLMClient, db: Database, teacher_model_name: str = "gpt-4o"):
        """Initialize with LLM client, database connection, and teacher model name."""
        self.llm_client = llm_client
        self.db = db
        self.teacher_model_name = teacher_model_name
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load system prompts from files."""
        # Ensure prompts directory exists
        os.makedirs(PROMPTS_DIR, exist_ok=True) # Use imported PROMPTS_DIR
        
        # Define prompt paths
        adaptive_path = PROMPTS_DIR / "adaptive_system.txt" # Use imported PROMPTS_DIR
        control_path = PROMPTS_DIR / "control_system.txt" # Use imported PROMPTS_DIR
        
        # Create default prompts if they don't exist
        if not adaptive_path.exists():
            with open(adaptive_path, "w") as f:
                f.write("You are Synapz, an adaptive teaching assistant specialized for neurodiverse learners.\n\n"
                        "LEARNER PROFILE:\n{learner_profile_json}\n\n"
                        "CONCEPT TO TEACH:\n{concept_json}\n\n"
                        "TEACHING HISTORY:\n{interaction_history}\n\n"
                        "CURRENT TURN: {turn_number}\n"
                        "PREVIOUS CLARITY RATING: {previous_clarity}\n\n"
                        "Adapt your teaching strategy based on the learner's cognitive profile and their clarity ratings. "
                        "Use different approaches for different cognitive styles.\n\n"
                        "Respond in JSON format with the following structure:\n"
                        "- teaching_strategy: Brief explanation of your teaching approach\n"
                        "- explanation: The teaching content\n"
                        "- pedagogy_tags: List of tags describing the pedagogical techniques used\n"
                        "- follow_up: A question asking the learner to rate the clarity from 1-5")
        
        if not control_path.exists():
            with open(control_path, "w") as f:
                f.write("You are Synapz, a teaching assistant specialized for learners.\n\n"
                        "CONCEPT TO TEACH:\n{concept_json}\n\n"
                        "TEACHING HISTORY:\n{interaction_history}\n\n"
                        "CURRENT TURN: {turn_number}\n\n"
                        "Teach the concept using a consistent approach regardless of the learner's response or clarity ratings.\n\n"
                        "Respond in JSON format with the following structure:\n"
                        "- teaching_strategy: Brief explanation of your teaching approach\n"
                        "- explanation: The teaching content\n"
                        "- pedagogy_tags: List of tags describing the pedagogical techniques used\n"
                        "- follow_up: A question asking the learner to rate the clarity from 1-5")
        
        # Load prompt templates
        with open(adaptive_path, "r") as f:
            self.adaptive_prompt = f.read()
        
        with open(control_path, "r") as f:
            self.control_prompt = f.read()
    
    def create_session(self, learner_id: str, concept_id: str, 
                       is_adaptive: bool = True) -> str:
        """Create a new teaching session."""
        experiment_type = "adaptive" if is_adaptive else "control"
        return self.db.create_session(learner_id, concept_id, experiment_type)
    
    def _build_system_prompt(self, session_id: str, is_adaptive: bool = True) -> str:
        """Build system prompt with context from session history."""
        # Get session info
        session_data = self.db.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
            
        learner_id, concept_id = session_data["learner_id"], session_data["concept_id"]
        
        # Load learner profile and concept
        # Use imported DATA_DIR
        # data_dir = Path(__file__).parent.parent / "data" # Old way
        profiles_dir = DATA_DIR / "profiles" # Use imported DATA_DIR
        concepts_dir = DATA_DIR / "concepts" # Use imported DATA_DIR
        
        # Ensure directories exist
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(concepts_dir, exist_ok=True)
        
        # Load profile and concept
        profile_path = profiles_dir / f"{learner_id}.json"
        concept_path = concepts_dir / f"{concept_id}.json"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Learner profile not found: {profile_path}")
        
        if not concept_path.exists():
            raise FileNotFoundError(f"Concept not found: {concept_path}")
        
        with open(profile_path, "r") as f:
            learner_profile = json.load(f)
        
        with open(concept_path, "r") as f:
            concept = json.load(f)
        
        # Get previous interactions
        interactions = self.db.get_session_history(session_id)
        
        # Determine turn number and previous clarity
        turn_number = len(interactions) + 1
        previous_clarity = None
        if interactions:
            previous_clarity = interactions[-1].get("clarity_score")
        
        # Format interaction history for context
        interaction_history = ""
        if interactions:
            for i, interaction in enumerate(interactions[-3:], 1):  # Only use last 3
                interaction_history += f"Turn {i}:\n"
                interaction_history += f"Strategy: {interaction.get('teaching_strategy', 'N/A')}\n"
                interaction_history += f"Tags: {', '.join(interaction.get('pedagogy_tags', []))}\n"
                interaction_history += f"Clarity: {interaction.get('clarity_score', 'N/A')}/5\n\n"
        
        # Choose prompt template based on mode
        template = self.adaptive_prompt if is_adaptive else self.control_prompt
        
        # Fill in template placeholders
        prompt = template.replace("{learner_profile_json}", json.dumps(learner_profile, indent=2))
        prompt = prompt.replace("{concept_json}", json.dumps(concept, indent=2))
        prompt = prompt.replace("{turn_number}", str(turn_number))
        prompt = prompt.replace("{previous_clarity}", str(previous_clarity) if previous_clarity is not None else "None")
        prompt = prompt.replace("{interaction_history}", interaction_history)
        
        return prompt
    
    def generate_explanation(self, session_id: str) -> Dict[str, Any]:
        """Generate a teaching explanation for the current turn."""
        # Get session type
        session_data = self.db.get_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")
            
        is_adaptive = (session_data["experiment_type"] == "adaptive")
        
        # Build system prompt
        system_prompt = self._build_system_prompt(session_id, is_adaptive)
        
        # Simple user prompt - the system prompt has all the context
        user_prompt = "Please teach this concept based on the learner's profile and history."
        
        # Get completion with structured output
        result = self.llm_client.get_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.teacher_model_name,  # Use the configured model
            temperature=0.7,
            max_tokens=1500  # Increased max_tokens
        )
        
        # Parse response
        response_content = result["content"]
        
        # Validate response structure
        required_fields = ["teaching_strategy", "explanation", "pedagogy_tags", "follow_up"]
        for field in required_fields:
            if field not in response_content:
                # Log the problematic response for debugging
                logger.error(f"Missing field '{field}' in LLM response. Response content: {response_content}")
                raise ValueError(f"Missing field in response: {field}")

        # Sanitize LLM outputs before logging and returning
        exp_val = response_content.get("explanation")
        ts_val = response_content.get("teaching_strategy")
        pt_val = response_content.get("pedagogy_tags")
        follow_up_val = response_content.get("follow_up")

        if not isinstance(exp_val, str):
            logger.warning(f"LLM returned non-string for explanation (type: {type(exp_val)}). Forcing to string. Value: {str(exp_val)[:100]}...")
            exp_val = str(exp_val)
        if not isinstance(ts_val, str):
            logger.warning(f"LLM returned non-string for teaching_strategy (type: {type(ts_val)}). Forcing to string. Value: {str(ts_val)[:100]}...")
            ts_val = str(ts_val)
        if not isinstance(follow_up_val, str):
            logger.warning(f"LLM returned non-string for follow_up (type: {type(follow_up_val)}). Forcing to string. Value: {str(follow_up_val)[:100]}...")
            follow_up_val = str(follow_up_val)

        if not isinstance(pt_val, list):
            logger.warning(f"LLM returned non-list for pedagogy_tags (type: {type(pt_val)}). Attempting to convert. Value: {str(pt_val)[:100]}...")
            if isinstance(pt_val, str):
                try:
                    parsed_pt = json.loads(pt_val)
                    if isinstance(parsed_pt, list):
                        pt_val = [str(item) for item in parsed_pt]
                    else:
                        pt_val = [str(pt_val)] # Treat as single item list if parsed but not list
                except json.JSONDecodeError:
                    pt_val = [str(pt_val)] # If string but not JSON list, make it a list of that string
            else: # If some other type (e.g., dict, int)
                pt_val = [str(pt_val)] # Convert to string and wrap in a list
        else: # It is a list, ensure all items are strings
            pt_val = [str(item) for item in pt_val]
        
        # Log interaction (without clarity score yet)
        interactions = self.db.get_session_history(session_id)
        turn_number = len(interactions) + 1
        
        interaction_id = self.db.log_interaction(
            session_id=session_id,
            turn_number=turn_number,
            explanation=exp_val,
            clarity_score=None,  # Will be updated later
            teaching_strategy=ts_val,
            pedagogy_tags=pt_val,
            tokens_in=result["usage"]["tokens_in"],
            tokens_out=result["usage"]["tokens_out"],
            cost=result["usage"]["cost"]  # Corrected: access cost via result["usage"]["cost"]
        )
        
        return {
            "interaction_id": interaction_id,
            "explanation": exp_val,
            "follow_up": follow_up_val,
            "pedagogy_tags": pt_val,
            "teaching_strategy": ts_val
        }
    
    def get_reasoning_response(
        self, 
        prompt_text: str, 
        max_tokens: int,
        model_name: Optional[str] = None, # Allow overriding model for this specific call
        temperature: Optional[float] = None # Allow overriding temperature
    ) -> Tuple[Dict[str, Any], int, int, float]:
        """
        Gets a structured response from the LLM based on a provided prompt.
        This method is designed for experiments like the reasoning_experiment
        where the prompt is constructed externally.

        Args:
            prompt_text: The full system prompt to send to the LLM.
            max_tokens: The maximum number of tokens for the completion.
            model_name: Optional model name to use, defaults to self.teacher_model_name.
            temperature: Optional temperature to use, defaults to a standard value (e.g. 0.5).

        Returns:
            A tuple containing:
                - response_data (dict): The parsed JSON response from the LLM.
                - tokens_in (int): Number of input tokens used.
                - tokens_out (int): Number of output tokens generated.
                - cost (float): The cost of the API call.
        """
        user_prompt = "Provide your response as a valid JSON object, based on the detailed instructions and context given in the system prompt." # Modified to include JSON instruction
        
        current_model = model_name if model_name else self.teacher_model_name
        current_temp = temperature if temperature is not None else 0.5 # Default temp if not specified

        try:
            result = self.llm_client.get_json_completion(
                system_prompt=prompt_text,
                user_prompt=user_prompt, 
                model=current_model,
                temperature=current_temp,
                max_tokens=max_tokens
            )
            
            response_content = result.get("content", {})
            usage_data = result.get("usage", {"tokens_in": 0, "tokens_out": 0, "cost": 0.0})
            
            # Ensure response_content is a dict
            if not isinstance(response_content, dict):
                logger.error(f"LLM response content is not a dict. Type: {type(response_content)}. Content: {str(response_content)[:200]}")
                # Depending on how strict we want to be, we could raise an error or return a default error structure
                # For reasoning_experiment, it expects a dict. If baseline returns string, it's handled there.
                # If it's for reasoning and not a dict, it is an issue.
                # For now, let it pass, reasoning_experiment.py checks its type.
                # Or, we can try to parse it if it's a string that looks like JSON.
                if isinstance(response_content, str):
                    try:
                        response_content = json.loads(response_content)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse string response_content to dict. Content: {str(response_content)[:200]}")
                        # Fallback: wrap in a dict if it's a string and baseline might produce this.
                        # However, get_json_completion should ideally always return a dict if successful.
                        # This path indicates an issue with get_json_completion or the LLM response.
                        # For now, we'll assume get_json_completion handles the JSON parsing.
                        # If it fails, it should raise an error or return a dict with an error field.
                        # Let's assume `result["content"]` IS the parsed dict.
                        pass # Rely on reasoning_experiment.py to handle non-dict if it's from baseline

            return (
                response_content,
                usage_data.get("tokens_in", 0),
                usage_data.get("tokens_out", 0),
                usage_data.get("cost", 0.0)
            )
        except Exception as e:
            logger.error(f"Error in get_reasoning_response: {e}", exc_info=True)
            # Return an error structure or re-raise
            # For now, re-raise to make it visible to the caller
            raise

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Counts the number of tokens in a given text for a specified model."""
        # This should ideally delegate to self.llm_client if it has a tokenizer
        if hasattr(self.llm_client, 'count_tokens'):
            return self.llm_client.count_tokens(text, model if model else self.teacher_model_name)
        else:
            # Fallback simple tokenization (not accurate for OpenAI models)
            logger.warning("LLMClient does not have count_tokens method. Using basic word count as a fallback.")
            return len(text.split())


    def calculate_cost(self, tokens_in: int, tokens_out: int, model: Optional[str] = None) -> float:
        """Calculates the estimated cost of an API call."""
        # This should ideally delegate to self.llm_client if it has pricing info
        if hasattr(self.llm_client, 'calculate_cost'):
            return self.llm_client.calculate_cost(tokens_in, tokens_out, model if model else self.teacher_model_name)
        else:
            # Fallback generic cost (highly inaccurate)
            logger.warning("LLMClient does not have calculate_cost method. Using a placeholder cost.")
            return (tokens_in + tokens_out) * 0.000002 # Placeholder cost

    def record_feedback(self, interaction_id: str, clarity_score: int) -> None:
        """Record clarity feedback for an interaction."""
        if not 1 <= clarity_score <= 5:
            raise ValueError("Clarity score must be between 1 and 5")
            
        self.db.update_interaction_clarity(interaction_id, clarity_score) 