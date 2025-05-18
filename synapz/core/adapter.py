"""Adapter module for generating adapted educational content."""

import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import logging

from .api_client import APIClient
from .profiles import CognitiveProfile, ProfileStrategy, ProfileManager
from .budget import BudgetTracker

# Conditional import to avoid circular imports
try:
    from synapz.models.learner_profiles import get_profile_for_adaptation
    DETAILED_PROFILES_AVAILABLE = True
except ImportError:
    DETAILED_PROFILES_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAdapter:
    """
    Adapts educational content based on cognitive profiles.
    
    This class is responsible for loading prompts and using them to generate
    content tailored to different cognitive profiles.
    """
    
    def __init__(
        self, 
        api_client: APIClient,
        profile_manager: ProfileManager,
        prompts_dir: Path
    ):
        """
        Initialize the content adapter.
        
        Args:
            api_client: Client for API calls
            profile_manager: Manager for cognitive profiles
            prompts_dir: Directory containing prompt templates
        """
        self.api_client = api_client
        self.profile_manager = profile_manager
        self.prompts_dir = prompts_dir
        self.prompt_templates = self._load_prompt_templates()
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load all prompt templates from the prompts directory."""
        templates = {}
        
        for file in os.listdir(self.prompts_dir):
            if file.endswith(".txt"):
                template_key = os.path.splitext(file)[0]
                with open(self.prompts_dir / file, "r") as f:
                    templates[template_key] = f.read()
                    
        return templates
    
    def get_system_prompt(
        self, 
        experiment_type: str,
        concept: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the appropriate system prompt based on experiment type.
        
        Args:
            experiment_type: Either "adaptive" or "control"
            concept: The concept data to teach
            profile: Profile data (required for adaptive prompts)
            context: Teaching context information
            
        Returns:
            Formatted system prompt string
        """
        if context is None:
            context = {
                "turn_number": 1,
                "previous_clarity": None,
                "interaction_history": []
            }
            
        if experiment_type == "adaptive":
            if profile is None:
                raise ValueError("Profile data is required for adaptive experiments")
                
            # Load adaptive system prompt
            template = self.prompt_templates.get("adaptive_system")
            if not template:
                raise ValueError("Adaptive system prompt not found")
                
            # Format prompt
            return template.format(
                learner_profile_json=json.dumps(profile, indent=2),
                concept_json=json.dumps(concept, indent=2),
                turn_number=context["turn_number"],
                previous_clarity=context["previous_clarity"] or "None",
                interaction_history=json.dumps(context["interaction_history"], indent=2)
            )
        else:
            # Load control system prompt
            template = self.prompt_templates.get("control_system")
            if not template:
                raise ValueError("Control system prompt not found")
                
            # Format prompt
            return template.format(
                concept_json=json.dumps(concept, indent=2),
                turn_number=context["turn_number"],
                previous_clarity=context["previous_clarity"] or "None"
            )
    
    def generate_adapted_content(
        self,
        topic: str,
        profile: CognitiveProfile,
        learning_objective: str,
        background_knowledge: Optional[str] = None,
        max_tokens: int = 800,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Generate content adapted to a specific cognitive profile.
        
        Args:
            topic: The educational topic to explain
            profile: The cognitive profile to adapt content for
            learning_objective: What the student should learn
            background_knowledge: Optional prior knowledge to consider
            max_tokens: Maximum output tokens
            model: Model to use
            
        Returns:
            Dictionary with generated content and metadata
        """
        # Get adaptation strategy for this profile
        strategy = self.profile_manager.get_strategy(profile)
        
        # Get the appropriate prompt template
        template_key = strategy.prompt_template_key
        if template_key not in self.prompt_templates:
            raise ValueError(f"Prompt template not found: {template_key}")
            
        prompt_template = self.prompt_templates[template_key]
        
        # Get detailed profile information if available
        detailed_profile = None
        if DETAILED_PROFILES_AVAILABLE:
            try:
                detailed_profile = get_profile_for_adaptation(profile.value)
                logger.info(f"Using detailed profile for {profile.value}")
            except Exception as e:
                logger.warning(f"Could not load detailed profile: {str(e)}")
        
        # Prepare prompt variables
        prompt_vars = {
            "topic": topic,
            "learning_objective": learning_objective,
            "background_knowledge": background_knowledge or "No specific background knowledge",
            "profile": profile.value,
            "instruction_modifiers": "\n".join([f"- {mod}" for mod in strategy.instruction_modifiers]),
            "example_count": strategy.example_count,
            "use_visuals": "true" if strategy.use_visuals else "false",
            "chunk_size": strategy.chunk_size
        }
        
        # Add detailed profile information if available
        if detailed_profile:
            # Add cognitive traits
            traits = detailed_profile["cognitive_traits"]
            traits_str = "\n".join([f"- {k}: {v}" for k, v in traits.items() 
                                   if k != "strengths" and k != "challenges"])
            
            # Add strengths and challenges
            if "strengths" in traits:
                traits_str += "\n- Strengths: " + ", ".join(traits["strengths"])
            if "challenges" in traits:
                traits_str += "\n- Challenges: " + ", ".join(traits["challenges"])
                
            prompt_vars["cognitive_traits"] = traits_str
            
            # Add modality preferences
            modality = detailed_profile["modality_preferences"]
            modality_str = "- Primary: " + ", ".join(modality["primary"])
            if modality.get("secondary"):
                modality_str += "\n- Secondary: " + ", ".join(modality["secondary"])
            if modality.get("avoid"):
                modality_str += "\n- Avoid: " + ", ".join(modality["avoid"])
                
            prompt_vars["modality_preferences"] = modality_str
            
            # Use adaptation parameters from detailed profile if available
            if "adaptation" in detailed_profile:
                adaptation = detailed_profile["adaptation"]
                if "chunk_size" in adaptation:
                    prompt_vars["chunk_size"] = adaptation["chunk_size"]
                if "example_count" in adaptation:
                    prompt_vars["example_count"] = adaptation["example_count"]
        
        # Format prompt with variables
        prompt = prompt_template.format(**prompt_vars)
        
        # Generate content
        response = self.api_client.chat_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens
        )
        
        # Return response with metadata
        return {
            "content": response["content"],
            "profile": profile.value,
            "topic": topic,
            "is_control": strategy.is_control,
            "usage": response["usage"],
            "detailed_profile_used": detailed_profile is not None
        } 