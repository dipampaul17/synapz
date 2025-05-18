"""Module for handling cognitive profiles and adaptive learning strategies."""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

class CognitiveProfile(Enum):
    """Enumeration of supported cognitive profiles for adaptation."""
    ADHD = "adhd"
    DYSLEXIC = "dyslexic"
    VISUAL = "visual"
    CONTROL = "control"  # No adaptation - used as baseline
    
    @staticmethod
    def from_string(profile_str: str) -> 'CognitiveProfile':
        """Convert string to CognitiveProfile enum."""
        try:
            return CognitiveProfile(profile_str.lower())
        except ValueError:
            raise ValueError(f"Unknown profile: {profile_str}. "
                            f"Valid options: {[p.value for p in CognitiveProfile]}")

@dataclass
class ProfileStrategy:
    """Defines adaptation strategies for a cognitive profile."""
    profile: CognitiveProfile
    prompt_template_key: str
    instruction_modifiers: List[str]
    example_count: int = 2
    use_visuals: bool = False
    chunk_size: int = 3  # Number of concepts to present at once
    
    @property
    def is_control(self) -> bool:
        """Check if this is a control group profile (no adaptation)."""
        return self.profile == CognitiveProfile.CONTROL


class ProfileManager:
    """Manages cognitive profiles and their adaptation strategies."""
    
    # Default strategies for each profile
    DEFAULT_STRATEGIES = {
        CognitiveProfile.ADHD: ProfileStrategy(
            profile=CognitiveProfile.ADHD,
            prompt_template_key="adhd_instruction",
            instruction_modifiers=[
                "Break content into smaller, focused chunks",
                "Include frequent recaps and summaries",
                "Use conversational, engaging language",
                "Highlight key points with bold formatting"
            ],
            example_count=2,
            chunk_size=2
        ),
        CognitiveProfile.DYSLEXIC: ProfileStrategy(
            profile=CognitiveProfile.DYSLEXIC,
            prompt_template_key="dyslexic_instruction",
            instruction_modifiers=[
                "Use simple, clear language with shorter sentences",
                "Avoid idioms and ambiguous phrasing",
                "Use bullet points and numbered lists",
                "Include phonetic pronunciation for difficult terms"
            ],
            example_count=3
        ),
        CognitiveProfile.VISUAL: ProfileStrategy(
            profile=CognitiveProfile.VISUAL,
            prompt_template_key="visual_instruction",
            instruction_modifiers=[
                "Use spatial organization like tables, diagrams, mind maps",
                "Emphasize visual patterns and relationships",
                "Describe concepts with visual metaphors",
                "Use color coding for different concept categories"
            ],
            use_visuals=True
        ),
        CognitiveProfile.CONTROL: ProfileStrategy(
            profile=CognitiveProfile.CONTROL,
            prompt_template_key="standard_instruction",
            instruction_modifiers=[],  # No adaptation
            example_count=1
        )
    }
    
    def __init__(self):
        """Initialize with default profile strategies."""
        self.strategies = self.DEFAULT_STRATEGIES.copy()
    
    def get_strategy(self, profile: CognitiveProfile) -> ProfileStrategy:
        """Get adaptation strategy for a profile."""
        return self.strategies[profile]
        
    def update_strategy(self, profile: CognitiveProfile, strategy_updates: Dict[str, Any]) -> None:
        """Update strategy parameters for a profile."""
        current = self.strategies[profile]
        
        # Create a new strategy with updated values
        updated = ProfileStrategy(
            profile=profile,
            prompt_template_key=strategy_updates.get('prompt_template_key', current.prompt_template_key),
            instruction_modifiers=strategy_updates.get('instruction_modifiers', current.instruction_modifiers),
            example_count=strategy_updates.get('example_count', current.example_count),
            use_visuals=strategy_updates.get('use_visuals', current.use_visuals),
            chunk_size=strategy_updates.get('chunk_size', current.chunk_size)
        )
        
        self.strategies[profile] = updated 