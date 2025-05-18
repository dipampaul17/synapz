"""Student simulator for automated testing."""

import json
import time
import random
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os
import textstat
import re # Import re for regex operations

# Use PROMPTS_DIR from synapz package
from synapz import PROMPTS_DIR
from .llm_client import LLMClient
# Removed BudgetTracker import as it's not directly used in this file based on current context
# from .budget import BudgetTracker # Assuming LLMClient handles budget

class SimulatedResponse(BaseModel):
    """Structured output for simulated student responses."""
    clarity_rating: int = Field(..., description="Clarity rating on scale 1-5 based on LLM's assessment")
    engagement_rating: int = Field(..., description="Engagement rating on scale 1-5 based on LLM's assessment")
    detailed_feedback: str = Field(..., description="Detailed qualitative feedback from the LLM")
    confusion_points: List[str] = Field(..., description="Specific points of confusion identified by LLM")
    helpful_elements: List[str] = Field(..., description="Specific helpful elements identified by LLM")
    improvement_suggestions: List[str] = Field(..., description="LLM's suggestions for improving the material for the profile")
    
    # Heuristic scores can be added by the calling function after LLM response
    heuristic_profile_match_score: Optional[float] = Field(None, description="Heuristic assessment of profile match (0-1)")
    # Final clarity score can be blended by the calling function
    final_clarity_score: Optional[float] = Field(None, description="Potentially blended clarity score (heuristic + LLM)")
    # Added to ensure detailed heuristic metrics are available in the output
    heuristic_metrics_detail: Optional[Dict[str, Any]] = Field(None, description="Detailed heuristic metrics calculated for the explanation")


class StudentSimulator:
    """Simulates student responses to explanations based on cognitive profiles."""
    
    def __init__(self, llm_client: LLMClient, simulator_model_name: str = "gpt-4o"):
        """Initialize with LLM client and simulator model name."""
        self.llm_client = llm_client
        self.simulator_model_name = simulator_model_name
        self._load_prompt()
        self.seed = int(time.time())  # For controlled randomness
    
    def _load_prompt(self) -> None:
        """Load simulator prompt from file."""
        # Use imported PROMPTS_DIR
        simulator_prompt_path = PROMPTS_DIR / "student_sim.txt"
        
        # Create prompt file if it doesn't exist with the new structure
        os.makedirs(PROMPTS_DIR, exist_ok=True) 
        if not simulator_prompt_path.exists():
            default_prompt_content = """You are a student simulator for Synapz, an adaptive learning system. Your goal is to provide feedback that helps us create irrefutable evidence that adaptive teaching, tailored to cognitive profiles, is superior.

Your persona is a student with the following cognitive profile:
COGNITIVE PROFILE:
{learner_profile_details}

You have just been presented with an explanation of a concept.
LEARNING MATERIAL:
{explanation_text}

HEURISTIC ASSESSMENT OF THE MATERIAL (scores 0.0 to 1.0, higher is better):
- Readability Score (e.g., Flesch-Kincaid): {readability_score} (Lower indicates easier to read)
- Sentence Complexity Score: {sentence_complexity_score} (Lower indicates less complex)
- Vocabulary Richness Score: {vocabulary_richness_score} (Lower indicates simpler vocabulary)
- Abstractness Score: {abstractness_score} (Lower indicates more concrete language)
- Profile Match Score (how well the text matches your profile's preferences heuristically): {profile_match_score}

Based on your cognitive profile, the learning material, AND the heuristic assessment provided above, please evaluate the explanation.

AUTHENTIC SIMULATION GUIDELINES:

*For ADHD cognitive style:*
- Notice when content maintains vs. loses your attention.
- Recognize when information feels overwhelming or unfocused.
- Respond positively to clear structure, visual variety, and engaging examples.
- Show difficulty with walls of text, long sentences, and abstract concepts without concrete anchors.
- Express frustration with sequential steps that lack clear progression markers.
- Appreciate content that respects working memory limitations.
- React authentically to pace, organization, and stimulation level.

*For dyslexic cognitive style:*
- Notice text complexity, sentence length, and terminology difficulty.
- Respond to clarity of organization and information chunking.
- Show appreciation for multimodal representations of ideas.
- Express difficulty with dense text or complex vocabulary.
- Acknowledge when instruction respects processing speed needs.
- Recognize when information connects to big-picture understanding.
- React authentically to font, spacing, and text presentation.

*For visual learner cognitive style:*
- Evaluate the presence and quality of visual elements, diagrams, or spatial organization (even if described in text).
- Notice when concepts lack visual anchors or spatial relationships.
- Show strong comprehension when information has clear visual structure.
- Express difficulty following purely verbal/text explanations if they don't evoke imagery.
- Appreciate metaphors and imagery that create mental pictures.
- Recognize when spatial relationships between concepts are clear vs. unclear.
- React authentically to the visual coherence of the explanation.

RATING SYSTEM:
Rate clarity and engagement from 1-5 where:
1: Completely unsuited, impossible to follow/engage with.
2: Poorly aligned, required tremendous effort.
3: Somewhat accessible/engaging but with significant issues.
4. Generally well-adapted/engaging with minor areas for improvement.
5: Perfectly tailored, optimal for my learning and engagement.

Respond in JSON format. Be specific, constructive, and always relate your feedback to your cognitive profile.

{{
  "clarity_rating": <Integer from 1-5>,
  "engagement_rating": <Integer from 1-5>,
  "detailed_feedback": "<Your overall experience. How did the material make you feel as a student with your profile? Did the heuristic scores align with your experience?>",
  "confusion_points": [
    "<Specific element 1 that hindered understanding/engagement, and WHY from your profile's perspective.>",
    "<Specific element 2...>"
  ],
  "helpful_elements": [
    "<Specific element 1 that aided understanding/engagement, and WHY from your profile's perspective.>",
    "<Specific element 2...>"
  ],
  "improvement_suggestions": [
    "<Suggestion 1 for how this material could be better adapted to YOUR cognitive profile.>",
    "<Suggestion 2...>"
  ]
}}

IMPORTANT: Your response must precisely reflect how the explanation meets or fails to meet YOUR specific cognitive needs. Connect your feedback directly to the characteristics in your profile and the heuristic data. Your insights are crucial for proving adaptive learning works!
"""
            with open(simulator_prompt_path, "w") as f:
                f.write(default_prompt_content)
        
        # Load prompt template
        with open(simulator_prompt_path, "r") as f:
            self.simulator_prompt = f.read()
    
    def _count_syllables(self, word: str) -> int:
        """Rudimentary syllable counter."""
        word = word.lower()
        if not word:
            return 0
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        if not word:
            return 0

        syllable_count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            syllable_count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                syllable_count += 1
        if word.endswith("e"):
            syllable_count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        if syllable_count == 0:
            syllable_count = 1 # Ensure at least one syllable for any word
        return syllable_count

    def _count_passive_voice(self, text: str) -> int:
        """Counts potential passive voice instances. Rudimentary."""
        # Looks for common passive voice indicators: "be" verbs + past participle (often ends in "ed", "en", "t")
        # This is a simplification and may have false positives/negatives.
        passive_indicators = re.compile(r'\b(am|is|are|was|were|be|being|been)\s+\w+(?:ed|en|t)\b', re.IGNORECASE)
        return len(passive_indicators.findall(text))

    def _assess_explanation_features(self, explanation: str, learner_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess text features of an explanation against learner needs.
        Returns dict of metrics and a profile_match_score (0-1).
        """
        # Basic text statistics
        word_count = textstat.lexicon_count(explanation)
        sentence_count = textstat.sentence_count(explanation)
        # Ensure words list is available for metrics that might need it, even if not avg_word_length_syllables
        words = explanation.split()


        metrics = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": textstat.avg_sentence_length(explanation),
            "avg_word_length_chars": textstat.avg_character_per_word(explanation),
            # Changed to use direct textstat method for average syllables per word
            "avg_word_length_syllables": textstat.avg_syllables_per_word(explanation),
            "difficult_words_pct": (textstat.difficult_words(explanation) / max(1, word_count)) * 100 if word_count > 0 else 0,
            "flesch_reading_ease": textstat.flesch_reading_ease(explanation), # Higher is easier
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(explanation), # Lower is easier
            "gunning_fog": textstat.gunning_fog(explanation), # Lower is easier
            "smog_index": textstat.smog_index(explanation), # Lower is easier
            "coleman_liau_index": textstat.coleman_liau_index(explanation), # Lower is easier
            "dale_chall_readability_score": textstat.dale_chall_readability_score(explanation), # Lower is easier
            "paragraph_count": explanation.count("\\\\n\\\\n") + 1,
            "list_item_count": explanation.count("\\\\n- ") + explanation.count("\\\\n* ") + sum(1 for line in explanation.splitlines() if line.strip().startswith(tuple(f"{i}." for i in range(10)))),
            "bold_text_count": explanation.count("**") // 2 + explanation.count("__") // 2, # Approx.
            "code_block_count": explanation.count("```") // 2,
            "passive_voice_sentences": self._count_passive_voice(explanation), # Lower is generally better
            "explicit_definitions_count": len(re.findall(r'\\\\b(is defined as|means that|refers to|is the term for)\\\\b', explanation, re.IGNORECASE)),
            "spatial_prepositions_count": len(re.findall(r'\\\\b(above|below|next to|inside|outside|between|among|behind|in front of|under|over|through|across|around|top|bottom|left|right)\\\\b', explanation, re.IGNORECASE)),
            "analogies_count": len(re.findall(r'\\\\b(is like|as if|similar to|can be thought of as|analogous to|imagine that)\\\\b', explanation, re.IGNORECASE)),
            "transition_phrases_count": len(re.findall(r'\\\\b(firstly|secondly|finally|in contrast|however|therefore|furthermore|consequently|in addition|for example|for instance|specifically|in summary|in conclusion|next|then|also|thus|hence|so|because|since|although|while|if|unless|otherwise|moreover|subsequently|ultimately)\\\\b', explanation, re.IGNORECASE)),
            "interactive_questions_count": len(re.findall(r'\\\\S+\\\\s*\\\\?', explanation)) # Counts sentences ending with '?'
        }
        metrics["avg_paragraph_length_words"] = metrics["word_count"] / max(1, metrics["paragraph_count"])

        # Add raw and normalized Flesch-Kincaid Grade as abstractness score
        metrics["abstractness_fkg_raw"] = metrics["flesch_kincaid_grade"] # Use the already calculated FKG
        metrics["abstractness_fkg_normalized"] = min(max(metrics["abstractness_fkg_raw"], 0.0), 20.0) / 20.0


        # Features that appeal to specific learning styles (can be boolean or density based)
        metrics["has_visual_elements_strong_raw"] = ( # Keep raw counts for detailed metrics
            metrics["code_block_count"] > 0 or
            bool(re.search(r'\\\\b(diagram|chart|table|figure|illustration|graph|plot)\\\\b', explanation, re.IGNORECASE)) or
            any(c in explanation for c in ["│", "┌", "└", "├", "─", "┬", "┴", "┼"]) # ASCII art
        )
        metrics["has_visual_elements_weak_raw"] = (
            metrics["list_item_count"] > 2 or
            bool(re.search(r'->|=>|\\\\*\\\\*|__', explanation)) or # Arrows, emphasis as structure
            any(line.strip().endswith(":") for line in explanation.splitlines())
        )
        metrics["has_examples_raw"] = (
            bool(re.search(r'\\\\b(example:|e\\\\.g\\\\.|for instance:|such as:)\\\\b', explanation, re.IGNORECASE)) or
            explanation.lower().count("example") > 1
        )
        metrics["has_clear_steps_raw"] = metrics["list_item_count"] > 2 and any(line.strip().startswith(tuple(f"{i}." for i in range(10))) for line in explanation.splitlines())


        # --- Derived Feature Scores (normalized or binary, 0-1 where 1 is good for profile characteristic) ---
        # These are intermediate scores used by the profile weights.
        # Readability scores (higher is better for these derived scores)
        # Flesch Reading Ease: 0-100, higher is better. Normalize to 0-1.
        derived_scores = {
            "flesch_ease_score": metrics["flesch_reading_ease"] / 100.0,
            # Flesch-Kincaid Grade: Lower is better. Invert and scale (e.g., 1 - grade/20, capped at 0-1)
            "fk_grade_score": max(0, min(1, 1.0 - (metrics["flesch_kincaid_grade"] / 18.0))), # Assuming 18 is a high grade level
            # Dale-Chall: Lower is better. (e.g. 1 - (score-4)/10, if typical range 4-14)
            "dale_chall_score": max(0, min(1, 1.0 - ((metrics["dale_chall_readability_score"] - 4.0) / 10.0))),
            "low_passive_voice_score": max(0, min(1, 1.0 - (metrics["passive_voice_sentences"] / max(1, metrics["sentence_count"] * 0.25)))), # Penalize if >25% passive

            "short_sentence_score": 1.0 if metrics["avg_sentence_length"] < 15 else max(0, (25 - metrics["avg_sentence_length"]) / 10), # Favor <15, scale down up to 25
            "short_paragraph_score": 1.0 if metrics["avg_paragraph_length_words"] < 80 else max(0, (150 - metrics["avg_paragraph_length_words"]) / 70), # Favor <80, scale down to 150

            "high_explicit_definitions_score": min(1, metrics["explicit_definitions_count"] / 3.0), # Cap at 3 definitions for max score
            "high_spatial_prepositions_score": min(1, metrics["spatial_prepositions_count"] / 10.0), # Cap at 10
            "high_analogies_score": min(1, metrics["analogies_count"] / 2.0), # Cap at 2
            "high_transition_phrases_score": min(1, metrics["transition_phrases_count"] / (max(1, metrics["sentence_count"] / 5.0))), # Relative to sentence count
            "high_interactive_questions_score": min(1, metrics["interactive_questions_count"] / 3.0), # Cap at 3

            "visual_strong_score": 1.0 if metrics["has_visual_elements_strong_raw"] else 0.0,
            "visual_weak_score": 1.0 if metrics["has_visual_elements_weak_raw"] else 0.0,
            "examples_score": 1.0 if metrics["has_examples_raw"] else 0.0,
            "clear_steps_score": 1.0 if metrics["has_clear_steps_raw"] else 0.0,
            "low_difficulty_score": max(0, min(1, 1.0 - (metrics["difficult_words_pct"] / 20.0))), # Penalize if >20% difficult words

            "varied_formatting_score": 1.0 if (metrics["bold_text_count"] > 2 or metrics["list_item_count"] > 1 or metrics["code_block_count"] > 0) else 0.0,
            "chunking_friendliness_score": (
                (1.0 if metrics["avg_paragraph_length_words"] < 100 else 0.5) + \
                (1.0 if metrics["list_item_count"] > 0 else 0.0)
            ) / 2.0
        }

        # Calculate abstractness score (0=concrete, 1=abstract)
        # Higher values for features that make text concrete will reduce abstractness.
        concreteness_contribution = (
            derived_scores.get('examples_score', 0.0) * 0.4 +          # Examples make things concrete
            derived_scores.get('visual_strong_score', 0.0) * 0.3 +    # Strong visuals anchor concepts
            derived_scores.get('high_spatial_prepositions_score', 0.0) * 0.2 + # Spatial language grounds text
            derived_scores.get('high_analogies_score', 0.0) * 0.1      # Analogies relate to knowns
        )
        derived_scores["abstractness_score_concreteness_based"] = max(0.0, min(1.0, 1.0 - concreteness_contribution))

        # Add derived scores to the main metrics dict so they are also part of heuristic_metrics_detail
        metrics.update(derived_scores)


        # --- Profile Matching Score Calculation (0-1) ---
        profile_score = 0.0
        total_weight_for_profile = 0.0
        
        cognitive_style = learner_profile.get("cognitive_style", "").lower()
        # preferences = learner_profile.get("learning_preferences", []) # Currently not used directly, but could be
        # challenges = learner_profile.get("challenges", []) # Currently not used directly

        active_weights = {}

        if cognitive_style == "adhd":
            adhd_weights = {
                "short_sentence_score": 2.0,
                "short_paragraph_score": 1.5,
                "high_interactive_questions_score": 2.0,
                "examples_score": 1.5,
                "clear_steps_score": 1.0,
                "varied_formatting_score": 1.0, # Includes bolding, lists
                "high_transition_phrases_score": 1.0, # Helps with flow
                "low_difficulty_score": 1.0, # Avoid cognitive overload from complex words
                "chunking_friendliness_score": 1.5, # Short paragraphs and lists
                "visual_weak_score": 0.5 # Some visual structure helps
            }
            active_weights = adhd_weights
        
        elif cognitive_style == "dyslexic":
            dyslexic_weights = {
                "flesch_ease_score": 2.0,       # Directly prefers high readability
                "fk_grade_score": 1.5,          # Directly prefers lower grade level
                "dale_chall_score": 1.5,        # Prefers simpler vocabulary
                "low_passive_voice_score": 1.0, # Active voice is clearer
                "short_sentence_score": 2.0,
                "short_paragraph_score": 1.5,   # Good for chunking
                "high_explicit_definitions_score": 1.5,
                "examples_score": 1.0,
                "clear_steps_score": 1.0,       # Structure is key
                "low_difficulty_score": 2.0,    # Very important
                "chunking_friendliness_score": 2.0, # Combines short paragraphs and lists
                 # Less emphasis on visual metaphors unless very simple
                "visual_strong_score": 0.5, # Simple diagrams can be good
            }
            active_weights = dyslexic_weights

        elif cognitive_style == "visual":
            visual_weights = {
                "visual_strong_score": 3.0,     # Primary preference
                "visual_weak_score": 1.5,       # Secondary visual cues
                "high_spatial_prepositions_score": 2.0, # Language that evokes imagery
                "high_analogies_score": 1.5,    # Metaphors create mental pictures
                "examples_score": 1.0,          # Visualizable examples
                "clear_steps_score": 0.5,       # Structure can be visual
                "chunking_friendliness_score": 1.0, # Visually distinct chunks
                # Less penalty on text complexity if visually well-supported
                "flesch_ease_score": 0.5,
                "short_paragraph_score": 0.5,
            }
            active_weights = visual_weights

        if active_weights:
            for feature_key, weight in active_weights.items():
                # Use .get on derived_scores to default to 0 if a score isn't applicable or calculated
                # (though all should be present if defined above)
                feature_value = derived_scores.get(feature_key, 0.0) 
                profile_score += feature_value * weight
                total_weight_for_profile += weight
            
            if total_weight_for_profile > 0:
                profile_score = profile_score / total_weight_for_profile
            else: # Should not happen if weights are defined
                profile_score = 0.0
        
        metrics["profile_match_score"] = max(0, min(1, profile_score)) # Ensure it's capped 0-1

        # Return all metrics, including raw, derived, and final profile_match_score
        return metrics
    
    def _calculate_base_clarity(self, metrics: Dict[str, Any]) -> float:
        """Calculate base clarity score from metrics (profile_match_score) with controlled randomness."""
        # Initialize random generator with seed for this specific calculation if needed for reproducibility of *this step*
        # However, self.seed is already incremented per generate_feedback call.
        # random.seed(self.seed) 
        
        profile_match_score = metrics.get("profile_match_score", 0.5) # Default to 0.5 if not found
        
        # Base clarity now directly scaled from profile_match_score (0-1 to 1-5 range)
        # A score of 0 should map to 1 (lowest clarity)
        # A score of 1 should map to 5 (highest clarity)
        base_clarity = 1 + (profile_match_score * 4.0)
        
        # Add small random variation (+/- 0.3) to avoid complete predictability
        # The previous variation was +/- 0.5, reducing it slightly
        variation = random.uniform(-0.3, 0.3) 
        
        # Ensure clarity stays between 1-5
        final_base_clarity = max(1.0, min(5.0, base_clarity + variation))
        
        # Log the components for debugging if needed
        # logger.debug(f"Profile Match: {profile_match_score:.2f}, Initial Base: {base_clarity:.2f}, Variation: {variation:.2f}, Final Base: {final_base_clarity:.2f}")

        return final_base_clarity
    
    def generate_feedback(
        self,
        explanation: str,
        learner_profile: Dict[str, Any],
        pedagogy_tags: List[str],
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Generate simulated student feedback.

        Combines heuristic assessment with LLM-based evaluation if use_llm is True.
        
        Args:
            explanation: Teaching explanation to evaluate.
            learner_profile: Cognitive profile of the simulated student.
            pedagogy_tags: Tags describing the pedagogical approach (for future use).
            use_llm: Whether to use the LLM for detailed feedback.

        Returns:
            A dictionary containing feedback details, matching SimulatedResponse structure.
        """
        self.seed += 1 # Increment seed for varied randomness in sub-calls if any

        # 1. Assess explanation features and get all metrics
        all_metrics = self._assess_explanation_features(explanation, learner_profile)
        
        # 2. Calculate base clarity from heuristics
        base_clarity = self._calculate_base_clarity(all_metrics)

        # Prepare output structure compliant with SimulatedResponse
        # The heuristic_metrics_detail should contain all_metrics
        sim_response_data: Dict[str, Any] = {
            "clarity_rating": 0, # Will be updated by LLM or heuristic
            "engagement_rating": 0, # Will be updated by LLM or heuristic
            "detailed_feedback": "N/A (LLM not used)" if not use_llm else "",
            "confusion_points": [],
            "helpful_elements": [],
            "improvement_suggestions": [],
            "heuristic_profile_match_score": all_metrics.get("profile_match_score"),
            "final_clarity_score": base_clarity, # Start with heuristic clarity
            "heuristic_metrics_detail": all_metrics # Ensure all metrics are passed here
        }


        if use_llm:
            prompt_params = {
                "learner_profile_details": json.dumps(learner_profile, indent=2),
                "explanation_text": explanation,
                "readability_score": f"{all_metrics.get('flesch_kincaid_grade', 'N/A'):.2f} (FK Grade)", # Example
                "sentence_complexity_score": f"{all_metrics.get('avg_sentence_length', 'N/A'):.2f} words/sent", # Example
                "vocabulary_richness_score": f"{all_metrics.get('difficult_words_pct', 'N/A'):.2f}% difficult", # Example
                "abstractness_score": f"{all_metrics.get('abstractness_score_concreteness_based', 'N/A'):.2f}", # Now uses calculated score
                "profile_match_score": f"{all_metrics.get('profile_match_score', 0.0):.2f}"
            }
            
            # Add more heuristic details to the prompt for the LLM to consider
            prompt_params["heuristic_breakdown"] = (
                f"- Avg Sentence Length: {all_metrics.get('avg_sentence_length', 0):.1f} words\\\\n"
                f"- Difficult Words: {all_metrics.get('difficult_words_pct', 0):.1f}%\\\\n"
                f"- Visual Elements (strong): {'Yes' if all_metrics.get('visual_strong_score', 0) > 0 else 'No'}\\\\n"
                f"- Explicit Definitions: {all_metrics.get('explicit_definitions_count', 0)}\\\\n"
                f"- Interactive Questions: {all_metrics.get('interactive_questions_count', 0)}\\\\n"
                f"- Abstractness (FKG Grade Normalized): {all_metrics.get('abstractness_fkg_normalized', 'N/A'):.2f}"
            )

            # Ensure the student_sim.txt prompt template has {heuristic_breakdown} placeholder
            # It might be good to ensure student_sim.txt is updated to use these new params

            formatted_prompt = self.simulator_prompt.format(**prompt_params)
            
            try:
                # Changed to use get_json_completion
                # System prompt is generic as get_json_completion handles JSON mode.
                system_prompt_for_json = "You are a helpful assistant that always responds in JSON format."
                llm_response = self.llm_client.get_json_completion(
                    system_prompt=system_prompt_for_json, # System prompt for get_json_completion
                    user_prompt=formatted_prompt,        # The detailed prompt for the simulator
                    model=self.simulator_model_name,
                    temperature=0.5,
                    max_tokens=1024 # Max tokens for simulator response
                    # parser=SimulatedResponse # Optional: if get_json_completion can take a Pydantic model directly for parsing
                )

                # llm_response['content'] is already a dict (parsed JSON)
                llm_data = llm_response.get("content", {})

                # Update sim_response_data with LLM outputs
                sim_response_data["clarity_rating"] = llm_data.get("clarity_rating", int(base_clarity * 5.0)) 
                sim_response_data["engagement_rating"] = llm_data.get("engagement_rating", int(base_clarity * 2.0 + 1.0))
                sim_response_data["detailed_feedback"] = llm_data.get("detailed_feedback", "")
                sim_response_data["confusion_points"] = llm_data.get("confusion_points", [])
                sim_response_data["helpful_elements"] = llm_data.get("helpful_elements", [])
                sim_response_data["improvement_suggestions"] = llm_data.get("improvement_suggestions", [])

                # Blend LLM rating with heuristic clarity for final_clarity_score
                # Example: 60% LLM, 40% Heuristic (if LLM rating is 1-5, base_clarity 0-1)
                llm_clarity_normalized = sim_response_data["clarity_rating"] / 5.0
                sim_response_data["final_clarity_score"] = 0.6 * llm_clarity_normalized + 0.4 * base_clarity

            except Exception as e: # Catch other potential errors from LLM call
                print(f"Error during LLM call in simulator: {e}")
                sim_response_data["detailed_feedback"] = "Error in LLM call. Using heuristic data only."
                sim_response_data["clarity_rating"] = int(base_clarity * 5)
                sim_response_data["engagement_rating"] = int(base_clarity * 3 + 1)
                sim_response_data["final_clarity_score"] = base_clarity
        else: # Not using LLM
            sim_response_data["clarity_rating"] = int(base_clarity * 5) # Scale heuristic 0-1 to 1-5
            sim_response_data["engagement_rating"] = int(base_clarity * 3 + 1) # Crude heuristic engagement
            sim_response_data["detailed_feedback"] = "LLM simulation was disabled. Feedback based on heuristics."
            # heuristic_profile_match_score and final_clarity_score already set

        # Validate with Pydantic model before returning, helps catch issues
        try:
            validated_response = SimulatedResponse(**sim_response_data)
            return validated_response.model_dump()
        except Exception as e: # Pydantic validation error
            print(f"Pydantic validation error in StudentSimulator: {e}")
            # Return the raw data anyway, or a more robust error structure
            sim_response_data["detailed_feedback"] = f"Pydantic validation error: {e}. " + sim_response_data.get("detailed_feedback", "")
            return sim_response_data 