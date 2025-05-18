"""Student simulator for automated testing."""

import json
import time
import random
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os
import textstat

# Use PROMPTS_DIR from synapz package
from synapz import PROMPTS_DIR
from .llm_client import LLMClient
from .budget import BudgetTracker

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
    
    def _assess_explanation_features(self, explanation: str, learner_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess text features of an explanation against learner needs.
        Returns dict of metrics and a profile_match_score (0-1).
        """
        # Initialize random generator with seed for reproducibility
        # random.seed(self.seed) # Seeding here might make feature assessment too deterministic if not careful
        # self.seed += 1  # Increment seed for next call

        # Basic text statistics
        word_count = textstat.lexicon_count(explanation)
        sentence_count = textstat.sentence_count(explanation)
        
        metrics = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": textstat.avg_sentence_length(explanation),
            "avg_word_length": textstat.avg_character_per_word(explanation),
            "difficult_words_pct": (textstat.difficult_words(explanation) / max(1, word_count)) * 100 if word_count > 0 else 0,
            "flesch_reading_ease": textstat.flesch_reading_ease(explanation),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(explanation),
            "gunning_fog": textstat.gunning_fog(explanation),
            "smog_index": textstat.smog_index(explanation),
            "coleman_liau_index": textstat.coleman_liau_index(explanation),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(explanation),
            "paragraph_count": explanation.count("\\n\\n") + 1,
            "list_item_count": explanation.count("\\n- ") + explanation.count("\\n* ") + sum(1 for line in explanation.splitlines() if line.strip().startswith(tuple(f"{i}." for i in range(10)))),
            "bold_text_count": explanation.count("**") // 2 + explanation.count("__") // 2, # Approx.
            "code_block_count": explanation.count("```") // 2,
        }
        metrics["avg_paragraph_length_words"] = metrics["word_count"] / max(1, metrics["paragraph_count"])


        # Check for features that appeal to specific learning styles
        metrics["has_visual_elements_strong"] = (
            metrics["code_block_count"] > 0 or
            "diagram" in explanation.lower() or 
            "chart" in explanation.lower() or
            "table" in explanation.lower() or
            any(c in explanation for c in ["│", "┌", "└", "├"]) # ASCII table/diagram chars
        )
        metrics["has_visual_elements_weak"] = ( # More general visual structure
            metrics["list_item_count"] > 2 or
            "->" in explanation or 
            any(line.strip().endswith(":") for line in explanation.splitlines()) # Often used for key-value or definitions, check endswith
        )
        
        metrics["has_examples"] = (
            "example:" in explanation.lower() or
            "e.g." in explanation.lower() or
            "for instance:" in explanation.lower() or
            explanation.lower().count("example") > 1 # Multiple mentions
        )
        metrics["has_clear_steps"] = metrics["list_item_count"] > 2 and any(line.strip().startswith(tuple(f"{i}." for i in range(10))) for line in explanation.splitlines())


        # --- Profile Matching Score Calculation (0-1) ---
        profile_score = 0.0
        max_possible_score = 0.0 # Sum of weights for the active profile
        
        cognitive_style = learner_profile.get("cognitive_style", "").lower()
        preferences = learner_profile.get("learning_preferences", [])
        challenges = learner_profile.get("challenges", [])

        # ADHD Profile Weights & Logic
        if cognitive_style == "adhd":
            adhd_weights = {
                "short_sentences": 2, "short_paragraphs": 2, "low_word_count": 1,
                "clear_steps": 3, "examples": 2, "bolding": 1, "some_visuals_weak": 1
            }
            max_possible_score = sum(adhd_weights.values())

            if metrics["avg_sentence_length"] < 15: profile_score += adhd_weights["short_sentences"]
            elif metrics["avg_sentence_length"] > 25: profile_score -= adhd_weights["short_sentences"] * 0.5
            
            if metrics["avg_paragraph_length_words"] < 80: profile_score += adhd_weights["short_paragraphs"]
            elif metrics["avg_paragraph_length_words"] > 150: profile_score -= adhd_weights["short_paragraphs"] * 0.5

            if metrics["word_count"] < 200: profile_score += adhd_weights["low_word_count"] # Very short overall
            elif metrics["word_count"] < 400: profile_score += adhd_weights["low_word_count"] * 0.5
            elif metrics["word_count"] > 700: profile_score -= adhd_weights["low_word_count"] * 0.5

            if metrics["has_clear_steps"]: profile_score += adhd_weights["clear_steps"]
            if metrics["has_examples"]: profile_score += adhd_weights["examples"]
            if metrics["bold_text_count"] > 2: profile_score += adhd_weights["bolding"]
            if metrics["has_visual_elements_weak"]: profile_score += adhd_weights["some_visuals_weak"]
            
            if "sustained attention" in challenges and metrics["word_count"] > 500:
                 profile_score -= 1 # Penalty for long text if sustained attention is a challenge

        # Dyslexic Profile Weights & Logic
        elif cognitive_style == "dyslexic":
            dyslexic_weights = {
                "high_flesch": 3, "low_grade_level": 2, "short_avg_word": 2,
                "low_difficult_words": 2, "examples": 1, "lists": 1, "some_visuals_weak": 1
            }
            max_possible_score = sum(dyslexic_weights.values())

            if metrics["flesch_reading_ease"] > 70: profile_score += dyslexic_weights["high_flesch"]
            elif metrics["flesch_reading_ease"] < 50: profile_score -= dyslexic_weights["high_flesch"] * 0.5
            
            if metrics["flesch_kincaid_grade"] < 8: profile_score += dyslexic_weights["low_grade_level"]
            elif metrics["flesch_kincaid_grade"] > 12: profile_score -= dyslexic_weights["low_grade_level"] * 0.5

            if metrics["avg_word_length"] < 5: profile_score += dyslexic_weights["short_avg_word"]
            if metrics["difficult_words_pct"] < 10: profile_score += dyslexic_weights["low_difficult_words"]
            if metrics["has_examples"]: profile_score += dyslexic_weights["examples"]
            if metrics["list_item_count"] > 1: profile_score += dyslexic_weights["lists"]
            if metrics["has_visual_elements_weak"]: profile_score += dyslexic_weights["some_visuals_weak"]
            
            if "text processing" in challenges and metrics["flesch_reading_ease"] < 60:
                profile_score -= 1

        # Visual Learner Profile Weights & Logic
        elif cognitive_style == "visual":
            visual_weights = {
                "strong_visuals": 4, "weak_visuals": 2, "examples": 1, "clear_steps_as_visual_structure": 1
            }
            max_possible_score = sum(visual_weights.values())

            if metrics["has_visual_elements_strong"]: profile_score += visual_weights["strong_visuals"]
            elif metrics["has_visual_elements_weak"]: profile_score += visual_weights["weak_visuals"] # Add points if strong visuals are not present but weak ones are
            else: profile_score -= visual_weights["strong_visuals"] * 0.5 # Penalize lack of any visuals

            if metrics["has_examples"]: profile_score += visual_weights["examples"] # Examples can often be visual or aid visual understanding
            if metrics["has_clear_steps"]: profile_score += visual_weights["clear_steps_as_visual_structure"]
        
        else: # Default/Control or unknown profile
            # For control, we aim for a neutral score, or slightly positive if generally well-structured.
            # This ensures control isn't inherently "bad" but won't score high on specific adaptations.
            generic_weights = {"readable":1, "has_examples":1, "has_lists":0.5}
            max_possible_score = sum(generic_weights.values())
            if metrics["flesch_reading_ease"] > 60: profile_score += generic_weights["readable"]
            if metrics["has_examples"]: profile_score += generic_weights["has_examples"]
            if metrics["list_item_count"] > 1: profile_score += generic_weights["has_lists"]


        # Normalize score to 0-1 range
        if max_possible_score > 0:
            normalized_score = profile_score / max_possible_score
        else: # Avoid division by zero if a profile had no weights (should not happen with current logic)
            normalized_score = 0.5 
            
        metrics["profile_match_score"] = max(0.0, min(1.0, normalized_score)) # Clamp between 0 and 1
        
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

        # 1. Perform heuristic assessment
        metrics = self._assess_explanation_features(explanation, learner_profile)
        heuristic_profile_match_score = metrics.get("profile_match_score", 0.5) # Score from 0 to 1
        
        # Convert heuristic profile match score to a 1-5 scale for potential direct use or blending
        heuristic_clarity_1_to_5 = 1 + (heuristic_profile_match_score * 4.0)

        if not use_llm:
            # Fallback to purely heuristic-based feedback
            # Use the _calculate_base_clarity for some variation if desired, or just the scaled score
            # final_clarity = self._calculate_base_clarity(metrics) # This adds randomness
            final_clarity = heuristic_clarity_1_to_5 # Direct mapping for simplicity without LLM
            
            return {
                "clarity_rating": int(round(final_clarity)), # LLM field, but filled with heuristic
                "engagement_rating": int(round(heuristic_clarity_1_to_5 / 5.0 * 3.0 + 1.0)), # Crude engagement proxy
                "detailed_feedback": "Feedback generated heuristically. LLM not used.",
                "confusion_points": ["N/A - Heuristic assessment"] if final_clarity < 3 else [],
                "helpful_elements": ["N/A - Heuristic assessment"] if final_clarity >= 3 else [],
                "improvement_suggestions": ["N/A - Heuristic assessment"],
                "heuristic_profile_match_score": heuristic_profile_match_score,
                "final_clarity_score": final_clarity,
            }

        # 2. Prepare prompt for LLM
        learner_profile_str = json.dumps(learner_profile, indent=2)
        
        # Calculate abstractness score (0=concrete, 1=abstract)
        # Lower value for has_examples, has_visual_elements_strong, has_clear_steps means less abstract
        abstractness_metric = 1.0 - (
            metrics.get('has_examples', False) * 0.3 + 
            metrics.get('has_visual_elements_strong', False) * 0.4 + 
            metrics.get('has_clear_steps', False) * 0.3
        )
        abstractness_score = max(0.0, min(1.0, abstractness_metric))

        prompt_input = {
            "learner_profile_details": learner_profile_str,
            "explanation_text": explanation,
            "readability_score": round(metrics.get('flesch_kincaid_grade', 10.0), 1), # Default 10.0
            "sentence_complexity_score": round(metrics.get('avg_sentence_length', 20.0), 1), # Default 20.0
            "vocabulary_richness_score": round(metrics.get('difficult_words_pct', 15.0), 1), # Default 15.0
            "abstractness_score": round(abstractness_score, 2),
            "profile_match_score": round(heuristic_profile_match_score, 2)
        }
        
        formatted_prompt = self.simulator_prompt.format(**prompt_input)

        # 3. Call LLM
        response_json_str = None
        llm_response_data = {}
        # Ensure default values for all fields expected from LLM in case of failure
        default_llm_output = {
            "clarity_rating": int(round(heuristic_clarity_1_to_5)), # Fallback to heuristic
            "engagement_rating": int(round(heuristic_clarity_1_to_5 / 5.0 * 3.0 + 1)), # Crude fallback
            "detailed_feedback": "LLM call failed or produced invalid JSON. Using heuristic assessment.",
            "confusion_points": [],
            "helpful_elements": [],
            "improvement_suggestions": []
        }

        try:
            
            llm_api_response = self.llm_client.get_json_completion(
                system_prompt="You are an AI assistant acting as a student simulator. Follow the user prompt carefully to provide feedback in the specified JSON format.",
                user_prompt=formatted_prompt, # This contains the full student simulation instructions
                model=self.simulator_model_name, 
                temperature=0.5, 
                max_tokens=600 
                # response_format is handled by get_json_completion
            )
            # get_json_completion returns already parsed content if successful
            llm_output = llm_api_response["content"] 

            # Validate and use LLM output, with fallbacks for missing keys
            llm_response_data["clarity_rating"] = llm_output.get("clarity_rating", default_llm_output["clarity_rating"])
            llm_response_data["engagement_rating"] = llm_output.get("engagement_rating", default_llm_output["engagement_rating"])
            llm_response_data["detailed_feedback"] = llm_output.get("detailed_feedback", default_llm_output["detailed_feedback"])
            llm_response_data["confusion_points"] = llm_output.get("confusion_points", default_llm_output["confusion_points"])
            llm_response_data["helpful_elements"] = llm_output.get("helpful_elements", default_llm_output["helpful_elements"])
            llm_response_data["improvement_suggestions"] = llm_output.get("improvement_suggestions", default_llm_output["improvement_suggestions"])
                
        except json.JSONDecodeError: # This might be less likely now if get_json_completion handles it
            print(f"Error: Could not decode JSON from LLM (should have been handled by client): {llm_output if 'llm_output' in locals() else 'Unknown'}")
            llm_response_data = default_llm_output.copy()
        except ValueError as ve: # Catch ValueErrors from LLMClient (e.g. budget, API errors after retries)
            print(f"ValueError during LLM call in StudentSimulator: {ve}")
            llm_response_data = default_llm_output.copy()
        except Exception as e: # Catch other potential errors during LLM call
            print(f"Error during LLM call in StudentSimulator: {e}")
            llm_response_data = default_llm_output.copy()

        # 4. Combine Heuristic and LLM feedback
        # Blending: 40% heuristic (scaled 1-5), 60% LLM clarity rating
        final_clarity_score = (0.4 * heuristic_clarity_1_to_5) + (0.6 * llm_response_data["clarity_rating"])
        final_clarity_score = max(1.0, min(5.0, final_clarity_score)) # Clamp to 1-5

        return {
            "clarity_rating": llm_response_data["clarity_rating"], # This is the LLM's direct clarity rating
            "engagement_rating": llm_response_data["engagement_rating"],
            "detailed_feedback": llm_response_data["detailed_feedback"],
            "confusion_points": llm_response_data["confusion_points"],
            "helpful_elements": llm_response_data["helpful_elements"],
            "improvement_suggestions": llm_response_data["improvement_suggestions"],
            "heuristic_profile_match_score": heuristic_profile_match_score,
            "final_clarity_score": round(final_clarity_score, 2),
        } 