"""API client for handling OpenAI interactions with error handling and budget management."""

import time
import random
import logging
from typing import Dict, Any, Optional, List, Callable
import openai
from openai import OpenAI
from .budget import BudgetTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    """Handles OpenAI API calls with error handling, retry logic, and budget tracking."""
    
    def __init__(self, budget_tracker: BudgetTracker, api_key: Optional[str] = None):
        """Initialize with budget tracker and optional API key."""
        self.budget_tracker = budget_tracker
        self.client = OpenAI(api_key=api_key)
        self.max_retries = 7
        
    def _exponential_backoff(self, retry: int) -> float:
        """Calculate exponential backoff time with jitter."""
        base_delay = 10
        max_delay = 60
        delay = min(max_delay, base_delay * (2 ** retry))
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter
        
    def chat_completion(
        self, 
        prompt: str, 
        model: str = "gpt-4o-mini", 
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Make a chat completion call with budget checks and error handling.
        
        Args:
            prompt: The prompt text to send
            model: OpenAI model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for response generation
            
        Returns:
            Response from the API or error message
        """
        # Project cost before making call
        projected_cost = self.budget_tracker.project_cost(prompt, max_tokens, model)
        
        # Check if within budget
        if not self.budget_tracker.check_budget(projected_cost):
            logger.error(f"Budget exceeded. Projected cost: ${projected_cost:.4f}")
            raise ValueError(f"API call would exceed budget. Projected: ${projected_cost:.4f}")
            
        # Try API call with exponential backoff
        retry = 0
        while retry <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Log actual usage
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                self.budget_tracker.log_usage(model, tokens_in, tokens_out)
                
                # Return the response content
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost": self.budget_tracker.PRICE_MAP[model]["input"] * tokens_in / 1_000_000 +
                                self.budget_tracker.PRICE_MAP[model]["output"] * tokens_out / 1_000_000
                    }
                }
                
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                if retry == self.max_retries:
                    logger.error(f"Max retries reached: {str(e)}")
                    raise
                
                wait_time = self._exponential_backoff(retry)
                logger.warning(f"API error: {str(e)}. Retrying in {wait_time:.2f} seconds")
                time.sleep(wait_time)
                retry += 1
                
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise 