"""OpenAI API client with budget enforcement and structured outputs."""

import time
import json
import random
import logging
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Callable
import os
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from .budget import BudgetTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar('T')

class LLMClient:
    """OpenAI API client with budget enforcement, retries, and structured outputs."""
    
    def __init__(self, budget_tracker: BudgetTracker, max_retries: int = 8, api_key: Optional[str] = None):
        """
        Initialize with budget tracker and retry settings.
        
        Args:
            budget_tracker: Tracker for managing API costs
            max_retries: Maximum number of retry attempts (increased to 8)
            api_key: Optional API key (defaults to environment variable)
        """
        # Prioritize provided API key, then check environment variables
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError("API key must be provided either directly or via OPENAI_API_KEY environment variable")
            
        self.client = OpenAI(api_key=api_key)
        self.budget_tracker = budget_tracker
        self.max_retries = max_retries
        self.api_key = api_key
    
    def _exponential_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time with jitter for retries.
        
        Args:
            attempt: Current retry attempt number
            
        Returns:
            Delay in seconds before next retry
        """
        base_delay = 20  # Increased from 10
        max_delay = 120 # Also increasing max_delay
        delay = min(max_delay, base_delay * (2 ** attempt))
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter
    
    def _format_messages(self, system: str, user: str) -> List[Dict[str, str]]:
        """
        Format messages for chat completion.
        
        Args:
            system: System prompt
            user: User prompt
            
        Returns:
            Formatted messages list for the API
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    
    def get_completion(
        self, 
        system_prompt: str, 
        user_prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get completion with budget checking, retries, and optional JSON response format.
        
        Args:
            system_prompt: Instructions for the model
            user_prompt: The user query
            model: OpenAI model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            response_format: Optional response format to enforce JSON
            
        Returns:
            Dict with content and usage statistics
            
        Raises:
            ValueError: If budget limit reached or API errors
        """
        messages = self._format_messages(system_prompt, user_prompt)
        
        # Calculate full prompt for token counting
        full_prompt = system_prompt + "\n\n" + user_prompt
        
        # Project cost before API call
        projected_cost = self.budget_tracker.project_cost(full_prompt, max_tokens, model)
        
        # Check if we'd exceed budget for the current run
        if self.budget_tracker.run_budget_allowance > 0 and \
           not self.budget_tracker.check_budget(projected_cost):
            remaining_run_budget = self.budget_tracker.get_remaining_run_budget()
            raise ValueError(f"Budget limit for this run reached! Projected cost of ${projected_cost:.4f} "
                             f"would exceed the remaining run budget of ${remaining_run_budget:.4f}. "
                             f"(Run allowance: ${self.budget_tracker.run_budget_allowance:.2f}, "
                             f"Spend this run so far: ${self.budget_tracker.get_current_run_spend():.4f})" )
        
        # Try to get completion with retries
        attempts = 0
        while attempts < self.max_retries:
            try:
                # Set up the API call parameters
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Add response format if specified
                if response_format:
                    params["response_format"] = response_format
                
                # Make the API call
                response = self.client.chat.completions.create(**params)
                
                # Extract response content
                content = response.choices[0].message.content
                
                # Track token usage
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                
                # Log token usage
                self.budget_tracker.log_usage(model, tokens_in, tokens_out)
                
                # Calculate actual cost
                input_cost = self.budget_tracker.PRICE_MAP[model]["input"] * tokens_in / 1_000_000
                output_cost = self.budget_tracker.PRICE_MAP[model]["output"] * tokens_out / 1_000_000
                total_cost = input_cost + output_cost
                
                return {
                    "content": content,
                    "usage": {
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost": total_cost
                    },
                    "model": model
                }
                
            except RateLimitError as e:
                attempts += 1
                if attempts < self.max_retries:
                    delay = self._exponential_backoff(attempts)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds... (Attempt {attempts}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {attempts} attempts: {str(e)}")
                    raise ValueError(f"Rate limit exceeded after {attempts} attempts: {str(e)}") from e
                    
            except (APIError, APITimeoutError) as e:
                attempts += 1
                # Retry on 5xx errors or timeouts, but not on client-side errors (4xx) unless specifically rate limited
                is_server_error = hasattr(e, 'http_status') and e.http_status >= 500
                is_timeout = isinstance(e, APITimeoutError)

                if attempts < self.max_retries and (is_server_error or is_timeout):
                    delay = self._exponential_backoff(attempts)
                    logger.warning(f"API error ({type(e).__name__}), retrying in {delay:.2f} seconds... (Attempt {attempts}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"API error ({type(e).__name__}) not retryable or max retries reached: {str(e)}")
                    raise ValueError(f"API error: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise ValueError(f"Unexpected error: {str(e)}")
        
        raise ValueError(f"Failed to get completion after {self.max_retries} attempts")
    
    def get_json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 800,
        parser: Optional[Callable[[str], T]] = None,
        json_fix_attempts: int = 2 # Max attempts to ask LLM to fix its own JSON
    ) -> Dict[str, Any]:
        """
        Get completion with JSON format enforcement and optional parsing.
        Includes retries if the LLM produces invalid JSON.
        
        Args:
            system_prompt: Instructions for the model
            user_prompt: The user query
            model: OpenAI model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            parser: Optional function to parse the JSON string
            json_fix_attempts: How many times to ask the LLM to fix its own malformed JSON.
            
        Returns:
            Dict with parsed content and usage statistics
        """
        current_json_fix_attempt = 0
        original_user_prompt = user_prompt # Keep original for retry

        while current_json_fix_attempt <= json_fix_attempts:
            try:
                # Add JSON format requirement
                response = self.get_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt, # This might be modified in retries
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                # Parse the JSON content
                json_content_str = response["content"]
                json_content = json.loads(json_content_str)
                
                # Apply custom parser if provided
                if parser:
                    json_content = parser(json_content)
                    
                # Update the response with parsed content
                response["content"] = json_content
                return response # Success
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response (Attempt {current_json_fix_attempt + 1}/{json_fix_attempts + 1}): {str(e)}")
                logger.error(f"Raw content: {json_content_str if 'json_content_str' in locals() else 'Content not available'}")
                
                current_json_fix_attempt += 1
                if current_json_fix_attempt <= json_fix_attempts:
                    logger.warning(f"Retrying JSON parsing by asking LLM to fix. Attempt {current_json_fix_attempt}.")
                    # Modify user_prompt to include the bad JSON and ask for a fix
                    user_prompt = (
                        f"The previous response resulted in a JSON parsing error. Please regenerate the response, ensuring it is a single, valid JSON object that strictly adheres to the schema. \n"
                        f"Original Prompt was: --- {original_user_prompt[:1000]}... ---\n"
                        f"Faulty JSON was: --- {json_content_str if 'json_content_str' in locals() else 'Content not available'} ---"
                        f"Please provide the corrected and complete JSON output only."
                    )
                    # No need to change system_prompt, it should still guide overall behavior
                    time.sleep(self._exponential_backoff(0)) # Small delay before retry for JSON fix
                else:
                    logger.error(f"Max JSON fix attempts reached. Raising ValueError.")
                    raise ValueError(f"Failed to parse JSON response after {json_fix_attempts + 1} attempts: {str(e)}. Last raw content: {json_content_str if 'json_content_str' in locals() else 'N/A'}")
            except ValueError as ve: # Catch other ValueErrors from self.get_completion (like budget)
                logger.error(f"ValueError during get_completion call within get_json_completion: {ve}")
                raise # Re-raise as this is not a JSON parsing issue to be fixed by this loop

        # Should not be reached if logic is correct, but as a failsafe:
        raise ValueError(f"Failed to get valid JSON response after all attempts.")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> Dict[str, Any]:
        """
        Get embedding vector for text with budget tracking.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Dict with embedding vector and usage statistics
        """
        # Project cost
        projected_cost = self.budget_tracker.project_cost(text, 0, model)
        
        # Check budget
        if not self.budget_tracker.check_budget(projected_cost):
            raise ValueError(f"Budget limit reached! Projected cost of {projected_cost:.4f} "
                             f"would exceed the remaining budget.")
        
        # Try with retries
        attempts = 0
        while attempts < self.max_retries:
            try:
                # Get embedding
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                tokens = response.usage.prompt_tokens
                
                # Log token usage (output tokens are 0 for embeddings)
                self.budget_tracker.log_usage(model, tokens, 0)
                
                return {
                    "embedding": embedding,
                    "usage": {
                        "tokens": tokens,
                        "cost": self.budget_tracker.PRICE_MAP[model]["input"] * tokens / 1_000_000
                    },
                    "model": model
                }
                
            except RateLimitError as e:
                attempts += 1
                if attempts < self.max_retries:
                    delay = self._exponential_backoff(attempts)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds... (Attempt {attempts}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {attempts} attempts: {str(e)}")
                    raise ValueError(f"Rate limit exceeded after {attempts} attempts: {str(e)}") from e
                    
            except (APIError, APITimeoutError) as e:
                attempts += 1
                is_server_error = hasattr(e, 'http_status') and e.http_status >= 500
                is_timeout = isinstance(e, APITimeoutError)

                if attempts < self.max_retries and (is_server_error or is_timeout):
                    delay = self._exponential_backoff(attempts)
                    logger.warning(f"API error ({type(e).__name__}), retrying in {delay:.2f} seconds... (Attempt {attempts}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"API error ({type(e).__name__}) not retryable or max retries reached: {str(e)}")
                    raise ValueError(f"API error: {str(e)}")
        
        raise ValueError(f"Failed to get embedding after {self.max_retries} attempts") 