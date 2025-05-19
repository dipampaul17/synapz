import openai
import os
import json
import time
import tiktoken
from typing import Dict, Any, Tuple, Optional
import logging
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Ensure .env is in the root directory or adjust load_dotenv() call if needed.
# Example: load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')
load_dotenv()

# Default retry parameters
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 60.0     # seconds

# Pricing information (per token, not per 1k tokens)
# Source: https://openai.com/pricing (check for updates)
PRICING_INFO = {
    "gpt-4o": {"prompt_token_cost": 0.005 / 1000, "completion_token_cost": 0.015 / 1000},
    "gpt-4-turbo": {"prompt_token_cost": 0.01 / 1000, "completion_token_cost": 0.03 / 1000},
    "gpt-3.5-turbo": {"prompt_token_cost": 0.0005 / 1000, "completion_token_cost": 0.0015 / 1000},
    # Add other models as their pricing becomes relevant
}

class LLMClient:
    """Client for interacting with OpenAI's Large Language Models."""

    def __init__(self, 
                 api_key: Optional[str] = None, 
                 max_retries: int = DEFAULT_MAX_RETRIES, 
                 initial_backoff: float = DEFAULT_INITIAL_BACKOFF, 
                 max_backoff: float = DEFAULT_MAX_BACKOFF):
        """
        Initialize the LLMClient.

        Args:
            api_key: OpenAI API key. If None, loads from OPENAI_API_KEY env var.
            max_retries: Maximum number of retries for API calls.
            initial_backoff: Initial backoff time in seconds for retries.
            max_backoff: Maximum backoff time in seconds.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it directly.")
            raise ValueError("OpenAI API key not found.")
        
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self._tokenizer_cache: Dict[str, tiktoken.Encoding] = {}

    def _get_tokenizer(self, model_name: str) -> tiktoken.Encoding:
        """Get tokenizer for a given model, caching it for efficiency."""
        if model_name not in self._tokenizer_cache:
            try:
                self._tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"No tokenizer found for model {model_name}. Using cl100k_base as default.")
                self._tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer_cache[model_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count the number of tokens in a text string for a given model.
        """
        if not text:
            return 0
        tokenizer = self._get_tokenizer(model_name)
        return len(tokenizer.encode(text))

    def calculate_cost(self, tokens_in: int, tokens_out: int, model_name: str) -> float:
        """
        Calculate the estimated cost of an API call based on token counts and model.
        """
        model_pricing = PRICING_INFO.get(model_name)
        if not model_pricing:
            # Try to find a base model if it's a versioned model e.g. gpt-4-turbo-preview
            base_model_name = '-'.join(model_name.split('-')[:2]) # e.g. "gpt-4"
            model_pricing = PRICING_INFO.get(base_model_name)
            if not model_pricing:
                logger.warning(f"Pricing info not found for model {model_name} or base {base_model_name}. Using gpt-3.5-turbo pricing as fallback.")
                model_pricing = PRICING_INFO.get("gpt-3.5-turbo")
                if not model_pricing: # Should not happen if gpt-3.5-turbo is in PRICING_INFO
                    logger.error("Fallback pricing for gpt-3.5-turbo not found. Cost will be 0.")
                    return 0.0
        
        prompt_cost = tokens_in * model_pricing["prompt_token_cost"]
        completion_cost = tokens_out * model_pricing["completion_token_cost"]
        return prompt_cost + completion_cost

    def get_json_completion(
        self,
        system_prompt: str,
        user_prompt: str, 
        model: str,
        temperature: float = 0.5,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Get a completion from the LLM, expecting a JSON response.
        Uses exponential backoff for retries on failures.

        Returns:
            Dict with "content" (parsed JSON) and "usage" (tokens_in, tokens_out, cost).
            If JSON parsing fails or API error occurs after retries, 
            content will be a dict with "error" key and details.
        """
        current_backoff = self.initial_backoff
        retries = 0
        
        # Pre-calculate initial prompt tokens for logging and cost estimation of failed attempts
        projected_prompt_tokens = self.count_tokens(system_prompt + user_prompt, model)

        while retries <= self.max_retries:
            try:
                logger.info(f"Attempting API call to {model} (Attempt {retries + 1}/{self.max_retries + 1}). Projected prompt tokens: {projected_prompt_tokens}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}, 
                )
                
                response_content_str = response.choices[0].message.content
                
                # Use actual token counts from response if available
                actual_prompt_tokens = response.usage.prompt_tokens if response.usage else projected_prompt_tokens
                completion_tokens = response.usage.completion_tokens if response.usage else self.count_tokens(response_content_str or "", model)
                
                cost = self.calculate_cost(actual_prompt_tokens, completion_tokens, model)
                logger.info(f"API call successful. Prompt Tokens: {actual_prompt_tokens}, Completion Tokens: {completion_tokens}, Cost: ${cost:.6f}")

                if response_content_str is None: # Should not happen with successful API call asking for content
                    response_content_str = "{}"
                    logger.warning("API returned None for response content. Treating as empty JSON.")
                
                try:
                    parsed_json = json.loads(response_content_str)
                    return {
                        "content": parsed_json,
                        "usage": {
                            "tokens_in": actual_prompt_tokens,
                            "tokens_out": completion_tokens,
                            "cost": cost,
                        },
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}. Response: {response_content_str[:500]}")
                    return {
                        "content": {"error": "JSONDecodeError", "message": str(e), "raw_response": response_content_str},
                        "usage": {"tokens_in": actual_prompt_tokens, "tokens_out": completion_tokens, "cost": cost}
                    }

            except openai.APIConnectionError as e:
                logger.warning(f"API Connection Error for model {model}: {e}. Retrying in {current_backoff:.2f}s...")
            except openai.RateLimitError as e:
                logger.warning(f"Rate Limit Error for model {model}: {e}. Retrying in {current_backoff:.2f}s...")
            except openai.APITimeoutError as e:
                logger.warning(f"API Timeout Error for model {model}: {e}. Retrying in {current_backoff:.2f}s...")
            except openai.APIStatusError as e:
                logger.error(f"API Status Error for model {model} (status {e.status_code}): {e.message}. Retrying in {current_backoff:.2f}s...")
            except Exception as e:
                logger.error(f"Unexpected error during API call to {model}: {e}", exc_info=True)
            
            retries += 1
            if retries > self.max_retries:
                logger.error(f"Max retries ({self.max_retries}) exceeded for API call to {model}.")
                # Calculate cost based on all failed prompt attempts
                total_failed_prompt_cost = self.calculate_cost(projected_prompt_tokens * retries, 0, model)
                return {
                    "content": {"error": "MaxRetriesExceeded", "message": f"Failed after {retries} attempts."},
                    "usage": {"tokens_in": projected_prompt_tokens * retries, "tokens_out": 0, "cost": total_failed_prompt_cost}
                }
            
            time.sleep(current_backoff)
            current_backoff = min(current_backoff * 2, self.max_backoff) # Exponential backoff
        
        # Fallback, should ideally not be reached if loop logic is correct
        logger.critical("Exited retry loop unexpectedly in get_json_completion.")
        return {
            "content": {"error": "UnexpectedLoopExit", "message": "Critical error in retry logic."},
            "usage": {"tokens_in": 0, "tokens_out": 0, "cost": 0.0}
        }

if __name__ == '__main__':
    # To run this test: python -m synapz.core.llm_client (from the project root)
    # Ensure you have a .env file in the project root with your OPENAI_API_KEY
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("LLMClient Test Script Started")
    
    try:
        client = LLMClient()
        logger.info("LLMClient initialized successfully.")

        test_model = "gpt-4o" # Or "gpt-3.5-turbo"
        
        # Test token counting
        test_text_tokens = "This is a test sentence for token counting."
        count = client.count_tokens(test_text_tokens, test_model)
        logger.info(f"Token count for '{test_text_tokens}' with {test_model}: {count}")

        # Test cost calculation
        prompt_tokens_cost = 150
        completion_tokens_cost = 250
        cost_estimate = client.calculate_cost(prompt_tokens_cost, completion_tokens_cost, test_model)
        logger.info(f"Estimated cost for {prompt_tokens_cost} prompt tokens and {completion_tokens_cost} completion tokens with {test_model}: ${cost_estimate:.6f}")

        # Test JSON completion
        logger.info(f"\nTesting JSON completion with model: {test_model}...")
        sys_prompt = "You are an assistant that provides information about planets. Respond in JSON format."
        usr_prompt = "Tell me about Mars. Include its diameter and primary atmospheric component."
        
        api_result = client.get_json_completion(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            model=test_model,
            temperature=0.2,
            max_tokens=200
        )
        
        logger.info("API Response Received:")
        print(json.dumps(api_result, indent=2))

        if "content" in api_result and isinstance(api_result["content"], dict) and "error" not in api_result["content"]:
            logger.info("Successfully received and parsed JSON response from API.")
        else:
            logger.error("Failed to get a valid JSON response from API or an error was reported in the content.")
            if "content" in api_result and "error" in api_result["content"]:
                logger.error(f"Error details: {api_result['content']['error']} - {api_result['content'].get('message')}")

    except ValueError as ve:
        logger.error(f"ValueError during LLMClient test setup or execution: {ve}", exc_info=True)
    except Exception as ex:
        logger.error(f"An unexpected error occurred during LLMClient test: {ex}", exc_info=True)

    logger.info("LLMClient Test Script Finished")