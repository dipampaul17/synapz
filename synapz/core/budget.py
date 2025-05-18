from typing import Dict, Tuple, Any
import tiktoken
import time
import sqlite3
import os
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    timestamp: float

class BudgetTracker:
    """Track token usage and costs with pre-call projections and hard limits."""
    
    # Current API prices ($/1M tokens) - May 2024
    PRICE_MAP = {
        # Latest GPT-4 models
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        
        # Older GPT-4 models
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        
        # GPT-3.5 models
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.0},
        
        # Embeddings models
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        
        # Fallback rates for unknown models
        "default": {"input": 10.0, "output": 30.0}  # Conservative estimate
    }
    
    def __init__(self, db_path: str, max_budget: float = 20.0):
        """Initialize with SQLite connection and budget ceiling."""
        self.db_path = db_path
        self.max_budget = max_budget
        self.total_cost = 0.0
        self._setup_db()
        
        # Load current spend from database
        self.total_cost = self.get_current_spend()
        logger.info(f"Budget tracker initialized with current spend: ${self.total_cost:.4f}")
    
    def _setup_db(self) -> None:
        """Set up SQLite database with WAL journaling for concurrent access."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")  # Prevent write-locks
        conn.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cost REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """)
        conn.commit()
        conn.close()
    
    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens for a given text and model."""
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except KeyError:
            # If model not found, use cl100k_base as fallback
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Using fallback tokenizer for unknown model: {model}")
                return len(enc.encode(text))
            except Exception as e:
                logger.error(f"Token counting error: {str(e)}")
                # Rough estimate: 1 token per 4 chars
                return len(text) // 4
    
    def get_model_rates(self, model: str) -> Dict[str, float]:
        """Get pricing rates for a model with fallback to similar models."""
        # Direct match
        if model in self.PRICE_MAP:
            return self.PRICE_MAP[model]
        
        # Try partial match (for model variants)
        for known_model in self.PRICE_MAP:
            if model.startswith(known_model):
                logger.info(f"Using price of {known_model} for {model}")
                return self.PRICE_MAP[known_model]
        
        # Use default pricing (conservative estimate) for unknown models
        logger.warning(f"Unknown model {model}, using default pricing.")
        return self.PRICE_MAP["default"]
    
    def project_cost(self, prompt: str, max_tokens: int, model: str) -> float:
        """Project cost before making API call."""
        prompt_tokens = self._count_tokens(prompt, model)
        # Assume worst case: max_tokens will be used
        rates = self.get_model_rates(model)
        cost = (prompt_tokens * rates["input"] + max_tokens * rates["output"]) / 1_000_000
        return cost
    
    def check_budget(self, projected_cost: float) -> bool:
        """Check if projected cost fits within remaining budget."""
        current_total = self.get_current_spend()
        
        # Update internal tracker
        self.total_cost = current_total
        
        # Would this call exceed our budget?
        return (current_total + projected_cost) < self.max_budget
    
    def log_usage(self, model: str, tokens_in: int, tokens_out: int) -> None:
        """Log token usage to SQLite and update running total."""
        rates = self.get_model_rates(model)
        cost = (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO token_usage (model, tokens_in, tokens_out, cost, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, tokens_in, tokens_out, cost, time.time())
        )
        conn.commit()
        conn.close()
        
        self.total_cost += cost
        logger.info(f"Logged usage: {model}, {tokens_in} in, {tokens_out} out, ${cost:.6f}, total: ${self.total_cost:.4f}")
        
    def get_current_spend(self) -> float:
        """Get current total spend from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT SUM(cost) FROM token_usage")
            result = cur.fetchone()
            conn.close()
            return result[0] if result[0] is not None else 0.0
        except Exception as e:
            logger.error(f"Error retrieving current spend: {str(e)}")
            return self.total_cost  # Fallback to in-memory tracker
    
    def is_exceeded(self) -> bool:
        """Check if the total cost has met or exceeded the max budget."""
        # Ensure total_cost is up-to-date with the database
        self.total_cost = self.get_current_spend()
        return self.total_cost >= self.max_budget

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of token usage by model."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Get usage by model
            cur.execute("""
                SELECT model, 
                       SUM(tokens_in) as total_in, 
                       SUM(tokens_out) as total_out, 
                       SUM(cost) as total_cost,
                       COUNT(*) as call_count
                FROM token_usage 
                GROUP BY model
            """)
            
            models_usage = []
            for row in cur.fetchall():
                models_usage.append({
                    "model": row[0],
                    "tokens_in": row[1],
                    "tokens_out": row[2],
                    "cost": row[3],
                    "calls": row[4]
                })
            
            # Get total usage
            cur.execute("""
                SELECT SUM(tokens_in), SUM(tokens_out), SUM(cost), COUNT(*)
                FROM token_usage
            """)
            total = cur.fetchone()
            
            conn.close()
            
            return {
                "total_tokens_in": total[0] if total[0] else 0,
                "total_tokens_out": total[1] if total[1] else 0,
                "total_cost": total[2] if total[2] else 0.0,
                "total_calls": total[3] if total[3] else 0,
                "remaining_budget": self.max_budget - (total[2] if total[2] else 0.0),
                "by_model": models_usage
            }
            
        except Exception as e:
            logger.error(f"Error generating usage summary: {str(e)}")
            return {
                "total_cost": self.total_cost,
                "remaining_budget": self.max_budget - self.total_cost,
                "error": str(e)
            } 