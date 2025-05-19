"""Database models and schema for Synapz."""

import sqlite3
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import logging

from synapz import PACKAGE_ROOT, DATA_DIR

# Setup logger for this module
logger = logging.getLogger(__name__)

class Database:
    """SQLite database with WAL journaling to handle concurrent access."""
    
    def __init__(self, db_path: str = str(DATA_DIR / "synapz.db")):
        """Initialize database connection with WAL mode."""
        self.db_path = db_path
        self._setup_db()
    
    def _setup_db(self) -> None:
        """Create tables and set WAL journaling mode."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode to prevent write locks
        conn.execute("PRAGMA journal_mode=WAL;")
        
        # Create learner profiles table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS learner_profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            profile_data TEXT NOT NULL
        )
        """)
        
        # Create concepts table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            difficulty INTEGER NOT NULL,
            description TEXT NOT NULL
        )
        """)
        
        # Create sessions table with experiment type (control or adaptive)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            experiment_type TEXT NOT NULL,  -- 'control' or 'adaptive'
            start_time REAL NOT NULL,
            end_time REAL,
            FOREIGN KEY (learner_id) REFERENCES learner_profiles (id),
            FOREIGN KEY (concept_id) REFERENCES concepts (id)
        )
        """)
        
        # Create interactions table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            explanation TEXT NOT NULL,
            clarity_score INTEGER,
            teaching_strategy TEXT,
            pedagogy_tags TEXT,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cost REAL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        """)
        
        # Create metrics table for experiment results
        conn.execute("DROP TABLE IF EXISTS experiment_metrics;") # Drop old table if exists for clean re-creation
        conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_metrics (
            id TEXT PRIMARY KEY,                        -- Unique ID for this metric entry (e.g., uuid)
            experiment_pair_id TEXT UNIQUE NOT NULL,    -- Identifier for the (learner, concept, timestamp) trial 
            adaptive_session_id TEXT NOT NULL,          -- FK to sessions table
            control_session_id TEXT NOT NULL,           -- FK to sessions table
            learner_id TEXT NOT NULL,                   -- Store for easier querying
            concept_id TEXT NOT NULL,                   -- Store for easier querying
            turns_per_session INT NOT NULL,             -- Number of turns in each session of the pair
            
            -- Key clarity metrics
            initial_adaptive_clarity REAL,
            final_adaptive_clarity REAL,
            adaptive_clarity_improvement REAL,
            initial_control_clarity REAL,
            final_control_clarity REAL,
            control_clarity_improvement REAL,
            clarity_improvement_difference REAL,        -- (Adaptive_Imp - Control_Imp)
            
            -- Key readability metrics (example: Flesch Reading Ease for final explanation)
            final_adaptive_flesch_ease REAL,
            final_control_flesch_ease REAL,
            readability_diff_flesch_ease REAL,
            
            -- Key content difference metrics (final explanation)
            text_similarity_final_explanation REAL,     -- e.g., Levenshtein similarity
            tag_jaccard_similarity_final REAL,      -- Jaccard similarity for pedagogy tags
            
            -- Cost metrics
            adaptive_session_cost REAL,
            control_session_cost REAL,
            total_pair_cost REAL,

            -- Full detailed metrics (JSON blob)
            metrics_data TEXT NOT NULL,                 -- JSON blob with all detailed comparisons, turn-by-turn data, etc.
            timestamp REAL NOT NULL,                    -- Timestamp of when this metric entry was created
            
            FOREIGN KEY (adaptive_session_id) REFERENCES sessions (id),
            FOREIGN KEY (control_session_id) REFERENCES sessions (id),
            FOREIGN KEY (learner_id) REFERENCES learner_profiles (id),
            FOREIGN KEY (concept_id) REFERENCES concepts (id)
        )
        """)
        
        self._create_reasoning_details_table(conn)
        
        conn.commit()
        conn.close()
    
    def _create_reasoning_details_table(self, conn: sqlite3.Connection) -> None:
        """Create the reasoning_details table if it doesn't exist."""
        conn.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_details (
            id TEXT PRIMARY KEY,
            interaction_id TEXT NOT NULL,
            condition TEXT NOT NULL, -- "baseline", "visible_reasoning", "hidden_reasoning"
            reasoning_process_text TEXT,
            metacognitive_supports_json TEXT, -- JSON list
            clarity_check_text TEXT,
            clarity_rating_initial INTEGER,
            clarity_rating_final INTEGER,
            clarity_improvement INTEGER,
            -- Optional: Token counts for individual components if captured
            -- tokens_reasoning_process INTEGER,
            -- tokens_metacognitive_supports INTEGER,
            -- tokens_clarity_check INTEGER,
            timestamp REAL NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
        """)
        logger.info("Reasoning details table schema checked/created.")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details by ID."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def update_interaction_clarity(self, interaction_id: str, clarity_score: int) -> bool:
        """Update the clarity score for an interaction."""
        if not (isinstance(clarity_score, int) and 1 <= clarity_score <= 5):
            #return False
            raise ValueError(f"Clarity score must be an integer between 1 and 5, got {clarity_score} (type: {type(clarity_score)})")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "UPDATE interactions SET clarity_score = ? WHERE id = ?",
            (clarity_score, interaction_id)
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
        
    def create_session(self, learner_id: str, concept_id: str, 
                      experiment_type: str) -> str:
        """Create a new teaching session and return its ID."""
        # Generate a unique session ID to avoid conflicts
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}_{learner_id}_{concept_id}"
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO sessions (id, learner_id, concept_id, experiment_type, start_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, learner_id, concept_id, experiment_type, time.time())
        )
        conn.commit()
        conn.close()
        
        return session_id
    
    def log_interaction(self, session_id: str, turn_number: int, explanation: str,
                        clarity_score: Optional[int], teaching_strategy: str,
                        pedagogy_tags: List[str], tokens_in: int, tokens_out: int,
                        cost: Optional[float]) -> str:
        """Log an interaction in a teaching session."""
        interaction_id = f"interaction_{int(time.time())}_{uuid.uuid4().hex[:8]}_{session_id}_{turn_number}"
        
        # Ensure pedagogy_tags is properly serialized to JSON
        if isinstance(pedagogy_tags, list):
            pedagogy_tags_json = json.dumps(pedagogy_tags)
        elif isinstance(pedagogy_tags, str):
            # Already a JSON string
            try:
                # Validate it's proper JSON
                json.loads(pedagogy_tags)
                pedagogy_tags_json = pedagogy_tags
            except json.JSONDecodeError:
                # Not valid JSON, wrap it as a single-item list
                pedagogy_tags_json = json.dumps([pedagogy_tags])
        else:
            # Unknown type, convert to string and wrap in list
            logger.warning(f"Pedagogy tags were not a list or string, received type {type(pedagogy_tags)}. Converting to string list.")
            pedagogy_tags_json = json.dumps([str(pedagogy_tags)])

        # --- DEBUG PRINTS (converted to logger.debug) ---
        current_time = time.time() # Moved up for logging
        logger.debug(f"DB.LOG_INTERACTION - interaction_id: {interaction_id} (type: {type(interaction_id)})")
        logger.debug(f"DB.LOG_INTERACTION - session_id: {session_id} (type: {type(session_id)})")
        logger.debug(f"DB.LOG_INTERACTION - turn_number: {turn_number} (type: {type(turn_number)})")
        logger.debug(f"DB.LOG_INTERACTION - explanation (first 60 chars): {explanation[:60]}... (type: {type(explanation)})")
        logger.debug(f"DB.LOG_INTERACTION - clarity_score: {clarity_score} (type: {type(clarity_score)})")
        logger.debug(f"DB.LOG_INTERACTION - teaching_strategy: {teaching_strategy} (type: {type(teaching_strategy)})")
        logger.debug(f"DB.LOG_INTERACTION - pedagogy_tags_json: {pedagogy_tags_json} (type: {type(pedagogy_tags_json)})")
        logger.debug(f"DB.LOG_INTERACTION - tokens_in: {tokens_in} (type: {type(tokens_in)})")
        logger.debug(f"DB.LOG_INTERACTION - tokens_out: {tokens_out} (type: {type(tokens_out)})")
        logger.debug(f"DB.LOG_INTERACTION - cost: {cost} (type: {type(cost)})")
        logger.debug(f"DB.LOG_INTERACTION - timestamp: {current_time} (type: {type(current_time)})")
        # --- END DEBUG PRINTS ---
            
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO interactions (id, session_id, turn_number, explanation, "
                "clarity_score, teaching_strategy, pedagogy_tags, tokens_in, tokens_out, cost, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (interaction_id, session_id, turn_number, explanation, clarity_score,
                 teaching_strategy, pedagogy_tags_json, tokens_in, tokens_out, cost, current_time)
            )
            conn.commit()
        except Exception as e:
            # Log the error with more detail using the logger
            logger.error(f"DB.LOG_INTERACTION - SQLite Error: {type(e).__name__}: {e}")
            logger.error(f"  Parameters: interaction_id={interaction_id}, session_id={session_id}, turn_number={turn_number}, clarity_score={clarity_score} (type: {type(clarity_score)}), ...")
            # Remove detailed print to console, rely on logger and raised exception
            # print(f"ERROR in db.log_interaction execute: {type(e).__name__}: {e}")
            # print(f"  Parameters passed to execute:")
            # print(f"    interaction_id: {interaction_id} (type: {type(interaction_id)})")
            # print(f"    session_id: {session_id} (type: {type(session_id)})")
            # print(f"    turn_number: {turn_number} (type: {type(turn_number)})")
            # print(f"    explanation: {explanation[:60]}... (type: {type(explanation)})")
            # print(f"    clarity_score: {clarity_score} (type: {type(clarity_score)}) <--- PARAMETER 4")
            # print(f"    teaching_strategy: {teaching_strategy} (type: {type(teaching_strategy)})")
            # print(f"    pedagogy_tags_json: {pedagogy_tags_json} (type: {type(pedagogy_tags_json)})")
            # print(f"    tokens_in: {tokens_in} (type: {type(tokens_in)})")
            # print(f"    tokens_out: {tokens_out} (type: {type(tokens_out)})")
            # print(f"    cost: {cost} (type: {type(cost)})")
            # print(f"    current_time: {current_time} (type: {type(current_time)})")
            raise
        finally:
            conn.close()
        
        return interaction_id
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all interactions for a session."""
        conn = self._get_connection()
        
        cursor = conn.execute(
            "SELECT * FROM interactions WHERE session_id = ? ORDER BY turn_number",
            (session_id,)
        )
        
        interactions = []
        for row in cursor:
            interaction = dict(row)
            interaction['pedagogy_tags'] = json.loads(interaction['pedagogy_tags'])
            interactions.append(interaction)
        
        conn.close()
        return interactions
    
    def log_experiment_metrics(self, session_id: str, metrics: Dict[str, Any]) -> str:
        """Save experiment metrics for a completed session."""
        # This method is now for the OLD schema. It will be replaced / updated.
        # For now, let's make it a pass or log a warning if called.
        logger.warning("log_experiment_metrics for single session is deprecated. Use log_paired_experiment_metrics.")
        return f"deprecated_metrics_log_for_{session_id}"

    def log_paired_experiment_metrics(
        self,
        experiment_pair_id: str,
        adaptive_session_id: str,
        control_session_id: str,
        learner_id: str,
        concept_id: str,
        turns_per_session: int,
        key_metrics: Dict[str, Any], # Dict containing the pre-calculated key scalar metrics for direct columns
        full_metrics_data: Dict[str, Any] # The complete JSON blob
    ) -> str:
        """Save metrics for a paired adaptive/control experiment."""
        metrics_entry_id = f"exp_metrics_{uuid.uuid4().hex[:12]}"
        current_time = time.time()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO experiment_metrics (
                    id, experiment_pair_id, adaptive_session_id, control_session_id, 
                    learner_id, concept_id, turns_per_session,
                    initial_adaptive_clarity, final_adaptive_clarity, adaptive_clarity_improvement,
                    initial_control_clarity, final_control_clarity, control_clarity_improvement,
                    clarity_improvement_difference,
                    final_adaptive_flesch_ease, final_control_flesch_ease, readability_diff_flesch_ease,
                    text_similarity_final_explanation, tag_jaccard_similarity_final,
                    adaptive_session_cost, control_session_cost, total_pair_cost,
                    metrics_data, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (   metrics_entry_id, experiment_pair_id, adaptive_session_id, control_session_id,
                    learner_id, concept_id, turns_per_session,
                    key_metrics.get('initial_adaptive_clarity'), key_metrics.get('final_adaptive_clarity'), key_metrics.get('adaptive_clarity_improvement'),
                    key_metrics.get('initial_control_clarity'), key_metrics.get('final_control_clarity'), key_metrics.get('control_clarity_improvement'),
                    key_metrics.get('clarity_improvement_difference'),
                    key_metrics.get('final_adaptive_flesch_ease'), key_metrics.get('final_control_flesch_ease'), key_metrics.get('readability_diff_flesch_ease'),
                    key_metrics.get('text_similarity_final_explanation'), key_metrics.get('tag_jaccard_similarity_final'),
                    key_metrics.get('adaptive_session_cost'), key_metrics.get('control_session_cost'), key_metrics.get('total_pair_cost'),
                    json.dumps(full_metrics_data), current_time
                )
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite error in log_paired_experiment_metrics: {e} - experiment_pair_id: {experiment_pair_id}")
            logger.error(f"Key Metrics: {key_metrics}")
            raise
        finally:
            conn.close()
        return metrics_entry_id
        
    def complete_session(self, session_id: str) -> None:
        """Mark a session as completed by setting its end time."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (time.time(), session_id)
        )
        conn.commit()
        conn.close()
    
    def get_control_adaptive_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Retrieve paired control and adaptive sessions for comparison."""
        conn = self._get_connection()
        
        # Get all completed sessions
        cursor = conn.execute(
            "SELECT * FROM sessions WHERE end_time IS NOT NULL"
        )
        
        sessions = [dict(row) for row in cursor.fetchall()]
        
        # Separate control and adaptive sessions
        control_sessions = [s for s in sessions if s["experiment_type"] == "control"]
        adaptive_sessions = [s for s in sessions if s["experiment_type"] == "adaptive"]
        
        # Find matching pairs (same learner and concept)
        pairs = []
        for adaptive in adaptive_sessions:
            for control in control_sessions:
                if (adaptive["learner_id"] == control["learner_id"] and 
                    adaptive["concept_id"] == control["concept_id"]):
                    pairs.append((adaptive, control))
                    break
        
        conn.close()
        return pairs

    def log_reasoning_detail(
        self,
        interaction_id: str,
        condition: str,
        reasoning_process_text: Optional[str],
        metacognitive_supports: Optional[List[str]],
        clarity_check_text: Optional[str],
        clarity_ratings: Dict[str, Optional[int]]
    ) -> str:
        """Logs a reasoning detail entry linked to an interaction."""
        reasoning_detail_id = f"reasoning_{uuid.uuid4().hex[:12]}_{interaction_id}"
        metacognitive_supports_json = json.dumps(metacognitive_supports if metacognitive_supports is not None else [])
        
        sql = """
            INSERT INTO reasoning_details (
                id, interaction_id, condition, reasoning_process_text,
                metacognitive_supports_json, clarity_check_text,
                clarity_rating_initial, clarity_rating_final, clarity_improvement,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            reasoning_detail_id,
            interaction_id,
            condition,
            reasoning_process_text,
            metacognitive_supports_json,
            clarity_check_text,
            clarity_ratings.get("initial"),
            clarity_ratings.get("final"),
            clarity_ratings.get("improvement"),
            time.time()
        )
        
        conn = self._get_connection()
        try:
            conn.execute(sql, params)
            conn.commit()
            logger.info(f"Logged reasoning detail {reasoning_detail_id} for interaction {interaction_id}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while logging reasoning detail for interaction {interaction_id}: {e}")
            # conn.rollback() # Not strictly necessary if commit isn't reached or if connection context manager used
            raise # Re-raise the exception to make the caller aware
        finally:
            conn.close()
            
        return reasoning_detail_id 