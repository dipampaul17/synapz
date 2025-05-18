"""Data storage and management for experimental results."""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class ExperimentStorage:
    """Store and retrieve experiment results for analysis."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        self._setup_db()
        
    def _setup_db(self) -> None:
        """Set up the SQLite database with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        
        # Create experiments table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile TEXT NOT NULL,
            topic TEXT NOT NULL,
            learning_objective TEXT NOT NULL,
            is_control INTEGER NOT NULL,
            content TEXT NOT NULL,
            tokens_in INTEGER NOT NULL,
            tokens_out INTEGER NOT NULL,
            cost REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """)
        
        # Create evaluation table for measuring effectiveness
        conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            score REAL NOT NULL,
            feedback TEXT,
            metric TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        """)
        
        conn.commit()
        conn.close()
        
    def store_experiment(
        self,
        profile: str,
        topic: str,
        learning_objective: str,
        is_control: bool,
        content: str,
        tokens_in: int,
        tokens_out: int,
        cost: float
    ) -> int:
        """
        Store experiment results.
        
        Returns:
            Experiment ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO experiments 
            (profile, topic, learning_objective, is_control, content, 
             tokens_in, tokens_out, cost, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile, 
                topic, 
                learning_objective, 
                1 if is_control else 0, 
                content,
                tokens_in,
                tokens_out,
                cost,
                time.time()
            )
        )
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return experiment_id
    
    def store_evaluation(
        self,
        experiment_id: int,
        score: float,
        metric: str,
        feedback: Optional[str] = None
    ) -> int:
        """
        Store evaluation results for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            score: Numeric score for the evaluation
            metric: Type of metric (e.g., "readability", "comprehension")
            feedback: Optional detailed feedback
            
        Returns:
            Evaluation ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO evaluations 
            (experiment_id, score, feedback, metric, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (experiment_id, score, feedback, metric, time.time())
        )
        
        evaluation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return evaluation_id
    
    def get_experiment_by_id(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        result = cursor.fetchone()
        
        if not result:
            return None
            
        experiment = dict(result)
        experiment["is_control"] = bool(experiment["is_control"])
        
        conn.close()
        return experiment
    
    def get_paired_experiments(self, topic: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get pairs of control/adapted experiments for the same topic.
        
        Returns list of (adapted, control) experiment pairs.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all experiments for this topic
        cursor.execute(
            "SELECT * FROM experiments WHERE topic = ? ORDER BY timestamp",
            (topic,)
        )
        experiments = [dict(row) for row in cursor.fetchall()]
        
        # Split into control and adapted
        control_exps = [e for e in experiments if e["is_control"]]
        adapted_exps = [e for e in experiments if not e["is_control"]]
        
        # Pair them by profile
        pairs = []
        for adapted in adapted_exps:
            profile = adapted["profile"]
            # Find matching control experiment
            for control in control_exps:
                if control["learning_objective"] == adapted["learning_objective"]:
                    pairs.append((adapted, control))
                    break
        
        conn.close()
        return pairs 