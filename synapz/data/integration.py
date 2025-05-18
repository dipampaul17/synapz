"""Integration utilities for the new Database and existing ExperimentStorage."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import os
import json

from synapz.core.models import Database
from synapz.data.storage import ExperimentStorage
from synapz.models.learner_profiles import get_profile_for_adaptation
from synapz.models.concepts import load_concept
from synapz.core import CognitiveProfile

logger = logging.getLogger(__name__)

def migrate_experiment_storage_data(source_db_path: str, target_db_path: str) -> int:
    """
    Migrate data from ExperimentStorage to the new Database.
    
    Args:
        source_db_path: Path to the ExperimentStorage database
        target_db_path: Path to the new Database
        
    Returns:
        Number of migrated records
    """
    if not os.path.exists(source_db_path):
        logger.warning(f"Source database does not exist: {source_db_path}")
        return 0
        
    # Connect to both databases
    experiment_storage = ExperimentStorage(source_db_path)
    new_db = Database(target_db_path)
    
    # Migrate data
    migrated_count = 0
    
    # TODO: Implement migration logic from ExperimentStorage to Database
    # This would involve:
    # 1. Reading experiments from ExperimentStorage
    # 2. Creating corresponding sessions in Database
    # 3. Converting experiment content to interactions
    
    return migrated_count

def create_unified_experiment_from_concept(
    db: Database,
    concept_id: str,
    learner_profile_id: str,
    experiment_type: str = "adaptive",
    with_control: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Create a unified experiment session for a concept and learner profile.
    
    Args:
        db: Database instance
        concept_id: ID of the concept to teach
        learner_profile_id: ID of the learner profile
        experiment_type: "adaptive" or "control"
        with_control: Whether to create a paired control experiment
        
    Returns:
        Tuple of (experiment_id, control_id or None)
    """
    # Load concept and profile
    try:
        concept = load_concept(concept_id)
        profile = get_profile_for_adaptation(learner_profile_id)
    except ValueError as e:
        logger.error(f"Failed to load concept or profile: {str(e)}")
        raise
    
    # Create adaptive session
    session_id = db.create_session(
        learner_id=learner_profile_id,
        concept_id=concept_id,
        experiment_type=experiment_type
    )
    
    # Create control session if requested
    control_id = None
    if with_control and experiment_type == "adaptive":
        control_id = db.create_session(
            learner_id=learner_profile_id,
            concept_id=concept_id,
            experiment_type="control"
        )
    
    return session_id, control_id

def store_initial_interaction(
    db: Database,
    session_id: str,
    content: str,
    teaching_strategy: str,
    pedagogy_tags: List[str],
    tokens_in: int,
    tokens_out: int
) -> str:
    """
    Store the initial teaching interaction for a session.
    
    Args:
        db: Database instance
        session_id: ID of the session
        content: Teaching content
        teaching_strategy: Strategy used for teaching
        pedagogy_tags: List of pedagogy tags
        tokens_in: Tokens used in the prompt
        tokens_out: Tokens generated in the response
        
    Returns:
        Interaction ID
    """
    return db.log_interaction(
        session_id=session_id,
        turn_number=1,
        explanation=content,
        clarity_score=None,  # No clarity score for initial interaction
        teaching_strategy=teaching_strategy,
        pedagogy_tags=pedagogy_tags,
        tokens_in=tokens_in,
        tokens_out=tokens_out
    ) 