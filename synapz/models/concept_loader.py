import json
from pathlib import Path
from typing import Dict, Any
import logging
import os

# print("CRITICAL_PRINT: synapz.models.concept_loader.py module PARSED AND LOADED") # Top-level print - REMOVED

logger = logging.getLogger(__name__)

def load_concept(concept_path_str: str) -> Dict[str, Any]:
    # Single prominent log message at the very start
    # logger.critical(f"CRITICAL_LOG: synapz.models.concept_loader.load_concept CALLED WITH: '{concept_path_str}'") # REMOVED DEBUG
    
    concept_path_obj = Path(concept_path_str)
    
    if not concept_path_obj.exists():
        logger.error(f"File does not exist: {concept_path_obj.resolve()} (Called from concept_loader.py)")
        raise FileNotFoundError(f"Concept file not found (from synapz.models.concept_loader.load_concept): {concept_path_obj.resolve()}")
    
    try:
        with open(concept_path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON from: {concept_path_obj.resolve()} (Called from concept_loader.py)")
            return data
    except FileNotFoundError: 
        logger.error(f"FileNotFoundError during open() for: {concept_path_obj.resolve()} (Called from concept_loader.py)")
        raise 
    except json.JSONDecodeError as e: # Specifically catch JSON errors
        logger.error(f"Invalid JSON in concept file: {concept_path_obj.resolve()}. Error: {e} (Called from concept_loader.py)")
        raise ValueError(f"Invalid JSON in concept file: {concept_path_obj.resolve()}. Error: {e}")
    except Exception as e:
        logger.error(f"Other exception during open/json.load for {concept_path_obj.resolve()}: {e} (Called from concept_loader.py)")
        raise 