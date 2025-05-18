# Synapz - Database and System Prompts

This module contains the core functionality for the Synapz adaptive learning system.

## Recent Additions

### Database Schema

The `synapz.core.models.Database` class implements a SQLite database with WAL journaling for concurrent access. It includes tables for:

- Learner profiles
- Concepts
- Teaching sessions
- Interactions
- Experiment metrics

Example usage:

```python
from synapz.core import Database
from synapz.data import (
    create_unified_experiment_from_concept,
    store_initial_interaction
)

# Initialize the database
db = Database()

# Create a teaching session
session_id, control_id = create_unified_experiment_from_concept(
    db=db,
    concept_id="variables", 
    learner_profile_id="adhd",
    experiment_type="adaptive",
    with_control=True
)

# Log an interaction
interaction_id = store_initial_interaction(
    db=db,
    session_id=session_id,
    content="Teaching content here...",
    teaching_strategy="visual",
    pedagogy_tags=["visual", "example-driven"],
    tokens_in=100,
    tokens_out=250
)

# Get history for a session
history = db.get_session_history(session_id)
```

### System Prompts

The system includes two types of prompts:

1. **Adaptive System Prompt** (`synapz/prompts/adaptive_system.txt`)
   - Tailors teaching to different cognitive profiles
   - Adapts based on clarity ratings and previous interactions
   - Returns structured JSON responses

2. **Control System Prompt** (`synapz/prompts/control_system.txt`)
   - Provides consistent teaching regardless of feedback
   - Used as a baseline for comparison
   - Returns the same JSON structure as adaptive prompts

To use these prompts:

```python
from synapz.core import ContentAdapter
from synapz.models.concepts import load_concept
from synapz.models.learner_profiles import get_profile_for_adaptation

# Get a concept and profile
concept = load_concept("variables")
profile = get_profile_for_adaptation("adhd")

# Get the appropriate prompt
adapter = ContentAdapter(api_client, profile_manager, PROMPTS_DIR)
prompt = adapter.get_system_prompt(
    experiment_type="adaptive",
    concept=concept,
    profile=profile,
    context={
        "turn_number": 1,
        "previous_clarity": None,
        "interaction_history": []
    }
)
```

## Integration with Existing Components

The `synapz.data.integration` module provides utilities to ensure the new Database class works seamlessly with the existing ExperimentStorage class. 