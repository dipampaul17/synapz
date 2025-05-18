# core module

> central components of the synapz adaptive learning system

## module contents

the `synapz` module contains the core functionality for creating adaptive learning experiences:

### key components

- 🧠 **`core/`**: core engine components
  - `adapter.py`: content adaptation for cognitive profiles
  - `api_client.py`: communication with external systems
  - `budget.py`: budget tracking and enforcement
  - `llm_client.py`: llm interaction with error handling
  - `models.py`: database schema and access
  - `profiles.py`: cognitive profile definitions
  - `simulator.py`: student feedback simulation
  - `teacher.py`: adaptive teaching agent

- 📊 **`data/`**: data management and metrics
  - `metrics.py`: learning effectiveness measurement
  - `analysis.py`: statistical analysis for adaptation
  - `visualization.py`: results visualization
  - `storage.py`: data persistence
  - `/concepts`: educational concept definitions
  - `/profiles`: cognitive profile definitions

- 📋 **`evaluate.py`**: batch experiment system for comparing adaptive vs control teaching

## database schema

the system uses sqlite with wal journaling for concurrent access, with tables for:

- learner profiles
- concepts
- teaching sessions
- interactions
- experiment metrics

```python
# example of core functionality
from synapz.core.models import Database
from synapz.core.teacher import TeacherAgent
from synapz.core.llm_client import LLMClient
from synapz.core.budget import BudgetTracker

# initialize components
db = Database()
budget = BudgetTracker(db_path="synapz.db", max_budget=5.0)
llm_client = LLMClient(budget_tracker=budget)
teacher = TeacherAgent(llm_client=llm_client, db=db)

# create a teaching session
session_id = teacher.create_session(
    learner_id="dyslexic_learner", 
    concept_id="equations",
    is_adaptive=True
)

# generate adaptive explanation
result = teacher.generate_explanation(session_id)
```

## system prompts

the system uses two types of prompts for experiments:

1. **adaptive system prompt** (`prompts/adaptive_system.txt`)
   - tailors teaching to different cognitive profiles
   - adapts based on feedback and learning patterns
   - structured for scientific comparison

2. **control system prompt** (`prompts/control_system.txt`)
   - provides consistent teaching regardless of profile
   - serves as scientific baseline for comparison
   - uses identical output structure as adaptive prompts

```python
# loading prompts example
from pathlib import Path
from synapz import PROMPTS_DIR

# adaptive prompt has full context of learner profile
with open(PROMPTS_DIR / "adaptive_system.txt", "r") as f:
    adaptive_prompt = f.read()
    
# formatted with learner profile, concept data, and interaction history
formatted_prompt = adaptive_prompt.format(
    learner_profile_json="...",
    concept_json="...",
    turn_number=1,
    previous_clarity="None",
    interaction_history=""
)
```

## integration with other components

the `synapz.data.integration` module provides utilities for compatibility between components:

```python
from synapz.core.models import Database
from synapz.data.integration import (
    export_experiment_metrics,
    import_legacy_data
)

# example of integrating with existing analysis tools
db = Database()
metrics_data = export_experiment_metrics(db, experiment_id="exp-123")
```
