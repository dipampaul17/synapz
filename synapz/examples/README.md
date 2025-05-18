# examples

> demonstration scripts for synapz library components

## overview

this directory contains example scripts showing how to use various synapz components:

- `llm_client_example.py`: demonstrates the `LLMClient` class with budget tracking
- `teacher_example.py`: shows how to use the `TeacherAgent` for adaptive teaching
- `simple_eval_example.py`: provides a minimal example of evaluation metrics

## key features demonstrated

- ⚙️ budget enforcement with pre-call cost projection
- 🔄 automatic retries with exponential backoff
- 📊 token usage tracking
- 🧠 adaptive teaching based on cognitive profiles
- 📝 structured json output handling

## running examples

```bash
# ensure environment is set up
export OPENAI_API_KEY='your-api-key-here'
# or use a .env file at project root

# run an example
python -m synapz.examples.llm_client_example
```

## api key management

the examples use environment variables to manage your openai api key:

```python
# example of secure api key handling
api_key = os.environ.get("OPENAI_API_KEY")
```

❗ **note**: examples may incur charges to your openai account. each example sets a small budget limit to prevent excessive spending. 