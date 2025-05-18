# Synapz Examples

This directory contains example scripts demonstrating how to use the Synapz library components.

## LLM Client Example

The `llm_client_example.py` script demonstrates how to use the new `LLMClient` class for OpenAI API interactions with:

1. Budget enforcement
2. Automatic retries with exponential backoff
3. Token tracking
4. Pre-call cost projection
5. Structured JSON output
6. Embedding generation

### API Key Management

The example uses python-dotenv to securely manage your OpenAI API key:

1. When you first run the example, it will create a `.env` file at the project root
2. The `.env` file is excluded from version control via `.gitignore`
3. The API key is loaded from the environment, not hardcoded in source files

If you want to use a different API key, simply edit the `.env` file.

### Running the Example

```bash
# Install required dependencies
pip install -r requirements.txt

# Run the example
python -m synapz.examples.llm_client_example
```

The example will demonstrate:
- Basic text completion
- JSON-formatted completion
- Embedding generation
- Budget tracking

## Disclaimer

These examples may incur charges to your OpenAI account. The example script sets a small budget limit ($1) to prevent excessive spending. 