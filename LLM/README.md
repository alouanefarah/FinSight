# FinSightLLM

A lightweight local conversational module using Phi-3 Mini through Ollama.

## Overview

FinSightLLM provides a simple Python interface for interacting with local language models through the Ollama API. It's specifically designed for financial analysis and insights using the Phi-3 Mini model, but can be configured for other models as well.

## Features

- 🤖 **Local LLM Integration**: Connect to Ollama-hosted models
- 💬 **Interactive Chat**: Command-line chat interface for testing
- ⚙️ **Configurable**: JSON-based configuration for model parameters
- 🔄 **Streaming Support**: Real-time response streaming
- 📊 **Financial Focus**: Optimized prompts for financial analysis

## Project Structure

```
FinsightLLM/
├── config/
│   └── llm_config.json    # Model configuration
├── llm_client.py          # Main LLM client module
├── chat_app.py           # Command-line chat application
├── __init__.py           # Package initialization
└── README.md             # This file
```

## Quick Start

1. **Install Ollama** and download the Phi-3 Mini model:
   ```bash
   ollama pull phi3:mini
   ```

2. **Configure the model** in `config/llm_config.json`

3. **Run the chat application**:
   ```bash
   python chat_app.py
   ```

## Configuration

Edit `config/llm_config.json` to customize model behavior:

```json
{
  "model_name": "phi3:mini",
  "base_url": "http://localhost:11434",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

## Usage

### Using the LLM Client

```python
from llm_client import FinSightLLMClient

client = FinSightLLMClient()
response = client.send_message("Analyze the current market trends")
print(response)
```

### Running the Chat App

```bash
python chat_app.py
```

## Requirements

- Python 3.8+
- Ollama server running locally
- Phi-3 Mini model (or compatible model)

## Development Status

🚧 **This project is currently in development.** Core functionality is being implemented.

## License

MIT License - See LICENSE file for details.