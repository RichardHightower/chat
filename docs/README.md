# Multi-Provider Chat Application

A Streamlit-based chat application that supports multiple LLM providers, including OpenAI, Anthropic, Google Gemini, Perplexity, and now Ollama for local model inference.

## Features

- Multi-provider support (OpenAI, Anthropic, Google Gemini, Perplexity, Ollama)
- Conversation management (save, load, export)
- Temperature control
- Conversation context maintenance
- Responsive UI with Streamlit

## Setup and Installation

### Prerequisites

- Python 3.12 or later
- Poetry (for dependency management)

### Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

### API Keys

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
PERPLEXITY_API_KEY=your_perplexity_key
CONVERSATION_STORAGE_DIR=conversations
```

## Running the Application

```bash
poetry run streamlit run src/chat/app.py
```

## Using Ollama with the Chat App

### Installing Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start the Ollama server

### Installed Models

Your system has these powerful models installed:

```
gemma3:27b         - Google's 27B parameter model
qwen3:32b          - Alibaba's newer 32B parameter model
qwen:72b           - Alibaba's 72B parameter multilingual model
deepseek-r1:70b    - 70B parameter specialized reasoning model
llama3.3:latest    - Meta's Llama 3.3 model
llama4:scout       - Meta's newest Llama 4 Scout model
```

To install additional models:

```bash
# Pull additional models as needed
ollama pull mistral
ollama pull mixtral
ollama pull phi3
```

### Using Ollama in the Chat App

1. Start the chat application
2. In the sidebar, select "Ollama" as the provider
3. Choose your desired model from the dropdown
4. Verify the Ollama base URL (default: http://localhost:11434)
5. Click "Check Ollama Status" to ensure connectivity

## Model Capabilities

Here's what to expect from each of your installed Ollama models:

- **gemma3:27b**: Google's large 27B parameter model with excellent instruction following and reasoning
- **qwen3:32b**: Alibaba's newer 32B parameter model with improved capabilities over earlier versions
- **qwen:72b**: Alibaba's powerful 72B parameter model with strong multilingual capabilities
- **deepseek-r1:70b**: A 70B parameter model specialized for enhanced reasoning and problem-solving
- **llama3.3:latest**: Meta's Llama 3.3 model with improved capabilities over previous versions
- **llama4:scout**: Meta's newest model, part of the Llama 4 family, optimized for efficiency and performance

## Using Different Providers

The application supports multiple LLM providers:

- **OpenAI**: Powerful models like GPT-4o
- **Google Gemini**: Google's latest large language models
- **Anthropic**: Claude models with strong reasoning
- **Perplexity**: Research-focused models with web search capabilities
- **Ollama**: Local inference with open-source models

Select the provider and model in the sidebar to switch between them.

## Notes

- For large local models, ensure your system has sufficient RAM
- The first request to Ollama may take longer as the model is loaded into memory
- Adjust the temperature to control response creativity (lower for more deterministic answers)


# Multi-Provider Chat App

A Streamlit-based chat application that supports multiple LLM providers including OpenAI, Google Gemini, Perplexity, 
and Anthropic Claude.
This application provides a flexible chat interface that can connect to multiple LLM providers (OpenAI, Anthropic Claude, 
Google Gemini, and Perplexity) using LiteLLM. It includes conversation persistence functionality to save and load chat histories.

## Features

- Chat with multiple LLM providers from a single interface
- Switch between providers with a simple dropdown selection
- Adjust model parameters like temperature in real-time
- Persistent chat history across provider changes
- Clear chat history with a single click
- **Multiple LLM Providers**: Connect to OpenAI, Anthropic Claude, Google Gemini, or Perplexity
- **Model Selection**: Choose from different models for each provider
- **Conversation Persistence**: Save, load, and manage conversations
- **Conversation Context**: Maintain context across multiple messages
- **Temperature Control**: Adjust the temperature parameter for response creativity
- **Error Handling**: Robust error handling for API issues
- **Message Continuation**: Automatically continues responses that exceed token limits

## Supported Providers

- **OpenAI**: GPT-4o and GPT-4.1 models
- **Google Gemini**: Gemini-2-flash and Pro models
- **Perplexity**: Sonar Pro and online models
- **Anthropic**: Claude 3.7 Sonnet and other Claude models


## Project Structure

```
├── src/
│   └── chat/
│       ├── app.py                  # Main Streamlit application
│       ├── llm_provider.py         # Abstract base class for LLM providers
│       ├── open_ai.py              # OpenAI provider implementation
│       ├── anthropic.py            # Anthropic Claude provider implementation
│       ├── google_gemini.py        # Google Gemini provider implementation
│       ├── perplexity.py           # Perplexity provider implementation
│       ├── conversation.py         # Conversation and Message models
│       ├── conversation_storage.py # Conversation persistence utilities
│       ├── json_util.py            # JSON handling utilities
│       └── logging.py              # Logging configuration
├── conversations/                  # Directory for saved conversations
└── .env                            # Environment variables for API keys
```

## Environment Variables

The application uses the following environment variables:

- `OPENAI_API_KEY`: API key for OpenAI
- `ANTHROPIC_API_KEY`: API key for Anthropic
- `GOOGLE_API_KEY`: API key for Google Gemini
- `PERPLEXITY_API_KEY`: API key for Perplexity
- `CONVERSATION_STORAGE_DIR`: Directory for storing conversation files (default: "conversations")

## Conversation Model

The application uses Pydantic models to represent conversations and messages:

- `Message`: Represents an individual message in the conversation
  - `timestamp`: When the message was created
  - `message_type`: Either `INPUT` (user) or `OUTPUT` (assistant)
  - `content`: The actual message content
  - `role`: The role associated with the message (typically "user" or "assistant")

- `Conversation`: Represents a full conversation
  - `id`: Unique identifier for the conversation
  - `title`: Optional title for the conversation
  - `messages`: List of Message objects
  - `created_at`: When the conversation was created
  - `updated_at`: When the conversation was last updated

## Conversation Persistence

Conversations are saved as JSON files in the `conversations` directory. The `ConversationStorage` class provides methods for:

- Saving conversations
- Loading conversations
- Listing available conversations
- Deleting conversations
- Generating titles for conversations

## Usage

1. Set up your environment variables in a `.env` file
2. Run the application: `streamlit run src/chat/app.py`
3. Select a provider and model in the sidebar
4. Start chatting!
5. Use the conversation management tools in the sidebar to save, load, or export conversations

## API Usage

The application provides a consistent interface for different LLM providers through the `LLMProvider` abstract base class. 

Example usage:

```python
from chat.ai.open_ai import OpenAIProvider
from chat.conversation.conversation import Conversation

# Initialize provider
provider = OpenAIProvider(model="gpt-4o-2024-08-06")

# Create a conversation
conversation = Conversation(id="unique-id")

# Generate a completion with conversation context
response = await provider.generate_completion(
  prompt="Hello, how are you?",
  output_format="text",
  options={"temperature": 0.7},
  conversation=conversation  # Optional: provide conversation context
)

# The conversation object will be automatically updated with the new message
print(f"Response: {response}")
print(f"Messages in conversation: {len(conversation.messages)}")
```


## Dependencies

- Streamlit: Web application framework
- LiteLLM: Universal API for multiple LLM providers
- python-dotenv: Environment variable management
- jsonschema: JSON validation

## License

MIT

# Installation 

Install Python 3.12.9 if you need to. 

```
pyenv install 3.12.9
```

Ensure current project directory is using python 3.12.9.
```
pyenv local 3.12.9
```

Set up uv to create virtual environment in .venv.
```
uv venv
```

Activate the virtual environment.
```
source .venv/bin/activate
```

Init Poetry

```
poetry init --no-interaction --name chat-app \
            --description "A Streamlit chat app using LiteLLM and OpenAI" \
            --python "^3.12"
```



```
poetry add streamlit
poetry add litellm
poetry add python-dotenv
```
