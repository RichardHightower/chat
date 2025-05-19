# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Installation

```bash
# Install dependencies using Poetry
poetry install

# Run the Streamlit application
poetry run streamlit run src/chat/app.py
```

## Architecture Overview

This is a multi-provider chat application built with Streamlit that supports various LLM providers:

1. **Provider System**: The application uses an abstract `LLMProvider` base class that defines a common interface for all LLM provider implementations:
   - `OpenAIProvider`: Integrates with OpenAI models
   - `AnthropicProvider`: Integrates with Anthropic Claude models
   - `GoogleGeminiProvider`: Integrates with Google Gemini models
   - `PerplexityProvider`: Integrates with Perplexity models
   - `OllamaProvider`: Integrates with local Ollama models

2. **Conversation Management**: The application maintains conversation state using:
   - `Conversation`: Pydantic model representing a chat conversation with messages
   - `ConversationStorage`: Handles saving/loading conversations to/from JSON files

3. **UI Components**: Streamlit-based UI is organized into:
   - `sidebar.py`: Provider selection, settings, and conversation management
   - `chat.py`: Chat interface and message handling
   - `conversation_manager.py`: UI logic for conversation state management
   
4. **Application Flow**:
   - User selects a provider and model in the sidebar
   - User input is processed through the selected provider
   - Conversation history is maintained and can be saved/loaded
   - Providers handle different model-specific parameters and formats

5. **Environment Configuration**:
   - API keys for different providers are stored in a `.env` file
   - `CONVERSATION_STORAGE_DIR` configures where conversations are saved

The codebase follows a clean modular architecture with separation of concerns between providers, conversation management, and UI components.