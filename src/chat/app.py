"""
Multi-Provider Chat Application

This is the main application file that ties together all components of the chat application.
It uses Streamlit for the UI and supports multiple LLM providers.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Import utility modules
from chat.util.logging_util import logger as llm_logger

# Import application modules
from chat.conversation.conversation_storage import ConversationStorage
from chat.ai.provider_manager import initialize_provider, get_available_providers
from chat.ui.conversation_manager import get_conversation, initialize_conversation_id
from chat.ui.sidebar import (
    render_provider_settings,
    render_conversation_management,
    render_current_conversation_details
)
from chat.ui.chat import (
    display_chat_messages,
    handle_user_input,
    initialize_chat_history
)


def setup_environment():
    """Load environment variables and configure logging."""
    # Load environment variables
    if load_dotenv():
        llm_logger.info(".env file loaded successfully.")
    else:
        llm_logger.warning(".env file not found. Relying on pre-set environment variables.")


def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Multi-Provider Chat App",
        page_icon="🤖",
        layout="wide"
    )
    
    # App title and description
    st.title("🤖 Multi-Provider Chat App")
    st.caption("Chat with multiple LLM providers using Streamlit and LiteLLM.")


@st.cache_resource
def get_conversation_storage():
    """Initialize and retrieve the conversation storage."""
    storage_dir = os.environ.get("CONVERSATION_STORAGE_DIR", "conversations")
    return ConversationStorage(storage_dir)


def main():
    """Main application function."""
    # Setup environment and page
    setup_environment()
    setup_page()

    # Get available providers
    providers = get_available_providers()

    # Get conversation storage
    conversation_storage = get_conversation_storage()

    # Create tabs in the sidebar
    sidebar_tabs = st.sidebar.tabs(["Chat Settings", "RAG"])

    from chat.rag.torch_fix import apply_torch_fix
    apply_torch_fix()

    # Initialize RAG service if not already done
    if "rag_service" not in st.session_state:
        try:
            from chat.rag.rag_service import RAGService
            st.session_state.rag_service = RAGService()
        except Exception as e:
            st.sidebar.error(f"Error initializing RAG service: {str(e)}")
            st.session_state.rag_service = None
            st.session_state.rag_enabled = False

    # Render standard sidebar components in the Chat Settings tab
    with sidebar_tabs[0]:
        # Provider settings
        selected_provider, selected_model, temperature, use_streaming = render_provider_settings(providers)

        # Conversation management
        render_conversation_management(conversation_storage, selected_provider, selected_model)

        # Current conversation details
        render_current_conversation_details(conversation_storage, selected_provider, selected_model)

    # Render RAG sidebar components in the RAG tab
    with sidebar_tabs[1]:
        # Import and use the RAG sidebar
        from chat.ui.sidebar_rag import render_rag_sidebar
        render_rag_sidebar()

    # Initialize provider (using selections from first tab)
    llm_provider, error_message = initialize_provider(selected_provider, selected_model)

    # Display error message if provider initialization failed
    if error_message:
        st.error(error_message)
        st.sidebar.error(f"Provider failed: {error_message}")

    # Initialize chat history
    initialize_chat_history(selected_provider, selected_model)

    # Initialize conversation
    initialize_conversation_id()
    conversation = get_conversation(conversation_storage)

    # Display existing chat messages
    display_chat_messages(st.session_state.messages)

    # Handle user input
    handle_user_input(
        llm_provider=llm_provider,
        conversation=conversation,
        conversation_storage=conversation_storage,
        selected_provider=selected_provider,
        selected_model=selected_model,
        temperature=temperature,
        use_streaming=use_streaming
    )

if __name__ == "__main__":
    main()
