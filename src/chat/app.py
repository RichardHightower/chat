"""
Multi-Provider Chat Application

This is the main application file that ties together all components of the chat application.
It uses Streamlit for the UI and supports multiple LLM providers.
"""

import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add src directory to Python path if not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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
    sidebar_tabs = st.sidebar.tabs(["Chat Settings", "RAG", "Search"])

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
    
    # Render Search interface in the Search tab
    with sidebar_tabs[2]:
        # Debug information
        has_rag_service = "rag_service" in st.session_state and st.session_state.rag_service is not None
        has_rag_project = "rag_project" in st.session_state and st.session_state.rag_project is not None
        has_project_id = "current_rag_project_id" in st.session_state and st.session_state.current_rag_project_id is not None
        
        # Show debug info
        with st.expander("Debug Info", expanded=False):
            st.write(f"RAG Service exists: {has_rag_service}")
            st.write(f"RAG Project exists: {has_rag_project}")
            st.write(f"Project ID exists: {has_project_id}")
            if has_project_id:
                st.write(f"Project ID: {st.session_state.current_rag_project_id}")
            st.write("Session state keys:", list(st.session_state.keys()))
        
        if has_rag_service and has_project_id:
            # Try to get the project if we only have the ID
            if not has_rag_project and st.session_state.rag_service:
                try:
                    projects = st.session_state.rag_service.list_projects()
                    project = next((p for p in projects if p.id == st.session_state.current_rag_project_id), None)
                    if project:
                        st.session_state.rag_project = project
                        has_rag_project = True
                except Exception as e:
                    st.error(f"Error loading project: {str(e)}")
        
        if has_rag_service and has_rag_project:
            try:
                from chat.ui.search_interface import render_search_interface, render_search_statistics
            except ImportError as e:
                st.error(f"Failed to import search interface: {e}")
                render_search_interface = None
                render_search_statistics = None
            
            # Add a toggle to show search in main area
            show_search_main = st.checkbox(
                "Show search in main area",
                value=st.session_state.get("show_search_main", False),
                help="Toggle between chat and search interface in the main area"
            )
            st.session_state.show_search_main = show_search_main
            
            if show_search_main:
                st.info("Search interface is now shown in the main area")
            
            # Show search statistics in sidebar
            if render_search_statistics:
                render_search_statistics(
                    st.session_state.rag_service,
                    st.session_state.rag_project.id
                )
        else:
            st.warning("Please select a project in the RAG tab first")
            if not has_rag_service:
                st.error("RAG service is not initialized")
            if not has_project_id:
                st.info("No project ID found in session state")
            if has_project_id and not has_rag_project:
                st.info(f"Project ID {st.session_state.current_rag_project_id} exists but project object not loaded")

    # Initialize provider (using selections from first tab)
    llm_provider, error_message = initialize_provider(selected_provider, selected_model)

    # Display error message if provider initialization failed
    if error_message:
        st.error(error_message)
        st.sidebar.error(f"Provider failed: {error_message}")

    # Check if we should show search interface or chat
    if st.session_state.get("show_search_main", False) and st.session_state.get("rag_service") and st.session_state.get("rag_project"):
        # Show search interface in main area
        try:
            from chat.ui.search_interface import render_search_interface
            render_search_interface(
                st.session_state.rag_service,
                st.session_state.rag_project.id
            )
        except ImportError as e:
            st.error(f"Failed to import search interface: {e}")
            st.info("Falling back to chat interface")
    else:
        # Show chat interface (default)
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
