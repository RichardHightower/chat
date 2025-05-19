"""
Conversation management module for the chat application.

This module contains functions for managing conversations, including
creating, loading, and retrieving conversation objects.
"""

import streamlit as st
import uuid


from chat.conversation.conversation import Conversation, MessageType
from chat.conversation.conversation_storage import ConversationStorage
from chat.util.logging_util import logger as llm_logger


def initialize_conversation_id() -> str:
    """
    Initialize the conversation ID in session state if it doesn't exist.
    
    Returns:
        The current conversation ID
    """
    if "conversation_id" not in st.session_state:
        # Create a new conversation ID
        st.session_state.conversation_id = str(uuid.uuid4())
        llm_logger.info(f"Created new conversation with ID: {st.session_state.conversation_id}")
    
    return st.session_state.conversation_id


def get_conversation(conversation_storage: ConversationStorage) -> Conversation:
    """
    Create or retrieve the current conversation object.
    
    Args:
        conversation_storage: Instance of ConversationStorage
        
    Returns:
        The current Conversation object
    """
    # Ensure we have a conversation ID
    conversation_id = initialize_conversation_id()
    
    # Create or retrieve the conversation object
    if "conversation_obj" not in st.session_state or st.session_state.get(
            "current_conversation_id") != conversation_id:
        # Try to load an existing conversation
        conversation = conversation_storage.load_conversation(conversation_id)
        
        if not conversation:
            # Create a new conversation object
            conversation = Conversation(id=conversation_id)
            
            # Add existing messages from session state if any
            if "messages" in st.session_state:
                for msg in st.session_state.messages:
                    msg_type = MessageType.INPUT if msg["role"] == "user" else MessageType.OUTPUT
                    conversation.add_message(msg["content"], msg_type, role=msg["role"])
        
        st.session_state.conversation_obj = conversation
        st.session_state.current_conversation_id = conversation_id
        
        # Generate a title if none exists
        if not conversation.title and conversation.messages:
            conversation.title = conversation_storage.generate_conversation_title(conversation)
            # Save the conversation with the new title
            conversation_storage.save_conversation(conversation)
    
    return st.session_state.conversation_obj


def sync_conversation_with_messages(conversation: Conversation) -> None:
    """
    Synchronize the conversation object with the messages in session state.
    
    This ensures that the conversation object and the displayed messages are in sync.
    
    Args:
        conversation: The conversation object to synchronize
    """
    if "messages" in st.session_state and conversation:
        # Clear existing messages in the conversation
        conversation.messages.clear()
        
        # Add messages from session state to the conversation
        for msg in st.session_state.messages:
            msg_type = MessageType.INPUT if msg["role"] == "user" else MessageType.OUTPUT
            conversation.add_message(msg["content"], msg_type, role=msg["role"])
