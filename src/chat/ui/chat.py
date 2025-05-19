"""
Chat UI components for the chat application.

This module contains functions for rendering the chat UI components
including message display, input handling, and response generation.
"""

import streamlit as st
import asyncio
from typing import Dict, Optional, List

from chat.conversation.conversation import Conversation, MessageType
from chat.conversation.conversation_storage import ConversationStorage
from chat.ai.llm_provider import LLMProvider
from chat.util.logging_util import logger as llm_logger


def display_chat_messages(messages: List[Dict[str, str]]) -> None:
    """
    Display the chat message history.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    """
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input(
    llm_provider: Optional[LLMProvider],
    conversation: Optional[Conversation],
    conversation_storage: ConversationStorage,
    selected_provider: str,
    selected_model: str,
    temperature: float,
    system_prompt: str = "You are a helpful and concise chat assistant designed for providing accurate and relevant information."
) -> None:
    """
    Handle user input, generate responses, and update the conversation.
    
    Args:
        llm_provider: The initialized LLM provider
        conversation: The current conversation object
        conversation_storage: Instance of ConversationStorage
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
        temperature: Temperature setting for response generation
        system_prompt: System prompt to use for the LLM
    """
    # Get user input from chat input
    if prompt := st.chat_input("Your message..."):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant's response
        if llm_provider:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ Thinking...")
                full_response_content = ""
                try:
                    # Options for the provider's generate_completion method
                    generation_options = {
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                    }
                    
                    llm_logger.info(f"User prompt for {selected_provider}: {prompt}")
                    
                    # Decide whether to use conversation context
                    maintain_context = st.session_state.get("maintain_context", True)
                    active_conversation = conversation if maintain_context else None
                    
                    # Since generate_completion is async, run it in an event loop
                    full_response_content = asyncio.run(
                        llm_provider.generate_completion(
                            prompt=prompt,
                            output_format="text",
                            options=generation_options,
                            conversation=active_conversation
                        )
                    )
                    
                    llm_logger.info(f"{selected_provider} response received, length: {len(full_response_content)}")
                    if not full_response_content:
                        full_response_content = "I received an empty response. Could you try rephrasing?"
                        llm_logger.warning("LLM returned an empty response.")
                    message_placeholder.markdown(full_response_content)
                    
                except ValueError as ve:
                    error_msg = f"Input/Output Error: {ve}"
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    full_response_content = f"I encountered an issue with the data format: {ve}"
                except ImportError as ie:
                    error_msg = f"Import error: {ie}. Ensure all dependencies for the LLM provider are installed."
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    full_response_content = "There's a configuration problem with the LLM provider."
                except Exception as e:
                    error_msg = f"Sorry, an unexpected error occurred: {type(e).__name__} - {e}"
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    full_response_content = "I ran into an unexpected problem trying to respond."
            
            # Add assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})
            
            # If not using the conversation object for context (which would have been updated automatically),
            # we need to manually add the response to the conversation
            if not st.session_state.get("maintain_context", True) and conversation:
                conversation.add_message(full_response_content, MessageType.OUTPUT)
            
            # Auto-save the conversation after each interaction
            try:
                if conversation_storage.save_conversation(conversation):
                    llm_logger.info(f"Auto-saved conversation {conversation.id}")
                else:
                    llm_logger.warning(f"Failed to auto-save conversation {conversation.id}")
            except Exception as e:
                llm_logger.error(f"Error during auto-save of conversation: {e}", exc_info=True)
        
        else:
            # This block is reached if llm_provider was None from the start
            with st.chat_message("assistant"):
                st.error("LLM Provider is not available. Cannot process messages.")
            if not any(m["role"] == "assistant" and "Provider is not available" in m["content"] for m in
                       st.session_state.messages):
                st.session_state.messages.append({"role": "assistant",
                                                 "content": "LLM Provider is not available due to an initialization error. Please check the logs."})


def initialize_chat_history(selected_provider: str, selected_model: str) -> None:
    """
    Initialize the chat history in session state if it doesn't exist.
    Update provider information if the provider has changed.
    
    Args:
        selected_provider: Currently selected provider name
        selected_model: Currently selected model name
    """
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": f"Hello! I'm using {selected_provider}'s {selected_model}. How can I help you today?"}
        ]
    # If provider changed, add a system message
    elif "current_provider" in st.session_state and st.session_state.current_provider != selected_provider:
        st.session_state.messages.append(
            {"role": "assistant",
             "content": f"I've switched to {selected_provider}'s {selected_model}. How can I help you?"}
        )
    
    # Update current provider
    st.session_state.current_provider = selected_provider
