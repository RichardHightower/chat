"""
Chat UI components for the chat application.

This module contains functions for rendering the chat UI components
including message display, input handling, and response generation.
"""
import logging

import streamlit as st
import asyncio
from typing import Dict, Optional, List, Union, Any

from chat.conversation.conversation import Conversation, MessageType
from chat.conversation.conversation_storage import ConversationStorage
from chat.ai.llm_provider import LLMProvider
from chat.util.logging_util import logger as llm_logger
from chat.ui.chat_utils import handle_message_with_streaming
from chat.rag.rag_chat import get_rag_context_for_prompt, format_rag_response


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
    system_prompt: str = "You are a helpful and concise chat assistant designed for providing accurate and relevant information.",
    use_streaming: bool = False
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
        use_streaming: Whether to use streaming mode for response generation
    """
    # Get user input from chat input
    if prompt := st.chat_input("Your message..."):
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        logging.info(f"PROMPT: {prompt}")
        # Check for RAG context
        enhanced_prompt = get_rag_context_for_prompt(prompt)
        logging.info(f"ENHANCED_PROMPT: {enhanced_prompt}")


        # Use enhanced prompt if available, otherwise use original prompt
        final_prompt = enhanced_prompt if enhanced_prompt else prompt

        # If RAG is enabled and enhanced prompt is available, indicate that RAG is being used
        if enhanced_prompt:
            with st.chat_message("system"):
                st.info("Using document context to enhance response...")
        
        # Generate and display assistant's response
        if not llm_provider:
            with st.chat_message("assistant"):
                st.error("LLM Provider is not available. Cannot process messages.")
            if not any(m["role"] == "assistant" and "Provider is not available" in m["content"] for m in
                       st.session_state.messages):
                st.session_state.messages.append({"role": "assistant",
                                                 "content": "LLM Provider is not available due to an initialization error. Please check the logs."})
            return

        # Prepare generation options
        generation_options = {
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        
        llm_logger.info(f"User prompt for {selected_provider}: {prompt}")
        
        # Decide whether to use conversation context
        maintain_context = st.session_state.get("maintain_context", True)
        active_conversation = conversation if maintain_context else None
        
        # Check if we should use streaming mode for any provider that supports it
        should_stream = (use_streaming and 
                         hasattr(llm_provider, 'generate_completion_stream'))
        
        # Handle streaming separately (it creates its own chat message context)
        if should_stream:
            llm_logger.info(f"Using streaming mode for {selected_provider}")
            
            try:
                # Handle streaming
                handle_message_with_streaming(
                    provider=llm_provider,
                    prompt=final_prompt,
                    conversation=active_conversation,
                    conversation_storage=conversation_storage,
                    options=generation_options
                )
            except Exception as e:
                llm_logger.error(f"Error in streaming mode: {e}", exc_info=True)
                # Fallback to non-streaming mode on error
                with st.chat_message("assistant"):
                    st.error(f"Streaming failed: {str(e)}. Trying standard response...")
                    should_stream = False
        
        # Non-streaming path
        if not should_stream:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ Thinking...")
                
                try:
                    # Use standard non-streaming approach
                    full_response_content = asyncio.run(
                        llm_provider.generate_completion(
                            prompt=final_prompt,
                            output_format="text",
                            options=generation_options,
                            conversation=active_conversation
                        )
                    )
                    
                    # Update UI after completion
                    llm_logger.info(f"{selected_provider} response received, length: {len(full_response_content or '')}")
                    
                    if not full_response_content:
                        full_response_content = "I received an empty response. Could you try rephrasing?"
                        llm_logger.warning("LLM returned an empty response.")

                    # Format response to highlight citations if RAG was used
                    if enhanced_prompt:
                        full_response_content = format_rag_response(full_response_content)
                    
                    # Display the response
                    message_placeholder.markdown(full_response_content)
                    
                    # Add assistant's response to session state
                    st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                    
                    # Update conversation if needed
                    if active_conversation:
                        if not maintain_context:  # Only needed if not already updated via context
                            active_conversation.add_message(full_response_content, MessageType.OUTPUT)
                        
                        # Auto-save the conversation
                        try:
                            if conversation_storage.save_conversation(active_conversation):
                                llm_logger.info(f"Auto-saved conversation {active_conversation.id}")
                            else:
                                llm_logger.warning(f"Failed to auto-save conversation {active_conversation.id}")
                        except Exception as e:
                            llm_logger.error(f"Error during auto-save of conversation: {e}", exc_info=True)
                        
                except ValueError as ve:
                    error_msg = f"Input/Output Error: {ve}"
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"I encountered an issue with the data format: {ve}"})
                    
                except ImportError as ie:
                    error_msg = f"Import error: {ie}. Ensure all dependencies for the LLM provider are installed."
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": "There's a configuration problem with the LLM provider."})
                    
                except Exception as e:
                    error_msg = f"Sorry, an unexpected error occurred: {type(e).__name__} - {e}"
                    llm_logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": "I ran into an unexpected problem trying to respond."})

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