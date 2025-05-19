"""
Utility functions for chat UI components.

This module contains utility functions specifically for chat UI operations.
"""

import streamlit as st
import asyncio
from typing import Optional, Dict, Any, Callable

from chat.ai.llm_provider import LLMProvider
from chat.conversation.conversation import Conversation, MessageType
from chat.conversation.conversation_storage import ConversationStorage
from chat.util.logging_util import logger as llm_logger

# Alias the logger for convenient access
logger = llm_logger


def handle_message_with_streaming(
    provider: LLMProvider,
    prompt: str,
    conversation: Optional[Conversation],
    conversation_storage: Optional[ConversationStorage],
    options: Dict[str, Any]
):
    """
    Handle streaming message display in the chat UI.
    
    Args:
        provider: The LLM provider instance
        prompt: User's input prompt
        conversation: Current conversation object
        conversation_storage: Instance of ConversationStorage
        options: Options for the provider (system_prompt, temperature, etc.)
    """
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Note: We don't display the user message here because it should already
    # be displayed in the main chat UI before this function is called
    
    # Create placeholder for assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")
        streaming_indicator = st.empty()
        full_response = ""
        
        # Function to update UI with each chunk
        def update_message_placeholder(chunk: str):
            nonlocal full_response
            full_response += chunk
            # Add blinking cursor to show it's still streaming
            message_placeholder.markdown(full_response + "▌")
        
        try:
            # Start streaming response
            async def stream_response():
                nonlocal full_response
                try:
                    maintain_context = st.session_state.get("maintain_context", True)
                    active_conversation = conversation if maintain_context else None
                    
                    # Show streaming indicator 
                    streaming_indicator.markdown("*Streaming response...*")
                    
                    # Get the async generator
                    llm_logger.info(f"Starting streaming with provider {provider.__class__.__name__}")
                    generator = provider.generate_completion_stream(
                        prompt=prompt,
                        output_format="text",
                        options=options,
                        conversation=active_conversation,
                        callback=update_message_placeholder
                    )
                    llm_logger.info(f"Got generator of type: {type(generator)}")
                    
                    # Manually iterate over the generator
                    while True:
                        try:
                            chunk = await anext(generator)
                            # Processing is handled by callback
                        except StopAsyncIteration:
                            break
                    
                    # Clear streaming indicator when done
                    streaming_indicator.empty()
                    
                    # Remove cursor and display final message
                    if full_response:
                        message_placeholder.markdown(full_response)
                    return full_response
                except Exception as e:
                    logger.error(f"Error in stream_response: {e}", exc_info=True)
                    raise
            
            # Run the streaming in asyncio
            try:
                full_response = asyncio.run(stream_response())
            except Exception as e:
                logger.error(f"Error running stream_response: {e}", exc_info=True)
                raise
            
            # Add to session state, making sure we avoid duplicate entries
            # Check if the message was already added (can happen if this is interrupted and rerun)
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # If not using the conversation object for context (which would have been updated automatically),
            # we need to manually add the response to the conversation
            if not st.session_state.get("maintain_context", True) and conversation:
                conversation.add_message(full_response, MessageType.OUTPUT)
            
            # Save conversation
            if conversation_storage and conversation:
                try:
                    if conversation and conversation_storage.save_conversation(conversation):
                        llm_logger.info(f"Auto-saved conversation {conversation.id}")
                    elif conversation:
                        llm_logger.warning(f"Failed to auto-save conversation {conversation.id}")
                except Exception as e:
                    llm_logger.error(f"Error during auto-save of conversation: {e}", exc_info=True)
                
        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            llm_logger.error(error_msg, exc_info=True)
            message_placeholder.error(error_msg)
            streaming_indicator.empty()  # Clear streaming indicator on error
            
            # Add error message to session state if not already done
            if not st.session_state.messages or "Error during streaming" not in st.session_state.messages[-1]["content"]:
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                
            # Try to fall back to non-streaming if possible
            try:
                llm_logger.info("Attempting fallback to non-streaming mode")
                message_placeholder.markdown("⏳ Trying non-streaming mode as fallback...")
                
                # Use standard non-streaming approach as fallback
                maintain_context = st.session_state.get("maintain_context", True)
                active_conversation = conversation if maintain_context else None
                
                fallback_response = asyncio.run(
                    provider.generate_completion(
                        prompt=prompt,
                        output_format="text",
                        options=options,
                        conversation=active_conversation
                    )
                )
                
                if fallback_response:
                    message_placeholder.markdown(fallback_response)
                    # Update session state with the fallback response
                    st.session_state.messages[-1] = {"role": "assistant", "content": fallback_response}
                    
                    # Update conversation if needed
                    if conversation:
                        conversation.add_message(fallback_response, MessageType.OUTPUT)
                        if conversation_storage:
                            conversation_storage.save_conversation(conversation)
                            
                    llm_logger.info("Fallback to non-streaming mode succeeded")
                    return  # Exit function successfully
                    
            except Exception as fallback_error:
                # If fallback fails too, just log the error
                llm_logger.error(f"Fallback to non-streaming also failed: {fallback_error}", exc_info=True)
                message_placeholder.error(f"Both streaming and non-streaming attempts failed. Please try again.")