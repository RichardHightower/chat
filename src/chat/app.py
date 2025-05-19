import streamlit as st
import os
import asyncio
import uuid
from datetime import datetime
from dotenv import load_dotenv

from chat.logging_util import logging

# Import the custom providers
try:
    # Import all provider classes
    from chat.open_ai import OpenAIProvider
    from chat.google_gemini import GoogleGeminiProvider
    from chat.perplexity import PerplexityProvider
    from chat.anthropic import AnthropicProvider
    from chat.conversation import Conversation, MessageType
    from chat.conversation_storage import ConversationStorage
    from chat.logging_util import logger as llm_logger
except ImportError as e:
    st.error(
        f"Failed to import providers: {e}. Make sure all provider files are in the correct path and dependencies are installed.")
    import logging_util

    llm_logger = logging.getLogger("llm_provider_fallback")
    st.stop()

# --- Load Environment Variables ---
if load_dotenv():
    llm_logger.info(".env file loaded successfully.")
else:
    llm_logger.warning(".env file not found. Relying on pre-set environment variables.")

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Provider Chat App", page_icon="🤖", layout="wide")

# --- App Title and Description ---
st.title("🤖 Multi-Provider Chat App")
st.caption("Chat with multiple LLM providers using Streamlit and LiteLLM.")

# --- Provider Selection ---
# Define available providers and their default models
PROVIDERS = {
    "OpenAI": {
        "class": OpenAIProvider,
        "models": ["gpt-4o-2024-08-06", "gpt-4.1-2025-04-14",
                   "gpt-4o", "gpt-4.1", "o4-mini", "o3", "o3-mini",
                   "chatgpt-4o-latest"]
    },
    "Google Gemini": {
        "class": GoogleGeminiProvider,
        "models": ["gemini-2.5-pro-preview-05-06",
                   "gemini-2.0-flash-001", "gemini-2.0-flash-lite-001",
                   "gemini-2.5-flash-preview-04-17",
                   "gemini-2.0-flash-live-preview-04-09"]
    },
    "Perplexity": {
        "class": PerplexityProvider,
        "models": ["sonar-pro", "sonar", "sonar-deep-research",
                   "sonar-reasoning-pro", "sonar-reasoning", "r1-1776"
                   ]
    },
    "Anthropic": {
        "class": AnthropicProvider,
        "models": ["claude-3-7-sonnet-latest",
                   "claude-3-5-haiku-latest", "claude-3-opus-latest"]
    }
}

# --- Initialize Conversation Storage ---
@st.cache_resource
def get_conversation_storage():
    """Initialize and retrieve the conversation storage."""
    storage_dir = os.environ.get("CONVERSATION_STORAGE_DIR", "conversations")
    return ConversationStorage(storage_dir)


# Get conversation storage instance
conversation_storage = get_conversation_storage()
llm_logger.info("Conversation storage initialized")

# --- Initialize or Load Conversation in Session State ---
if "conversation_id" not in st.session_state:
    # Create a new conversation ID
    st.session_state.conversation_id = str(uuid.uuid4())
    llm_logger.info(f"Created new conversation with ID: {st.session_state.conversation_id}")


# --- Create or retrieve the conversation object ---
def get_conversation():
    """Create or retrieve the current conversation object."""
    if "conversation_obj" not in st.session_state or st.session_state.get(
            "current_conversation_id") != st.session_state.conversation_id:
        # Try to load an existing conversation
        conversation = conversation_storage.load_conversation(st.session_state.conversation_id)

        if not conversation:
            # Create a new conversation object
            conversation = Conversation(id=st.session_state.conversation_id)

            # Add existing messages from session state if any
            if "messages" in st.session_state:
                for msg in st.session_state.messages:
                    msg_type = MessageType.INPUT if msg["role"] == "user" else MessageType.OUTPUT
                    conversation.add_message(msg["content"], msg_type, role=msg["role"])

        st.session_state.conversation_obj = conversation
        st.session_state.current_conversation_id = st.session_state.conversation_id

        # Generate a title if none exists
        if not conversation.title and conversation.messages:
            conversation.title = conversation_storage.generate_conversation_title(conversation)
            # Save the conversation with the new title
            conversation_storage.save_conversation(conversation)

    return st.session_state.conversation_obj

# Sidebar for provider and model selection
with st.sidebar:
    st.header("Provider Settings")
    selected_provider = st.selectbox("Select Provider", list(PROVIDERS.keys()))

    # Get models for the selected provider
    provider_info = PROVIDERS[selected_provider]
    selected_model = st.selectbox("Select Model", provider_info["models"])

    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Conversation Management
    st.header("Conversation Management")
    maintain_context = st.checkbox("Maintain Conversation Context", value=True,
                                   help="When enabled, the app will send the entire conversation history to the LLM.")

    # Load existing conversations
    st.subheader("Saved Conversations")
    try:
        # Create storage directory if it doesn't exist
        os.makedirs(os.environ.get("CONVERSATION_STORAGE_DIR", "conversations"), exist_ok=True)

        conversation_list = conversation_storage.list_conversations()

        if conversation_list:
            # Create a selectbox of saved conversations
            conversation_options = ["Current"] + [
                f"{c['title']} ({datetime.fromisoformat(c['updated_at']).strftime('%Y-%m-%d %H:%M')})"
                for c in conversation_list
            ]
            selected_conversation = st.selectbox("Select Conversation", conversation_options)

            # Load the selected conversation
            if selected_conversation != "Current" and conversation_list:
                # Find the selected conversation index
                selected_idx = conversation_options.index(selected_conversation) - 1  # Adjust for "Current"

                if st.button("Load Selected Conversation"):
                    # Get the conversation ID
                    conversation_id = conversation_list[selected_idx]["id"]

                    # Load the conversation
                    st.session_state.conversation_id = conversation_id
                    # Clear the current conversation object to force reload
                    if "conversation_obj" in st.session_state:
                        del st.session_state.conversation_obj

                    # Update messages in session state from the loaded conversation
                    loaded_conversation = conversation_storage.load_conversation(conversation_id)
                    if loaded_conversation:
                        st.session_state.messages = [
                            {"role": "user" if msg.message_type == MessageType.INPUT else "assistant",
                             "content": msg.content}
                            for msg in loaded_conversation.messages
                        ]
                        st.success(f"Loaded conversation: {loaded_conversation.title or conversation_id[:8]}")
                        st.rerun()
        else:
            st.info("No saved conversations found. Start chatting to create one!")
    except Exception as e:
        llm_logger.error(f"Error loading conversation list: {e}", exc_info=True)
        st.error("Could not load saved conversations")

    # Conversation actions
    st.subheader("Conversation Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("New Conversation"):
            # Create a new conversation
            st.session_state.conversation_id = str(uuid.uuid4())
            if "conversation_obj" in st.session_state:
                del st.session_state.conversation_obj

            # Clear messages
            st.session_state.messages = [
                {"role": "assistant",
                 "content": f"Hello! I'm using {selected_provider}'s {selected_model}. How can I help you today?"}
            ]
            st.rerun()

    with col2:
        if st.button("Save Conversation"):
            # Get the current conversation
            current_conversation = get_conversation()

            # Generate a title if none exists
            if not current_conversation.title:
                current_conversation.title = conversation_storage.generate_conversation_title(current_conversation)

            # Save the conversation
            if conversation_storage.save_conversation(current_conversation):
                st.success(f"Conversation saved: {current_conversation.title or current_conversation.id[:8]}")
            else:
                st.error("Failed to save conversation")

# --- Instantiate Provider ---
llm_provider = None
try:
    # Initialize the selected provider with the selected model
    provider_class = PROVIDERS[selected_provider]["class"]
    llm_provider = provider_class(model=selected_model)
    st.sidebar.success(f"Provider initialized: {selected_provider} with model: {selected_model}")
except ValueError as e:
    st.error(f"Error initializing provider: {e}")
    st.sidebar.error(f"Provider failed: {e}")
    llm_provider = None
except Exception as e:
    st.error(f"An unexpected error occurred during provider initialization: {e}")
    st.sidebar.error(f"Provider critical error: {e}")
    llm_provider = None

# --- Initialize Chat History in Session State ---
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

# --- Initialize or Load Conversation in Session State ---
if "conversation_id" not in st.session_state:
    # Create a new conversation ID
    st.session_state.conversation_id = str(uuid.uuid4())
    llm_logger.info(f"Created new conversation with ID: {st.session_state.conversation_id}")


# Get the current conversation
conversation = get_conversation()

# --- Display Existing Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input and Generate Response ---
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
                    "system_prompt": "You are a helpful and concise chat assistant designed for providing accurate and relevant information.",
                    "temperature": temperature,
                }

                llm_logger.info(f"User prompt for {selected_provider}: {prompt}")

                # Decide whether to use conversation context
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
        if not maintain_context and conversation:
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

# --- Optional: Add buttons for conversation management ---
with st.sidebar:
    st.subheader("Current Conversation")

    try:
        # Edit conversation title
        conversation = get_conversation()
        current_title = conversation.title or conversation_storage.generate_conversation_title(conversation)
        new_title = st.text_input("Conversation Title", value=current_title)

        if new_title != current_title:
            conversation.title = new_title
            try:
                if conversation_storage.save_conversation(conversation):
                    st.success("Title updated")
                else:
                    st.error("Failed to update title")
            except Exception as e:
                llm_logger.error(f"Error updating conversation title: {e}", exc_info=True)
                st.error("Error saving title")

        # Delete conversation button
        if st.button("Delete Current Conversation"):
            try:
                if conversation_storage.delete_conversation(st.session_state.conversation_id):
                    # Create a new conversation
                    st.session_state.conversation_id = str(uuid.uuid4())
                    if "conversation_obj" in st.session_state:
                        del st.session_state.conversation_obj

                    # Clear messages
                    st.session_state.messages = [
                        {"role": "assistant",
                         "content": f"Previous conversation deleted. I'm ready for a new conversation using {selected_provider}'s {selected_model}!"}
                    ]
                    st.success("Conversation deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete conversation")
            except Exception as e:
                llm_logger.error(f"Error deleting conversation: {e}", exc_info=True)
                st.error("Error deleting conversation")

        # Export conversation
        if st.button("Export Conversation"):
            try:
                # Create a download link for the conversation history
                conversation_data = "\n".join([
                    f"[{msg['role']}] ({datetime.now().strftime('%Y-%m-%d %H:%M')}): {msg['content']}"
                    for msg in st.session_state.messages
                ])

                title_slug = current_title.replace(" ", "_").lower()[:30]
                st.download_button(
                    label="Download Conversation",
                    data=conversation_data,
                    file_name=f"{title_slug}_{st.session_state.conversation_id[:8]}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                llm_logger.error(f"Error exporting conversation: {e}", exc_info=True)
                st.error("Error generating export")

        # Display conversation statistics
        st.text(f"ID: {conversation.id[:8]}...")
        st.text(f"Messages: {len(conversation.messages)}")
        st.text(f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M')}")
        st.text(f"Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        llm_logger.error(f"Error in conversation management sidebar: {e}", exc_info=True)
        st.error("Error loading conversation details")
        st.text("Try starting a new conversation.")