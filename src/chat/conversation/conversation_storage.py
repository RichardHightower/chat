import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from chat.conversation.conversation import Conversation, MessageType
from chat.util.logging_util import logger


class ConversationStorage:
    """Utility class for storing and retrieving conversations."""

    def __init__(self, storage_dir: Union[str, Path] = "conversations"):
        """
        Initialize the conversation storage.

        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ConversationStorage in directory: {self.storage_dir}")

    def save_conversation(self, conversation: Conversation) -> bool:
        """
        Save a conversation to a JSON file.

        Args:
            conversation: The conversation to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.storage_dir.mkdir(parents=True, exist_ok=True)

            # Prepare file path
            file_path = self.storage_dir / f"{conversation.id}.json"

            # Convert to serializable format
            conversation_dict = conversation.dict()

            # Convert datetime objects to strings for JSON serialization
            conversation_dict['created_at'] = conversation_dict['created_at'].isoformat()
            conversation_dict['updated_at'] = conversation_dict['updated_at'].isoformat()

            for msg in conversation_dict['messages']:
                msg['timestamp'] = msg['timestamp'].isoformat()

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved conversation {conversation.id} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation {conversation.id}: {e}", exc_info=True)
            return False

    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation from a JSON file.

        Args:
            conversation_id: The ID of the conversation to load

        Returns:
            The loaded conversation or None if not found
        """
        try:
            file_path = self.storage_dir / f"{conversation_id}.json"

            if not file_path.exists():
                logger.warning(f"Conversation file not found: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_dict = json.load(f)

            # Convert string timestamps back to datetime objects
            conversation_dict['created_at'] = datetime.fromisoformat(conversation_dict['created_at'])
            conversation_dict['updated_at'] = datetime.fromisoformat(conversation_dict['updated_at'])

            # Convert message timestamps
            for msg in conversation_dict['messages']:
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])

            # Recreate the Conversation object
            conversation = Conversation(**conversation_dict)
            logger.info(f"Loaded conversation {conversation_id} from {file_path}")

            return conversation

        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}", exc_info=True)
            return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation file.

        Args:
            conversation_id: The ID of the conversation to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.storage_dir / f"{conversation_id}.json"

            if not file_path.exists():
                logger.warning(f"Cannot delete: Conversation file not found: {file_path}")
                return False

            os.remove(file_path)
            logger.info(f"Deleted conversation {conversation_id} from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
            return False

    def list_conversations(self) -> List[Dict]:
        """
        List all available conversations with metadata.

        Returns:
            List of conversation metadata dictionaries
        """
        try:
            conversations = []

            # Check if directory exists
            if not self.storage_dir.exists():
                logger.warning(f"Storage directory {self.storage_dir} does not exist")
                # Create it
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                return []

            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract basic metadata
                    conversations.append({
                        'id': data['id'],
                        'title': data.get('title', 'Untitled Conversation'),
                        'created_at': data['created_at'],
                        'updated_at': data['updated_at'],
                        'message_count': len(data['messages'])
                    })
                except Exception as e:
                    logger.error(f"Error reading conversation file {file_path}: {e}")

            # Sort by updated_at (most recent first)
            conversations.sort(key=lambda x: x['updated_at'], reverse=True)

            logger.info(f"Listed {len(conversations)} conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error listing conversations: {e}", exc_info=True)
            return []

    def generate_conversation_title(self, conversation: Conversation, max_length: int = 50) -> str:
        """
        Generate a title for a conversation based on its content.

        Args:
            conversation: The conversation to generate a title for
            max_length: Maximum length of the title

        Returns:
            A generated title
        """
        # Find the first user message to use as the title
        for msg in conversation.messages:
            if msg.message_type == MessageType.INPUT:
                # Get the first line of the message
                first_line = msg.content.split('\n')[0].strip()

                # Truncate if needed
                if len(first_line) > max_length:
                    title = first_line[:max_length - 3] + "..."
                else:
                    title = first_line

                return title

        # If no user messages found, use a default title
        return f"Conversation {conversation.id[:8]}"

    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """
        Update the title of a conversation.

        Args:
            conversation_id: The ID of the conversation to update
            new_title: The new title

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the conversation
            conversation = self.load_conversation(conversation_id)

            if not conversation:
                logger.warning(f"Cannot update title: Conversation {conversation_id} not found")
                return False

            # Update the title
            conversation.title = new_title

            # Save the conversation
            return self.save_conversation(conversation)

        except Exception as e:
            logger.error(f"Error updating title for conversation {conversation_id}: {e}", exc_info=True)
            return False