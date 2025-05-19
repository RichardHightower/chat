from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Enum for message types."""
    INPUT = "input"
    OUTPUT = "output"


class Message(BaseModel):
    """A model for individual messages in a conversation."""
    timestamp: datetime = Field(default_factory=datetime.now)
    message_type: MessageType
    content: str
    role: str = Field(default="user")  # Default role for backward compatibility with LLM APIs

    def to_llm_message(self) -> dict:
        """Convert to a format suitable for LLM API calls."""
        role = "user" if self.message_type == MessageType.INPUT else "assistant"
        return {
            "role": role,
            "content": self.content
        }


class Conversation(BaseModel):
    """A model for storing conversation history."""
    id: str
    title: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, content: str, message_type: MessageType, role: Optional[str] = None) -> Message:
        """Add a new message to the conversation."""
        message = Message(
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            role=role or ("user" if message_type == MessageType.INPUT else "assistant")
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def to_llm_messages(self) -> List[dict]:
        """Convert conversation history to a format suitable for LLM APIs."""
        return [msg.to_llm_message() for msg in self.messages]

    def ensure_alternating_messages(self) -> List[dict]:
        """Ensure that messages alternate between user and assistant roles."""
        result = []
        last_role = None

        for msg in self.messages:
            current_role = "user" if msg.message_type == MessageType.INPUT else "assistant"

            # If this role is the same as the last one, combine them
            if current_role == last_role and result:
                result[-1]["content"] += "\n\n" + msg.content
            else:
                # Otherwise add as a new message
                result.append({
                    "role": current_role,
                    "content": msg.content
                })

            last_role = current_role

        return result

    class Config:
        arbitrary_types_allowed = True