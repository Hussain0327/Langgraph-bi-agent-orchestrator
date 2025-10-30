"""Conversation memory for maintaining chat history."""
from typing import List, Dict
from collections import deque


class ConversationMemory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_messages: int = 10):
        """Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content
        })

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation history.

        Returns:
            List of message dictionaries
        """
        return list(self.messages)

    def clear(self):
        """Clear the conversation history."""
        self.messages.clear()

    def get_context_string(self) -> str:
        """Get conversation history as a formatted string.

        Returns:
            Formatted conversation history
        """
        context = []
        for msg in self.messages:
            context.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(context)
