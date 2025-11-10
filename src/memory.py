from typing import List, Dict, Deque
from collections import deque


class ConversationMemory:

    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages: int = max_messages
        self.messages: Deque[Dict[str, str]] = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content
        })

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def get_context_string(self) -> str:
        context: List[str] = []
        for msg in self.messages:
            context.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(context)
