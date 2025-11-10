import pytest
from src.memory import ConversationMemory


class TestConversationMemory:
    
    def test_initialization(self):
        memory = ConversationMemory(max_messages=5)
        assert memory.max_messages == 5
        assert len(memory.get_messages()) == 0
    
    def test_add_message(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
    
    def test_sliding_window(self):
        memory = ConversationMemory(max_messages=3)
        
        memory.add_message("user", "Message 1")
        memory.add_message("assistant", "Response 1")
        memory.add_message("user", "Message 2")
        memory.add_message("assistant", "Response 2")
        
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "Response 1"
        assert messages[-1]["content"] == "Response 2"
    
    def test_get_context_string(self):
        memory = ConversationMemory()
        memory.add_message("user", "What is AI?")
        memory.add_message("assistant", "AI is artificial intelligence.")
        
        context = memory.get_context_string()
        assert "USER: What is AI?" in context
        assert "ASSISTANT: AI is artificial intelligence." in context
    
    def test_clear(self):
        memory = ConversationMemory()
        memory.add_message("user", "Test")
        memory.add_message("assistant", "Response")
        
        memory.clear()
        
        assert len(memory.get_messages()) == 0
    
    def test_multiple_conversations(self):
        memory = ConversationMemory(max_messages=10)
        
        for i in range(5):
            memory.add_message("user", f"Question {i}")
            memory.add_message("assistant", f"Answer {i}")
        
        messages = memory.get_messages()
        assert len(messages) == 10
        assert messages[0]["content"] == "Question 0"
        assert messages[-1]["content"] == "Answer 4"
