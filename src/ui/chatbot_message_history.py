"""
StreamlitChatMessageHistory.py
========================
This module holds the StreamlitChatMessageHistory class, which is a wrapper for managing chat message history in a Streamlit application.

"""
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Any
from ..utils.logger import logger

# Custom class to handle chat history for the Streamlit chat interface
class StreamlitChatMessageHistory(BaseChatMessageHistory):
    # When an instance is created, it starts with an empty list of messages.
    def __init__(self):
        super().__init__()
        self.messages = []
        logger.debug("StreamlitChatMessageHistory initialized with empty messages list.")

    # This method adds a new message (either from the user or the AI) to the history list. 
    def add_message(self, message: Any) -> None:
        self.messages.append(message)
        logger.debug(f"Message added to history: {message}")

    # This method retrieves all messages from the history list.
    def get_messages(self) -> List[Any]:        
        return self.messages

    # This method clears the message history by resetting the messages list to an empty list.
    def clear(self) -> None:
        self.messages = []