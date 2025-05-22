"""
groq_chat_model.py
============================

This module defines the GroqChatModel class, which is a wrapper around the Groq API for chat-based interactions.
It provides methods for initializing the model, updating the model, and managing conversation history.
"""
# Import necessary libraries
from litellm import model_list
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable

from ..utils.logger import logger
from ..utils.constants import DEFAULT_MODEL

# Class representing a chat model using the Groq API.
class GroqChatModel(Runnable):

    # Initialize the GroqChatModel with the provided API key and model.
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        """
        Initializes the GroqChatModel with the provided API key and model.

        Args:
            api_key (str): The API key for authentication.
            model (str): The model to use for chat interactions. Defaults to DEFAULT_MODEL.
        """
        self.api_key = api_key
        self.model_name = model_name
        self._initialize_model()
        logger.info(f"GroqChatModel initialized with model: {self.model}")

    # initialize the GROQ Chat model
    def _initialize_model(self):
        try:
            self.model = ChatGroq(
               groq_api_key=self.api_key,
               model_name=self.model_name, 
            )
            logger.debug(f"Model initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise e

    # Update the model to the selected model   
    def update_model(self, selected_model_name: str):
        if selected_model_name != self.model_name:
            try:
                self.model_name = selected_model_name
                self._initialize_model()
                logger.info(f"Model updated to: {self.model_name}")
            except Exception as e:
                logger.error(f"Error updating model: {str(e)}")
                raise e
            
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input and return a response.
        
        Args:
            input (Dict[str, Any]): Input containing 'input' and 'history' keys
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        try:
            # Get the input message and history
            message = input.get('input', '')
            history = input.get('history', [])
            
            # Create the chat history
            messages = []
            for msg in history:
                if isinstance(msg, dict):
                    if msg.get('type') == 'human':
                        messages.append({"role": "user", "content": msg.get('content', '')})
                    elif msg.get('type') == 'ai':
                        messages.append({"role": "assistant", "content": msg.get('content', '')})
            
            # Add the current message
            messages.append({"role": "user", "content": message})
            
            # Get response from the model
            response = self.model.invoke(messages)
            
            return {"output": response.content}
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            raise
            
