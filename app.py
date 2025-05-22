"""
app.py
============================
This script serves as the entry point for the Streamlit application. 
It checks for required dependencies and initializes the ChatbotUI class.
"""
import streamlit as st
from src.utils import logger
from src.ui.chatbot_ui import ChatbotUI

def check_dependencies():
    try:
        import numpy as np
        import torch
        import transformers
        import sentence_transformers
        logger.info("All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        st.error(f"Missing required package: {str(e)}. Please install it using pip.")
        return False

if __name__ == "__main__":
    if not check_dependencies():
        st.error("Please install all required dependencies and restart the application")
        st.stop()
    # Create an instance of the ChatbotUI class
    chatbot = ChatbotUI()
    # Initialize the UI components
    chatbot.render()

