"""
chatbot_ui.py
=================
This module defines the ChatbotUI class, which provides a user interface for interacting with a chatbot.
It allows user to select a model, upload file and prompt the model with a message.

"""

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain.schema import Document
import os
import tempfile
from typing import Optional, List, Dict, Any

from .chatbot_message_history import StreamlitChatMessageHistory
from ..models.groq_chat_model import GroqChatModel
from ..vectorstore.chromadb_store import ChromaDBStore
from ..vectorstore.chroma_store import ChromaVectorStore
from ..utils.logger import logger
from ..utils.constants import (
    APP_NAME,
    DEFAULT_MODEL,
    GROQ_API_KEY_HELP,
    DEFAULT_GROQ_API_KEY,
    AVAILABLE_MODELS,
    MODEL_SELECTION_HELP,
    FILE_CHUNK_SIZE,
    FILE_CHUNK_OVERLAP    
)



# ChatbotUI class to create a user interface for the chatbot
class ChatbotUI:
    def __init__(self):
        self._initialize_session_state()
        logger.debug("ChatbotUI initialized with session state.")

    """
    Initialize the session state variables with default values.
    This method checks if the session state variables are already set, and if not, initializes them with default values.
    """
    def _initialize_session_state(self):
        default_values = {
            "selected_model": DEFAULT_MODEL,
            "chatbot_model": None,
            "file_processed": False,
            "messages": [],
            # "vector_store": ChromaVectorStore(),
            # "vector_store": ChromaDBStore(),
            "vector_store": None,
            "message_history": None,
        }

        # Initialize session state variables if not already set
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug(f"Session state variable '{key}' initialized with default value.")

        # Initialize vector store only when needed and not already initialized
        if st.session_state.vector_store is None:
            try:
                st.session_state.vector_store = ChromaDBStore()
                logger.debug("Vector store initialized in session state")
            except Exception as e:
                logger.error(f"Error initializing vector store: {str(e)}")
                st.error("Error initializing vector store. Please try again.")
                st.session_state.vector_store = None  # Reset to None if initialization fails

        logger.debug("Session state initialized with default values.")  


    # Render model selection in the sidebar
    # This method provides a dropdown menu for the user to select a model from the available models.
    def _render_model_selection(self):
        st.sidebar.title("Model Selection")
        st.sidebar.write(MODEL_SELECTION_HELP)
        choose_model = st.selectbox(
                "Select GROQ Model",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x],
                help=MODEL_SELECTION_HELP
        )

        # If "Other" is selected, show a text input
        if choose_model == "Other":
            custom_model = st.text_input("Enter custom GROQ model name:")
            selected_model = custom_model.strip()
        else:
            selected_model = choose_model

        # Display final selected model
        if selected_model:
            st.success(f"Using model: {selected_model}")
            
        logger.debug(f"Model selected: {selected_model}") 
        return selected_model
       
    # Render the API key input field in the sidebar
    def _render_api_key_input(self, selected_model: str):
        api_key = st.text_input(
                "Enter your GROQ API Key", 
                type="password",
                help=GROQ_API_KEY_HELP, 
                value=DEFAULT_GROQ_API_KEY
        )
        if not api_key:
            st.warning(GROQ_API_KEY_HELP)
            return  
        st.session_state.groq_api_key = api_key
        # Check if model has changed and initialize or update the chat model
        try:
            if (st.session_state.chatbot_model is None or 
                    selected_model != st.session_state.selected_model):
                if st.session_state.chatbot_model is None:
                    st.session_state.chatbot_model = GroqChatModel(
                        model_name=selected_model,
                        api_key=api_key
                    )
                    logger.debug(f"Chat model initialized with model: {selected_model}")
                else:
                    st.session_state.chatbot_model.update_model(
                        selected_model_name=selected_model
                    )
                    logger.debug(f"Chat model updated with new model: {selected_model}")
                
                # Update the selected model in session state
                st.session_state.selected_model = selected_model
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            logger.error(f"Error initializing model: {str(e)}")
            return
        return api_key, selected_model

    # # Initialize the message history
    def _initialize_message_history(self):
        try:
            # First check if we have a valid chat model
            if st.session_state.chatbot_model is None:
                logger.error("Cannot initialize message history: chat model is not initialized")
                st.error("Please select a model and enter your API key first")
                return

            # Initialize the message history if it doesn't exist        
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = StreamlitChatMessageHistory()
                logger.debug("Initialized chat history")

            st.session_state.message_history = RunnableWithMessageHistory(
                st.session_state.chatbot_model,
                lambda session_id: st.session_state.chat_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            logger.debug("Initialized message history with chat model")
        except Exception as e:
            logger.error(f"Error initializing message history: {str(e)}")
            st.error(f"Error initializing message history: {str(e)}")
            return


    # Process a PDF file: load and split into chunks. 
    def process_pdf_to_chunks(self, file_path: str):        
        logger.info(f"Processing PDF file: {file_path}")
        try:
            # Load and extract text from a PDF file.
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} pages from PDF")

            # Split documents into chunks while preserving semantic coherence as much as possible. 
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=FILE_CHUNK_SIZE,
                chunk_overlap=FILE_CHUNK_OVERLAP,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            logger.debug(f"Split PDF into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    # Render 
    def _render_file_upload(self) -> Optional[str]:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if not uploaded_file or st.session_state.file_processed:
            return None
        if uploaded_file is not None and not st.session_state.file_processed:
            try:
                # Save the uploaded file temporarily
                temp_path = "temp.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                logger.info("Saved uploaded file temporarily")

                # Process the PDF and create a vector store.
                with st.spinner("Processing PDF..."):
                    try:
                         # Load and split the PDF into chunks.
                        processed_chunks = self.process_pdf_to_chunks(temp_path)

                        if not processed_chunks:
                            raise ValueError("No chunks were generated from the file.")

                        # Create a vector store from the chunks.
                        st.session_state.vector_store.create_store(processed_chunks)
                        st.session_state.file_processed = True
                        logger.info("PDF processed successfully")
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        logger.error(f"Error processing PDF: {str(e)}")
                        st.error(f"Error processing PDF: {str(e)}")
                        st.session_state.file_processed = False
                        return None
                    
                # Clean up temporary file
                import os       
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug("Removed temporary file")
                return temp_path
                    
            except Exception as e:
                logger.error(f"Error handling file upload: {str(e)}")
                st.error(f"Error handling file upload: {str(e)}")
                st.session_state.file_processed = False
                return None
            
        return None






    # Render the sidebar for model selection, file upload and API key input
    def _render_sidebar(self):
        st.sidebar.title("Settings")
        st.sidebar.write("Configure your chat settings here.")

        with st.sidebar:
            # REnder model selection
            selected_model = self._render_model_selection()

            # Render API key input
            api_key, selected_model = self._render_api_key_input(selected_model)

            # REnder chat history messages
            self._initialize_message_history()

            # 
            self._render_file_upload()

        
    # Display the chat message history 
    def _render_chat_history(self):       
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])    

    # Render the chat interface and handle user input.
    def _render_chat_interface(self):
         # Check if the chat model is initialized and the PDF is processed
        def _validate_chat_state() -> bool:        
            if not st.session_state.chatbot_model:
                logger.warning("Chat model not initialized")
                st.error("Please set up your GROQ API key first!")
                return False
            
            if not st.session_state.file_processed:
                logger.warning("PDF not processed")
                st.error("Please upload a PDF file first!")
                return False
                
            return True
        
        # Get and process the relevant context from the vector store for the user's prompt.
        def _get_relevant_context(prompt: str) -> list:
            try:
                # Get relevant documents from the vector store
                relevant_docs = st.session_state.vector_store.search(prompt)
                logger.debug(f"Found {len(relevant_docs)} relevant documents")
                
                # Convert documents to a simple list of strings
                context_docs = []
                for doc in relevant_docs:
                    if isinstance(doc, Document):
                        context_docs.append(doc.page_content)
                    elif hasattr(doc, 'page_content'):
                        context_docs.append(doc.page_content)
                    else:
                        context_docs.append(str(doc))
                
                logger.debug(f"Processed context docs: {context_docs}")
                return context_docs
            except Exception as e:
                logger.error(f"Error getting relevant context: {str(e)}")
                st.error(f"Error getting relevant context: {str(e)}")
                raise

        # Handle user prompt and generate response.
        def _handle_user_prompt(prompt: str, context_docs: list):
            try:
                # Add the current message to history
                st.session_state.chat_history.add_message(
                    HumanMessage(content=prompt)
                )
                    
                # Create input dictionary with just the current query and context
                model_input = {
                    "input": prompt,
                    "context": context_docs
                }
                    
                # Get the response from the model
                logger.debug("Sending request to chat model")
                response = st.session_state.chatbot_model.invoke(model_input)
                logger.debug("Received response from chat model")

                # Extract the content from the response dictionary
                if isinstance(response, dict) and "output" in response:
                    response = response["output"]
                elif isinstance(response, str):
                    response = response
                else:
                    raise ValueError("Invalid response format from model")
                    
                # Add the response to history
                st.session_state.chat_history.add_message(
                    AIMessage(content=response)
                )
                return response
            except Exception as e:
                logger.error(f"Error handling user prompt: {str(e)}")
                st.error(f"Error handling user prompt: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

        # Handle user input
        if prompt := st.chat_input("Ask a question about your PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            logger.info(f"User question: {prompt}")

            # Get relevant context and generate response
            try:
                # Validate chat state
                if not _validate_chat_state():
                    return               
                
                # Get relevant context
                context_docs = _get_relevant_context(prompt)
                if not context_docs:
                    logger.warning("No relevant context found")
                    st.warning("No relevant context found in the PDF for your question. Please try rephrasing your question or ask about a different topic.")
                    return
                
                # Use message history to generate response
                session_id = "pdf_chat_session"

                # Handle user prompt and generate response
                response = _handle_user_prompt(prompt, context_docs)
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

    def _initialize_ui(self):
        st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")
        st.title(APP_NAME)
        st.write("Chat with your PDF documents effortlessly!")
        self._render_sidebar()
        self._render_chat_history()
        self._render_chat_interface()
        

    def render(self):
        self._initialize_ui()
        # self._render_model_selection()
        # self._render_file_upload()
        # self._render_chat_interface()
