"""
chromadb_store.py
=================

This module provides functionality to create and manage a ChromaDB vector store for document retrieval and storage.

"""

from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
from .vectorstore_base import VectorStore
from ..utils.constants import DEFULT_EMBEDDING_MODEL
from ..utils.logger import logger
import os
import traceback
import numpy as np

# ChromaDBStore class inherits from VectorStore and provides methods to create, load, and search the vector store.
class ChromaDBStore(VectorStore):
    
    # Initializes the ChromaDBStore with the specified persistence directory.
    def __init__(self, persist_directory: str = "chroma_db"):       
        self.persist_directory = persist_directory
        self.vectorstore = None

        # Create the persistnce directory first and then initialize the embeddings and vector store.
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Created persistence directory: {self.persist_directory}")
        else:
            logger.info(f"Persistence directory already exists: {self.persist_directory}")

        self._initialize_embeddings(model_name=DEFULT_EMBEDDING_MODEL)
        logger.debug(f"ChromaDBStore initialized with persistence directory: {self.persist_directory}")

    # Initialize the embeddings model with proper device handling.
    def _initialize_embeddings(self, model_name = DEFULT_EMBEDDING_MODEL):
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # initialize embeddings with specific model and device
            self.embeddings = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs = {
                    'device' : device,
                    # 'torch_dtype': torch.float16 if device == "cuda" else torch.float32,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,  # Add batch size for better performance
                    # 'show_progress_bar': True
                }
            )

            # Then manually move the model to the device
            if hasattr(self.embeddings, '_client'):
                try:
                    self.embeddings._client.to(device)
                except Exception as device_error:
                    logger.warning(f"Could not move model to {device}: {str(device_error)}")
                    # Continue without device placement if it fails

            logger.debug("Successfully initialized embeddings model")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    # Creates a vector store from the provided documents.
    def create_store(self, documents: List[Document], **kwargs) -> None:
        try:
            if not documents:
                raise ValueError("No documents provided to create vector store")
            
            logger.info(f"Creating vector store with {len(documents)} documents")

            # Check if the vector store already exists then clear it and create a new one.
            if self.vectorstore is not None:
                logger.info("Clearing existing vector store.")
                try:
                    self.vectorstore.delete_collection()
                    logger.info("Existing vector store cleared.")
                except Exception as e:
                    logger.error(f"Error clearing existing vector store: {str(e)}")
                    raise e
            # Process documents in smaller batches
            batch_size = 50  # Reduced batch size to avoid memory issues
            total_batches = (len(documents) + batch_size - 1) // batch_size

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents")
                
                try:
                    if i == 0:
                        # First batch - create new store
                        self.vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=self.persist_directory,
                            **kwargs
                        )
                    else:
                        # Subsequent batches - add to existing store
                        self.vectorstore.add_documents(batch)
                    
                    logger.info(f"Successfully processed batch {batch_num}")
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_num}: {str(batch_error)}")
                    raise        
           
            logger.info("Vector store created successfully.")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise e
        
    # Loads the vector store from the persistence directory.
    def load_store(self) -> None:
        try:
            # Check if the vectore store already exists, if not then throw an error.
            if not os.path.exists(self.persist_directory):
                raise FileNotFoundError(f"Persistence directory does not exist: {self.persist_directory}")
            
            # Load the vector store from the persistence directory.
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise e
        
    # Searches the vector store for the most relevant documents based on the query.
    def search(self, query: str, k: int = 5) -> List[Document]:
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store is not initialized. Please create or load the store first.")
            
            # Perform the search and return the results.
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Search completed with query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise e
        