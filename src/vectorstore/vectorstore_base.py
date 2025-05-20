"""
vectorstore_base.py

This module provides an abstract base class for vector stores, which are used to store and retrieve vectors (embeddings) for documents.
"""
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

# Abstract base class for vector stores.
class VectorStore(ABC):
    @abstractmethod
    def create_store(self, documents: List[Document], **kwargs) -> None:
        """
        Create a vector store from the provided documents.

        Args:
            documents (List[Document]): A list of documents to be stored in the vector store.
            **kwargs: Additional keyword arguments for specific vector store implementations.
        """
        pass

    @abstractmethod
    def load_store(self, **kwargs) -> None:
        """
        Load an existing vector store.

        Args:
            **kwargs: Additional keyword arguments for specific vector store implementations.
        """
        pass

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """
        Search the vector store for documents similar to the query.

        Args:
            query (str): The query string to search for.
            num_results (int): The number of results to return. Defaults to 5.

        Returns:
            List[Document]: A list of documents that match the query.
        """
        pass
    