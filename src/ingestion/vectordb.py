from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import Chunk


class VectorDB(ABC):
    """
    Abstract base class for vector database operations.
    """

    @abstractmethod
    def insert(
        self,
        vectors: np.ndarray,
        documents: List[Chunk],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert vectors, their associated documents, and optional metadata into the vector database.

        Args:
            vectors (np.ndarray): NumPy array of vector embeddings to insert.
            documents (List[Chunk]): List of Chunk objects associated with each vector.
            metadata (Optional[Dict[str, Any]]): Optional list of metadata associated with each vector.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement insert method")

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vector (np.ndarray): The query vector to search for.
            top_k (int): The number of most similar vectors to return.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing search results.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement search method")

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the database by their IDs.

        Args:
            ids (List[str]): List of vector IDs to delete.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement delete method")

    @abstractmethod
    def update(
        self,
        ids: List[str],
        vectors: np.ndarray,
        documents: List[Chunk],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update existing vectors, their associated documents, and optional metadata in the database.

        Args:
            ids (List[str]): List of vector IDs to update.
            vectors (np.ndarray): NumPy array of new vector embeddings.
            documents (List[Chunk]): List of new Chunk objects associated with each vector.
            metadata (Optional[Dict[str, Any]]): Optional list of new metadata associated with each vector.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement update method")
