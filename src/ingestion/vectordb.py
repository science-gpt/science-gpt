from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from .embedding import Embedding


class VectorDB(ABC):
    """
    Abstract base class for vector database operations.
    """

    @abstractmethod
    def insert(self, embeddings: List[Embedding]) -> None:
        """
        Insert embeddings into the vector database.

        Args:
            embeddings (List[Embedding]): List of Embedding objects to insert.
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vector (np.ndarray): The query vector to search for.
            top_k (int): The number of most similar vectors to return.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing search results.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the database by their IDs.

        Args:
            ids (List[str]): List of vector IDs to delete.
        """
        pass

    @abstractmethod
    def update(self, ids: List[str], embeddings: List[Embedding]) -> None:
        """
        Update existing vectors in the database.

        Args:
            ids (List[str]): List of vector IDs to update.
            embeddings (List[Embedding]): List of new Embedding objects.
        """
        pass
