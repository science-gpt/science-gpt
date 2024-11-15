import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from .embedding import Embedding


@dataclass
class SearchResult:
    id: str  # TODO: add this
    distance: float
    metadata: Dict[str, Any]
    document: str


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
    def search(
        self, query_vectors: List[np.ndarray], top_k: int = 5
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors (List[np.ndarray]): The query vectors to search for.
            top_k (int): The number of most similar vectors to return for each query.

        Returns:
            List[List[SearchResult]]: List of lists of SearchResult objects containing search results.
                                      The i-th inner list corresponds to the results for the i-th query vector.
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


class ChromaDB(VectorDB):
    """
    Concrete implementation of VectorDB using Chroma.
    """

    def __init__(self, collection_name: str):
        """
        Initialize the ChromaDB instance with the specified collection name.

        Args:
            collection_name (str): The name of the collection to create or use.
        """
        chromadb_path = f"{os.getcwd()}/vectorstore/chromadb/"
        os.makedirs(chromadb_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def insert(self, embeddings: List[Embedding]) -> None:
        """
        Insert embeddings into the vector database.

        Args:
            embeddings (List[Embedding]): List of Embedding objects to insert.
        """
        documents = [embedding.text for embedding in embeddings]
        ids = [embedding.name for embedding in embeddings]
        vectors = [embedding.vector.tolist() for embedding in embeddings]
        self.collection.add(ids=ids, embeddings=vectors, documents=documents)

    def search(
        self, query_vectors: List[np.ndarray], top_k: int = 5, where_document: Optional[dict]=None
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors (List[np.ndarray]): The query vectors to search for.
            top_k (int): The number of most similar vectors to return for each query.
            where_documents: An optional dictionary that can filter by content in the dictionary

        Returns:
            List[List[SearchResult]]: List of lists of SearchResult objects containing search results.
                                      The i-th inner list corresponds to the results for the i-th query vector.
        """

        if not where_document: #defaults where document to None if it doesn't exist or if it is an empty dict
            where_document = None

        query_embeddings = [vector.tolist() for vector in query_vectors]
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where_document=where_document
        )

        all_results = []
        for i in range(len(query_vectors)):
            query_results = [
                SearchResult(
                    id=id, distance=distance, metadata=metadata, document=document
                )
                for id, distance, metadata, document in zip(
                    results["ids"][i],
                    results["distances"][i],
                    results["metadatas"][i],
                    results["documents"][i],
                )
            ]
            all_results.append(query_results)

        return all_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the database by their IDs.

        Args:
            ids (List[str]): List of vector IDs to delete.
        """
        self.collection.delete(ids=ids)

    def update(self, ids: List[str], embeddings: List[Embedding]) -> None:
        """
        Update existing vectors in the database.

        Args:
            ids (List[str]): List of vector IDs to update.
            embeddings (List[Embedding]): List of new Embedding objects.
        """
        documents = [embedding.text for embedding in embeddings]
        vectors = [embedding.vector.tolist() for embedding in embeddings]
        self.collection.update(ids=ids, embeddings=vectors, documents=documents)
