import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import chromadb
import numpy as np
from pymilvus import DataType, MilvusClient

from .embedding import Embedding


@dataclass
class SearchResult:
    id: str  # TODO: add this
    distance: float
    metadata: Mapping[str, Any]
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
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 5,
        keywords: Optional[list[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors (List[np.ndarray]): The query vectors to search for.
            top_k (int): The number of most similar vectors to return for each query.
            keywords (List[str]): The keywords to use for keyword search.

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

    @abstractmethod
    def get_all_ids(self) -> List[str]:
        """
        Retrieve all IDs from the database.

        Returns:
            List[str]: List of all document IDs in the database.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all records from the collection.
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
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 5,
        keywords: Optional[list[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors (List[np.ndarray]): The query vectors to search for.
            top_k (int): The number of most similar vectors to return for each query.
            keywords: An optional list that has filter for content in the list

        Returns:
            List[List[SearchResult]]: List of lists of SearchResult objects containing search results.
                                      The i-th inner list corresponds to the results for the i-th query vector.
        """

        where_document = None
        if keywords:
            if len(keywords) > 1:
                where_document = {
                    "$or": [{"$contains": keyword} for keyword in keywords]
                }
            else:
                where_document = {"$contains": keywords[0]}

        query_embeddings = [vector.tolist() for vector in query_vectors]
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where_document=where_document,
        )

        all_results = []
        for i in range(len(query_vectors)):
            query_results = [
                SearchResult(
                    id=_id, distance=distance, metadata=metadata, document=document
                )
                for _id, distance, metadata, document in zip(
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

    def get_all_ids(self) -> List[str]:
        """
        Retrieve all IDs from the database.

        Returns:
            List[str]: List of all document IDs in the database.
        """
        return self.collection.get()["ids"]

    def clear(self) -> None:
        """
        Clear all records from the collection.
        """
        all_ids = self.get_all_ids()
        if all_ids:
            self.collection.delete(ids=all_ids)


class MilvusDB(VectorDB):
    """
    Concrete implementation of VectorDB using Milvus.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "standalone",
        port: str = "19530",
        dim: int = 1536,
    ):
        """
        Initialize the MilvusDB instance with the specified collection name.

        Args:
            collection_name (str): The name of the collection to create or use.
            host (str): The host of the Milvus server (defaults to "standalone")
            port (str): The port of the Milvus server (defaults to "19530")
            dim (int): Dimension of the vectors to be stored (defaults to 1536 for OpenAI embeddings)
        """
        self.collection_name = collection_name
        self.client = MilvusClient(uri=f"http://{host}:{port}")

        if not self.client.has_collection(collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )

            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
            )
            schema.add_field(
                field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim
            )
            schema.add_field(
                field_name="document",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                enable_match=True,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

        self.client.load_collection(collection_name)

    def insert(self, embeddings: List[Embedding]) -> None:
        """
        Insert embeddings into the vector database.
        """
        entities = [
            {
                "id": embedding.name,
                "vector": embedding.vector.tolist(),
                "document": embedding.text,
            }
            for embedding in embeddings
        ]

        self.client.insert(collection_name=self.collection_name, data=entities)

    def search(
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database with optional keyword filtering.
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        filter_expr = (
            f"TEXT_MATCH(document, '{' '.join(keywords)}')" if keywords else None
        )

        results = self.client.search(
            collection_name=self.collection_name,
            data=[v.tolist() for v in query_vectors],
            anns_field="vector",
            search_params=search_params,
            limit=top_k,
            filter=filter_expr,
            output_fields=["document"],
        )

        all_results = []
        for hits in results:
            query_results = []
            for hit in hits:
                query_results.append(
                    SearchResult(
                        id=str(hit.get("id")),
                        distance=hit.get("distance"),
                        metadata={},
                        document=hit.get("entity", {}).get("document"),
                    )
                )
            all_results.append(query_results)
        return all_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the database by their IDs.
        """
        self.client.delete(collection_name=self.collection_name, filter=f"id in {ids}")

    def update(self, ids: List[str], embeddings: List[Embedding]) -> None:
        """
        Update existing vectors in the database.
        """
        # Milvus doesn't support direct updates, so delete and reinsert
        self.delete(ids)
        self.insert(embeddings)

    def get_all_ids(self) -> List[str]:
        """
        Retrieve all IDs from the database.

        Returns:
            List[str]: List of all document IDs in the database.
        """
        # no milvus built-in for "grab everything", so hacking it
        results = self.client.query(
            collection_name=self.collection_name,
            filter="id != 'NULL'",
            output_fields=["id"],
        )
        return [result["id"] for result in results]

    def clear(self) -> None:
        """
        Clear all records from the collection.
        """
        self.client.delete(collection_name=self.collection_name, filter='id != "NULL"')
