import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

import chromadb
import numpy as np
import scipy
import torch
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    WeightedRanker,
    connections,
    utility,
)

from .embedding import Embedding, BGEM3Embedder
from .reranker import Reranker

# Get a logger for this module.
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str  # TODO: add this
    distance: float
    metadata: Mapping[str, Any]
    document: str
    embedding: List[float]


class VectorDB(ABC):
    """
    Abstract base class for vector database operations.
    """

    @abstractmethod
    def insert(
        self, embedding: Embedding, metadatum: Optional[List[dict]] = None
    ) -> None:
        """
        Insert embeddings into the vector database.


        Args:
            embedding (Embedding): Embedding object to insert.
            metadatum (Optional[List[dict]]): Optional metadata list.
        """
        pass

    @abstractmethod
    def search(
        self,
        queries: List[str],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        hybrid_weighting: float = 0.5,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ) -> List[List[SearchResult]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vectors (List[np.ndarray]): The query vectors to search for.
            top_k (int): The number of most similar vectors to return for each query.
            keywords (List[str]): The keywords to use for keyword search.
            filenames (List[str]): The filenames to filter by.
            hybrid_weighting (float): Weight for hybrid search (dense vs sparse).
            reranker_model (str): Name of the reranker model to use.

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

    def __init__(
        self,
        collection_name: str,
    ):
        """
        Initialize the ChromaDB instance with the specified collection name.

        Args:
            collection_name (str): The name of the collection to create or use.
        """
        chromadb_path = os.path.join(os.getcwd(), "vectorstore", "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def insert(
        self, embeddings: List[Embedding], metadatum: Optional[List[dict]] = None
    ) -> None:
        """
        Insert an aggregated Embedding into the Chroma collection.
        """
        # Expecting aggregated embedding with:
        #   embedding.name -> list of IDs
        #   embedding.docs -> list of document texts
        #   embedding.dense_vector -> 2D numpy array (one row per document)
        documents = [embedding.docs for embedding in embeddings]
        ids = [embedding.name for embedding in embeddings]
        # Convert each dense vector row into a list.
        vectors = [embedding.dense_vector.tolist() for embedding in embeddings]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatum=metadatum,
        )

    def search(
        self,
        query_embeddings: List[Embedding],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        hybrid_weighting: float = 0.5,
    ) -> List[List[SearchResult]]:
        """
        Process query strings internally to compute embeddings, then perform the search.
        """
        where_document = None
        if keywords:
            if len(keywords) > 1:
                where_document = {"$or": [{"$contains": kw} for kw in keywords]}
            else:
                where_document = {"$contains": keywords[0]}
        where = {"source": {"$in": filenames}} if filenames else None

        dense_vectors = [
            embedding.dense_vector.tolist() for embedding in query_embeddings
        ]

        results = self.collection.query(
            query_embeddings=dense_vectors,
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "embeddings", "metadatum", "distances"],
        )

        all_results = []
        for i in range(len(dense_vectors)):
            query_results = [
                SearchResult(
                    id=_id,
                    distance=distance,
                    metadata=metadata,
                    document=document,
                    embedding=embedding,
                )
                for _id, distance, metadata, document, embedding in zip(
                    results["ids"][i],
                    results["distances"][i],
                    results["metadatum"][i],
                    results["documents"][i],
                    results["embeddings"][i],
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
    Unified Milvus Vector Database that supports both standard and BGEM3 hybrid embeddings.

    The collection schema is always:
        - id (VARCHAR primary key)
        - text (VARCHAR)
        - sparse_vector (SPARSE_FLOAT_VECTOR)
        - dense_vector (FLOAT_VECTOR)

    When if_hybrid_search is True, the sparse vector is computed via BGEM3 and used during search.
    Otherwise, sparse_vector is filled with empty dictionaries and only dense_vector is used.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "standalone",
        port: str = "19530",
        dense_dim: int = 1536,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        """
        Args:
            collection_name (str): Name of the Milvus collection.
            host (str): Milvus host.
            port (str): Milvus port.
            dense_dim (int): Dimension for the dense vector.
            reranker_model (str): Model name for the BGE reranker
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dim = dense_dim

        self._setup_milvus()

        # Initialize reranker
        self.reranker = Reranker(model_name=reranker_model)

    def _setup_milvus(self):
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

        if not self.client.has_collection(self.collection_name):
            # Define the collection schema.
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=65535,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                FieldSchema(
                    name="filename",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                    enable_match=True,
                ),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(
                    name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim
                ),
            ]
            schema = CollectionSchema(fields, "Hybrid search collection")

            # Create index params
            index_params = self.client.prepare_index_params()

            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="FLAT",
                metric_type="IP",
            )

            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_inverted_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )

        self.client.load_collection(self.collection_name)

    def insert(
        self, embeddings: List[Embedding], metadatum: Optional[List[dict]] = None
    ) -> None:
        """
        Inserts documents into Milvus.

        Computes the dense vector using the provided dense_embedder.
        Computes the sparse vector using BGEM3 if hybrid search is enabled;
        otherwise, fills the sparse_vector field with empty dicts.

        Args:
            embedding: An object that contains:
                - docs: a list of document texts.
                - name: a list of unique IDs.
            metadatum (Optional[List[dict]]): Additional metadata for each document.
        """
        entities = [
            {
                "id": embedding.name,
                "text": embedding.docs,
                "filename": metadata["source"],
                "dense_vector": embedding.dense_vector.tolist(),
                "sparse_vector": embedding.sparse_vector,
            }
            for embedding, metadata in zip(embeddings, metadatum)
        ]

        self.client.insert(collection_name=self.collection_name, data=entities)
        self.client.flush(collection_name=self.collection_name)
        logger.info(
            "Inserted %d documents into Milvus collection '%s'.",
            len(entities),
            self.collection_name,
        )

    def search(
        self,
        query_embeddings: List[Embedding],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        hybrid_weighting: float = 0.5,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ) -> List[List[SearchResult]]:
        """
        Searches Milvus for relevant documents and applies reranking.

        For each query:
        - The dense embedding is computed using the dense embedder.
        - The sparse embedding is computed using BGEM3.
        - Both embeddings are used for hybrid search.
        - Results are reranked using BGE reranker.

        Args:
            query_embeddings (List[Embedding]): Query embeddings.
            top_k (int): Number of results to return per query.
            keywords (Optional[List[str]]): Keywords for filtering.
            filenames (Optional[List[str]]): Filenames for filtering.
            hybrid_weighting (float): Weight for sparse vector in hybrid search (1-weight for dense).
            reranker_model (str): Name of the reranker model to use.

        Returns:
            List[List[SearchResult]]: A list of search result lists.
        """
        print(f"reranker_model: {reranker_model}")
        print(f"Hybrid weighting: {hybrid_weighting}")
        print(f"Reranker_model: {reranker_model}")
        self.reranker = Reranker(model_name=reranker_model)

        filter_list = []
        if keywords:
            filter_list.append(f"TEXT_MATCH(text, '{' '.join(keywords)}')")

        if filenames:
            filter_list.append(f"TEXT_MATCH(filename, '{' '.join(filenames)}')")

        filter_expr = " AND ".join(filter_list) if len(filter_list) > 0 else None

        dense_req = AnnSearchRequest(
            [embedding.dense_vector.tolist() for embedding in query_embeddings],
            "dense_vector",
            {"metric_type": "IP"},
            expr=filter_expr,
            limit=top_k,
        )

        sparse_req = AnnSearchRequest(
            [embedding.sparse_vector for embedding in query_embeddings],
            "sparse_vector",
            {"metric_type": "IP"},
            expr=filter_expr,
            limit=top_k,
        )

        # Perform hybrid search
        hybrid_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[sparse_req, dense_req],
            ranker=WeightedRanker(hybrid_weighting, 1 - hybrid_weighting),
            limit=top_k,
            output_fields=["id", "text", "filename", "dense_vector"],
        )

        all_results = []
        # Milvus currently only supports 1 query per hybrid search
        for i, (embedding, hits) in enumerate(zip(query_embeddings, hybrid_results)):
            # Rerank the results for this query
            reranked_hits = self.reranker.rerank_milvus_results(
                query=embedding.docs, hits=hits, top_k=top_k
            )

            query_results = []
            for hit in reranked_hits:
                query_results.append(
                    SearchResult(
                        id=str(hit["id"]),
                        distance=hit["distance"],
                        metadata={"filename": hit["entity"].get("filename", "")},
                        document=hit["entity"].get("text", ""),
                        embedding=hit["entity"].get("dense_vector", []),
                    )
                )
            all_results.append(query_results)

        return all_results

    def delete(self, ids: List[str]) -> None:
        """Deletes documents from the collection by their IDs."""
        self.client.delete(collection_name=self.collection_name, expr=f"id in {ids}")

    def get_all_ids(self) -> List[str]:
        """Retrieves all document IDs from the collection."""
        results = self.client.query(
            collection_name=self.collection_name,
            filter="id != 'NULL'",
            output_fields=["id"],
        )
        return [result["id"] for result in results]

    def clear(self) -> None:
        """Clears all documents from the collection."""
        self.client.delete(collection_name=self.collection_name, expr='id != "NULL"')
