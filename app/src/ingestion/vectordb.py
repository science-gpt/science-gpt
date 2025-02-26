
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Any, List, Mapping, Optional, Callable
import os
import scipy

import chromadb
import numpy as np
from pymilvus import AnnSearchRequest, CollectionSchema, DataType, FieldSchema, MilvusClient, RRFRanker, WeightedRanker, Collection

from .embedding import Embedding
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import logging

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
    def insert(self, embedding: Embedding, metadatum: Optional[List[dict]] = None) -> None:
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

    def __init__(self, collection_name: str, 
                 embedder: Optional[Callable[[List[str]], dict]] = None,
                 embedding_dimension: int = 768):
        """
        Initialize the ChromaDB instance with the specified collection name.

        Args:
            collection_name (str): The name of the collection to create or use.
        """
        chromadb_path = os.path.join(os.getcwd(), "vectorstore", "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_dimension = embedding_dimension
        # If no embedder is provided, fallback to zero vectors.
        self.ef = embedder if embedder is not None else lambda texts: {"dense": np.array([np.zeros(self.embedding_dimension) for _ in texts])}

    def insert(self, embedding: Embedding, metadatum: Optional[List[dict]] = None) -> None:
        """
        Insert an aggregated Embedding into the Chroma collection.
        """
        # Expecting aggregated embedding with:
        #   embedding.name -> list of IDs
        #   embedding.docs -> list of document texts
        #   embedding.dense_vector -> 2D numpy array (one row per document)
        documents = embedding.docs
        ids = embedding.name
        # Convert each dense vector row into a list.
        vectors = [vec.tolist() for vec in embedding.dense_vector]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatum=metadatum,
        )

    def search(
        self,
        queries: List[str],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
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

        # Process queries into embeddings.
        query_emb_dict = self.ef(queries)
        query_vectors = query_emb_dict["dense"]
        query_embeddings = [vec.tolist() for vec in query_vectors]

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "embeddings", "metadatum", "distances"],
        )

        all_results = []
        for i in range(len(queries)):
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

    def update(self, ids: List[str], embedding: Embedding, metadatum: Optional[List[dict]] = None) -> None:
        """
        Update existing vectors in the database.

        Args:
            ids (List[str]): List of vector IDs to update.
            embeddings (List[Embedding]): List of new Embedding objects.
        """
        self.delete(ids)
        self.insert(embedding, metadatum)

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
        dense_embedder: Optional[Callable[[List[str]], dict]] = None,
        if_hybrid_search: bool = True,
        use_reranker: bool = True,
    ):
        """
        Args:
            collection_name (str): Name of the Milvus collection.
            host (str): Milvus host.
            port (str): Milvus port.
            dense_dim (int): Dimension for the dense vector.
            dense_embedder (Callable): Function that takes a list of texts and returns a dict with a "dense" key.
            if_hybrid_search (bool): If True, use hybrid search (dense + sparse); if False, use only dense search.
            use_reranker (bool): Whether to use a reranker during search.
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.if_hybrid_search = if_hybrid_search
        self.use_reranker = use_reranker
        self.dense_dim = dense_dim

        if dense_embedder is None:
            raise ValueError("A dense_embedder function must be provided.")
        self.dense_embedder = dense_embedder

        # Always use BGEM3 for the sparse vector component.
        self.bge_m3_func = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

        self._setup_milvus()

    def _setup_milvus(self):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        
        print("Connecting to MilvusBGE at %s:%s", self.host, self.port)
        connections.connect("default", host=self.host, port=self.port)
        print("Connected to MilvusBGE successfully.")

        # Define the collection schema.
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
        ]
        
        schema = CollectionSchema(fields, "Hybrid search collection")
        
        # For debugging: drop any pre-existing collection.
        if utility.has_collection(self.collection_name):
            print("Collection %s already exists. Dropping it to re-create.", self.collection_name)
            utility.drop_collection(self.collection_name)
        
        # Create the collection.
        print("Creating collection %s", self.collection_name)
        self.collection = Collection(self.collection_name, schema, consistency_level="Strong")
        
        # Create indexes on the vector fields.
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        print("Creating index on 'sparse_vector' with params: %s", sparse_index)
        self.collection.create_index("sparse_vector", sparse_index)

        print("Creating index on 'dense_vector' with params: %s", dense_index)
        self.collection.create_index("dense_vector", dense_index)
        
        # Load the collection so it can serve search requests immediately.
        print("Loading collection %s", self.collection_name)
        self.collection.load()
        print("Collection %s loaded successfully", self.collection_name)

    def insert(self, embedding, metadatum: Optional[List[dict]] = None) -> None:
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
        docs = embedding.docs
        ids = embedding.name
        dense_embeddings = embedding.dense_vector
        
        # Compute sparse embeddings if hybrid search is enabled.
        try:
            if self.if_hybrid_search:
                sparse_result = self.bge_m3_func(docs)
                sparse_embeddings = sparse_result["sparse"]
                print("Sparse embeddings when inserting: ", sparse_embeddings)
            else:
                sparse_embeddings = [dict() for _ in docs]
        except Exception as e:
            logger.error("Error computing sparse embeddings: %s", str(e))
            sparse_embeddings = [dict() for _ in docs]
        
        entities = [
            ids,
            docs,
            sparse_embeddings,
            dense_embeddings,
        ]
        
        # debug print out the entities ids docs sparse_embeddings dense_embeddings
        print("ids: ", ids)
        print("docs: ", docs)
        print("sparse_embeddings: ", sparse_embeddings)
        print("dense_embeddings: ", dense_embeddings)
        
        self.collection.insert(entities)
        self.collection.flush()
        logger.info("Inserted %d documents into Milvus collection '%s'.", len(docs), self.collection_name)

    def search(
        self,
        queries: List[str],
        top_k: int = 5,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Searches Milvus for relevant documents.
        
        For each query:
        - The dense embedding is computed using the chosen dense embedder (with the query wrapped as a Chunk).
        - The sparse embedding is computed using BGEM3.
        
        If if_hybrid_search is True, both embeddings are used to create a hybrid search request.
        Otherwise, only the dense query is used.
        
        Args:
            queries (List[str]): Query texts.
            top_k (int): Number of results to return per query.
            keywords (Optional[List[str]]): Keywords for filtering (not used in this snippet).
            filenames (Optional[List[str]]): Filenames for filtering (not used in this snippet).
        
        Returns:
            List[List[SearchResult]]: A list of search result lists.
        """
        from ingestion.chunking import Chunk
        all_results = []
        
        for query in queries:
            print(f"Processing query: {query}")
            # Wrap the query into a Chunk for the dense embedder.
            query_chunk = Chunk(text=query, name=f"Query-{query[:10]}", data_type="query")
            # Compute dense query embedding using the dense embedder.
            dense_query = self.dense_embedder([query_chunk]).dense_vector
            # Compute sparse query embedding using BGEM3.
            sparse_query = self.bge_m3_func([query])["sparse"]
            
            
            print("Dense query:", dense_query)
            print("Sparse query:", sparse_query)
        
            if self.if_hybrid_search:
                # Create AnnSearchRequest objects for both dense and sparse fields.
                dense_req = AnnSearchRequest(dense_query, "dense_vector", {"metric_type": "IP"}, limit=top_k)
                sparse_req = AnnSearchRequest(sparse_query, "sparse_vector", {"metric_type": "IP"}, limit=top_k)
                
                # Call hybrid_search with reranking if enabled.
                if self.use_reranker:
                    hybrid_results = self.collection.hybrid_search(
                        reqs=[sparse_req, dense_req],
                        rerank=RRFRanker() if self.use_reranker else None,
                        limit=top_k,
                        output_fields=["text", "id"],
                    )[0]
                    
                    result_texts = [hit.fields["text"] for hit in hybrid_results]
                    bge_rf = BGERerankFunction(device='cpu')
                    reranked_results = bge_rf(query, result_texts, top_k=top_k)
                    query_results = []
                    for i, hit in enumerate(reranked_results):
                        print(f'text: {hit.text} distance {hit.score}')
                        query_results.append(
                            SearchResult(
                                id=hybrid_results[i].fields.get("id", ""),
                                distance=float(hit.score),
                                metadata={},
                                document=hit.text,
                                embedding=[]
                            )
                        )
                    all_results.append(query_results)
            else:
                # Dense-only search: use only the dense query.
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 512}}
                # Convert dense_query to a list of lists if necessary.
                query_vectors = [vec.tolist() if hasattr(vec, "tolist") else vec for vec in dense_query]
                
                results = self.collection.search(
                    data=query_vectors,
                    anns_field="dense_vector",
                    param=search_params,
                    limit=top_k,
                    expr="",
                    output_fields=["text", "id"],
                )
                
                query_results = []
                for hit in results[0]:
                    query_results.append(
                        SearchResult(
                            id=str(hit.id),
                            distance=hit.distance,
                            metadata={},
                            document=hit.entity.get("text", ""),
                            embedding=hit.entity.get("dense_vector", [])
                        )
                    )
                all_results.append(query_results)
        return all_results


    
    def delete(self, ids: List[str]) -> None:
        """Deletes documents from the collection by their IDs."""
        self.collection.delete(expr=f"id in {ids}")
    
    def update(self, ids: List[str], embedding, metadatum: Optional[List[dict]] = None) -> None:
        """Updates documents by deleting and reinserting them."""
        self.delete(ids)
        self.insert(embedding, metadatum)
    
    def get_all_ids(self) -> List[str]:
        """Retrieves all document IDs from the collection."""
        results = self.collection.query(expr="id != 'NULL'", output_fields=["id"])
        return [result["id"] for result in results]
    
    def clear(self) -> None:
        """Clears all documents from the collection."""
        self.collection.delete(expr='id != "NULL"')

class MilvusBGE_DB(VectorDB):
    """
    Concrete implementation of VectorDB for hybrid search with BGE-M3 embeddings.
    
    This class creates a Milvus collection designed for storing both dense and sparse vectors.
    It uses separate fields ("dense_vector" and "sparse_vector") to support BGE-M3's output.
    """
    def __init__(self, collection_name: str, host: str = "localhost", port: str = "19530", 
                 use_bge_m3: bool = True, use_reranker: bool = True):
        """
        Initialize the MilvusBGE_DB instance.
        
        Args:
            collection_name (str): The name of the collection to create or use.
            host (str): The host of the Milvus server.
            port (str): The port of the Milvus server.
            use_bge_m3 (bool): Whether to use the BGEM3 embedding function.
            use_reranker (bool): Whether to use a reranker after search.
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.use_bge_m3 = use_bge_m3
        self.use_reranker = use_reranker
        self.dense_dim = 768

        # Use the BGEM3 embedder if enabled.
        if self.use_bge_m3:
            print("Using BGEM3 embedder")
            self.ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
            self.dense_dim = self.ef.dim["dense"]
            print("Using dense_dimension: %d", self.dense_dim)
        else:
            self.ef = self.random_embedding
            print("Using fallback embedder (if implemented)")



        self._setup_milvus()


    def _setup_milvus(self):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        
        print("Connecting to MilvusBGE at %s:%s", self.host, self.port)
        connections.connect("default", host=self.host, port=self.port)
        print("Connected to MilvusBGE successfully.")


        # Define the collection schema.
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
        ]
        
        schema = CollectionSchema(fields, "Hybrid search collection")
        
        # For debugging: drop any pre-existing collection.
        if utility.has_collection(self.collection_name):
            print("Collection %s already exists. Dropping it to re-create.", self.collection_name)
            utility.drop_collection(self.collection_name)
        

        # Create the collection.
        print("Creating collection %s", self.collection_name)
        self.collection = Collection(self.collection_name, schema, consistency_level="Strong")
        

        # Create indexes on the vector fields.
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        print("Creating index on 'sparse_vector' with params: %s", sparse_index)
        self.collection.create_index("sparse_vector", sparse_index)

        print("Creating index on 'dense_vector' with params: %s", dense_index)
        self.collection.create_index("dense_vector", dense_index)
        

        # Load the collection so it can serve search requests immediately.
        print("Loading collection %s", self.collection_name)
        self.collection.load()
        print("Collection %s loaded successfully", self.collection_name)

    
    def insert(self, embedding: Embedding, metadatum: Optional[List[dict]] = None) -> None:
        """
        Insert hybrid embeddings into the Milvus collection.
        """
        # milbus expects one embedding at a time for bgem3 embedder
        # Extract the raw documents and ids from the embedding objects.
        docs = embedding.docs

        print("Inserting %d docs into collection %s" % (len(docs), self.collection_name))
        
        # Recompute embeddings using BGEM3 embedder.
        docs_embeddings = self.ef(docs)
        print("type(docs_embeddings): ", type(docs_embeddings))
        
        # Prepare the entities payload with all four required fields in order:
        # [id, text, sparse_vector, dense_vector]
        entities = [
            [f"{i}" for i in range(len(docs))],  # Generate unique id for each doc
            docs,                       # text field
            docs_embeddings["sparse"],  # sparse_vector field
            docs_embeddings["dense"]    # dense_vector field
        ]
        
        print("[DEBUG] Entities structure:")
        print("- ids:", type(entities[0]), len(entities[0]))
        print("- docs:", type(entities[1]), len(entities[1]))
        print("- sparse:", type(entities[2]))
        print("- dense:", type(entities[3]))

        self.collection.insert(entities)
        print("Successfully inserted %d entities" % len(entities))
        self.collection.flush()

    def search(
        self,
        queries: List[str],
        top_k: int = 2,
        keywords: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
    ) -> List[List[SearchResult]]:
        print("Using hybrid search in MilvusBGE_DB")
        print("[DEBUG] MilvusBGE_DB.search() received queries:", queries)
        
        print("top_k: ", top_k)
        all_results = []
        for query in queries:
            query_results = []
            
            # Get embeddings for this query
            query_embeddings = self.ef([query])
            
            # Prepare the search requests for both vector fields
            sparse_req = AnnSearchRequest(
                query_embeddings["sparse"],
                "sparse_vector",
                {"metric_type": "IP"},
                limit=top_k
            )
            dense_req = AnnSearchRequest(
                query_embeddings["dense"],
                "dense_vector",
                {"metric_type": "IP"},
                limit=top_k
            )
            
            # Perform hybrid search
            # Currently Milvus only support 1 query in the same hybrid search request, so
            # we inspect res[0] directly. In future release Milvus will accept batch
            # hybrid search queries in the same call.
            hybrid_results = self.collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=RRFRanker() if self.use_reranker else None,
                limit=top_k,
                output_fields=["text", "id"]
            )[0]
            
            # Process results
            if self.use_reranker:
                result_texts = [hit.fields["text"] for hit in hybrid_results]
                # https://milvus.io/api-reference/pymilvus/v2.4.x/Rerankers/BGERerankFunction/BGERerankFunction.md
                # here by default the reranker model is BAAI/bge-reranker-v2-m3
                bge_rf = BGERerankFunction(device='cpu')
                
                try:
                    # Match the example code's reranker call signature
                    # rerank the results using BGE CrossEncoder model
                    reranked_results = bge_rf(query, result_texts, top_k=top_k)
                    for hit in reranked_results:
                        print(f'text: {hit.text} distance {hit.score}')
                        
                    for i, hit in enumerate(reranked_results):
                        query_results.append(
                            SearchResult(
                                id=hybrid_results[i].fields.get("id", ""),
                                distance=float(hit.score),
                                metadata={},
                                document=hit.text,
                                embedding=[]
                            )
                        )
                except Exception as e:
                    print(f"[ERROR] Reranking failed: {str(e)}")
                    # Fall back to non-reranked results
                    for hit in hybrid_results:
                        query_results.append(
                            SearchResult(
                                id=hit.fields.get("id", ""),
                                distance=float(hit.distance),
                                metadata={},
                                document=hit.fields["text"],
                                embedding=[]
                            )
                        )
            else:
                for hit in hybrid_results:
                    query_results.append(
                        SearchResult(
                            id=hit.fields.get("id", ""),
                            distance=float(hit.distance),
                            metadata={},
                            document=hit.fields["text"],
                            embedding=[]
                        )
                    )
            
            all_results.append(query_results)
        
        print("all_results: ", all_results)
        return all_results


    # def search(
    #     self,
    #     queries: List[str],
    #     top_k: int = 2,
    #     keywords: Optional[List[str]] = None,
    #     filenames: Optional[List[str]] = None,
    # ) -> List[List[SearchResult]]:
    #     print("Using hybrid search in MilvusBGE_DB")
    #     print("[DEBUG] MilvusBGE_DB.search() received queries:", queries)
        
    #     print("top_k: ", top_k)
    #     all_results = []
    #     for query in queries:
    #         query_results = []
            
    #         # Get embeddings for this query
    #         query_embeddings = self.ef([query])
            
    #         # Prepare the search requests for both vector fields
    #         sparse_req = AnnSearchRequest(
    #             query_embeddings["sparse"],
    #             "sparse_vector",
    #             {"metric_type": "IP"},
    #             limit=top_k
    #         )
    #         dense_req = AnnSearchRequest(
    #             query_embeddings["dense"],
    #             "dense_vector",
    #             {"metric_type": "IP"},
    #             limit=top_k
    #         )
    #         sparse_weight = 1.0
    #         dense_weight = 1.0
    #         reranker = WeightedRanker(sparse_weight, dense_weight)
            
    #         hybrid_results = self.collection.hybrid_search(
    #             [sparse_req, dense_req],
    #             rerank=reranker,
    #             limit=top_k,
    #             output_fields=["text", "id"]
    #         )[0]
            
    #         all_results.append(hybrid_results)
        
    #     print("all_results: ", all_results)
    #     return all_results
    
    def delete(self, ids: List[str]) -> None:
        print("Deleting ids %s from collection %s", ids, self.collection_name)
        self.collection.delete(expr=f"id in {ids}")


    def update(self, ids: List[str], embeddings: List) -> None:
        print("Updating %d embeddings", len(ids))
        self.delete(ids)
        self.insert(embeddings)


    def get_all_ids(self) -> List[str]:
        results = self.collection.query(expr="id != 'NULL'", output_fields=["id"])
        ids = [result["id"] for result in results]
        print("Retrieved %d ids from collection", len(ids))
        return ids


    def clear(self) -> None:
        print("Clearing collection %s", self.collection_name)
        self.collection.delete(expr='id != "NULL"')