from typing import List

from ingestion.chunking import Chunk, Chunker
from ingestion.embedding import Embedder
from ingestion.extraction import TextExtract
from ingestion.raw_data import RawData
from ingestion.vectordb import SearchResult, VectorDB
from orchestrator.config import SystemConfig


class DataBroker:
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.

    Attributes:
        config (SystemConfig): Configuration object containing settings for
            embedding model, vector store, chunking method, etc.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Instantiates an object of this class.

        Args:
            config (SystemConfig): Configuration object containing settings for
                embedding model, vector store, chunking method, etc.
        """
        self.config = config
        self.extractor = self._create_extractor(config)
        self.chunker = self._create_chunker(config)
        self.embedder = self._create_embedder(config)
        self.vector_store = self._create_vector_store(config)

    def insert(self, data: RawData) -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        Args:
            data (RawData): The raw data to be processed and inserted

        Raises:
            ValueError: If any step in the process fails due to unsupported
                data types or methods
        """
        # TODO better logging and error handling
        text = self.extractor(data)
        chunks = self.chunker(text)
        embeddings = self.embedder(chunks)
        self.vector_store.insert(embeddings)

    def search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        Searches the vector store for the most relevant docs based on the given queries.

        Args:
            queries (List[str]): List of search queries
            top_k (int): The number of results to return for each query (default: 5)

        Returns:
            List[List[SearchResult]]: A list of lists of SearchResult objects containing
                the search results for each query, sorted by relevance

        Raises:
            ValueError: If the search operation fails
        """
        try:
            # Create Chunk objects from the queries
            query_chunks = [
                Chunk(text=query, title=f"Query_{i}", data_type="query")
                for i, query in enumerate(queries)
            ]

            # Embed the query chunks
            query_embeddings = self.embedder(query_chunks)

            # Extract vectors from embeddings
            query_vectors = [embedding.vector for embedding in query_embeddings]

            # Search the vector store
            results = self.vector_store.search(query_vectors, top_k)

            return results
        except Exception as e:
            raise ValueError(f"Search operation failed: {str(e)}")

    @staticmethod
    def _create_extractor(config: SystemConfig, data_type: str) -> TextExtract:
        """
        Creates an extractor based on the configured extraction method and data type.

        Args:
            config (SystemConfig): Configuration object containing settings for
                embedding model, vector store, chunking method, etc.

        Returns:
            TextExtract: A function that extracts text from the given data

        Raises:
            ValueError: If the configured extraction method is not supported
            ValueError: If the data type is not supported
        """
        # Implement based on config
        raise NotImplementedError("Extractor creation not yet implemented")

    @staticmethod
    def _create_chunker(config: SystemConfig) -> Chunker:
        """
        Creates a chunker based on the configured chunking method.

        Args:
            config (SystemConfig): Configuration object containing settings for
                embedding model, vector store, chunking method, etc.

        Returns:
            Chunker: A function that splits the input text into chunks

        Raises:
            ValueError: If the configured chunking method is not supported
        """
        # Implement based on config
        raise NotImplementedError("Embedding creation not yet implemented")

    @staticmethod
    def _create_embedder(config: SystemConfig) -> Embedder:
        """
        Creates an embedder based on the configured embedding model.

        Args:
            config (SystemConfig): Configuration object containing settings for
                embedding model, vector store, chunking method, etc.

        Returns:
            Embedder: A function that embeds the given chunks

        Raises:
            NotImplementedError: If the embedding creation is not yet implemented
        """
        # Implement based on config
        raise NotImplementedError("Embedding creation not yet implemented")

    @staticmethod
    def _create_vector_store(config: SystemConfig) -> VectorDB:
        """
        Creates a vector store based on the configured vector store.

        Args:
            config (SystemConfig): Configuration object containing settings for
                embedding model, vector store, chunking method, etc.

        Returns:
            VectorDB: A function that inserts vectors into the vector store

        Raises:
            NotImplementedError: If the vector store creation is not yet implemented
        """
        # Implement based on config
        raise NotImplementedError("Vector store creation not yet implemented")
