from typing import Dict, List

from ingestion.chunking import Chunk, Chunker, SplitSentencesChunker
from ingestion.embedding import Embedder, HuggingFaceSentenceTransformerEmbedder
from ingestion.extraction import PyPDF2Extract, TextExtract
from ingestion.raw_data import Data
from ingestion.vectordb import ChromaDB, SearchResult, VectorDB
from orchestrator.config import SystemConfig


# TODO: error handling throughout this classis absent or inconsistent
class DataBroker:
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Instantiates an object of this class.

        Args:
            config (SystemConfig): Configuration object containing settings
        """
        self.config = config
        self.extractors = self._create_extractors(config)
        self.chunker = self._create_chunker(config)
        self.embedder = self._create_embedder(config)
        self.vector_store = self._create_vector_store(config)

    def insert(self, data: Data) -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        Args:
            data (Data): The raw data to be processed and inserted
        """
        # TODO better logging and error handling
        extractor = self.extractors.get(data.data_type)
        text = extractor(data)
        chunks = self.chunker(text)
        embeddings = self.embedder(chunks)
        self.vector_store.insert(embeddings)

    def search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        Searches the vector store for the most relevant docs based on the given queries.

        Args:
            queries (List[str]): List of search queries
            top_k (int): The number of results to return for each query

        Returns:
            List[List[SearchResult]]: A list of lists of SearchResult objects containing
                the search results for each query, sorted by relevance
        """
        # TODO better logging and error handling
        query_chunks = [
            Chunk(text=query, title=f"Query_{i}", data_type="query")
            for i, query in enumerate(queries)
        ]
        query_embeddings = self.embedder(query_chunks)
        query_vectors = [embedding.vector for embedding in query_embeddings]
        results = self.vector_store.search(query_vectors, top_k)

        return results

    @staticmethod
    def _create_extractors(config: SystemConfig) -> Dict[str, TextExtract]:
        """
        Creates a dictionary of extractors for different data types.
        Each of the supported data types receives its own extractor.
        Extractors are set using the config.

        Args:
            config (SystemConfig): Configuration object containing settings

        Returns:
            Dict[str, TextExtract]: A dictionary mapping data types to their respective extractors
        """
        extractors = {}
        if config.pdf_extraction_method == "pypdf2":
            extractors["pdf"] = PyPDF2Extract()
        return extractors

    @staticmethod
    def _create_chunker(config: SystemConfig) -> Chunker:
        """
        Creates a chunker based on the configured chunking method.

        Args:
            config (SystemConfig): Configuration object containing settings

        Returns:
            Chunker: An instance of the appropriate Chunker subclass

        Raises:
            ValueError: If the configured chunking method is not supported
        """
        if config.chunking_method == "split_sentences":
            return SplitSentencesChunker()
        else:
            raise ValueError(f"Unsupported chunking method: {config.chunking_method}")

    @staticmethod
    def _create_embedder(config: SystemConfig) -> Embedder:
        """
        Creates an embedder based on the configured embedding model.

        Args:
            config (SystemConfig): Configuration object containing settings.

        Returns:
            Embedder: An instance of the appropriate Embedder subclass

        Raises:
            ValueError: If the configured embedding method is not supported
        """
        if config.embedding_method == "huggingface-sentence-transformer":
            return HuggingFaceSentenceTransformerEmbedder(
                model_name=config.embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding method: {config.embedding_method}")

    @staticmethod
    def _create_vector_store(config: SystemConfig) -> VectorDB:
        """
        Creates a vector store based on the configured vector store.

        Args:
            config (SystemConfig): Configuration object containing settings

        Returns:
            VectorDB: An instance of the appropriate VectorDB subclass

        Raises:
            ValueError: If the configured vector store is not supported
        """
        if config.vector_db == "local-chroma":
            return ChromaDB(collection_name=config.vector_db_name)
        else:
            raise ValueError(f"Unsupported vector store: {config.vector_db}")
