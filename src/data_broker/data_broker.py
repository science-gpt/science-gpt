import logging
import os
from typing import Dict, List

from ingestion.chunking import Chunk, Chunker, SplitSentencesChunker
from ingestion.embedding import Embedder, HuggingFaceEmbedder, OllamaEmbedder
from ingestion.extraction import PDFData, PyPDF2Extract, TextExtract
from ingestion.raw_data import Data
from ingestion.vectordb import ChromaDB, SearchResult, VectorDB
from orchestrator.config import SystemConfig
from orchestrator.utils import load_config

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# TODO: error handling throughout this class is absent or inconsistent
class DataBroker(metaclass=SingletonMeta):
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.
    """

    def __init__(self) -> None:
        """
        Instantiates an object of this class.
        """

        self.embedder = None
        self.config: SystemConfig = load_config(
            config_name="system_config", config_dir=f"{os.getcwd()}/src/configs"
        )
        self.extractors = self._create_extractors(self.config)
        self.chunker = self._create_chunker(self.config)
        self.embedder = self._create_embedder(self.config)
        self.vector_store = self._create_vector_store(self.config)

        self.files: SystemConfig = load_config(
            config_name="data_config", config_dir=f"{os.getcwd()}/src/configs"
        )
        data_root = f"{os.getcwd()}/data/"
        # List all PDF files in the data directory
        pdf_files = [
            file for file in os.listdir(data_root) if file.endswith('.pdf')
        ]

        # Process each PDF file
        for pdf_file in pdf_files:
            pdf = PDFData(filepath=os.path.join(data_root, pdf_file), name=pdf_file, data_type="pdf")
            try:
                self.insert(pdf)
            except IOError as e:
                logger.error(f"Failed to insert {pdf.name} into the vector store: {e}")

    def get_embedding_model(self):
        """Returns the currently set embedding model."""
        return self.embedding_model

    def load_embedding_model(self, model_name: str):
        self.embedding_model = model_name  # Update the current embedding model name
        ollama_models = ["mxbai-embed-large:latest"]
        hface_models = ["all-mpnet-base-v2"]
        if model_name in ollama_models:
            self.embedder = OllamaEmbedder(model_name=model_name)
        elif model_name in hface_models:
            self.embedder = HuggingFaceEmbedder(model_name=model_name)

        self.clear_db()
        self.ingest_and_process_data()

    def ingest_and_process_data(self):
        """
        Orchestrates the ingestion, chunking, embedding, and storing of data.
        """
        data_root = f"{os.getcwd()}/data/"
        
        # List all PDF files in the data directory
        pdf_files = [
            file for file in os.listdir(data_root) if file.endswith('.pdf')
        ]

        # Process each PDF file
        for pdf_file in pdf_files:
            pdf = PDFData(filepath=os.path.join(data_root, pdf_file), name=pdf_file, data_type="pdf")
            try:
                print("inserting", pdf)
                self.insert(pdf)
            except IOError as e:
                logger.error(f"Failed to insert {pdf.name} into the vector store: {e}")

    def insert(self, data: Data) -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        Args:
            data (Data): The raw data to be processed and inserted
        """
        # Check if the collection exists, and if not, recreate it
        try:
            self.vector_store.collection = (
                self.vector_store.client.get_or_create_collection(
                    name=self.vector_store.collection.name
                )
            )
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            return
        # TODO better logging and error handling
        extractor = self.extractors.get(data.data_type)
        text = extractor(data)
        chunks = self.chunker(text)

        # Choose embedder based on selected embedding model
        embeddings = self.embedder(chunks)

        try:
            # Attempt to insert embeddings into the vector store
            self.vector_store.insert(embeddings)
        except:
            logger.warning(
                "Embedding dimension mismatch detected. Clearing the DB and regenerating embeddings."
            )
            print("I couldn't insert stuff... :( clearing DB and trying again")
            self.clear_db()  # Clear the database to reset the collection
            # Regenerate embeddings after clearing the database
            self.ingest_and_process_data()  # Call to retry embedding after clearing the DB
            return

    ### added clear db fuction here
    def clear_db(self):
        """
        Clears all vectors from the vector store.
        """
        collection_name = (
            self.vector_store.collection.name
        )  # Retrieve the collection name
        print("I'm clearing the db:", collection_name)
        self.vector_store.client.delete_collection(collection_name)

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
            Chunk(text=query, name=f"Query_{i}", data_type="query")
            for i, query in enumerate(queries)
        ]

        query_embeddings = self.embedder(query_chunks)
        query_vectors = [embedding.vector for embedding in query_embeddings]

        try:
            results = self.vector_store.search(query_vectors, top_k)
        except:
            logger.error(
                "Connect search. probably an issue with the DB not initialized and nothing returned"
            )
            # You could prompt the user to reprocess data or clear and reset the DB here
            print(
                "Could not search, vector DB probably doesn't exist. We should flag this and tell the user the error"
            )
            results = []

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
        extraction_config = config.extraction
        if extraction_config.pdf_extract_method == "pypdf2":
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
        chunking_config = config.chunking
        if chunking_config.method == "split_sentences":
            return SplitSentencesChunker()
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_config.method}")

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
        embedding_config = config.embedding
        if embedding_config.method == "huggingface-sentence-transformer":
            return HuggingFaceEmbedder(model_name=embedding_config.model)
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_config.method}")

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
        vector_db_config = config.vector_db
        if vector_db_config.type == "local-chromadb":
            return ChromaDB(collection_name=vector_db_config.instance_name)
        else:
            raise ValueError(f"Unsupported vector store: {vector_db_config.type}")
