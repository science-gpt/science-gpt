import logging
import os
import string
from types import SimpleNamespace
from typing import Dict, List, Optional

import toml
from ingestion.chunking import (
    Chunk,
    Chunker,
    RecursiveCharacterChunker,
    SplitSentencesChunker,
)
from ingestion.embedding import Embedder, HuggingFaceEmbedder, OllamaEmbedder
from ingestion.extraction import PDFData, PyPDF2Extract, TextExtract
from ingestion.raw_data import Data
from ingestion.vectordb import ChromaDB, MilvusDB, SearchResult, VectorDB
from orchestrator.utils import SingletonMeta

logger = logging.getLogger(__name__)  # using custom logger causes circular dependency


class DataBroker(metaclass=SingletonMeta):
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.
    """

    def __init__(
        self,
        database_config: SimpleNamespace = None,
        secrets_path: str = "secrets.toml",
    ) -> None:
        """
        Instantiates an object of this class.
        """
        self._database_config = database_config
        self._secrets = toml.load(secrets_path)
        if database_config is not None:
            self.data_cache = {
                "base": {},
                "user": {},
            }
            self._init_databroker_pipeline(database_config)

    def get_database_config(self) -> SimpleNamespace:
        """
        Returns the database configuration.
        """
        return self._database_config

    def get_embedding_model(self) -> str:
        """
        Returns the currently set embedding model.
        """
        return self._database_config.embedding_model

    def _create_embedder(self) -> Embedder:
        """
        Creates an embedder based on the configured embedding model.
        Returns:
            Embedder: An instance of the appropriate Embedder subclass
        Raises:
            ValueError: If the configured embedding method is not supported
        """
        OLLAMA_MODELS = ["mxbai-embed-large", "nomic-embed-text"]
        HFACE_MODELS = ["all-mpnet-base-v2"]

        embedding_model = self._database_config.embedding_model
        if embedding_model in OLLAMA_MODELS:
            macbook_endpoint = self._secrets["localmodel"]["macbook_endpoint"]
            embedder = OllamaEmbedder(
                model_name=embedding_model, endpoint=macbook_endpoint
            )
            try:
                embedder.test_connection()
            except RuntimeError:
                logger.error(
                    "Failed to connect to the Ollama model. Defaulting to HuggingFace embeddings."
                )
                embedder = HuggingFaceEmbedder(model_name="all-mpnet-base-v2")
        elif embedding_model in HFACE_MODELS:
            embedder = HuggingFaceEmbedder(model_name=embedding_model)
        else:
            raise ValueError(
                f"Unsupported embedding method: {self._database_config.embedding_model}"
            )

        return embedder

    def _create_chunker(self) -> Chunker:
        """
        Creates a chunker based on the configured chunking method.
        Returns:
            Chunker: An instance of the appropriate Chunker subclass
        Raises:
            ValueError: If the configured chunking method is not supported
        """
        if self._database_config.chunking_method == "split_sentences":
            chunker = SplitSentencesChunker()
        elif self._database_config.chunking_method == "recursive_character":
            chunker = RecursiveCharacterChunker(
                chunk_size=1500,
                chunk_overlap=250,
            )
        elif (
            self._database_config.chunking_method == "recursive_character:large_chunks"
        ):
            chunker = RecursiveCharacterChunker(
                chunk_size=3000,
                chunk_overlap=500,
            )
        elif (
            self._database_config.chunking_method == "recursive_character:small_chunks"
        ):
            chunker = RecursiveCharacterChunker(
                chunk_size=750,
                chunk_overlap=250,
            )
        else:
            raise ValueError(
                f"Unsupported chunking method: {self._database_config.chunking_method}"
            )
        return chunker

    def _create_extractors(self) -> Dict[str, TextExtract]:
        """
        Creates a dictionary of extractors for different data types.
        Each of the supported data types receives its own extractor.
        Extractors are set using the config.
        Returns:
            Dict[str, TextExtract]: A dictionary mapping data types to their respective extractors
        """
        extractors = {}
        if self._database_config.pdf_extractor.extraction_method == "pypdf2":
            extractors["pdf"] = PyPDF2Extract()
        return extractors

    def _create_vectorstore(self, embedding_dimension: int) -> Dict[str, VectorDB]:
        if self._database_config.vector_store.database == "chromadb":
            vectorstore = {
                "base": ChromaDB(collection_name=self.collection_name["base"]),
                "user": ChromaDB(collection_name=self.collection_name["user"]),
            }
        elif self._database_config.vector_store.database == "milvus":
            vectorstore = {
                "base": MilvusDB(
                    collection_name=self.collection_name["base"],
                    dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                ),
                "user": MilvusDB(
                    collection_name=self.collection_name["user"],
                    dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                ),
            }
        else:
            raise ValueError(
                f"Unsupported chunking method: {self._database_config.chunking_method}"
            )
        return vectorstore

    def _init_databroker_pipeline(self, database_config: SimpleNamespace) -> None:
        """
        Initializes the data broker pipeline.
        """
        logger.info("Initializing data broker pipeline")
        self._database_config = database_config
        if self._database_config is None:
            raise ValueError("Database configuration is not set")

        self.data_roots = {
            "base": f"{os.getcwd()}/data/",
            "user": self._database_config.userpath,
        }

        def strip(s):
            """
            Remove punctuation from a string
            """
            return s.translate(str.maketrans("", "", string.punctuation))

        suffix = f"_{strip(self._database_config.embedding_model)}_{strip(self._database_config.chunking_method)}"

        self.collection_name = {
            "base": self._database_config.vector_store.database + suffix,
            "user": self._database_config.username + suffix,
        }

        self.data_cache["base"][self.collection_name["base"]] = {}
        self.data_cache["user"][self.collection_name["user"]] = {}

        self.embedder = self._create_embedder()
        self.chunker = self._create_chunker()
        self.extractors = self._create_extractors()
        self.vectorstore = self._create_vectorstore(
            embedding_dimension=self.embedder.embedding_dimension
        )

        self._ingest_root_data(collection="base")
        self._ingest_root_data(collection="user")
        self._ingest_and_prune_data(collection="user")

    def _ingest_root_data(self, collection="base"):
        """
        Orchestrates the ingestion, chunking, embedding, and storing of data.
        """
        data_root = self.data_roots[collection]
        collection_name = self.collection_name[collection]

        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]
        existing_files = list(self.data_cache[collection][collection_name].keys())
        new_files = list(set(pdf_files) - set(existing_files))

        for pdf_file in new_files:
            pdf = PDFData(
                filepath=os.path.join(data_root, pdf_file),
                name=pdf_file,
                data_type="pdf",
            )
            try:
                logger.info("Inserting", pdf)
                chunk_ids = self.insert(pdf, collection=collection)
                self.data_cache[collection][collection_name][pdf_file] = chunk_ids
            except IOError as e:
                logger.error(f"Failed to insert {pdf.name} into the vector store: {e}")

    def _ingest_and_prune_data(self, collection="user"):

        data_root = self.data_roots[collection]
        collection_name = self.collection_name[collection]

        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]
        existing_files = list(self.data_cache[collection][collection_name].keys())
        remove_files = list(set(existing_files) - set(pdf_files))

        del_chunks = []
        for pdf_file in remove_files:
            del_chunks.extend(self.data_cache[collection][pdf_file])
            self.data_cache[collection][collection_name].pop(pdf_file)

        if del_chunks:
            self.vectorstore[collection].delete(ids=del_chunks)

    def insert(self, data: Data, collection="base") -> List[str]:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        Args:
            data (Data): The raw data to be processed and inserted
            collection (str, optional): Which collection to insert into. Defaults to "base".
        """
        extractor = self.extractors.get(data.data_type)
        text = extractor(data)
        chunks = self.chunker(text)

        existing_ids = self.vectorstore[collection].get_all_ids()

        # more looping over every entry in the db?
        new_chunks = []
        for chunk in chunks:
            if chunk.name not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            embeddings = self.embedder(new_chunks)
            try:
                self.vectorstore[collection].insert(embeddings)
            except Exception as e:
                logger.error(f"Failed to get or create collection: {e}")
                return []
        else:
            print("No new documents to add")

        return [chunk.name for chunk in chunks]

    def clear_db(self, collection="base"):
        """
        Clears all vectors from the vector store.
        """
        logging.info("Clearing the database")
        self.vectorstore[collection].clear()

    def search(
        self,
        queries: List[str],
        top_k: int = 5,
        collection="base",
        keywords: Optional[list[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Searches the vector store for the most relevant docs based on the given queries.

        Args:
            queries (List[str]): List of search queries
            top_k (int): The number of results to return for each query
            collection (str, optional): Which collection to search for. Defaults to "base".
            keywords (List[str], optional): List of keywords to search for. Defaults to None.

        Returns:
            List[List[SearchResult]]: A list of lists of SearchResult objects containing
                the search results for each query, sorted by relevance
        """
        query_chunks = [
            Chunk(text=query, name=f"Query_{i}", data_type="query")
            for i, query in enumerate(queries)
        ]

        query_embeddings = self.embedder(query_chunks)
        query_vectors = [embedding.vector for embedding in query_embeddings]

        results = self.vectorstore[collection].search(query_vectors, top_k, keywords)
        return results
