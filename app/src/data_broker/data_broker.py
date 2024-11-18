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
from ingestion.vectordb import ChromaDB, SearchResult, VectorDB

logger = logging.getLogger(__name__)  # using custom logger causes circular dependency


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


class DataBroker(metaclass=SingletonMeta):
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.
    """

    def __init__(self, database_config: SimpleNamespace = None) -> None:
        """
        Instantiates an object of this class.
        """
        self._database_config = database_config
        self.secrets = toml.load("secrets.toml")
        self.init_databroker_pipeline()
        if database_config != None:
            print("---INIT---")
            self.load_database_config(database_config)
            self.data_cache = {
                "base": {},
                "user": {},
            }

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
            macbook_endpoint = self.secrets["localmodel"]["macbook_endpoint"]
            embedder = OllamaEmbedder(
                model_name=embedding_model, endpoint=macbook_endpoint
            )
            try:
                embedder.test_connection()
            except RuntimeError as e:
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
        if self._database_config.pdf_extractor.pdf_extract_method == "pypdf2":
            extractors["pdf"] = PyPDF2Extract()
        return extractors

    def _create_vectorstore(self) -> VectorDB:

        if self._database_config.vector_store.type == "local-chromadb":
            vector_store = {
                "base": ChromaDB(collection_name=self.db_instance_names["base"]),
                "user": ChromaDB(collection_name=self.db_instance_names["user"]),
            }
        elif self._database_config.vector_store.type == "local-milvus":
            vector_store = {}
        else:
            raise ValueError(
                f"Unsupported chunking method: {self._database_config.chunking_method}"
            )
        return vector_store

    def init_databroker_pipeline(self) -> None:
        """
        Initializes the data broker pipeline.
        """
        logger.info("Initializing data broker pipeline")
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

        self.db_instance_names = {
            "base": self._database_config.vector_store.instance_name + suffix,
            "user": self._database_config.username + suffix,
        }

        self.data_cache["base"][self.collection_name["base"]] = {}
        self.data_cache["user"][self.collection_name["user"]] = {}

        self.embedder = self._create_embedder()
        self.chunker = self._create_chunker()
        self.extractors = self._create_extractors()
        self.vector_store = self._create_vectorstore()

        self.ingest_and_process_data(collection="base")
        self.ingest_and_process_data(collection="user")
        self.ingest_and_prune_data(collection="user")

    def ingest_and_process_data(self, collection="base"):
        """
        Orchestrates the ingestion, chunking, embedding, and storing of data.
        """
        data_root = self.data_roots[collection]
        collection_name = self.collection_name[collection]

        # List all PDF files in the data directory
        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]
        existing_files = list(self.data_cache[collection][collection_name].keys())
        new_files = list(set(pdf_files) - set(existing_files))

        # Process each PDF file
        for pdf_file in new_files:
            pdf = PDFData(
                filepath=os.path.join(data_root, pdf_file),
                name=pdf_file,
                data_type="pdf",
            )
            try:
                print("inserting", pdf)
                chunk_ids = self.insert(pdf, collection=collection)
                self.data_cache[collection][collection_name][pdf_file] = chunk_ids
            except IOError as e:
                logger.error(f"Failed to insert {pdf.name} into the vector store: {e}")

    def ingest_and_prune_data(self, collection="user"):

        data_root = self.data_root[collection]
        collection_name = self.collection_name[collection]

        # List all PDF files in the data directory
        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]
        existing_files = list(self.data_cache[collection][collection_name].keys())
        remove_files = list(set(existing_files) - set(pdf_files))

        chunks_ids = []
        for pdf_file in remove_files:
            chunks_ids.extend(self.data_cache[collection][pdf_file])
            self.data_cache[collection][collection_name].pop(pdf_file)

        existing_items = self.vector_store[collection].collection.get(include=[])
        existing_ids = list(set(existing_items["ids"]))

        del_chunks = []
        for id in existing_ids:
            if id not in chunks_ids:
                del_chunks.append(id)

        if len(del_chunks) > 0:
            print("count before", self.vector_store[collection].collection.count())
            self.vector_store[collection].delete(ids=del_chunks)
            print("count after", self.vector_store[collection].collection.count())

    def insert(self, data: Data, collection="base") -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        Args:
            data (Data): The raw data to be processed and inserted
        """
        # why accessing the collection directly? this violates the abstraction
        # now the databroker only works with chroma db, and no other vector stores
        # we should have a method in the vector stores that handles this
        # also why creating the collection here at all, this is the insert method not the init
        # why would there ever be a case where the collection doesn't exist?
        try:
            self.vector_store[collection].collection = self.vector_store[
                collection
            ].client.get_or_create_collection(name=self.db_instance_names[collection])
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            return

        extractor = self.extractors.get(data.data_type)
        text = extractor(data)
        chunks = self.chunker(text)

        existing_items = self.vector_store[collection].collection.get(include=[])
        existing_ids = set(existing_items["ids"])

        # Only add missing chunks
        new_chunks = []
        for chunk in chunks:
            if chunk.name not in existing_ids:
                new_chunks.append(chunk)

        # Choose embedder based on selected embedding model
        if len(new_chunks):
            embeddings = self.embedder(new_chunks)
            try:
                # Attempt to insert embeddings into the vector store
                self.vector_store[collection].insert(embeddings)
            except Exception as e:
                logger.error(f"Failed to get or create collection: {e}")
                return []
        else:
            print("✅ No new documents to add")

        return [chunk.name for chunk in chunks]

    ### added clear db fuction here
    def clear_db(self, collection="base"):
        """
        Clears all vectors from the vector store.
        """
        collection_name = self.db_instance_names[
            collection
        ]  # Retrieve the collection name
        print("I'm clearing the db:", collection_name)
        self.vector_store[collection].client.delete_collection(collection_name)

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
            results = self.vector_store[collection].search(
                query_vectors, top_k, keywords
            )
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
