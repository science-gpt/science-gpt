import logging
import os
import string
from typing import Dict, List

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

    def __init__(self, database_config=None) -> None:
        """
        Instantiates an object of this class.
        """
        if database_config != None:
            print("---INIT---")
            self.load_database_config(database_config)

    def get_embedding_model(self):
        """Returns the currently set embedding model."""
        return self.embedding_model

    def load_database_config(self, database_config):
        self.embedding_model = database_config.embedding_model

        secrets = toml.load("secrets.toml")

        ollama_models = ["mxbai-embed-large:latest", "nomic-embed-text"]
        hface_models = ["all-mpnet-base-v2"]
        macbook_endpoint = secrets["localmodel"]["macbook_endpoint"]
        if self.embedding_model in ollama_models:
            self.embedder = OllamaEmbedder(
                model_name=self.embedding_model, endpoint=macbook_endpoint
            )
        elif self.embedding_model in hface_models:
            self.embedder = HuggingFaceEmbedder(model_name=self.embedding_model)

        if database_config.chunking_method == "split_sentences":
            self.chunker = SplitSentencesChunker()
        elif database_config.chunking_method == "recursive_character":
            self.chunker = RecursiveCharacterChunker(
                chunk_size=1500,
                chunk_overlap=250,
            )
        elif database_config.chunking_method == "recursive_character:large_chunks":
            self.chunker = RecursiveCharacterChunker(
                chunk_size=3000,
                chunk_overlap=500,
            )
        elif database_config.chunking_method == "recursive_character:small_chunks":
            self.chunker = RecursiveCharacterChunker(
                chunk_size=750,
                chunk_overlap=250,
            )

        self.extractors = {}
        if database_config.pdf_extractor.pdf_extract_method == "pypdf2":
            self.extractors["pdf"] = PyPDF2Extract()

        if database_config.vector_store.type == "local-chromadb":
            self.collection_name = database_config.vector_store.instance_name
            self.collection_name += "_" + self.embedding_model.translate(
                str.maketrans("", "", string.punctuation)
            )
            self.collection_name += "_" + database_config.chunking_method.translate(
                str.maketrans("", "", string.punctuation)
            )

            print("---")
            print(self.collection_name)
            print("---")
            self.vector_store = ChromaDB(collection_name=self.collection_name)

        self.ingest_and_process_data()

    def ingest_and_process_data(self):
        """
        Orchestrates the ingestion, chunking, embedding, and storing of data.
        """
        data_root = f"{os.getcwd()}/data/"

        # List all PDF files in the data directory
        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]

        # Process each PDF file
        for pdf_file in pdf_files:
            pdf = PDFData(
                filepath=os.path.join(data_root, pdf_file),
                name=pdf_file,
                data_type="pdf",
            )
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
                    name=self.collection_name
                )
            )
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            return

        # TODO better logging and error handling
        extractor = self.extractors.get(data.data_type)
        text = extractor(data)
        chunks = self.chunker(text)

        existing_items = self.vector_store.collection.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(len(existing_ids))

        # Only add missing chunks
        new_chunks = []
        for chunk in chunks:
            if chunk.name not in existing_ids:
                new_chunks.append(chunk)
        print(len(new_chunks))

        # Choose embedder based on selected embedding model
        if len(new_chunks):
            embeddings = self.embedder(new_chunks)
            try:
                # Attempt to insert embeddings into the vector store
                self.vector_store.insert(embeddings)
            except Exception as e:
                logger.error(f"Failed to get or create collection: {e}")
                return
        else:
            print("✅ No new documents to add")

    ### added clear db fuction here
    def clear_db(self):
        """
        Clears all vectors from the vector store.
        """
        collection_name = self.collection_name  # Retrieve the collection name
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
