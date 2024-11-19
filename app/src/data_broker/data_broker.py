import logging
import os
import string
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
from orchestrator.config import SystemConfig
from orchestrator.utils import load_config

logger = logging.getLogger(__name__)


def squish(s):
    return s.translate(str.maketrans("", "", string.punctuation))


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
            self.data_cache = {
                "base": {},
                "user": {},
            }

    def get_embedding_model(self):
        """Returns the currently set embedding model."""
        return self.embedding_model

    def load_database_config(self, database_config):
        self.embedding_model = database_config.embedding_model
        print(self.embedding_model)

        secrets = toml.load("secrets.toml")

        ollama_models = ["mxbai-embed-large", "nomic-embed-text"]
        hface_models = ["all-mpnet-base-v2"]
        macbook_endpoint = secrets["localmodel"]["macbook_endpoint"]
        if self.embedding_model in ollama_models:
            self.embedder = OllamaEmbedder(
                model_name=self.embedding_model, endpoint=macbook_endpoint
            )

            try:
                self.embedder.test_connection()
            except RuntimeError as e:
                logger.error(
                    "Failed to connect to the Ollama model. Defaulting to HuggingFace embeddings."
                )
                self.embedder = HuggingFaceEmbedder(model_name="all-mpnet-base-v2")
                self.embedding_model = "all-mpnet-base-v2"

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
            suffix = "_" + squish(self.embedding_model)
            suffix += "_" + squish(database_config.chunking_method)

            self.data_root = {
                "base": f"{os.getcwd()}/data/",
                "user": database_config.userpath,
            }

            self.collection_name = {
                "base": database_config.vector_store.instance_name + suffix,
                "user": database_config.username + suffix,
            }

            self.vector_store = {
                "base": ChromaDB(collection_name=self.collection_name["base"]),
                "user": ChromaDB(collection_name=self.collection_name["user"]),
            }

            self.data_cache["base"][self.collection_name["base"]] = {}
            self.data_cache["user"][self.collection_name["user"]] = {}

            print("---")
            print(self.collection_name)
            print("---")

        self.ingest_and_process_data(collection="base")
        self.ingest_and_process_data(collection="user")
        self.ingest_and_prune_data(collection="user")

    def ingest_and_process_data(self, collection="base"):
        """
        Orchestrates the ingestion, chunking, embedding, and storing of data.
        """
        data_root = self.data_root[collection]
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
        # Check if the collection exists, and if not, recreate it
        try:
            self.vector_store[collection].collection = self.vector_store[
                collection
            ].client.get_or_create_collection(name=self.collection_name[collection])
        except Exception as e:
            logger.error(f"Failed to get or create collection: {e}")
            return

        # TODO better logging and error handling
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
        collection_name = self.collection_name[
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
