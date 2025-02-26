import logging
import os
import string
from types import SimpleNamespace
from typing import Dict, List, Optional

import toml
from ingestion.chunking import (
    Chunk,
    Chunker,
    DoclingHierarchicalChunker,
    DoclingHybridChunker,
    RecursiveCharacterChunker,
    SplitSentencesChunker,
)
from ingestion.embedding import Embedder, HuggingFaceEmbedder, OllamaEmbedder, BGEM3Embedder
from ingestion.extraction import (
    ContentExtractor,
    DoclingPDFExtract,
    PDFData,
    PyPDF2Extract,
)
from ingestion.raw_data import Data
from ingestion.vectordb import ChromaDB, MilvusDB, SearchResult, VectorDB, MilvusBGE_DB
from orchestrator.utils import SingletonMeta

# Carter: I've left the logger as python's default for now because
# there are many log statements in the databroker that we might not want to
# write to azure. We can change this later when we're ready.
logger = logging.getLogger(__name__)


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
        **embedder_kwargs,
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
        OLLAMA_MODELS = ["mxbai-embed-large", "nomic-embed-text", "bge-m3:567m"]
        HFACE_MODELS = ["sentence-transformers/all-mpnet-base-v2"]
        BGEM3_MODELS = ["BAAI/bge-m3"]

        embedding_model = self._database_config.embedding_model
        print("Using embedding model: ", embedding_model)
        if embedding_model in OLLAMA_MODELS:
            macbook_endpoint = self._secrets["localmodel"]["macbook_endpoint"]
            embedder = OllamaEmbedder(model_name=embedding_model, endpoint=macbook_endpoint)
            try:
                embedder.test_connection()
            except RuntimeError:
                logger.error("Failed to connect to the Ollama model. Defaulting to HuggingFace embeddings.")
                embedder = HuggingFaceEmbedder(model_name=embedding_model)
        elif embedding_model in HFACE_MODELS:
            print("Using HuggingFaceEmbedder")
            embedder = HuggingFaceEmbedder(model_name=embedding_model)
        elif embedding_model in BGEM3_MODELS:
            print("Using BGEM3Embedder")

            embedder = BGEM3Embedder()
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_model}")

        return embedder


    def _create_chunker(self) -> Chunker:
        """
        Creates a chunker based on the configured chunking method.
        Returns:
            Chunker: An instance of the appropriate Chunker subclass
        Raises:
            ValueError: If the configured chunking method is not supported
        """
        if self._database_config.chunking_method == "docling_hybrid":
            chunker = DoclingHybridChunker()
        elif self._database_config.chunking_method == "docling_hierarchical":
            chunker = DoclingHierarchicalChunker()
        elif self._database_config.chunking_method == "split_sentences":
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

    def _create_extractors(self) -> Dict[str, ContentExtractor]:
        """
        Creates a dictionary of extractors for different data types.
        Each of the supported data types receives its own extractor.
        Extractors are set using the config.
        Returns:
            Dict[str, ContentExtractor]: A dictionary mapping data types to their respective extractors
        """
        extractors = {}
        if self._database_config.pdf_extractor.extraction_method == "pypdf2":
            extractors["pdf"] = PyPDF2Extract()
        elif self._database_config.pdf_extractor.extraction_method == "docling":
            extractors["pdf"] = DoclingPDFExtract()
        return extractors

    def _create_vectorstore(self, embedding_dimension: int) -> Dict[str, VectorDB]:
        if self._database_config.vector_store.database == "chromadb":
            print("Creating ChromaDB vector store")
            vectorstore = {
                "base": ChromaDB(collection_name=self.collection_name["base"]),
                "user": ChromaDB(collection_name=self.collection_name["user"]),
            }
        elif self._database_config.vector_store.database == "milvus":
            print("Creating Milvus vector store")
            vectorstore = {
                "base": MilvusDB(
                    collection_name=self.collection_name["base"],
                    dense_dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                    dense_embedder=self.embedder,
                    if_hybrid_search=True, # TODO: make this configurable
                    use_reranker=True,

                ),
                "user": MilvusDB(
                    collection_name=self.collection_name["user"],
                    dense_dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                    dense_embedder=self.embedder,
                    if_hybrid_search=True, # TODO: make this configurable
                    use_reranker=True,

                ),
            }
        else:
            raise ValueError(f"Unsupported vector store type: {self._database_config.vector_store.database}")
        
        return vectorstore


    def _validate_extractor_chunker_compatibility(self):
        """
        Validates that the configured extractor and chunker are compatible.
        Raises:
            ValueError: If the combination is not supported
        """
        extraction_method = self._database_config.pdf_extractor.extraction_method
        chunking_method = self._database_config.chunking_method

        if (
            chunking_method in ["docling_hierarchical", "docling_hybrid"]
            and extraction_method != "docling"
        ):
            raise ValueError(
                """
                Docling chunking requires Docling extraction.
                Please set the extraction method to 'docling,' or use a non-docling chunker.
                """
            )

        if extraction_method == "docling" and chunking_method not in [
            "docling_hierarchical",
            "docling_hybrid",
        ]:
            logger.warning(
                """
                Using DoclingExtract with a basic chunker. 
                This will lose document structure information.
                Consider using DoclingChunker to preserve document structure.
                """
            )

    def _init_databroker_pipeline(self, database_config: SimpleNamespace) -> None:
        """
        Initializes the data broker pipeline.
        """
        logger.info("Initializing data broker pipeline")
        self._database_config = database_config
        if self._database_config is None:
            raise ValueError("Database configuration is not set")

        self._validate_extractor_chunker_compatibility()

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
        Supports both standard embeddings and BGEM3 hybrid embeddings.
        """
        extractor = self.extractors.get(data.data_type)
        extracted_content = extractor(data)

        chunks = self.chunker(extracted_content)
        print("chunks: ", chunks)
        existing_ids = self.vectorstore[collection].get_all_ids()

        new_chunks = []
        metadatum = []
        for chunk in chunks:
            if chunk.name not in existing_ids:
                new_chunks.append(chunk)
                metadatum.append({"source": data.name, "id": chunk.name})

        if not new_chunks:
            print("No new documents to add")
            return []

        # # Handle BGEM3 Hybrid Embeddings
        # if self._database_config.vector_store.database == "milvus_bge":
        #     embedding = self.embedder(new_chunks)
        #     print("embedding: ", embedding)

        #     # Insert using MilvusBGE_DB
        #     try:
        #         self.vectorstore[collection].insert(embedding, metadatum)
        #         print(f"Successfully inserted {len(new_chunks)} hybrid embeddings into MilvusBGE.")
        #     except Exception as e:
        #         logger.error(f"Failed to insert into MilvusBGE_DB: {e}")


        # else:
        #     embeddings = self.embedder(new_chunks)  # Standard embedding call
        #     try:
        #         self.vectorstore[collection].insert(embeddings, metadatum)
        #         print(f"Successfully inserted {len(new_chunks)} standard embeddings.")
        #     except Exception as e:
        #         logger.error(f"Failed to insert into vector store: {e}")
        #         return []
        # for i, chunk in enumerate(new_chunks):
        #     if not hasattr(chunk, "name"):
        #         print("Chunk at index %d is not a proper Chunk object: %s", i, type(chunk))
        #     else:
        #         print("Chunk %d has name: %s", i, chunk.name)

        try:
            embedding = self.embedder(new_chunks)
            print("This is true")
            self.vectorstore[collection].insert(embedding, metadatum)
            print("This is true2")
            print(f"Successfully inserted {len(new_chunks)} standard embeddings.")
        except Exception as e:
            print(f"Failed to insert into vector store: {e}")
            return []

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
        top_k: int = 2,
        collection="base",
        keywords: Optional[list[str]] = None,
        filenames: Optional[list[str]] = None,
    ) -> List[List[SearchResult]]:
        """
        Searches the vector store for the most relevant docs based on the given queries.

        Args:
            queries (List[str]): List of search queries
            top_k (int): The number of results to return for each query
            collection (str, optional): Which collection to search for. Defaults to "base".
            keywords (List[str], optional): List of keywords to search for. Defaults to None.
            filenames (List[str], optional): List of filenames to search for. Defaults to None.

        Returns:
            List[List[SearchResult]]: A list of lists of SearchResult objects containing
                the search results for each query, sorted by relevance
        """
        results = self.vectorstore[collection].search(queries, top_k, keywords, filenames)
        return results
    
        # # Handle MilvusBGE hybrid search
        # if self._database_config.vector_store.database == "milvus_bge":
        #     print("Using hybrid search in MilvusBGE_DB")
        #     results = self.vectorstore[collection].search(queries, top_k, keywords, filenames)

        # else:
        #     query_chunks = [
        #         Chunk(text=query, name=f"Query_{i}", data_type="query")
        #         for i, query in enumerate(queries)
        #     ]
        #     query_embeddings = self.embedder(query_chunks)
        #     query_vectors = [embedding.vector for embedding in query_embeddings]
        #     results = self.vectorstore[collection].search(query_vectors, top_k, keywords, filenames)

        # return results

