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
from ingestion.embedding import (
    BGEM3Embedder,
    Embedder,
    HuggingFaceEmbedder,
    OllamaEmbedder,
)
from ingestion.extraction import (
    ContentExtractor,
    DoclingPDFExtract,
    PDFData,
    PyPDF2Extract,
)
from ingestion.raw_data import Data
from ingestion.vectordb import ChromaDB, MilvusDB, SearchResult, VectorDB
from ingestion.reranker import Reranker
from orchestrator.utils import SingletonMeta
from tqdm import tqdm

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
            embedder = OllamaEmbedder(
                model_name=embedding_model, endpoint=macbook_endpoint
            )
            try:
                embedder.test_connection()
            except RuntimeError:
                logger.error(
                    "Failed to connect to the Ollama model. Defaulting to HuggingFace embeddings."
                )
                embedder = HuggingFaceEmbedder(model_name=HFACE_MODELS[0])
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
            vectorstore = {
                "base": ChromaDB(collection_name=self.collection_name["base"]),
                "user": ChromaDB(collection_name=self.collection_name["user"]),
            }
        elif self._database_config.vector_store.database == "milvus":
            vectorstore = {
                "base": MilvusDB(
                    collection_name=self.collection_name["base"],
                    dense_dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                ),
                "user": MilvusDB(
                    collection_name=self.collection_name["user"],
                    dense_dim=embedding_dimension,
                    host=self._database_config.vector_store.host,
                    port=self._database_config.vector_store.port,
                ),
            }
        else:
            raise ValueError(
                f"Unsupported vector store type: {self._database_config.vector_store.database}"
            )

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

    def _init_databroker_cache(self, collection="base"):
        chunks = self.vectorstore[collection].get_all_ids()
        collection_name = self.collection_name[collection]
        for chunk in tqdm(chunks):
            file = chunk.split(" - Chunk ")[0]
            if file not in self.data_cache[collection][collection_name]:
                self.data_cache[collection][collection_name][file] = []
            self.data_cache[collection][collection_name][file].append(chunk)

    def _create_reranker(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> Reranker:
        """
        Creates a reranker based on the specified model name.

        Args:
            model_name (str): Name of the reranker model to use

        Returns:
            Reranker: An instance of the Reranker class
        """
        return Reranker(model_name=model_name)

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
            "base": "{}_{}".format(self._database_config.vector_store.database, suffix),
            "user": "{}_{}".format(strip(self._database_config.username), suffix),
        }

        if self.collection_name["base"] not in self.data_cache["base"]:
            self.data_cache["base"][self.collection_name["base"]] = {}

        if self.collection_name["user"] not in self.data_cache["user"]:
            self.data_cache["user"][self.collection_name["user"]] = {}

        self.embedder = self._create_embedder()
        self.chunker = self._create_chunker()
        self.extractors = self._create_extractors()
        self.vectorstore = self._create_vectorstore(
            embedding_dimension=self.embedder.embedding_dimension
        )
        self.reranker = self._create_reranker()
        self.current_reranker_model = "BAAI/bge-reranker-v2-m3"

        self._init_databroker_cache(collection="base")
        self._init_databroker_cache(collection="user")

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

        for pdf_file in tqdm(new_files):
            pdf = PDFData(
                filepath=os.path.join(data_root, pdf_file),
                name=pdf_file,
                data_type="pdf",
            )
            try:
                logger.info("Inserting", pdf)
                self.insert(pdf, collection=collection)
            except IOError as e:
                logger.error(f"Failed to insert {pdf.name} into the vector store: {e}")

    def _ingest_and_prune_data(self, collection="user"):

        data_root = self.data_roots[collection]
        collection_name = self.collection_name[collection]

        pdf_files = [file for file in os.listdir(data_root) if file.endswith(".pdf")]
        existing_files = list(self.data_cache[collection][collection_name].keys())
        remove_files = list(set(existing_files) - set(pdf_files))

        del_chunks = []
        for pdf_file in tqdm(remove_files):
            del_chunks.extend(self.data_cache[collection][pdf_file])
            self.data_cache[collection][collection_name].pop(pdf_file)

        if del_chunks:
            self.vectorstore[collection].delete(ids=del_chunks)

    def insert(self, data: Data, collection="base") -> List[str]:
        """
        Process and insert the given raw data into the vector store.
        Supports both standard embeddings and BGEM3 hybrid embeddings.
        """
        collection_name = self.collection_name[collection]
        extractor = self.extractors.get(data.data_type)
        extracted_content = extractor(data)

        chunks = self.chunker(extracted_content)
        existing_ids = {id: "" for id in self.vectorstore[collection].get_all_ids()}

        new_chunks = []
        metadatum = []
        for chunk in chunks:
            if chunk.name not in existing_ids:
                new_chunks.append(chunk)
                metadatum.append({"source": data.name, "id": chunk.name})

        self.data_cache[collection][collection_name][data.name] = chunks

        if len(new_chunks) > 0:
            embeddings = self.embedder(new_chunks)
            self.vectorstore[collection].insert(embeddings, metadatum)
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
        top_k: int = 2,
        collection="base",
        hybrid_weighting: float = 0.5,
        keywords: Optional[list[str]] = None,
        filenames: Optional[list[str]] = None,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ) -> List[List[SearchResult]]:
        """
        Searches the vector store for the most relevant docs based on the given queries.

        Args:
            queries (List[str]): List of search queries
            top_k (int): The number of results to return for each query
            collection (str, optional): Which collection to search for. Defaults to "base".
            hybrid_weighting (float, optional): Weight between dense and sparse search. Defaults to 0.5.
            keywords (List[str], optional): List of keywords to search for. Defaults to None.
            filenames (List[str], optional): List of filenames to search for. Defaults to None.
            reranker_model (str, optional): Name of the reranker model to use. Defaults to "BAAI/bge-reranker-v2-m3".

        Returns:
            List[List[SearchResult]]: A list of lists of SearchResult objects containing
                the search results for each query, sorted by relevance
        """
        query_chunks = [
            Chunk(text=query, name=f"Query_{i}", data_type="query")
            for i, query in enumerate(queries)
        ]
        query_embeddings = self.embedder(query_chunks)

        raw_results = self.vectorstore[collection].search(
            query_embeddings,
            top_k + 15,  # Get more results than needed for reranking
            keywords,
            filenames,
            hybrid_weighting,
        )

        if reranker_model != self.current_reranker_model:
            self.reranker = self._create_reranker(model_name=reranker_model)
            self.current_reranker_model = reranker_model
            print("Current reranker model: ", self.current_reranker_model)

        reranked_results = []

        for query, result_list in zip(queries, raw_results):
            if not result_list:
                reranked_results.append([])
                continue

            reranked_items = self.reranker.rerank(
                query=query, results=result_list, top_k=min(top_k, len(result_list))
            )

            reranked_results.append(reranked_items)

        return reranked_results
