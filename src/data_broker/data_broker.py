from typing import List
from ingestion.chunking import CustomTextSplitter, Chunker
from ingestion.extraction import PDFExtract, Extractor
from ingestion.embedding import Embedder
from ingestion.vector_db import VectorDB
from ingestion.raw_data import RawData
from orchestrator.config import SystemConfig


class DataBroker:
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.

    A config is expected to set the embedding model, vector store,
    chunking method etc.

    As a user of this class you can:
    1- Test the vectorstore connection.
    2- Given a text source, embed and insert into a vector store.
    3- Given a query, retrieve a list of relevant documents.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Instantiates an object of this class.

        :param config: Configuration object containing settings for
            embedding model, vector store, chunking method, etc.
        """
        self.config = config

    def insert(self, data: RawData) -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        :param data: The raw data to be processed and inserted
        :raises ValueError: If any step in the process fails due to unsupported
            data types or methods
        """
        extractor = self._create_extractor()
        chunker = self._create_chunker()
        embedder = self._create_embedder()
        vector_store = self._create_vector_store()

        text = extractor(data)
        chunks = chunker(text)
        vectors = embedder(chunks)
        vector_store.insert(vectors)

    def _create_extractor(self, data_type: str) -> Extractor:
        """
        Creates an extractor based on the given data type.

        :param data_type: The type of data to extract
        :return: A function that extracts text from the given data
        :raises ValueError: If the data type is not recognized or supported
        """
        # Implement based on config
        raise NotImplementedError("Embedding creation not yet implemented")

    def _create_chunker(self) -> Chunker:
        """
        Creates a chunker based on the configured chunking method.

        :return: A function that splits the input text into chunks
        :raises ValueError: If the configured chunking method is not supported
        """
        # Implement based on config
        raise NotImplementedError("Embedding creation not yet implemented")

    def _create_embedder(self) -> Embedder:
        """
        Creates an embedder based on the configured embedding model.

        :return: A function that embeds the given chunks
        :raises NotImplementedError: If the embedding creation is not yet implemented
        """
        # Implement based on config
        raise NotImplementedError("Embedding creation not yet implemented")

    def _create_vector_store(self) -> VectorDB:
        """
        Creates a vector store based on the configured vector store.

        :return: A function that inserts vectors into the vector store
        :raises NotImplementedError: If the vector store creation is not yet implemented
        """
        # Implement based on config
        raise NotImplementedError("Vector store creation not yet implemented")
