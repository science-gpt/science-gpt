from ingestion.chunking import CustomTextSplitter
from ingestion.extraction import PDFExtract, Text
from ingestion.raw_data import RawData
from typing import List
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

    def _extract(self, data: RawData) -> Text:
        """
        Extract text from the given raw data.

        :param data: The raw data object to extract text from
        :return: The extracted text
        :raises ValueError: If the data type is not recognized or supported
        """
        if data.data_type == "pdf":
            extractor = PDFExtract()
        else:
            # TODO handle case when datatype is unrecognized
            raise ValueError

        return extractor(data=data)

    def _chunk(self, text: Text) -> List[Chunk]:
        """
        Split the input text into chunks using the configured chunking method.

        :param text: The input text to be chunked
        :return: A list of text chunks
        :raises ValueError: If the configured chunking method is not supported
        """
        if self.config.chunking_method == "custom-text-splitter":
            chunker = CustomTextSplitter()
        else:
            # TODO unsupported chunking method
            raise ValueError

        return chunker(text=text)
    
    def _embed(self, chunks: List[Chunk]):
        """
        Embed the given chunks using the configured embedding model.

        :param chunks: A list of text chunks to be embedded
        :return: A list of embedded vectors
        """
        raise NotImplementedError("Embedding method not yet implemented")
        

    def insert(self, data: RawData) -> None:
        """
        Process and insert the given raw data into the vector store.

        This method orchestrates the extraction, chunking, embedding, and storage
        of the input data.

        :param data: The raw data to be processed and inserted
        :raises ValueError: If any step in the process fails due to unsupported
            data types or methods
        """
        text = self._extract(data=data)
        chunks = self._chunk(text=text)
        vectors = self._embed(chunks=chunks)
        

        # embed
        # insert into vector store
        # return confirmation