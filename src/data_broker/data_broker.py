from ingestion.chunking import CustomTextSplitter
from ingestion.extraction import PDFExtract, Text
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

    def __init__(self, config: SystemConfig):
        # TODO docstring
        self.config = config

    def _extract(self, data: RawData) -> Text:
        # TODO docstring
        if data.data_type == "pdf":
            extractor = PDFExtract()
        else:
            # TODO handle case when datatype is unrecognized
            raise ValueError

        return extractor(data=data)

    def _chunk(self, text: Text):
        # TODO docstring
        if self.config.chunking_method == "custom-text-splitter":
            chunker = CustomTextSplitter()
        else:
            # TODO unsupported chunking method
            raise ValueError

        return chunker(text=text)

    def insert(self, data: RawData):
        # TODO docstring

        text = self._extract(data=data)
        chunks = self._chunk(text=text)

        # embed
        # insert into vector store
        pass
