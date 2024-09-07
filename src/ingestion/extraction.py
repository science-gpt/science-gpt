from abc import ABC, abstractmethod
from chunk import Chunk
from typing import List

from raw_data import RAW_DATA_TYPES, PDFData, RawData
from utils import OutputObject


class Text(OutputObject):
    """
    Represents extracted text from a data source.

    This class inherits from OutputObject and adds a text attribute
    to store the extracted content.
    """

    def __init__(self, text: str, title: str, data_type: RAW_DATA_TYPES):
        """
        Instantiates an object of this class.

        :param text: the extracted text
        :param title: a title for the extracted text
        :param data_type: the type of the original data source
        """
        super().__init__(title=title, data_type=data_type)
        self.text = text


class TextExtract(ABC):
    """
    Abstract base class for text extraction from various data sources.

    This class defines the interface for text extraction classes and
    provides a common initialization method.
    """

    def __init__(self, data_type: RAW_DATA_TYPES):
        """
        Instantiates a TextExtract object.

        :param data_type: type of raw data
        """
        self.data_type = data_type

    @abstractmethod
    def __call__(self, data: RawData) -> Text:
        """
        Abstract method to extract text from the given raw data.

        :param data: The raw data to extract text from
        :return: A Text object containing the extracted text
        """
        pass


class PDFExtract(TextExtract):
    """
    Concrete implementation of TextExtract for PDF data sources.

    This class provides functionality to extract text from PDF files.
    """

    def __init__(self, data_type: RAW_DATA_TYPES):
        """
        Instantiates a PDFExtract object.

        :param data_type: type of raw data, must be "pdf"
        """
        super().__init__(data_type=data_type)
        assert (
            data_type == "pdf"
        )  # TODO there should be a better way to validate this, pydantic?

    def __call__(self, data: PDFData) -> Text:
        """
        Extracts text from the given PDF data.

        :param data: The PDF data to extract text from
        :return: A Text object containing the extracted text from the PDF
        """
        # TODO add pdf extraction code
        pass
