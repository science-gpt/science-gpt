from abc import ABC, abstractmethod
from typing import List

import PyPDF2

from .raw_data import RAW_DATA_TYPES, PDFData, RawData
from .utils import OutputObject


class Text(OutputObject):
    """
    Represents extracted text from a data source.

    This class inherits from OutputObject and adds a text attribute
    to store the extracted content.
    """

    def __init__(self, text: str, title: str, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates an object of this class.

        :param text: the extracted text
        :param title: a title for the extracted text
        :param data_type: the type of the original data source
        """
        super().__init__(title=title, data_type=data_type)
        self.text = text

    def __str__(self) -> str:
        """
        Returns a string representation of the Text.
        """
        return f"""
        Text(
            title='{self.title}',
            text='{self.text[:50]}...',
            data_type={self.data_type}
        )
        """

    __repr__ = __str__


class TextExtract(ABC):
    """
    Abstract base class for text extraction from various data sources.

    This class defines the interface for text extraction classes and
    provides a common initialization method.
    """

    def __init__(self, data_type: RAW_DATA_TYPES) -> None:
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

    def __init__(self) -> None:
        """
        Instantiates a PDFExtract object.
        """
        super().__init__(data_type="pdf")

    def __call__(self, data: PDFData) -> Text:
        """
        Extracts text from the given PDF data.

        :param data: The PDF data to extract text from
        :return: A Text object containing the extracted text from the PDF
        :raises IOError: If there's an error reading the PDF file
        """
        # TODO: validate that data is a PDFData object
        try:
            with open(data.filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                # TODO: validate that text is not empty
                return Text(text=text.strip(), title=data.name, data_type="pdf")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")
