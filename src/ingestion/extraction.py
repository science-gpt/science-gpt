import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import PyPDF2

from .raw_data import RAW_DATA_TYPES, Data


@dataclass
class PDFData(Data):
    """
    Represents a raw .pdf data source.

    Attributes:
        name (str): The name of the PDF file.
        data_type (RAW_DATA_TYPES): The type of the data source.
        filepath (pathlib.Path): The path to the PDF file.
    """

    filepath: pathlib.Path

    def __post_init__(self):
        super().__init__(name=self.name, data_type="pdf")


@dataclass
class Text(Data):
    """
    Represents extracted text from a data source.

    Attributes:
        name (str): The name of the text.
        data_type (RAW_DATA_TYPES): The type of the raw data source.
        text (str): The extracted text content.
    """

    text: str

    def __post_init__(self):
        super().__init__(name=self.name, data_type=self.data_type)


class TextExtract(ABC):
    """
    Abstract base class for text extraction from various data sources.

    This class defines the interface for text extraction classes and
    provides a common initialization method.
    """

    def __init__(self, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates a TextExtract object.

        Args:
            data_type (RAW_DATA_TYPES): Type of raw data.
        """
        self.data_type = data_type

    @abstractmethod
    def __call__(self, data: Data) -> Text:
        """
        Abstract method to extract text from the given raw data.

        Args:
            data (Data): The raw data to extract text from.

        Returns:
            Text: A Text object containing the extracted text.
        """
        pass


class PyPDF2Extract(TextExtract):
    """
    Concrete implementation of TextExtract for PDF data sources using PyPDF2.

    This class provides functionality to extract text from PDF files using the PyPDF2 library.
    """

    def __init__(self) -> None:
        """
        Instantiates a PyPDF2Extract object.
        """
        super().__init__(data_type="pdf")

    def __call__(self, data: PDFData) -> Text:
        """
        Extracts text from the given PDF data using PyPDF2.

        Args:
            data (PDFData): The PDF data to extract text from.

        Returns:
            Text: A Text object containing the extracted text from the PDF.

        Raises:
            IOError: If there's an error reading the PDF file.
            ValueError: If there's an error extracting text from the PDF.
        """
        # TODO: validate that data is a PDFData object
        try:
            with open(data.filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                # TODO: validate that text is not empty
                return Text(text=text.strip(), name=data.name, data_type="pdf")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")
