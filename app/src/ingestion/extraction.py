import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pymupdf4llm
import PyPDF2
import fitz
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



class PyMuPDF4LLMExtract(TextExtract):
    """
    Concrete implementation of TextExtract for PDF data sources using PyMuPDF4LLM.

    This class provides functionality to extract Markdown-formatted text from PDF files using pymupdf4llm.
    """

    def __init__(self) -> None:
        """
        Instantiates a PyMuPDF4LLMExtract object.
        """
        super().__init__(data_type="pdf")

    def __call__(self, data: PDFData) -> Text:
        """
        Extracts Markdown-formatted text from the given PDF data using pymupdf4llm.

        Args:
            data (PDFData): The PDF data to extract text from.

        Returns:
            Text: A Text object containing the extracted Markdown-formatted text from the PDF.

        Raises:
            IOError: If there's an error reading the PDF file.
            ValueError: If there's an error extracting text from the PDF.
        """
        # Validate that the provided data is a PDFData object
        if not isinstance(data, PDFData):
            raise ValueError("Input data must be a PDFData object.")

        try:
            # Use pymupdf4llm to extract the document as Markdown
            text = ""
            md_text = pymupdf4llm.to_markdown(data.filepath)

            # Ensure the Markdown text is not empty
            if not md_text.strip():
                raise ValueError("Extracted text is empty.")

            # Return the Markdown text as a Text object with 'text' attribute
            return Text(text=md_text, name=data.name, data_type="pdf")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF using pymupdf4llm: {e}")

class PyMuPDFExtract(TextExtract):
    """
    Concrete implementation of TextExtract for PDF data sources using PyMuPDF.

    This class provides functionality to extract text from PDF files using the PyMuPDF library.
    """

    def __init__(self) -> None:
        """
        Instantiates a PyMuPDFExtract object.
        """
        super().__init__(data_type="pdf")

    def __call__(self, data: PDFData) -> Text:
        """
        Extracts text from the given PDF data using PyMuPDF.

        Args:
            data (PDFData): The PDF data to extract text from.

        Returns:
            Text: A Text object containing the extracted text from the PDF.

        Raises:
            IOError: If there's an error reading the PDF file.
            ValueError: If there's an error extracting text from the PDF.
        """
        try:
            with fitz.open(data.filepath) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text() + "\n"
                # Validate extracted text is not empty
                if not text.strip():
                    raise ValueError("No text extracted from the PDF.")
                return Text(text=text.strip(), name=data.name, data_type="pdf")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")

# class AzureAIDocumentExtract(TextExtract):
#     """
#     Concrete implementation of TextExtract for document data sources using Azure AI Document Intelligence.
#     """

#     def __init__(self, endpoint: str, credential: str, model: str = "prebuilt-layout") -> None:
#         """
#         Instantiates an AzureAIDocumentExtract object.

#         Args:
#             endpoint (str): The endpoint URL for Azure AI Document Intelligence.
#             credential (str): The credential (API key) for Azure AI Document Intelligence.
#             model (str): The model to use for document analysis (default is 'prebuilt-layout').
#         """
#         super().__init__(data_type="azure_document")
#         self.endpoint = "https://sciencegptditest2.cognitiveservices.azure.com/"
#         self.credential = "193d8ca004f8437084db4cf5f53ebb97"
#         self.model = model

#     def __call__(self, data: PDFData) -> Text:
#         """
#         Extracts text from the given Azure Document Intelligence data.

#         Args:
#             data (AzureAIDocumentData): The document data to extract text from.

#         Returns:
#             Text: A Text object containing the extracted text from the document.

#         Raises:
#             IOError: If there's an error reading the document file.
#             ValueError: If there's an error extracting text from the document using Azure.
#         """
#         # Validate that the provided data is an AzureAIDocumentData object
#         if not isinstance(data, PDFData):
#             raise ValueError("Input data must be an AzureAIDocumentData object.")

#         try:
#             # Instantiate the Azure client
#             from azure.ai.documentintelligence import DocumentIntelligenceClient
#             from azure.core.credentials import AzureKeyCredential

#             client = DocumentIntelligenceClient(
#                 self.endpoint, AzureKeyCredential(self.credential)
#             )

#             # Open the document and send it to Azure Document Intelligence for processing
#             with open(data.filepath, "rb") as file:
#                 poller = client.begin_analyze_document(
#                     self.model,
#                     analyze_request=file,
#                     content_type="application/octet-stream",
#                     output_content_format="text",
#                 )
#                 result = poller.result()

#             # Extract the text content from the result
#             text_content = result.content

#             # Ensure the text is not empty
#             if not text_content.strip():
#                 raise ValueError("Extracted text is empty.")

#             # Return the extracted text as a Text object
#             return Text(text=text_content.strip(), name=data.name, data_type="PDFData)

#         except IOError as e:
#             raise IOError(f"Error reading document file: {e}")
#         except Exception as e:
#             raise ValueError(f"Error extracting text from document using Azure AI: {e}")