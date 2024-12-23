import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import PyPDF2
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    PdfFormatOption,
)

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
class ExtractedContent(Data):
    """
    Base class for extracted content from data sources.

    Attributes:
        name (str): The name of the content.
        data_type (RAW_DATA_TYPES): The type of the raw data source.
    """

    def __post_init__(self):
        super().__init__(name=self.name, data_type=self.data_type)

    @abstractmethod
    def get_text(self) -> str:
        """Returns a plain text representation of the content"""
        pass


@dataclass
class Text(ExtractedContent):
    """
    Represents extracted text from a data source.

    Attributes:
        text (str): The extracted text content.
    """

    text: str

    def __post_init__(self):
        super().__post_init__()

    def get_text(self) -> str:
        return self.text


@dataclass
class DoclingDocument(ExtractedContent):
    """
    Represents the extracted docling document from a data source using Docling.

    Attributes:
        conv_result (ConversionResult): The docling document conversion result.
    """

    conv_result: ConversionResult

    def __post_init__(self):
        super().__post_init__()

    def get_text(self) -> str:
        return self.conv_result.document.export_to_markdown()


class ContentExtractor(ABC):
    """
    Abstract base class for content extraction from various data sources.
    """

    def __init__(self, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates a ContentExtractor object.

        Args:
            data_type (RAW_DATA_TYPES): Type of raw data.
        """
        self.data_type = data_type

    @abstractmethod
    def __call__(self, data: Data) -> ExtractedContent:
        """
        Abstract method to extract text from the given raw data.

        Args:
            data (Data): The raw data to extract text from.

        Returns:
            ExtractedContent: A ExtractedContent object containing the extracted content.
        """
        pass


class PyPDF2Extract(ContentExtractor):
    """
    Concrete implementation of ContentExtractor for PDF data sources using PyPDF2.
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
        try:
            with open(data.filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return Text(text=text.strip(), name=data.name, data_type="pdf")
        except IOError as e:
            raise IOError(f"Error reading PDF file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")


class DoclingPDFExtract(ContentExtractor):
    """
    Concrete implementation of ContentExtractor for PDF data sources using Docling.
    """

    def __init__(
        self,
        do_table_structure: bool = False,
        table_former_mode: Literal["fast", "accurate"] = "accurate",
    ) -> None:
        """
        Instantiates a DoclingPDFExtract object.

        Args:
            do_table_structure (bool): Whether to extract table structure from PDFs.
            table_former_mode ("fast" | "accurate"): Mode for table extraction.
                                                   "fast" is quicker but less precise,
                                                   "accurate" is slower but more precise.
        """
        super().__init__(data_type="pdf")

        if table_former_mode not in ["fast", "accurate"]:
            raise ValueError(
                f"Invalid table former mode: {table_former_mode}. Must be 'fast' or 'accurate'."
            )

        mode = (
            TableFormerMode.ACCURATE
            if table_former_mode == "accurate"
            else TableFormerMode.FAST
        )

        pipeline_options = PdfPipelineOptions(do_table_structure=do_table_structure)
        pipeline_options.table_structure_options.mode = mode

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        self.converter = DocumentConverter(format_options=format_options)

    def __call__(self, data: PDFData) -> DoclingDocument:
        """
        Converts a PDF into a docling document using Docling DocumentConverter.

        Args:
            data (PDFData): The PDF data to convert into a docling document.

        Returns:
            DoclingDocument: A DoclingDocument object containing the converted document.
        """
        result = self.converter.convert(data.filepath)
        return DoclingDocument(conv_result=result, name=data.name, data_type="pdf")
