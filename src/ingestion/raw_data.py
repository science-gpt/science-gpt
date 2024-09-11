import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Literal

RAW_DATA_TYPES = Literal[
    "pdf",
    # eventually: 'json', 'doc', 'web', ...
]


@dataclass
class RawData(ABC):
    """
    Represents the base class for data sources such as .pdf or .doc files, website links, etc.

    Attributes:
        name (str): The name of the data source.
        data_type (RAW_DATA_TYPES): The type of the data source.
    """

    name: str
    data_type: RAW_DATA_TYPES
    # TODO: do we want to add uuids to objects?


class PDFData(RawData):
    """
    Represents a raw .pdf data source.

    Attributes:
        name (str): The name of the PDF file.
        data_type (RAW_DATA_TYPES): The type of the data source (always "pdf" for this class).
        filepath (pathlib.Path): The path to the PDF file.
    """

    def __init__(self, name: str, filepath: pathlib.Path) -> None:
        """
        Instantiates a PDFData object.

        Args:
            name (str): The name of the PDF file.
            filepath (pathlib.Path): The path to the PDF file.
        """
        super().__init__(name=name, data_type="pdf")
        self.filepath = filepath
