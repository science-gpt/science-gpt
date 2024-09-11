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
    """

    name: str
    data_type: RAW_DATA_TYPES
    # TODO: do we want to add uuids to objects?


class PDFData(RawData):
    """
    Represents a raw .pdf data source.
    """

    def __init__(self, name: str, filepath: pathlib.Path) -> None:
        """
        Instantiates an object of this class.

        :param filepath: Path of the pdf file
        """
        super().__init__(name=name, data_type="pdf")
        self.filepath = filepath
