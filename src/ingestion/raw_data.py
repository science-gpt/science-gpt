import pathlib
from abc import ABC
from typing import Literal

RAW_DATA_TYPES = Literal[
    "pdf",
    # eventually: 'json', 'doc', 'web', ...
]


class RawData(ABC):
    """
    Represents the base class for data sources such as .pdf or .doc files, website links, etc.
    """

    def __init__(self, name: str, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates an object of this class.

        :param name: Name of the data source. ie: title of the .pdf file
        :param data_type: Type of data source
        """
        self.name = name
        self.data_type = data_type
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
