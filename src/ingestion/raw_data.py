from abc import ABC
from dataclasses import dataclass
from typing import Literal

RAW_DATA_TYPES = Literal[
    "pdf",
    # eventually: 'json', 'doc', 'web', ...
]


@dataclass
class Data(ABC):
    """
    Represents the base class for data sources such as .pdf or .doc files, website links, etc.

    Attributes:
        name (str): The name of the data source.
        data_type (RAW_DATA_TYPES): The type of the data source.
    """

    name: str
    data_type: RAW_DATA_TYPES
    # TODO: do we want to add uuids to objects?
