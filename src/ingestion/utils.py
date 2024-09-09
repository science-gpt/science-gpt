from abc import ABC
from typing import Any

from .raw_data import RAW_DATA_TYPES


# TODO decide if and how to generate/handle ids
class OutputObject(ABC):
    """
    An abstract base class representing an output object for data processing.

    This class serves as a foundation for various types of output objects in the data
    ingestion pipeline. It provides a common structure for storing and managing
    processed data, including a title and the original data type.

    Attributes:
        title (str): A descriptive title for the data object.
        data_type (RAW_DATA_TYPES): The type of the original data source.
    """

    def __init__(self, title: str, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates an object of this class.

        :param title: a title for the data object
        :param data_type: the type of the original data source
        """
        self.title = title
        self.data_type = data_type
