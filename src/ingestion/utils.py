from abc import ABC

from raw_data import RAW_DATA_TYPES


class OutputObject(ABC):
    # TODO add a docstring

    def __init__(self, title: str, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates an object of this class.

        :param title: a title for the data object
        :param data_type: the type of the original data source
        """
        self.title = title
        self.data_type = data_type
