from abc import ABC, abstractmethod
from typing import List

from extraction import Text
from raw_data import RAW_DATA_TYPES
from utils import OutputObject


class Chunk(OutputObject):
    # TODO docstring
    def __init__(self, text: str, title: str, data_type: RAW_DATA_TYPES):
        """
        Instantiates an object of this class.

        :param text: the extracted text
        :param title: a title for the chunked text
        :param data_type: the type of the original data source
        """
        super().__init__(title=title, data_type=data_type)
        self.text = text


class Chunker(ABC):
    # TODO docstring
    @abstractmethod
    def __call__(self, text: Text) -> List[Chunk]:
        # TODO docstring
        pass


class CustomTextSplitter(Chunker):
    def __call__(self, text: Text) -> List[Chunk]:
        # TODO docstring
        pass


# TODO find a place to put these
# TODO handle metadata tagging


class CustomTableProcessor(ABC):
    pass


class CustomFigureProcessor(ABC):
    pass
