from abc import ABC, abstractmethod
from typing import List

from extraction import Text
from raw_data import RAW_DATA_TYPES
from utils import OutputObject


class Chunk(OutputObject):
    """
    Represents a chunk of text extracted from a data source.
    """

    def __init__(self, text: str, title: str, data_type: RAW_DATA_TYPES) -> None:
        """
        Instantiates an object of this class.

        :param text: The extracted text.
        :param title: A title for the chunked text.
        :param data_type: The type of the original data source.
        """
        super().__init__(title=title, data_type=data_type)
        self.text = text


class Chunker(ABC):
    """
    Abstract base class for text chunking algorithms.
    """

    @abstractmethod
    def __call__(self, text: Text) -> List[Chunk]:
        """
        Chunks the given text into smaller pieces.

        :param text: The text to be chunked.
        :return: A list of Chunk objects.
        """
        pass


class CustomTextSplitter(Chunker):
    """
    A custom implementation of the Chunker class for splitting text.
    """

    def __call__(self, text: Text) -> List[Chunk]:
        """
        Splits the given text into chunks using a custom algorithm.

        :param text: The text to be split into chunks.
        :return: A list of Chunk objects.
        """
        pass


# Factory function for chunkers
def create_chunker(chunker_type: str, **kwargs) -> CustomTextSplitter:
    """
    Create and return an instance of the specified chunker type.

    :param chunker_type: Type of chunker to create
    :param kwargs: Additional keyword arguments for the chunker constructor
    :return: An instance of the specified Chunker subclass
    :raises ValueError: If an unsupported chunker type is specified
    """
    # TODO: Implement concrete chunker classes and add them to this factory function
    raise NotImplementedError(f"Chunker type '{chunker_type}' is not implemented yet.")



# TODO find a place to put these
# TODO handle metadata tagging
class CustomTableProcessor(ABC):
    """
    Abstract base class for processing tables.
    """
    pass


class CustomFigureProcessor(ABC):
    """
    Abstract base class for processing figures.
    """
    pass
