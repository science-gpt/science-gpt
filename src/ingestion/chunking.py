from abc import ABC, abstractmethod
<<<<<<< HEAD
=======
from typing import List

from .extraction import Text
from .raw_data import RAW_DATA_TYPES
from .utils import OutputObject
>>>>>>> 790cbb9 (fixed imports)


<<<<<<< HEAD
class CustomTextSplitter(ABC):
=======
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

    def __str__(self) -> str:
        """
        Returns a string representation of the Chunk.
        """
        return f"""
        Chunk(
            title='{self.title}',
            text='{self.text[:50]}...',
            data_type={self.data_type}
        )
        """

    __repr__ = __str__


class Chunker(ABC):
    """
    Abstract base class for text chunking algorithms.
    """

>>>>>>> 1c4a8d6 (added text representations to intermediary objects)
    @abstractmethod
    def __call__(self, chunk_size, chunk_overlap):
        pass


class CustomTableProcessor(ABC):
    pass


class CustomFigureProcessor(ABC):
    pass
