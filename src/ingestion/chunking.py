from abc import ABC, abstractmethod
<<<<<<< HEAD
=======
from typing import List

from .extraction import Text
from .raw_data import RAW_DATA_TYPES
from .utils import OutputObject
>>>>>>> 790cbb9 (fixed imports)


class CustomTextSplitter(ABC):
    @abstractmethod
    def __call__(self, chunk_size, chunk_overlap):
        pass


class CustomTableProcessor(ABC):
    pass


class CustomFigureProcessor(ABC):
    pass
