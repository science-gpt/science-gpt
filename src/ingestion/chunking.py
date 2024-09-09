from abc import ABC, abstractmethod


class CustomTextSplitter(ABC):
    @abstractmethod
    def __call__(self, chunk_size, chunk_overlap):
        pass


class CustomTableProcessor(ABC):
    pass


class CustomFigureProcessor(ABC):
    pass
