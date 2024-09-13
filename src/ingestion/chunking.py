from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from .extraction import Text
from .raw_data import RAW_DATA_TYPES, Data


@dataclass
class Chunk(Data):
    """
    Represents a chunk of text extracted from a data source.

    Attributes:
        name (str): The name of the chunk.
        data_type (RAW_DATA_TYPES): The type of the original data source.
        text (str): The content of the chunk.
    """

    text: str

    def __post_init__(self):
        super().__init__(name=self.name, data_type=self.data_type)


class Chunker(ABC):
    """
    Abstract base class for text chunking algorithms.
    """

    @abstractmethod
    def __call__(self, text: Text) -> List[Chunk]:
        """
        Split the given text into chunks.

        Args:
            text (Text): The text to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects representing the split text.
        """
        pass


class SplitSentencesChunker(Chunker):
    """
    A Chunker implementation that splits text into sentences using NLTK.
    """

    def __init__(self):
        """
        Initialize the SplitSentencesChunker and download the required NLTK data.
        """
        nltk.download("punkt_tab", quiet=True)

    def __call__(self, text: Text) -> List[Chunk]:
        """
        Split the given text into chunks, where each chunk is a sentence.

        Args:
            text (Text): The text to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects, each containing a single sentence.
        """
        sentences = sent_tokenize(text.text)
        return [
            Chunk(
                text=sentence,
                title=f"{text.title} - Sentence {i+1}",
                data_type=text.data_type,
            )
            for i, sentence in enumerate(sentences)
        ]
