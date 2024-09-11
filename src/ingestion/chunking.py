from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from .extraction import Text
from .raw_data import RAW_DATA_TYPES


@dataclass
class Chunk:
    """
    Represents a chunk of text extracted from a data source.
    """

    text: str
    title: str
    data_type: RAW_DATA_TYPES


class Chunker(ABC):
    """
    Abstract base class for text chunking algorithms.
    """

    @abstractmethod
    def __call__(self, chunk_size, chunk_overlap):
        pass


class SplitSentencesChunker(Chunker):
    """
    A Chunker implementation that splits text into sentences using NLTK.
    """

    def __init__(self):
        nltk.download("punkt_tab", quiet=True)

    def __call__(self, text: Text) -> List[Chunk]:
        """
        Splits the given text into chunks, where each chunk is a sentence.

        :param text: The text to be split into chunks.
        :return: A list of Chunk objects, each containing a single sentence.
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
