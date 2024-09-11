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

>>>>>>> 1c4a8d6 (added text representations to intermediary objects)
    @abstractmethod
    def __call__(self, chunk_size, chunk_overlap):
        pass


<<<<<<< HEAD
class CustomTableProcessor(ABC):
    pass
=======
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
        # TODO find the custom chunking code and put it here
        pass


class SplitSentencesChunker(Chunker):
    """
    A Chunker implementation that splits text into sentences using NLTK.
    """

    def __init__(self):
        # Download the punkt tokenizer if not already available
        nltk.download("punkt_tab", quiet=True)

    def __call__(self, text: Text) -> List[Chunk]:
        """
        Splits the given text into chunks, where each chunk is a sentence.

<<<<<<< HEAD
class CustomFigureProcessor(ABC):
    pass
=======
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
>>>>>>> 65b593d (implemented a simple sentence splitting chunker)
