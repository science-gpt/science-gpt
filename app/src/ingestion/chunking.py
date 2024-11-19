from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from .extraction import Text
from .raw_data import Data


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
                name=f"{text.name} - Sentence {i+1}",
                data_type=text.data_type,
            )
            for i, sentence in enumerate(sentences)
        ]


class RecursiveCharacterChunker(Chunker):
    def __init__(self, chunk_size=1600, chunk_overlap=160, is_separator_regex=False):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=is_separator_regex,
        )

    def __call__(self, text: Text) -> List[Chunk]:
        chunks = self.text_splitter.split_text(text.text)
        return [
            Chunk(
                text=c,
                name=f"{text.name} - Chunk {i+1}",
                data_type=text.data_type,
            )
            for i, c in tqdm(enumerate(chunks))
        ]

class MarkdownChunker(Chunker):
    """
    A Chunker implementation that splits Markdown text into sections based on headers,
    and then further splits within each section using a character-level splitter.
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 0):
        """
        Initialize the MarkdownChunker.

        Args:
            chunk_size (int): Maximum size of each chunk (number of characters).
            chunk_overlap (int): Overlap between consecutive chunks (number of characters).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Header-based splitting
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False,
        )

        # Recursive character-level splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def __call__(self, text: Text) -> List[Chunk]:
        """
        Split the given Markdown text into chunks.

        Args:
            text (Text): The text to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects representing the split Markdown text.
        """
        # Step 1: Split Markdown text into sections based on headers
        sections = self.markdown_splitter.split_text(text.text)

        # Step 2: Further split each section into smaller chunks
        chunks = []
        for section in sections:
            # Extract the actual text from the Document object
            section_text = section.page_content if hasattr(section, "page_content") else section

            # Avoid splitting very small sections
            if len(section_text) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        text=section_text,
                        name=f"{text.name} - Section",
                        data_type=text.data_type,
                    )
                )
                continue

            # Split larger sections using the RecursiveCharacterTextSplitter
            documents = self.text_splitter.split_text(section_text)
            for i, doc in enumerate(documents):
                chunk_text = doc.page_content if hasattr(doc, "page_content") else doc

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        name=f"{text.name} - Section {i+1}",
                        data_type=text.data_type,
                    )
                )

        return chunks