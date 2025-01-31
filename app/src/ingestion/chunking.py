from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import nltk
from docling.chunking import HierarchicalChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.transforms.chunker import DocMeta

from .extraction import DoclingDocument, ExtractedContent
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
    def __call__(self, content: ExtractedContent) -> List[Chunk]:
        """
        Split the given content into chunks.

        Args:
            content (ExtractedContent): The content to be split into chunks.

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

    def __call__(self, content: ExtractedContent) -> List[Chunk]:
        """
        Split the given content into chunks, where each chunk is a sentence.

        Args:
            content (ExtractedContent): The content to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects, each containing a single sentence.
        """
        text = content.get_text()
        sentences = sent_tokenize(text)
        return [
            Chunk(
                text=sentence,
                name=f"{content.name} - Sentence {i+1}",
                data_type=content.data_type,
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

    def __call__(self, content: ExtractedContent) -> List[Chunk]:
        text = content.get_text()
        chunks = self.text_splitter.split_text(text)
        return [
            Chunk(
                text=c,
                name=f"{content.name} - Chunk {i+1}",
                data_type=content.data_type,
            )
            for i, c in tqdm(enumerate(chunks))
        ]


class DoclingHierarchicalChunker(Chunker):
    """
    A Chunker implementation that uses Docling's HierarchicalChunker to split a DoclingDocument into chunks.
    """

    def __init__(self):
        """
        Initialize the DoclingHierarchicalChunker.
        """
        self.chunker = HierarchicalChunker()

    def __call__(self, content: DoclingDocument) -> List[Chunk]:
        """
        Split the given content into chunks, where each chunk is a sentence.

        Args:
            content (ExtractedContent): The content to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects.
        """
        if not isinstance(content, DoclingDocument):
            raise TypeError(
                f"DoclingChunker requires DoclingDocument input, but got {type(content)}. "
                "Use DoclingExtract or a different chunker."
            )
        chunks_iter = self.chunker.chunk(content.conv_result.document)
        chunks = [
            Chunk(
                text=chunk.text,
                name=f"{content.name} - Chunk {i+1}",
                data_type=content.data_type,
            )
            for i, chunk in tqdm(enumerate(chunks_iter))
        ]
        return chunks


class DoclingHybridChunker(Chunker):
    """
    A Hybrid Chunker implementation that processes Docling documents into chunks,
    including headings, captions, and metadata.
    """

    def __init__(self):
        """
        Initialize the HybridDoclingChunker.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer, max_tokens=8192, merge_peers=True
        )

    def __call__(self, content: DoclingDocument) -> List[Chunk]:
        """
        Split the given content into chunks, where each chunk contains text, headings, captions, and metadata.

        Args:
            content (DoclingDocument): The content to be split into chunks.

        Returns:
            List[Chunk]: A list of Chunk objects with processed data.
        """
        if not isinstance(content, DoclingDocument):
            raise TypeError(
                f"HybridDoclingChunker requires DoclingDocument input, but got {type(content)}. "
                "Use DoclingExtract or a different chunker."
            )

        # Use the conv_result.document structure for chunking (same as HierarchicalChunker)
        document = (
            content.conv_result.document
        )  # This matches the HierarchicalChunker's approach

        # Now, chunk the document
        chunk_iter = self.chunker.chunk(document)
        chunks = []

        # Generate chunk data with headings, captions, and metadata
        for i, chunk in tqdm(enumerate(chunk_iter)):
            chunk_text = chunk.text
            headings = chunk.meta.headings or []
            captions = chunk.meta.captions or []
            metadata = chunk.meta.dict()

            # Concatenate headings and captions into the text (add them at the start or end)
            combined_text = chunk_text
            if headings:
                combined_text = (
                    f"Headings: {headings}\n" + combined_text
                )  # Add headings at the top
            if captions:
                combined_text = (
                    f"Captions: {captions}\n" + combined_text
                )  # Add captions after headings
            # if metadata:
            #     combined_text += f"\nMetadata: {json.dumps(metadata)}"  # Add metadata at the end

            # Prepare chunk data
            chunk_data = {
                "text": combined_text,
                "headings": headings,
                "captions": captions,
                # "metadata": metadata,
                "document_name": content.name,
            }

            # Create the chunk with the combined text
            chunks.append(
                Chunk(
                    text=chunk_data["text"],
                    name=f"{content.name} - Chunk {i+1}",
                    data_type=content.data_type,  # Assuming 'data_type' exists in content
                )
            )
        return chunks
