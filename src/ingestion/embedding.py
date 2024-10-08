from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .chunking import Chunk
from .raw_data import Data


@dataclass
class Embedding(Data):
    """
    Represents an embedding of a text chunk.

    Attributes:
        name (str): The name of the embedding.
        data_type (RAW_DATA_TYPES): The type of the original data source.
        vector (np.ndarray): The embedding vector.
        text (str): The original text that was embedded.
    """

    vector: np.ndarray
    text: str

    def __post_init__(self):
        super().__init__(name=self.name, data_type=self.data_type)


class Embedder(ABC):
    """
    Abstract base class for embedding text chunks into vectors.
    """

    @abstractmethod
    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors.

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """
        pass


class HuggingFaceSentenceTransformerEmbedder(Embedder):
    """
    An embedder that uses HuggingFace's Sentence Transformer models to create embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "sentence-transformers/all-mpnet-base-v2".
        """
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors using the Sentence Transformer model.

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """
        embeddings = []
        for chunk in chunks:
            vector = self.model.embed_query(chunk.text)
            embedding = Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                vector=np.array(vector),
                text=chunk.text,
            )
            embeddings.append(embedding)
        return embeddings
