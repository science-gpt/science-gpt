from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .chunking import Chunk
from .raw_data import RAW_DATA_TYPES


@dataclass
class Embedding:
    """
    Represents an embedding of a text chunk.
    """

    vector: np.ndarray
    text: str
    title: str
    data_type: RAW_DATA_TYPES


class Embedder(ABC):
    """
    Abstract base class for embedding text chunks into vectors.
    """

    @abstractmethod
    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors.

        :param chunks: List of Chunk objects to be embedded
        :return: A list of Embedding objects
        """
        pass


class HuggingFaceSentenceTransformerEmbedder(Embedder):
    """
    An embedder that uses HuggingFace's Sentence Transformer models to create embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        :param model_name: The name of the Sentence Transformer model to use.
                           Defaults to "sentence-transformers/all-mpnet-base-v2".
        """
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors using the Sentence Transformer model.

        :param chunks: List of Chunk objects to be embedded.
        :return: A list of Embedding objects containing the embedded vectors and metadata.
        """
        embeddings = []
        for chunk in chunks:
            vector = self.model.embed_query(chunk.text)
            embedding = Embedding(
                vector=np.array(vector),
                text=chunk.text,
                title=chunk.title,
                data_type=chunk.data_type,
            )
            embeddings.append(embedding)
        return embeddings
