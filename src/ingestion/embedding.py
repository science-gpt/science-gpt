from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .chunking import Chunk
from .utils import OutputObject
from .raw_data import RAW_DATA_TYPES
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class Embedding(OutputObject):
    """
    Represents an embedding of a text chunk.
    """

    def __init__(
        self, vector: np.ndarray, text: str, title: str, data_type: RAW_DATA_TYPES
    ) -> None:
        """
        Instantiates an Embedding object.

        :param vector: The embedding vector
        :param text: The original text of the chunk
        :param title: The title of the chunk
        :param data_type: The type of the original data source
        """
        super().__init__(title=title, data_type=data_type)
        self.vector = vector
        self.text = text

    def __str__(self) -> str:
        """
        Returns a string representation of the Embedding.
        """
        return f"""
        Embedding(
            title='{self.title}',
            text='{self.text[:50]}...',
            data_type={self.data_type},
            vector_shape={self.vector.shape}
        )
        """

    __repr__ = __str__


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
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        embeddings = []
        for chunk in chunks:
            vector = self.model.embed_query(chunk.text)
            embedding = Embedding(
                vector=np.array(vector),
                text=chunk.text,
                title=chunk.title,
                data_type=chunk.data_type
            )
            embeddings.append(embedding)
        return embeddings


def create_embedder(embedder_type: str, **kwargs) -> Embedder:
    """
    Create and return an instance of the specified embedder type.

    :param embedder_type: Type of embedder to create
    :param kwargs: Additional keyword arguments for the embedder constructor
    :return: An instance of the specified Embedder subclass
    :raises ValueError: If an unsupported embedder type is specified
    """
    # TODO: Implement concrete embedder classes and add them to this factory function
    raise NotImplementedError(
        f"Embedder type '{embedder_type}' is not implemented yet."
    )
