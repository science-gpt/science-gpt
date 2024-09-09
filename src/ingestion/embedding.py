from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .chunking import Chunk


class Embedder(ABC):
    """
    Abstract base class for embedding text chunks into vectors.
    """

    @abstractmethod
    def embed(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Embed a list of text chunks into vectors.

        :param chunks: List of Chunk objects to be embedded
        :return: A numpy array of embedded vectors, where each row represents a chunk's embedding
        """
        pass


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
