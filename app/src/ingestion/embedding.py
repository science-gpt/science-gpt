from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

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

    def __init__(self):
        self.embedding_dimension: Optional[int] = None

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


class HuggingFaceEmbedder(Embedder):
    """
    An embedder that uses HuggingFace's Sentence Transformer models to create embeddings.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "sentence-transformers/all-mpnet-base-v2".
        """
        super().__init__()
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.embedding_dimension = self.model.client.get_sentence_embedding_dimension()

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors using the Sentence Transformer model.

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """
        embeddings = []
        for chunk in tqdm(chunks):
            vector = self.model.embed_query(chunk.text)
            embedding = Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                vector=np.array(vector),
                text=chunk.text,
            )
            embeddings.append(embedding)
        return embeddings


class OllamaEmbedder(Embedder):
    """
    An embedder that uses API calls to our Ollama instances hosting embedding models to generate embeddings.
    """

    def __init__(self, model_name: str, endpoint: str):
        """
        Initialize the embedding API call for the embedding model on Ollama.

        Args:
            model_name (str): The name of the embedding model to use.
                              Defaults to "mxbai-embed-large:latest".
        """
        super().__init__()
        self.model_name = model_name
        self.model = OllamaEmbeddings(model=self.model_name, base_url=endpoint)
        # Get dimension by running a test embedding
        test_embedding = self.model.embed_query("test")
        self.embedding_dimension = len(test_embedding)

    def test_connection(self):
        """(Carter) I've written this to test the connection to the macbook. We will default to Huggingface embeddings if this fails."""
        try:
            self.model.embed_query("test")
        except Exception as e:
            raise RuntimeError("Embedding model initialization failed") from e

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks into vectors using the Ollama hosted embedding model

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """

        embeddings = []
        for chunk in tqdm(chunks):
            vector = self.model.embed_query(chunk.text)
            embedding = Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                vector=np.array(vector),
                text=chunk.text,
            )
            embeddings.append(embedding)
        return embeddings
