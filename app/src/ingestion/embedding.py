import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from tqdm import tqdm

from .chunking import Chunk
from .raw_data import Data


@staticmethod
@dataclass
class Embedding(Data):
    """
    Represents an embedding of a text chunk.

    Attributes:
        name (str): The name of the embedding.
        data_type (RAW_DATA_TYPES): The type of the original data source.
        docs (List[str]): The list of document texts that were embedded.
        dense_vector (np.ndarray): The dense embedding vector.
        sparse_vector (Optional[dict[str, float]]): The sparse embedding vector (token-weight mapping), optional.
    """

    docs: str
    dense_vector: np.ndarray
    sparse_vector: Optional[Dict[str, float]] = None

    def __post_init__(self):
        # This calls the __init__ of Data to properly initialize name and data_type.
        super().__init__(name=self.name, data_type=self.data_type)


class Embedder(ABC):
    """
    Abstract base class for embedding text chunks into vectors.
    """

    def __init__(self):
        self.setup_device()
        self.embedding_dimension: Optional[int] = None

    def setup_device(self):
        self.device = "cpu"
        self.use_fp16 = False
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = True

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

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "sentence-transformers/all-mpnet-base-v2".
        """
        super().__init__()
        self.base_embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.sparse_embedder = BGEM3EmbeddingFunction(
            use_fp16=self.use_fp16, device=self.device
        )
        self.embedding_dimension = self.model.client.get_sentence_embedding_dimension()

    def __call__(self, chunks: List[Chunk]) -> Embedding:
        """
        Embed a list of text chunks into a single aggregated Embedding using the Sentence Transformer model.

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            Embedding: An Embedding object containing:
                - name: list of chunk names,
                - data_type: the data type of the first chunk,
                - docs: list of texts,
                - dense_vector: a stacked NumPy array of dense embeddings,
                - sparse_vector: a list of empty dicts (no sparse data available).
        """
        docs = [chunk.text for chunk in chunks]

        dense_list = []
        sparse_list = []
        for text in tqdm(docs, desc="HuggingFace Embedding"):
            vector = self.base_embedder.embed_query(text)
            sparse_embeddings = self.sparse_embedder([text])
            dense_list.append(vector)
            sparse_list.append(sparse_embeddings["sparse"])

        dense_vectors = np.stack(dense_list, axis=0)
        sparse_vectors = np.stack(sparse_list, axis=0)

        return [
            Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                docs=chunk.text,
                dense_vector=dense_vectors[i],
                sparse_vector=sparse_vectors[i],
            )
            for i, chunk in enumerate(chunks)
        ]


class OllamaEmbedder(Embedder):
    """
    An embedder that uses API calls to our Ollama instances hosting embedding models to generate embeddings.
    """

    def __init__(self, model_name: str, endpoint: str):
        """
        Initialize the embedding API call for the embedding model on Ollama.

        Args:
            model_name (str): The name of the embedding model to use.
            endpoint (str): The API endpoint for the Ollama instance.
        """
        super().__init__()
        self.model_name = model_name
        self.base_embedder = OllamaEmbeddings(model=self.model_name, base_url=endpoint)
        self.sparse_embedder = BGEM3EmbeddingFunction(
            use_fp16=self.use_fp16, device=self.device
        )

    def test_connection(self):
        """Test the connection to the embedding service. Fallback to HuggingFace embeddings if this fails."""
        try:
            test_embedding = self.model.embed_query("test")
            self.embedding_dimension = len(test_embedding)
        except Exception as e:
            raise RuntimeError("Embedding model initialization failed") from e

    def __call__(self, chunks: List[Chunk]) -> Embedding:
        """
        Embed a list of text chunks into a single aggregated Embedding using the Ollama hosted model.

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            Embedding: An Embedding object containing:
                - name: list of chunk names,
                - data_type: the data type of the first chunk,
                - docs: list of texts,
                - dense_vector: a stacked NumPy array of dense embeddings,
                - sparse_vector: a list of empty dicts (no sparse data available).
        """
        docs = [chunk.text for chunk in chunks]

        dense_list = []
        sparse_list = []
        for text in tqdm(docs, desc="Ollama Embedding"):
            vector = self.model.embed_query(text)
            sparse_embeddings = self.sparse_embedder([text])
            dense_list.append(vector)
            sparse_list.append(sparse_embeddings["sparse"])

        dense_vectors = np.stack(dense_list, axis=0)
        sparse_vectors = np.stack(sparse_list, axis=0)

        return [
            Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                docs=chunk.text,
                dense_vector=dense_vectors[i],
                sparse_vector=sparse_vectors[i],
            )
            for i, chunk in enumerate(chunks)
        ]


class BGEM3Embedder(Embedder):
    """
    BGEM3Embedder returns hybrid embeddings (dense and sparse vectors) for texts.
    If use_bge_m3 is True, it uses the BGEM3EmbeddingFunction from pymilvus;
    otherwise, it uses a random embedding generator.
    """

    def __init__(self):
        super().__init__()
        self.embedder = BGEM3EmbeddingFunction(
            use_fp16=self.use_fp16, device=self.device
        )
        self.embedding_dimension = self.embedder.dim["dense"]

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks using the BGEM3 model (both dense & sparse).

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """
        docs = [chunk.text for chunk in chunks]

        dense_list = []
        sparse_list = []
        for text in tqdm(docs, desc="BGEM3 Embedding"):
            embeddings = self.embedder(
                [text]
            )  # {"dense": np.ndarray, "sparse": list[dict[str, float]]}
            dense_list.extend(embeddings["dense"])
            sparse_list.append(embeddings["sparse"])

        dense_vectors = np.stack(dense_list, axis=0)
        sparse_vectors = np.stack(sparse_list, axis=0)

        return [
            Embedding(
                name=chunk.name,
                data_type=chunk.data_type,
                docs=chunk.text,
                dense_vector=dense_vectors[i],
                sparse_vector=sparse_vectors[i],
            )
            for i, chunk in enumerate(chunks)
        ]
