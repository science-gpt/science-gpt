from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import random
from .chunking import Chunk
from .raw_data import Data


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
    docs: List[str]
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

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "sentence-transformers/all-mpnet-base-v2".
        """
        super().__init__()
        self.model = HuggingFaceEmbeddings(model_name=model_name)
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
        names = [chunk.name for chunk in chunks]
        docs = [chunk.text for chunk in chunks]

        dense_list = []
        for text in tqdm(docs, desc="HuggingFace Embedding"):
            vector = self.model.embed_query(text)
            dense_list.append(vector)
        dense_vectors = np.stack(dense_list, axis=0)
        sparse_vectors = [{} for _ in docs]  # no sparse embeddings available

        return Embedding(
            name=names,
            data_type=chunks[0].data_type,
            docs=docs,
            dense_vector=dense_vectors,
            sparse_vector=sparse_vectors,
        )


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
        self.model = OllamaEmbeddings(model=self.model_name, base_url=endpoint)

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
        names = [chunk.name for chunk in chunks]
        docs = [chunk.text for chunk in chunks]

        dense_list = []
        for text in tqdm(docs, desc="Ollama Embedding"):
            vector = self.model.embed_query(text)
            dense_list.append(vector)
        dense_vectors = np.stack(dense_list, axis=0)
        sparse_vectors = [{} for _ in docs]  # no sparse embeddings available

        return Embedding(
            name=names,
            data_type=chunks[0].data_type,
            docs=docs,
            dense_vector=dense_vectors,
            sparse_vector=sparse_vectors,
        )


class BGEM3Embedder(Embedder):
    """
    BGEM3Embedder returns hybrid embeddings (dense and sparse vectors) for texts.
    If use_bge_m3 is True, it uses the BGEM3EmbeddingFunction from pymilvus;
    otherwise, it uses a random embedding generator.
    """
    def __init__(self, use_bge_m3=True, use_fp16=False, device="cpu"):
        super().__init__()
        self.use_bge_m3 = use_bge_m3
        if self.use_bge_m3:
            self.embedder = BGEM3EmbeddingFunction(use_fp16=use_fp16, device=device)
            self.embedding_dimension = self.embedder.dim["dense"]

    def __call__(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Embed a list of text chunks using the BGEM3 model (both dense & sparse).

        Args:
            chunks (List[Chunk]): List of Chunk objects to be embedded.

        Returns:
            List[Embedding]: A list of Embedding objects containing the embedded vectors and metadata.
        """
        print("using bge m3 embedder for processing the list of chunks and embedding them")
        print("chunks: ", chunks)
        docs = [chunk.text for chunk in chunks] # this is a list of chunking strings
        
        # TODO: list of names for the chunks
        names = [chunk.name for chunk in chunks]
        
        # print("docs: ", docs)
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.dense_dim = ef.dim["dense"]
        docs_embeddings = ef(docs)  # {"dense": np.ndarray, "sparse": list[dict[str, float]]}
        
        
        embedding = Embedding(
            # name = chunks[0].name, # TODO: for this list of chunks, it may have differnt file names.
            name = names,
            data_type = chunks[0].data_type,
            docs = docs,
            dense_vector = docs_embeddings["dense"],
            sparse_vector = docs_embeddings["sparse"],
        )

        return embedding