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
        dense_vector (np.ndarray): The dense embedding vector.
        sparse_vector (Optional[dict[str, float]]): The sparse embedding vector (token-weight mapping), optional.
        text (str): The original text that was embedded.
    """
    docs: List[str]
    dense_vector: np.ndarray
    sparse_vector: Optional[dict[str, float]] = None
    

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

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the HuggingFaceSentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
                              Defaults to "sentence-transformers/BAAI/bge-m3".
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
                dense_vector=np.array(vector),
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

    def test_connection(self):
        """(Carter) I've written this to test the connection to the macbook. We will default to Huggingface embeddings if this fails."""
        try:
            # Get dimension by running a test embedding
            test_embedding = self.model.embed_query("test")
            self.embedding_dimension = len(test_embedding)
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
                dense_vector=np.array(vector),
                text=chunk.text,
            )

            embeddings.append(embedding)
        return embeddings



# class BGEM3Embedder:
#     """
#     BGEM3Embedder returns hybrid embeddings (dense and sparse vectors) for texts.
#     If use_bge_m3 is True, it uses the BGEM3EmbeddingFunction from pymilvus;
#     otherwise, it uses a random embedding generator.
#     """
#     def __init__(self, use_bge_m3=True, use_fp16=False, device="cpu"):
#         self.use_bge_m3 = use_bge_m3
#         if self.use_bge_m3:
#             self.embedder = BGEM3EmbeddingFunction(use_fp16=use_fp16, device=device)
#             self.dense_dim = self.embedder.dim["dense"]
#         else:
#             self.embedder = random_embedding
#             self.dense_dim = 768

#     def __call__(self, texts):
#         results = self.embedder(texts)  # Expecting keys "dense" and "sparse"
#         # Ensure that the "sparse" result is always a list of dense 1D arrays.
#         if "sparse" in results:
#             # If it's not a list, wrap it.
#             if not isinstance(results["sparse"], list):
#                 if hasattr(results["sparse"], "toarray"):
#                     results["sparse"] = [results["sparse"].toarray().flatten()]
#                 else:
#                     results["sparse"] = [results["sparse"]]
#             else:
#                 new_sparse = []
#                 for vec in results["sparse"]:
#                     if hasattr(vec, "toarray"):
#                         new_sparse.append(vec.toarray().flatten())
#                     else:
#                         new_sparse.append(vec)
#                 results["sparse"] = new_sparse
#         return results
    
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
        # else:
        #     self.embedder = random_embedding
        #     self.embedding_dimension = 768

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
        
        # print("docs: ", docs)
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.dense_dim = ef.dim["dense"]
        docs_embeddings = ef(docs)  # {"dense": np.ndarray, "sparse": list[dict[str, float]]}
        
        
        embedding = Embedding(
            name = chunks[0].name, # TODO: for this list of chunks, it may have differnt file names.
            data_type = chunks[0].data_type,
            docs = docs,
            dense_vector = docs_embeddings["dense"],
            sparse_vector = docs_embeddings["sparse"],
        )

        return embedding
    
    def random_embedding(self, texts: List[str]) -> Dict[str, np.ndarray]:

        """
        Generate random embeddings (both dense and sparse) for testing purposes.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            dict: A dictionary with "dense" and "sparse" embeddings.
        """
        rng = np.random.default_rng()
        return {
            "dense": rng.random((len(texts), 768)),  # Dense embeddings
            "sparse": [
                {str(d): float(rng.random()) for d in random.sample(range(1000), random.randint(20, 30))}
                for _ in texts
            ],  # Sparse embeddings with correct format
        }


