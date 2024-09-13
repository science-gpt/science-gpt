from typing import Literal, Optional

from pydantic import BaseModel


class ModelParams(BaseModel):
    temperature: Optional[float]
    max_tokens: Optional[int]


class ModelAuth(BaseModel):
    url: Optional[str]
    api_key: Optional[str]
    version: Optional[str]


class Extraction(BaseModel):
    pdf_extract_method: Literal["pypdf2"]


class Chunking(BaseModel):
    method: Literal["split_sentences"]


class Embedding(BaseModel):
    method: Literal["huggingface-sentence-transformer"]
    model: Literal["sentence-transformers/all-mpnet-base-v2"]


class VectorDB(BaseModel):
    type: Literal["local-chromadb"]
    local_path: str
    instance_name: str
    search_strategy: Literal["similarity"]


class RAGParams(BaseModel):
    top_k_retrieval: int


class SystemConfig(BaseModel):
    model_name: str
    model_params: ModelParams
    model_auth: ModelAuth
    extraction: Extraction
    chunking: Chunking
    embedding: Embedding
    vector_db: VectorDB
    rag_params: RAGParams
