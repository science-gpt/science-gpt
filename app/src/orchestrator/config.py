from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal


class ModelAuth(BaseModel):
    macbook_endpoint: Optional[str]
    url: Optional[str]
    api_key: Optional[str]
    version: Optional[str]


class ModelParams(BaseModel):
    supported_models: List[str]
    model: str
    seed: int
    temperature: float
    top_p: float
    max_tokens: int
    num_ctx: int


class Extraction(BaseModel):
    supported_extractors: List[str]
    extraction_method: str


class Chunking(BaseModel):
    supported_chunkers: List[str]
    chunking_method: str


class Embedding(BaseModel):
    supported_embedders: List[str]
    embedding_model: str


class VectorDB(BaseModel):
    supported_databases: List[str]
    database: str
    host: Optional[str]
    port: Optional[int]


class RAGParams(BaseModel):
    top_k_retrieval: int
    keywords: Optional[str]


class SystemConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_params: ModelParams
    model_auth: ModelAuth
    extraction: Extraction
    chunking: Chunking
    embedding: Embedding
    vector_db: VectorDB
    rag_params: RAGParams
