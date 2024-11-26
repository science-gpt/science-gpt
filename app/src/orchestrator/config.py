from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ModelParams(BaseModel):
    seed: Optional[int]
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    num_ctx: Optional[int]


class ModelAuth(BaseModel):
    macbook_endpoint: Optional[str]
    url: Optional[str]
    api_key: Optional[str]
    version: Optional[str]


class Extraction(BaseModel):
    pdf_extract_method: str


class Chunking(BaseModel):
    method: str


class Embedding(BaseModel):
    model: str


class VectorDB(BaseModel):
    type: Literal["chromadb", "milvus"]
    host: Optional[str]
    port: Optional[int]


class RAGParams(BaseModel):
    top_k_retrieval: int


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
    supported_models: List[str]
