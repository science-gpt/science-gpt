from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal


class ModelAuth(BaseModel):
    macbook_endpoint: Optional[str]
    url: Optional[str]
    api_key: Optional[str]
    version: Optional[str]

class AgentParams(BaseModel):
    enable: bool
    type: str

class ModelParams(BaseModel):
    # model_config is added to suppress warnings
    model_config = ConfigDict(protected_namespaces=())
    supported_models: List[str]
    model_name: str
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
    use_rag: bool
    top_k: int
    moderationfilter: bool
    onlyusecontext: bool
    useknowledgebase: bool
    keywords: Optional[list[str]]
    filenames: Optional[list[str]]


class SystemConfig(BaseModel):
    # model_config is added to suppress warnings
    model_config = ConfigDict(protected_namespaces=())
    model_params: ModelParams
    model_auth: ModelAuth
    agent_params: AgentParams
    extraction: Extraction
    chunking: Chunking
    embedding: Embedding
    vector_db: VectorDB
    rag_params: RAGParams
