from pydantic.dataclasses import dataclass


@dataclass
class SystemConfig:
    model_name: str
    model_params: dict
    model_auth: dict
    extraction: dict
    chunking: dict
    embedding: dict
    vector_db: dict
    rag_params: dict
