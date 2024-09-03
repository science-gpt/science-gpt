from pydantic.dataclasses import dataclass


@dataclass
class SystemConfig:
    model_name: str
    model_params: dict
    model_auth: dict
    vectorstore: str
    chroma_path: str
    search_strategy: str
    rag_params: dict
