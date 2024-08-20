from pydantic.dataclasses import dataclass


@dataclass
class SystemConfig:
    model_name: str
    model_params: dict
    model_auth: dict
    vectorstore: str
    rag_params: dict
