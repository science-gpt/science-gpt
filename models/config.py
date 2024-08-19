from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str


@dataclass
class OpenAIModelConfig(ModelConfig):
    model_params: dict
