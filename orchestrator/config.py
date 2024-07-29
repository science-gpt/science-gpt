from pydantic.dataclasses import dataclass

# @dataclass
# class ModelConfig:
#     model_name: str


# @dataclass
# class OpenAIModelConfig(ModelConfig):
#     model_params: dict

@dataclass
class SystemConfig:
    model_name: str
    model_params: dict
    model_auth: dict
    vectorstore: str
    rag_params: dict