from models.config import ModelConfig
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    CustomOpenAIChatContentFormatter,
)
from langchain_community.llms.azureml_endpoint import AzureMLEndpointApiType

from abc import ABC, abstractmethod


class ChatModel(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(self, path: str):
        pass


class AzureMLChatModel(ChatModel):
    def __init__(self, config: ModelConfig):
        super.__init__(config)

        self.model = AzureMLChatOnlineEndpoint(
            endpoint_url=self.config.url,
            endpoint_api_type=AzureMLEndpointApiType.dedicated,
            endpoint_api_key=self.config.api_key,
            content_formatter=CustomOpenAIChatContentFormatter(),
            model_kwargs=self.config.params,
        )
