from abc import ABC, abstractmethod

from langchain_community.callbacks import (  # without specifying the model version, flat-rate 0.002 USD per 1k input and output tokens is used
    get_openai_callback,
)
from langchain_openai import AzureChatOpenAI

from orchestrator.config import SystemConfig


class ChatModel(ABC):
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(self, path: str):
        pass


class OpenAIChatModel(ChatModel):
    def __init__(self, config: SystemConfig):
        super().__init__(config)

        self.model = AzureChatOpenAI(
            openai_api_key=self.config.model_auth["api_key"],
            openai_api_version=self.config.model_auth["version"],
            azure_endpoint=self.config.model_auth["url"],
            max_tokens=self.config.model_params["max_tokens"],
            temperature=self.config.model_params["temperature"],
        )

    def __call__(self, query: str):
        with get_openai_callback() as cb:
            response = self.model.invoke(query)
            return response, cb

    def test_connection(self):
        return self.model.test_connection()
