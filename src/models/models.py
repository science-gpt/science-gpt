from abc import ABC, abstractmethod

from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI

from logs.logger import logger
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
            openai_api_key=self.config.model_auth.api_key,
            openai_api_version=self.config.model_auth.version,
            azure_endpoint=self.config.model_auth.url,
            seed=self.config.model_params.seed,
            max_tokens=self.config.model_params.max_tokens,
            temperature=self.config.model_params.temperature,
            top_p=self.config.model_params.top_p,
        )

    def __call__(self, query: str):
        with get_openai_callback() as cb:
            response = self.model.invoke(query)
            return response, cb

    def test_connection(self):
        # TODO: implement test connection method
        return True
