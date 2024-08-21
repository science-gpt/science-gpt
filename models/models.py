from abc import ABC, abstractmethod

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
        return self.model.invoke(query)
