from models.models import ChatModel
from orchestrator.config import SystemConfig
from prompt.base_prompt import ConcretePrompt
from prompt.prompts import TestDecorator
from prompt.retrieval import TestRetrieval


class LLMCallHandler:
    def __init__(
        self, model: ChatModel, prompt: ConcretePrompt, config: SystemConfig
    ) -> None:
        self.config = config
        self.model = model
        self.prompt = prompt

    def get_prompt(self, query: str):
        return self.prompt.get_prompt(query)

    def call_llm(self, query: str):
        """
        Returns the LLM response and the cost of the query
        """
        return self.model(query)
