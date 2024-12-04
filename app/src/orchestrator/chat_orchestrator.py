import os
from types import SimpleNamespace

import toml
from logs.logger import logger
from orchestrator.call_handlers import LLMCallHandler
from orchestrator.config import SystemConfig
from orchestrator.utils import DEFAULT_SYSTEM_PROMPT, SingletonMeta, load_config
from prompt.base_prompt import ConcretePrompt
from prompt.prompts import ModerationDecorator, OnlyUseContextDecorator
from prompt.retrieval import ContextRetrieval
from requests.exceptions import ConnectTimeout

from models.models import LocalAIModel, OpenAIChatModel


class ChatOrchestrator(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.config: SystemConfig = load_config(
            config_name="system_config", config_dir=f"{os.getcwd()}/src/configs"
        )
        self.model = None
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def load_model(self, model: str) -> None:
        """
        Load secrets from toml file into config object.
        """
        secrets = toml.load("secrets.toml")

        if model in ["GPT-3.5", "GPT-4.0"]:
            model_key = model.lower().replace("-", "").replace(".", "") + "-api"
            self.config.model_auth.version = secrets[model_key]["api_version"]
            self.config.model_auth.api_key = secrets[model_key]["api_key"]
            self.config.model_auth.url = secrets[model_key]["azure_endpoint"]
            self.model = OpenAIChatModel(self.config)
        else:
            self.config.model_auth.macbook_endpoint = (
                secrets["localmodel"]["macbook_endpoint"] + "/api/generate"
            )
            self.model = LocalAIModel(self.config)

        self.config.model_params.model_name = model

    def test_connection(self, model_name: str) -> bool:
        """
        Test connection to the local or remote chat model.
        """

        if model_name in ["GPT-3.5", "GPT-4.0"]:
            return OpenAIChatModel(self.config).test_connection()

        return LocalAIModel(self.config).test_connection()

    def triage_query(self, query: str, model: str) -> tuple[str, str, float]:
        """
        Given a user query, the orchestrator detects user intent and leverages
        appropriate agents to provide a response.

        Returns the response text content (str) and cost (float)
        """

        self.load_model(model)
        prompt = ConcretePrompt(self.system_prompt)

        if self.config.rag_params.use_rag:
            prompt = ContextRetrieval(
                prompt,
                self.config,
                rewrite_model=self.model,
            )

        # if useknowledgebase is enabled without uploading documents, the app errors out
        if self.config.rag_params.useknowledgebase:
            prompt = ContextRetrieval(
                prompt,
                self.config,
                collection="user",
                rewrite_model=self.model,
            )

        # look for moderation filter
        if self.config.rag_params.moderationfilter:
            prompt = ModerationDecorator(prompt)

        # look for only use context
        if self.config.rag_params.onlyusecontext:
            prompt = OnlyUseContextDecorator(prompt)

        try:
            handler = LLMCallHandler(self.model, prompt, self.config)
            llm_prompt, response, cost = handler.call_llm(query)
            logger.info(
                "Prompt: "
                + llm_prompt
                + " Response: "
                + response
                + " System Config: "
                + self.config.model_dump_json()
            )

        # Carter: we will want a better solution here but we need error handling for the time being.
        # This catches errors when the local models are offline
        except ConnectTimeout:
            logger.error("Unable to connect to local model.")
            return "N/A", "The model you selected is not online.", 0.0

        return llm_prompt, response, cost

    def direct_query(self, prompt):
        """
        This is only used during direct prompt modification where the user can test a prompt
        directly without additional prompt decorators.
        """
        response, cb = self.model(prompt)
        return prompt, response, cb
