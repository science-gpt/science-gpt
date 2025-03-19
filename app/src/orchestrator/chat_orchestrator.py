import os
from typing import List, Optional

import toml
from logs.logger import logger
from orchestrator.call_handlers import AgentCallHandler, LLMCallHandler
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
        self.model_key = "gpt40-api"
        self.secrets = secrets
        if model in ["GPT-3.5", "GPT-4.0"]:
            self.model_key = model.lower().replace("-", "").replace(".", "") + "-api"
            self.config.model_auth.version = secrets[self.model_key]["api_version"]
            self.config.model_auth.api_key = secrets[self.model_key]["api_key"]
            self.config.model_auth.url = secrets[self.model_key]["azure_endpoint"]
            self.model = OpenAIChatModel(self.config)
        else:
            self.config.model_auth.macbook_endpoint = (
                secrets["localmodel"]["macbook_endpoint"] + "/api/generate/"
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

    def triage_query(
        self, query: str, model: str
    ) -> tuple[str, str, float, list[str], str]:
        """
        Given a user query, the orchestrator detects user intent and leverages
        appropriate agents to provide a response.

        Returns the response text content (str) and cost (float)
        """

        chunks = []
        rewriten_query = ""
        self.load_model(model)
        prompt = ConcretePrompt(self.system_prompt)

        if self.config.rag_params.use_rag:
            prompt = ContextRetrieval(
                prompt,
                self.config,
                rewrite_model=self.model,
            )

        # WARNING: if useknowledgebase is enabled without uploading documents, the app errors out
        if self.config.rag_params.useknowledgebase:
            prompt = ContextRetrieval(
                prompt,
                self.config,
                collection="user",
                rewrite_model=self.model,
            )

        if self.config.rag_params.moderationfilter:
            prompt = ModerationDecorator(prompt)

        if self.config.rag_params.onlyusecontext:
            prompt = OnlyUseContextDecorator(prompt)

        try:
            if self.config.agent_params.enable:
                handler = AgentCallHandler(
                    self.model_key, self.secrets, prompt, self.config
                )
            else:
                handler = LLMCallHandler(self.model, prompt, self.config)

            llm_prompt, response, cost = handler.call_llm(query)
            chunks = prompt.get_chunks()

            filtered_config = self.config.model_dump(
                exclude={  # hides all the options. only shows you what you're using
                    "extraction": {"supported_extractors"},
                    "vector_db": {"supported_databases"},
                    "chunking": {"supported_chunkers"},
                    "embedding": {"supported_embedders"},
                    "model_auth": {"api_key", "macbook_endpoint"},
                    "model_params": {"supported_models"},
                }
            )

            rewriten_query = prompt.get_rewrite_query()
            logger.info(
                "LLM Call",
                configs=filtered_config,
                xtra={"prompt": llm_prompt, "response": response},
            )

            print("Model Params", filtered_config)
            print(type(filtered_config))

        # Carter: we will want a better solution here but we need error handling for the time being.
        # This catches errors when the local models are offline
        except ConnectTimeout:
            logger.error("Unable to connect to local model.")
            return "N/A", "The model you selected is not online.", 0.0

        return llm_prompt, response, cost, chunks, rewriten_query

    def direct_query(self, prompt):
        """
        This is only used during direct prompt modification where the user can test a prompt
        directly without additional prompt decorators.
        """
        response, cb = self.model(prompt)
        return prompt, response, cb
