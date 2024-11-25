import os

import requests
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

    def load_model(self, model: str = "gpt-3.5"):
        """
        Load secrets from toml file into config object.
        """
        secrets = toml.load("secrets.toml")

        # TODO: dynamically select model secrets based on 'model' str input
        # hardcoded to use gpt3.5 for now
        if model == "GPT-4.0":
            self.config.model_auth.version = secrets["gpt40-api"]["api_version"]
            self.config.model_auth.api_key = secrets["gpt40-api"]["api_key"]
            self.config.model_auth.url = secrets["gpt40-api"]["azure_endpoint"]
            self.config.model_name = model
            self.model = OpenAIChatModel(self.config)
        elif model == "GPT-3.5":  # Defaults to GPT-3.5
            self.config.model_auth.version = secrets["gpt35-api"]["api_version"]
            self.config.model_auth.api_key = secrets["gpt35-api"]["api_key"]
            self.config.model_auth.url = secrets["gpt35-api"]["azure_endpoint"]
            self.config.model_name = model
            self.model = OpenAIChatModel(self.config)
        else:
            self.config.model_auth.macbook_endpoint = (
                secrets["localmodel"]["macbook_endpoint"] + "/api/generate"
            )
            self.config.model_name = model
            self.model = LocalAIModel(self.config)

    def set_model_config(self, query_config):
        self.config.model_params.seed = query_config.seed
        self.config.model_params.temperature = query_config.temperature
        # self.config.model_params.max_tokens = query_config.max_tokens
        self.config.rag_params.top_k_retrieval = query_config.top_k
        self.config.model_params.top_p = query_config.top_p

    def test_connection(self, local=False):
        """
        Test connection to the local or remote chat model.
        """

        if local:
            model = LocalAIModel(self.config)
            response = model.test_connection()
        else:
            response = self.llm.test_connection()

        return response

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt

    def triage_query(
        self,
        model: str,
        query: str,
        query_config,
    ) -> tuple[str, float]:
        """
        Given a user query, the orchestrator detects user intent and leverages
        appropriate agents to provide a response.

        Returns the response text content (str) and cost (float)
        """

        # Set the model config and load the model
        self.set_model_config(query_config)
        self.load_model(model)
        logger.info(self.config.model_dump_json())

        prompt = ConcretePrompt(self.system_prompt)

        # involving user input.
        if query.lower().startswith("search:") or use_rag:
            query = query[7:] if query.lower().startswith("search:") else query
            prompt = ContextRetrieval(
                prompt,
                self.config,
                keyword_filter=query_config.keywords,
                rewrite_model=self.model,
            )
        # we want to avoid the case of wrapping the prompt in two ContextRetrival decorators.
        # note - if use_rag and useknowledgebase are on at the same time the app will not work.
        if query_config.useknowledgebase:
            prompt = ContextRetrieval(
                prompt,
                self.config,
                collection="user",
                keyword_filter=query_config.keywords,
                rewrite_model=self.model,
            )

        # look for moderation filter
        if query_config.moderationfilter:
            prompt = ModerationDecorator(prompt)

        # look for only use context
        if query_config.onlyusecontext:
            prompt = OnlyUseContextDecorator(prompt)

        try:
            handler = LLMCallHandler(self.model, prompt, self.config)
            llm_prompt, response, cost = handler.call_llm(query)

        # Carter: we will want a better solution here but we need error handling for the time being.
        # This catches errors when the local models are offline
        except ConnectTimeout:
            logger.error("Unable to connect to local model.")
            return "N/A", "The model you selected is not online.", 0.0

        return llm_prompt, response, cost
