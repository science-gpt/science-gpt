import os

import toml

from models.models import OpenAIChatModel
from orchestrator.call_handlers import LLMCallHandler
from orchestrator.config import SystemConfig
from orchestrator.utils import load_config
from prompt.base_prompt import ConcretePrompt
from prompt.prompts import ModerationDecorator, OnlyUseContextDecorator
from prompt.retrieval import ContextRetrieval

DEFAULT_SYSTEM_PROMPT: str = """ You are a helpful chatbot that answers questions from the perspective 
    of a regulatory toxicologist. You should answer the user's question in 
    plain and precise language based on the below context. If the context 
    doesn't contain any relevant information to the question, don't make 
    something up. Instead, just say "I don't have information on that 
    topic".
    """


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ChatOrchestrator(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.config: SystemConfig = load_config(
            config_name="system_config", config_dir=f"{os.getcwd()}/src/configs"
        )

        self.load_secrets()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def load_secrets(self, model: str = "gpt-3.5"):
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
        else:  # Defaults to GPT-3.5
            self.config.model_auth.version = secrets["gpt35-api"]["api_version"]
            self.config.model_auth.api_key = secrets["gpt35-api"]["api_key"]
            self.config.model_auth.url = secrets["gpt35-api"]["azure_endpoint"]

    def set_model_config(self, query_config):
        self.config.model_params.seed = query_config.seed
        self.config.model_params.temperature = query_config.temperature
        # self.config.model_params.max_tokens = query_config.max_tokens
        self.config.rag_params.top_k_retrieval = query_config.top_k

    def test_connection(self, local=False):
        """
        Test connection to the local or remote chat model.
        """

        if local:
            response = "Local models are not yet supported."
        else:
            response = self.llm.test_connection()

        return response

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt

    def triage_query(
        self, query: str, query_config, chat_history=None
    ) -> tuple[str, float]:
        """
        Given a user query, the orchestrator detects user intent and leverages
        appropriate agents to provide a response.

        Returns the response text content (str) and cost (float)
        """

        print(query_config)

        self.set_model_config(query_config)

        # Basic use case
        model = OpenAIChatModel(self.config)
        prompt = ConcretePrompt(self.system_prompt)

        # Retrieval use case
        # TODO: This is clunky - ideally we would have a LLM detect the intent for use cases
        #  involving user input.
        if query[:7].lower() == "search:":
            query = query[7:]
            prompt = ContextRetrieval(prompt, self.config)

        # look for moderation filter
        if query_config.moderationfilter:
            prompt = ModerationDecorator(prompt)

        # look for only use context
        if query_config.onlyusecontext:
            prompt = OnlyUseContextDecorator(prompt)

        handler = LLMCallHandler(model, prompt, self.config)

        response, cb = handler.call_llm(query)

        return response.content, cb.total_cost
