import os

import toml

from models.models import OpenAIChatModel
from orchestrator.config import SystemConfig
from orchestrator.utils import load_config


class ChatOrchestrator:
    def __init__(self) -> None:
        self.config: SystemConfig = load_config(
            config_name="system_config", config_dir=f"{os.getcwd()}/src/configs"
        )
        self.load_secrets()

        self.llm = OpenAIChatModel(self.config)

    def load_secrets(self, model: str = "gpt-3.5"):
        """
        Load secrets from toml file into config object.
        """
        secrets = toml.load("secrets.toml")

        # hardcoded to use gpt3.5 for now
        self.config.model_auth["version"] = secrets["gpt35-api"]["api_version"]
        self.config.model_auth["api_key"] = secrets["gpt35-api"]["api_key"]
        self.config.model_auth["url"] = secrets["gpt35-api"]["azure_endpoint"]

    # TODO - Implemement test connection method
    def test_connection(self, local=False):
        """
        Test connection to the local or remote chat model.
        """

        if local:
            response = "Local models are not yet supported."
        else:
            response = self.llm.test_connection()

        return response

    def triage_query(self, query: str, chat_history=None) -> str:
        """
        Given a user query, the orchestrator detects user intent and leverages
        appropriate agents to provide a response.
        """

        response, cb = self.llm(query)

        return response.content, cb.total_cost
