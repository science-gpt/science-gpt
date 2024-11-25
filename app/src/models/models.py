from abc import ABC, abstractmethod

import requests
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from orchestrator.config import SystemConfig
from requests.exceptions import ConnectTimeout


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
            max_tokens=self.config.model_params.max_tokens,
            temperature=self.config.model_params.temperature,
        )

    def __call__(self, query: str, override_config=None):
        with get_openai_callback() as cb:

            # for cases like query reformatting, we want to override specific
            # model parameters such as temperature. unfortunately I havent't found a nice way to do this.
            if override_config:
                old_params = {key: getattr(self.model, key) for key in override_config}
                for key, value in override_config.items():
                    setattr(self.model, key, value)

            response = self.model.invoke(query)

            # after we get the model response, reset the default model parameters
            if override_config:
                for key, value in old_params.items():
                    setattr(self.model, key, value)

            return str(response.content), cb.total_cost

    def test_connection(self):
        try:
            response, _ = self.__call__("Test connection")
            return True
        except ConnectTimeout:
            return False


class LocalAIModel(ChatModel):
    def __init__(self, config: SystemConfig):
        super().__init__(config)
        self.macbook_endpoint = self.config.model_auth.macbook_endpoint
        self.macbookmodel = {
            "model": self.config.model_name,
            # "prompt": query,
            "stream": False,
            "options": {
                "temperature": self.config.model_params.temperature,
                "seed": self.config.model_params.seed,
                "top_p": config.model_params.top_p,
                "num_ctx": self.config.model_params.num_ctx,
            },
        }

    def __call__(self, query: str, override_config=None):
        # Add the query to the request body when calling the model
        self.macbookmodel["prompt"] = query

        if override_config:
            temp_config = self.macbookmodel.copy()
            temp_config["options"].update(override_config)

            response = requests.post(
                self.macbook_endpoint, json=temp_config, stream=False
            )
        else:
            response = requests.post(
                self.macbook_endpoint, json=self.macbookmodel, stream=False
            )

        # Check if the response was successful
        if response.status_code == 200:
            response_json = response.json()  # Assuming the response is in JSON format
            # Extract the relevant content from the response
            return (
                str(response_json.get("response", "")),
                0.0,
            )  # Return the response content and a dummy cost
        else:
            # Handle error response
            print(f"Error: {response.status_code}, {response.text}")
            return "Error occurred", 0.0  # You can customize the error handling

    def test_connection(self):
        try:
            response, _ = self.__call__("Test connection")
            return True
        except ConnectTimeout:
            return False
