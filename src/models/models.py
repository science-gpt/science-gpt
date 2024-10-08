from abc import ABC, abstractmethod

from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI

from orchestrator.config import SystemConfig
import requests


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
            print(response)
            return response, cb

    def test_connection(self):
        # TODO: implement test connection method
        return True


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

    def __call__(self, query: str):
        # Add the query to the request body when calling the model
        self.macbookmodel["prompt"] = query
        response = requests.post(
            self.macbook_endpoint, json=self.macbookmodel, stream=False
        )
        print(response)
        print(response.json())
        #    return response
        # Check if the response was successful
        if response.status_code == 200:
            response_json = response.json()  # Assuming the response is in JSON format
            # Extract the relevant content from the response
            return response_json, 0.0  # Return the response content and a dummy cost
        else:
            # Handle error response
            print(f"Error: {response.status_code}, {response.text}")
            return "Error occurred", 0.0  # You can customize the error handling

    def test_connection(self):
        # TODO: implement test connection method
        return True
