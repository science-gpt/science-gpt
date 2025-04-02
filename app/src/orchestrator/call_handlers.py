from models.models import ChatModel
from orchestrator.config import SystemConfig
from prompt.base_prompt import PromptComponent
from reasoning.agents import (
    GoogleSearchTool,
    LLMTool,
    LocalSearchTool,
    RewooAgent,
    WikipediaTool,
)
from reasoning.llms import AzureChatOpenAI


class LLMCallHandler:
    def __init__(
        self, model: ChatModel, prompt: PromptComponent, config: SystemConfig
    ) -> None:
        self.config = config
        self.model = model
        self.prompt = prompt

    def get_prompt(self, query: str):
        return self.prompt.get_prompt(query), self.prompt.get_cost()

    def call_llm(self, query: str):
        """
        Returns the LLM response and the cost of the query
        """
        prompt, prompt_cost = self.get_prompt(query)
        print("-----The Prompt-----")
        print(prompt)
        print("--------------------")
        response, cost = self.model(prompt)
        return prompt, response, cost + prompt_cost


# TODO: Fix this class up - need to refactor all of models
class AgentCallHandler:
    def __init__(
        self, model_key, secrets, prompt: PromptComponent, config: SystemConfig
    ) -> None:
        self.config = config
        self.prompt = prompt

        # Hard coded model right now...
        llm = AzureChatOpenAI(
            api_key=secrets[model_key]["api_key"],
            api_version=secrets[model_key]["api_version"],
            azure_deployment="gpt-4o-mini",
            azure_endpoint=secrets[model_key]["azure_endpoint"],
        )

        plugins = [
            GoogleSearchTool(),
            WikipediaTool(),
            LLMTool(llm=llm),
            LocalSearchTool(),
        ]

        self.model = RewooAgent(planner_llm=llm, solver_llm=llm, plugins=plugins)

    def get_prompt(self, query: str):
        return self.prompt.get_prompt(query), self.prompt.get_cost()

    def call_llm(self, query: str):
        """
        Returns the LLM response and the cost of the query
        """
        prompt, prompt_cost = self.get_prompt(query)
        print("-----The Prompt-----")
        print(prompt)
        print("--------------------")
        response = self.model(prompt)
        response, cost = response.content, response.total_cost
        return prompt, response, cost + prompt_cost
