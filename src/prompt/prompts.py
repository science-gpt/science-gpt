from src.models.config import ModelConfig
from src.prompt.base_prompt import PromptComponent, PromptDecorator


class TestDecorator(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}
    Test Decorator Decorator Decorator
    """

    def __init__(
        self,
        prompt: PromptComponent,
    ) -> None:
        self._prompt = prompt

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(decorate=self.PromptTemplate)


class DefinitionsDecorator(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    The following are definitions to help you answer the specific query:
    {definitions}
    -------------
    {decorate}
    """

    def __init__(self, prompt: PromptComponent, definitions: list[str]) -> None:
        self._prompt = prompt
        self.definitions = definitions

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                definitions="\n\n".join(self.definitions), decorate="{decorate}"
            )
        )


class OnlyUseContextDecorator(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}
    Only use the information that was given to answer the question. If not enough information is given,
    answer with "I am unsure about the answer as there was not enough information given". Do not try to
    guess an answer.
    """

    def __init__(
        self,
        prompt: PromptComponent,
    ) -> None:
        self._prompt = prompt

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(decorate=self.PromptTemplate)


class ModerationDecorator(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}
    Please refuse to answer any query containing offensive, racist, homophobic, sexist, and pornographic 
    content. Do no answer with offensive, racist, homophophic, sexist, or pornographic answers. 
    """

    def __init__(
        self,
        prompt: PromptComponent,
    ) -> None:
        self._prompt = prompt

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(decorate=self.PromptTemplate)


class ExamplesDecorator(PromptDecorator):
    PromptTemplate: str = """
    The following are examples to help you answer the specific query:
    {examples}
    -------------
    {decorate}
    """

    def __init__(self, prompt: PromptComponent, question: str, examples: dict) -> None:
        self._prompt = prompt
        self.question = question
        self.examples = examples

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                examples="\n\n---\n\n"
                + "\n\n---\n\n".join(
                    [
                        "Answer the following question for the following example:\n"
                        + example
                        + f"\n\nQuestion: {self.question}"
                        + f"\nAnswer: {answer}"
                        for example, answer in self.examples
                    ]
                ),
                decorate="{decorate}",
            )
        )
