from data_broker.data_broker import DataBroker
from langchain_community.vectorstores import Chroma
from orchestrator.config import SystemConfig
from prompt.base_prompt import PromptComponent, PromptDecorator


class TestRetrieval(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {context}
    Test test test test
    {decorate}
    """

    def __init__(self, prompt: PromptComponent) -> None:
        self._prompt = prompt

    def get_prompt(self, query: str) -> str:
        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context="context \n context \n context", decorate="{decorate}"
            )
        )


class ContextRetrieval(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}
    
    Use the following context to answer the question:
    {context}
    """

    def __init__(self, prompt: PromptComponent) -> None:
        self._prompt = prompt
        self.data_broker = DataBroker()

    def get_prompt(self, query: str) -> str:
        print("Retrieval!\n")
        results = self.data_broker.search([query])
        print(results)
        context_text = "\n\n---\n\n".join([res.document for res in results[0]])
        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context=context_text, decorate="{decorate}"
            )
        )


# TODO: I'm broken PLEASE FIX LATER
class FilteredContextRetrieval(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}

    Use the following context to answer the question:
    {examples}
    """

    def __init__(
        self,
        prompt: PromptComponent,
        meta: str,
        filters: list[str],
    ) -> None:
        self._prompt = prompt
        self.data_broker = DataBroker()
        self.meta = meta
        self.filters = filters

    def get_prompt(self, query: str) -> str:
        k = max(2, self.config.rag_params.top_k_retrieval // len(self.filters))
        context_text = ""
        for f in self.filters:
            results = self.search(query, filter={self.meta: f}, k=k)
            f_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            context_text += f_text

        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context=context_text, decorate="{decorate}"
            )
        )
