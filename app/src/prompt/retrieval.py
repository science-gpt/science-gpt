import os

from langchain_community.vectorstores import Chroma
from orchestrator.config import SystemConfig
from prompt.base_prompt import PromptComponent, PromptDecorator

from data_broker.data_broker import DataBroker


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

    def __init__(self, prompt: PromptComponent, config: SystemConfig) -> None:
        self._prompt = prompt
        self.data_broker = DataBroker()
        self.config: SystemConfig = config

    def get_prompt(self, query: str, top_k=None) -> str:
        if top_k == None:
            top_k = self.config.rag_params.top_k_retrieval
        print("Retrieval!\n", str(top_k))
        results = self.data_broker.search([query], top_k=top_k)

        #### If no results found, we should log and we should also tell the user that their RAG search did not return any results for some reason
        if not results or len(results[0]) == 0:
            # No results found; handle the case here
            # logger.warning("No documents found for the query. Returning only the query as the prompt.")
            print(
                "no results returned... probably something wrong with the DB not existing but trying to be queried"
            )
            return query  # Return the query itself if no context is found

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
        config: SystemConfig,
        meta: str,
        filters: list[str],
    ) -> None:
        self._prompt = prompt
        self.data_broker = DataBroker()
        self.meta = meta
        self.filters = filters
        self.config: SystemConfig = config

    def get_prompt(self, query: str, top_k=None) -> str:
        if top_k == None:
            top_k = self.config.rag_params.top_k_retrieval
        k = max(2, top_k // len(self.filters))
        context_text = ""
        for f in self.filters:
            # Add filtering to search!
            results = self.search(query, filter={self.meta: f}, k=k)
            f_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            context_text += f_text

        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context=context_text, decorate="{decorate}"
            )
        )
