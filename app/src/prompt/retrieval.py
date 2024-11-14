import os

from langchain_community.vectorstores import Chroma
from orchestrator.config import SystemConfig
from prompt.base_prompt import PromptComponent, PromptDecorator

from data_broker.data_broker import DataBroker
from models.models import ChatModel

DEFAULT_QUERY_REWRITER: str = """
    You are an expert in simplifying scientific literature search queries for toxicology and pesticide research. 
    Your task is to rewrite verbose and detailed user queries into concise, focused search queries that retain only the most relevant scientific keywords.

    Here are some examples to guide you:

    Example 1:
    Verbose Query: What is the acceptable daily intake of glyphosate for humans?
    Simplified Query: Glyphosate acceptable daily intake humans

    Example 2:
    Verbose Query: What are the reproductive parameters evaluated in OECD TG 408?
    Simplified Query: Reproductive parameters OECD TG 408

    Example 3:
    Verbose Query: malathion and glyphosate monograph differences.
    Simplified Query: malathion and glyphosate monograph differences

    Example 4:
    Verbose Query: Tell me what studies say about aquatic ecotoxicology of triticonazole
    Simplified Query: triticonazole aquatic ecotoxicology

    Your task is to process the following query:

    {question}

    Return only the simplified query. If the query is already sufficiently concise, return it exactly as it is. 
    Do not include any additional text or labels such as "Original Query" or "Simplified Query"—only output the simplified query itself.
"""


class TestRetrieval(PromptDecorator):
    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {context}
    Test test test test
    {decorate}
    """

    def __init__(self, prompt: PromptComponent) -> None:
        self._prompt = prompt
        self.cost = self._prompt.cost

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
    Answer the question using the following context between XML tags <context></context>:
    <context>{context}</context>
    """

    def __init__(
        self,
        prompt: PromptComponent,
        config: SystemConfig,
        rewrite_model: ChatModel,
        collection="base",
    ) -> None:
        self._prompt = prompt
        self.data_broker = DataBroker()
        self.config: SystemConfig = config
        self.collection = collection
        self.rewrite_model = rewrite_model
        self.cost = self._prompt.cost

    def get_prompt(self, query: str, top_k=None) -> str:

        # Rewrite
        retrieval_query, cost = self.rewrite_model(
            DEFAULT_QUERY_REWRITER.format(question=query),
            override_config={"temperature": 0.0},
        )
        self.cost += cost
        print("retrieval_query\n", retrieval_query)

        if top_k == None:
            top_k = self.config.rag_params.top_k_retrieval
        print("Retrieval!\n", str(top_k))
        results = self.data_broker.search(
            [retrieval_query], top_k=top_k, collection=self.collection
        )

        #### If no results found, we should log and we should also tell the user that their RAG search did not return any results for some reason
        if len(results) == 0 or len(results[0]) == 0:
            # No results found; handle the case here
            # logger.warning("No documents found for the query. Returning only the query as the prompt.")
            print("no results returned...")

        print(results)
        context_text = "\n\n---\n\n".join(
            [
                f"Context Source: {chunk.id}\nDocument: {chunk.document}"
                for result in results
                for chunk in result
            ]
        )
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
