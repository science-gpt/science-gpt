from typing import List, Optional

from databroker.databroker import DataBroker
from logs.logger import logger
from orchestrator.config import SystemConfig
from prompt.base_prompt import PromptComponent, PromptDecorator

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
        self.chunks = self._prompt.chunks
        self.rewrite_query = self._prompt.rewrite_query

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
        hybrid_weight=0.5,
    ) -> None:
        self._prompt = prompt
        self.databroker = DataBroker()
        self.config: SystemConfig = config
        self.collection = collection
        self.rewrite_model = rewrite_model
        self.cost = self._prompt.cost
        self.chunks = self._prompt.chunks
        self.rewrite_query = self._prompt.rewrite_query
        self.hybrid_weight = (hybrid_weight,)

    def get_prompt(self, query: str) -> str:

        # Rewrite
        retrieval_query, cost = self.rewrite_model(
            DEFAULT_QUERY_REWRITER.format(question=query),
            override_config={"temperature": 0.0},
        )
        self.rewrite_query = retrieval_query
        self.cost += cost
        print("Query was rewritten. The retrieval query is:\n", retrieval_query)
        logger.info(
            "Retrieval query",
            xtra={
                "custom_dimensions": {
                    "retrieval_query": retrieval_query,
                    "original_query": query,
                }
            },
        )

        results = self.databroker.search(
            [retrieval_query],
            top_k=self.config.rag_params.top_k,
            collection=self.collection,
            keywords=self.config.rag_params.keywords,
            filenames=self.config.rag_params.filenames,
            hybrid_weighting=self.config.rag_params.hybrid_weight,
            reranker_model=self.config.rag_params.reranker_model,
            use_reranker=self.config.rag_params.use_reranker,
        )

        # No results were returned.
        if len(results) == 0 or len(results[0]) == 0:
            return "No results found for the query. Please relay that no documents were retrieved for the given query."

        self.chunks = [
            f"Context Source: {chunk.id}\nDocument: {chunk.document}\nDistance: {chunk.distance}"
            for result in results
            for chunk in result
        ]

        context_text = "\n\n---\n\n".join(self.chunks)

        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context=context_text, decorate="{decorate}"
            )
        )
