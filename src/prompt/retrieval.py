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

    def __init__(
        self, prompt: PromptComponent, config: SystemConfig, embedding_function
    ) -> None:
        self._prompt = prompt
        self.emb_fn = embedding_function
        self.config = config

        # set db
        if config.vectorstore == "chromadb":
            self.db = Chroma(
                persist_directory=config.chroma_path,
                embedding_function=embedding_function,
            )

        # set search
        if self.config.search_strategy == "similarity":
            self.search = self.db.similarity_search_with_score

    def get_prompt(self, query: str) -> str:
        # Change based on search strategy
        results = self.search(query, k=self.config.rag_params.top_k_retrieval)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return self._prompt.get_prompt(query).format(
            decorate=self.PromptTemplate.format(
                context=context_text, decorate="{decorate}"
            )
        )


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
        embedding_function,
    ) -> None:
        self._prompt = prompt
        self.emb_fn = embedding_function
        self.config = config
        self.meta = meta
        self.filters = filters

        # set db
        if config.vectorstore == "chromadb":
            self.db = Chroma(
                persist_directory=config.chroma_path,
                embedding_function=embedding_function,
            )

        # set search
        if self.config.search_strategy == "similarity":
            self.search = self.db.similarity_search_with_score

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
