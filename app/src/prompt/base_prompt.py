from abc import ABC, abstractmethod


class PromptComponent(ABC):
    """
    A class to represent a prompt.

    ...

    Attributes
    ----------
    PromptTemplate : str
        a prompt template

    Methods
    -------
    get_prompt(query):
        returns a prompt to be input into a LLM.
    """

    PromptTemplate = """
    {decorate}
    """

    @abstractmethod
    def get_prompt(self, query: str) -> str:
        """
        get_prompt: Query : str -> Prompt : str
        The get_prompt function takes as input the desired user query and populates
        and returns a prompt to be input into a LLM.
            Parameters:
                Query (str): The user specified query describing the task for the LLM

            Returns:
                Prompt (str): The prompt string that corresponds to the user specied
                query to feed to the LLM to complete the task
        """
        pass

    @abstractmethod
    def get_cost(self) -> float:
        pass


class PromptDecorator(PromptComponent):
    """
    A class to represent a prompt decorator (i.e. RAG).

    ...

    Attributes
    ----------
    _prompt : PromptComponent
        a prompt component to decorate
    PromptTemplate : str
        a prompt template

    Methods
    -------
    __init__(prompt):
        initialize the prompt decorator
    get_prompt(query):
        returns a prompt to be input into a LLM.
    """

    _prompt: PromptComponent = None
    PromptTemplate: str = """
    {decorate}
    """

    def __init__(self, prompt: PromptComponent) -> None:
        """
        Initialize the prompt decorator class
            Parameters:
                prompt (PromptComponent): The prompt class to decorate
        """
        self._prompt = prompt
        self.cost = self._prompt.cost
        self.chunks = self._prompt.chunks
        self.rewrite_query = self._prompt.rewrite_query

    def get_prompt(self, query: str) -> str:
        """
        get_prompt: Query : str -> Prompt : str
        The get_prompt function takes as input the desired user query and populates
        and returns a prompt to be input into a LLM.
            Parameters:
                Query (str): The user specified query describing the task for the LLM

            Returns:
                Prompt (str): The prompt string that corresponds to the user specied
                query to feed to the LLM to complete the task
        """
        # RAG -> RagString
        # format(decorate = RagString)
        print("Decorator Prompt")
        return self._prompt.get_prompt(query).format(decorate=self.PromptTemplate)

    def get_cost(self) -> float:
        return self.cost

    def get_chunks(self) -> list:
        return self.chunks

    def get_rewrite_query(self) -> str:
        return self.rewrite_query


class ConcretePrompt(PromptComponent):
    """
    The concrete class to represent a basic prompt.

    ...

    Attributes
    ----------
    PromptTemplate : str
        a prompt template

    Methods
    -------
    get_prompt(query):
        returns a prompt to be input into a LLM.
    """

    PromptTemplate: str = """
    
    {decorate}

    <query>
    {query}
    </query>
    """

    def __init__(self, system_prompt: str):
        """Overrides the prompt template if initialized with a different prompt"""
        self.PromptTemplate = system_prompt + ConcretePrompt.PromptTemplate
        self.cost = 0
        self.chunks = []
        self.rewrite_query = ""

    def get_prompt(self, query):
        print("Base Prompt")
        return self.PromptTemplate.format(query=query, decorate="{decorate}")

    def get_cost(self) -> float:
        return self.cost

    def get_chunks(self) -> list:
        return self.chunks

    def get_rewrite_query(self) -> str:
        return self.rewrite_query
