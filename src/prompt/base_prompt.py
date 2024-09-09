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
    def get_prompt(self, query: str, **kwargs) -> str:
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

    def get_prompt(self, query: str, **kwargs) -> str:
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
        return self._prompt.get_prompt(query, **kwargs).format(
            decorate=self.PromptTemplate
        )


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
    You are a helpful chatbot that answers questions from the perspective 
    of a regulatory toxicologist. You should answer the user's question in 
    plain and precise language based on the below context. If the context 
    doesn't contain any relevant information to the question, don't make 
    something up. Instead, just say "I don't have information on that 
    topic".

    {decorate}

    <query>
    {query}
    </query>
    """

    def get_prompt(self, query, **kwargs):
        return self.PromptTemplate.format(query=query, decorate="{decorate}")
