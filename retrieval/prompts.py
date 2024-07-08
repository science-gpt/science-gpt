from abc import ABC, abstractmethod

class Prompt(ABC):
    @abstractmethod
    def __call__(self):
        pass

class GenericSystemPrompt(Prompt):

    def __init__(self):
        self.system_message = """
        You are a helpful chatbot that answers questions from the perspective 
        of a regulatory toxicologist. You should answer the user's question in 
        plain and precise language based on the below context. If the context 
        doesn't contain any relevant information to the question, don't make 
        something up. Instead, just say "I don't have information on that 
        topic".

        <context>
        {context}
        </context>
        """

    def __call__(self):
        return self.system_message


