from langchain_community.embeddings.ollama import OllamaEmbeddings
from abc import ABC, abstractmethod

class Embedding(ABC):
    """
    A class to represent an embedding.
    
    ...
    
    Attributes
    ----------
    id : str
        a unique identifier for the embedding
    content : str
        the content of the embedding
    
    Methods
    -------
    get_id():
        returns the id of the embedding
    get_content():
        returns the content of the embedding
    """
    
    @abstractmethod
    def __call__(self, text: str, id: str):
        pass
    
    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content
    
    def get_id(self):
        return self.id
    
    def get_content(self):
        return self.content