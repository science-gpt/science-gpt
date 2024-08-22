from abc import ABC, abstractmethod

class Document(ABC):
    """
    A class to represent a document.

    ...

    Attributes
    ----------
    id : str
        a unique identifier for the document
    content : str
        the content of the document
    
    Methods
    -------
    get_id():
        returns the id of the document
    get_content():
        returns the content of the document
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

class Document_Reader(ABC):
    """
    A class to represent a document reader.

    ...

    Attributes
    ----------
    location : str
        the location of the documents

    Methods
    -------
    read_documents(location):
        returns a list of documents from the location
    """
    @abstractmethod
    
    def __init__(self, location: str):
        self.location = location
    
    def read_documents(self, location: str) -> List[Document]:
        pass