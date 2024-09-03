from abc import ABC, abstractmethod


class TextStorage(ABC):
    """
    A class to represent a text storage.

    ...
    Attributes
    ----------
    text : str
        the text to store

    Methods
    -------
    store_text(text):
        stores the text
    """

    @abstractmethod
    def __init__(self, text: str):
        self.text = text

    def store_text(self, text: str):
        pass
