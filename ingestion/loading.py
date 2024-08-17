from abc import ABC, abstractmethod
from langchain.document_loaders.pdf import PyPDFDirectoryLoader


class Loader(ABC):
    @abstractmethod
    def __call__(self, path: str):
        pass


class PDFLoader(Loader):
    def __call__(self, path: str):
        document_loader = PyPDFDirectoryLoader(path)
        return document_loader.load()
