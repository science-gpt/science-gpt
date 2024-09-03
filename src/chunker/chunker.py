from abc import ABC, abstractmethod


class Chunk(ABC):
    """
    A class to represent a chunk of text.

    ...

    Attributes
    ----------
    id : str
        a unique identifier for the chunk
    content : str
        the content of the chunk

    Methods
    -------
    get_id():
        returns the id of the chunk
    get_content():
        returns the content of the chunk
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


class Chunker(ABC):
    """
    A class to represent a chunker.

    ...

    Methods
    -------
    chunk(text_storage):
        returns a list of chunks from the text storage
    """

    @abstractmethod
    def chunk(self, text_storage: TextStorage) -> List[Chunk]:
        pass


class ChunkWithMetadata(ABC):
    """
    A class to represent a chunk of text with metadata.

    ...

    Attributes
    ----------
    id : str
        a unique identifier for the chunk
    content : str
        the content of the chunk
    metadata : Dict
        the metadata of the chunk

    Methods
    -------
    get_id():
        returns the id of the chunk
    get_content():
        returns the content of the chunk
    get_metadata():
        returns the metadata of the chunk
    """

    @abstractmethod
    def __call__(self, text: str, id: str, metadata: Dict):
        pass

    def __init__(self, id: str, content: str, metadata: Dict):
        self.id = id
        self.content = content
        self.metadata = metadata

    def get_id(self):
        return self.id

    def get_content(self):
        return self.content

    def get_metadata(self):
        return self.metadata


class MetadataAppender(ABC):
    """
    A class to represent a metadata appender.

    ...

    Methods
    -------
    append_metadata(chunks):
        appends metadata to the chunks
    """

    @abstractmethod
    def append_metadata(self, chunks: List[Chunk]) -> List[ChunkWithMetadata]:
        pass
