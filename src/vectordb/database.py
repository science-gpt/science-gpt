from abc import ABC, abstractmethod


class Database(ABC):
    """
    A class to represent a database.

    ...

    Attributes
    ----------
    _connection : Connection
        a connection to the database

    Methods
    -------
    __init__(connection):
        initialize the database
    get_connection():
        returns the connection to the database
    store_embeddings(embeddings):
        stores embeddings in the database
    """

    _connection: Connection = None

    @abstractmethod
    def __init__(self, connection: Connection):
        self._connection = connection

    def store_embeddings(self, embeddings: List[Embedding]):
        pass
