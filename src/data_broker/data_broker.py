from abc import ABC


class DataBroker(ABC):
    """
    The interface between the client (the app) and all data
    related operations. This class abstracts away the extraction,
    chunking, embedding, storage and retrieval of text data.

    It supports two methods:

    1- Given a text source, embed and store into a vector store.
    2- Given a query, retrieve a list of relevant documents.

    A config is expected to set the embedding model, vector store,
    chunking method etc.

    """

    pass


# TODO: add the two classes above
# TODO: design a configuration propogation method
