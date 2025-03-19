from typing import AnyStr, Optional, Type
from urllib.error import HTTPError

from databroker.databroker import DataBroker
from langchain_community.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field

from .base import BaseTool


class LocalSearchArgs(BaseModel):
    query: str = Field(..., description="a search query")


class LocalSearchTool(BaseTool):
    name: str = "local_search"
    description: str = (
        "A local search engine retrieving top search results as snippets from the user's document database."
        "Input should be a search query."
    )
    args_schema: Optional[Type[BaseModel]] = LocalSearchArgs

    def _run_tool(self, query: AnyStr) -> str:

        databroker = DataBroker()

        # hard code some hyperparameters
        results = []
        results.extend(
            databroker.search(
                [query],
                top_k=5,
                collection="base",
            )
        )

        results.extend(
            databroker.search(
                [query],
                top_k=5,
                collection="user",
            )
        )

        chunks = [
            f"Context Source: {chunk.id}\nDocument: {chunk.document}"
            for result in results
            for chunk in result
        ]

        output = ""
        if len(chunks) > 0:
            output = "\n\n---\n\n".join(chunks)

        return output
