from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

from prompt import base_prompt


class ChatOrchestrator:
    def __init__(self, config) -> None:
        pass
