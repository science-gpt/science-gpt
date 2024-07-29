from orchestrator.config import SystemConfig
from orchestrator.utils import load_config
from retrieval import prompts
#from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

class ChatOrchestrator:
    def __init__(self) -> None:
        self.config: SystemConfig = load_config(config_name="system_config", config_dir="./configs")
    
    def triage_query(query: str) -> str:
        """
        Given a user query, the orchestrator detects user intent and leverages 
        appropriate agents to provide a response.
        """

        return "hello"


