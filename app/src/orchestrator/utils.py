import hydra
from omegaconf import DictConfig

DEFAULT_SYSTEM_PROMPT: str = """ You are a helpful chatbot that answers questions from the perspective 
    of a regulatory toxicologist. You should answer the user's question in 
    plain and precise language based on the below context. If the context 
    doesn't contain any relevant information to the question, don't make 
    something up. Instead, just say "I don't have information on that 
    topic".
    """


DEFAULT_QUERY_REWRITER: str = """
    You are an expert in search queries for scientific literature review. 
    Your task is to simplify verbose and detailed user queries into concise and effective queries for information retrieval. 
    Focus on capturing the essential scientific keywords relevant to toxicology or pesticide research.

    Here are examples of the queries that need simplification, and what the simplification should look like:

    Example 1:
    Verbose Query: What is the acceptable daily intake of glyphosate for humans?
    Is simplification needed her: Yes.
    Simplified Query: Glyphosate acceptable daily intake humans

    Example 2:
    Verbose Query: What are the reproductive parameters evaluated in OECD TG 408?
    Is simplification needed her: Yes.
    Simplified Query: Reproductive parameters OECD TG 408

    Example 3:
    Verbose Query: malathion and glyphosate monograph differences.
    Is simplification needed her: No.
    Simplified Query: malathion and glyphosate monograph differences

    Example 4:
    Verbose Query: Tell me what studies say about aquatic ecotoxicology of triticonazole
    Is simplification needed her: Yes.
    Simplified Query: triticonazole aquatic ecotoxicology

    This is the user query:
    {question}

    Do not output anything except the simplified query. 
    If the query is already simple enough just return the original user question.
    Only output the simplified query, nothing else! 
"""


def load_config(config_name: str, config_dir: str):
    """
    Loads specified config from yaml into a python Object.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir, version_base=None):
        dict_config: DictConfig = hydra.compose(config_name)
        config = hydra.utils.instantiate(dict_config, _convert_="all")

    return config
