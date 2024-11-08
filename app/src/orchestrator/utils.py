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
    You are an expert in simplifying scientific literature search queries for toxicology and pesticide research. 
    Your task is to rewrite verbose and detailed user queries into concise, focused search queries that retain only the most relevant scientific keywords.

    Here are some examples to guide you:

    Example 1:
    Verbose Query: What is the acceptable daily intake of glyphosate for humans?
    Simplified Query: Glyphosate acceptable daily intake humans

    Example 2:
    Verbose Query: What are the reproductive parameters evaluated in OECD TG 408?
    Simplified Query: Reproductive parameters OECD TG 408

    Example 3:
    Verbose Query: malathion and glyphosate monograph differences.
    Simplified Query: malathion and glyphosate monograph differences

    Example 4:
    Verbose Query: Tell me what studies say about aquatic ecotoxicology of triticonazole
    Simplified Query: triticonazole aquatic ecotoxicology

    Your task is to process the following query:

    {question}

    Return only the simplified query. If the query is already sufficiently concise, return it exactly as it is. 
    Do not include any additional text or labels such as "Original Query" or "Simplified Query"—only output the simplified query itself.
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
