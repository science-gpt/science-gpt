import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from types import SimpleNamespace

from databroker.databroker import DataBroker

config = SimpleNamespace(
    embedding_model="all-mpnet-base-v2",
    chunking_method="recursive_character",
    pdf_extractor=SimpleNamespace(pdf_extract_method="pypdf2"),
    vector_store=SimpleNamespace(type="milvus"),
    username="test_user",
    userpath=os.getcwd() + "/test_user_data/",
)
secrets_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../secrets.toml")
)
db = DataBroker(config, secrets_path)
