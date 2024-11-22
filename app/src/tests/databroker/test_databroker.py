import os
import sys
from types import SimpleNamespace

import pytest

# Jank path fix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from databroker.databroker import DataBroker


@pytest.fixture
def secrets_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../secrets.toml")
    )


@pytest.fixture
def mock_database_config():
    def _config(vector_store_type):
        if vector_store_type == "chromadb":
            return SimpleNamespace(
                embedding_model="all-mpnet-base-v2",
                chunking_method="split_sentences",
                pdf_extractor=SimpleNamespace(pdf_extract_method="pypdf2"),
                vector_store=SimpleNamespace(type="chromadb"),
                username="test_user",
                userpath=os.getcwd() + "/test_user_data",
            )
        elif vector_store_type == "milvus":
            return SimpleNamespace(
                embedding_model="all-mpnet-base-v2",
                chunking_method="split_sentences",
                pdf_extractor=SimpleNamespace(pdf_extract_method="pypdf2"),
                vector_store=SimpleNamespace(
                    type="milvus", host="localhost", port=19530
                ),
                username="test_user",
                userpath=os.getcwd() + "/test_user_data",
            )

    return _config


@pytest.fixture
def databroker_instance(mock_database_config, secrets_path, vector_store_type):
    # Clear singleton instance before each test
    DataBroker._instances = {}
    config = mock_database_config(vector_store_type)
    return DataBroker(config, secrets_path)


@pytest.mark.parametrize("vector_store_type", ["milvus", "chromadb"])
class TestDataBroker:
    def test_singleton_pattern(
        self, mock_database_config, secrets_path, vector_store_type
    ):
        """Test that DataBroker maintains singleton pattern"""
        config = mock_database_config(vector_store_type)
        broker1 = DataBroker(config, secrets_path)
        broker2 = DataBroker(config, secrets_path)
        assert broker1 is broker2

    def test_initialization(self, databroker_instance):
        """Test proper initialization of DataBroker"""
        assert databroker_instance._database_config is not None
        assert hasattr(databroker_instance, "embedder")
        assert hasattr(databroker_instance, "chunker")
        assert hasattr(databroker_instance, "extractors")
        assert hasattr(databroker_instance, "vectorstore")

    def test_clear_functionality(self, databroker_instance):
        """Test clear functionality"""
        assert databroker_instance.vectorstore["base"].get_all_ids() != []
        databroker_instance.vectorstore["base"].clear()
        assert databroker_instance.vectorstore["base"].get_all_ids() == []

    def test_search_functionality(self, databroker_instance):
        """Tests search functionality"""
        search_results = databroker_instance.search(
            queries=[""], collection="base", top_k=1
        )
        assert len(search_results) == 1
