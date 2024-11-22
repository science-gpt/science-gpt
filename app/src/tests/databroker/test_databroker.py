import os
import sys

# Jank path fix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from databroker.databroker import DataBroker


@pytest.fixture
def secrets_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../secrets.toml")
    )


@pytest.fixture
def mock_database_config_chromadb():
    return SimpleNamespace(
        embedding_model="all-mpnet-base-v2",
        chunking_method="split_sentences",
        pdf_extractor=SimpleNamespace(pdf_extract_method="pypdf2"),
        vector_store=SimpleNamespace(type="chromadb"),
        username="test_user",
        userpath=os.getcwd() + "/test_user_data",
    )


@pytest.fixture
def mock_database_config_milvus():
    return SimpleNamespace(
        embedding_model="all-mpnet-base-v2",
        chunking_method="split_sentences",
        pdf_extractor=SimpleNamespace(pdf_extract_method="pypdf2"),
        vector_store=SimpleNamespace(type="milvus", host="localhost", port=19530),
        username="test_user",
        userpath=os.getcwd() + "/test_user_data",
    )


@pytest.fixture
def databroker_chromadb(mock_database_config_chromadb, secrets_path):
    # Clear singleton instance before each test
    DataBroker._instances = {}
    broker = DataBroker(mock_database_config_chromadb, secrets_path)
    return broker


@pytest.fixture
def databroker_milvus(mock_database_config_milvus, secrets_path):
    # Clear singleton instance before each test
    DataBroker._instances = {}
    broker = DataBroker(mock_database_config_milvus, secrets_path)
    return broker


class TestDataBrokerChromaDB:
    def test_singleton_pattern(self, mock_database_config_chromadb, secrets_path):
        """Test that DataBroker maintains singleton pattern"""
        broker1 = DataBroker(mock_database_config_chromadb, secrets_path)
        broker2 = DataBroker(mock_database_config_chromadb, secrets_path)
        assert broker1 is broker2

    def test_initialization(self, databroker_chromadb):
        """Test proper initialization of DataBroker"""
        assert databroker_chromadb._database_config is not None
        assert hasattr(databroker_chromadb, "embedder")
        assert hasattr(databroker_chromadb, "chunker")
        assert hasattr(databroker_chromadb, "extractors")
        assert hasattr(databroker_chromadb, "vectorstore")


class TestDataBrokerMilvus:
    def test_singleton_pattern(self, mock_database_config_milvus, secrets_path):
        """Test that DataBroker maintains singleton pattern"""
        broker1 = DataBroker(mock_database_config_milvus, secrets_path)
        broker2 = DataBroker(mock_database_config_milvus, secrets_path)
        assert broker1 is broker2

    def test_initialization(self, databroker_milvus):
        """Test proper initialization of DataBroker"""
        assert databroker_milvus._database_config is not None
        assert hasattr(databroker_milvus, "embedder")
        assert hasattr(databroker_milvus, "chunker")
        assert hasattr(databroker_milvus, "extractors")
        assert hasattr(databroker_milvus, "vectorstore")
