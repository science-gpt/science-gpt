import torch
from typing import Optional, List, Dict, Any, Tuple
from pymilvus.model.reranker import BGERerankFunction


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
        normalize: bool = True,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        Initialize the BGE reranker with given parameters.

        Args:
            model_name: The name of the BGE reranker model to use
                - "BAAI/bge-reranker-v2-m3"
                - "BAAI/bge-reranker-base"
                - "BAAI/bge-reranker-large"
                - "BAAI/bge-re-anchor-v2-gemma"
                - "BAAI/bge-reranker-v2-minicpm-layerwise"
            batch_size: The batch size used for computation
            normalize: Whether to normalize reranking scores
            device: Optional device specification (will auto-detect if None)
            use_fp16: Whether to use 16-bit floating-point precision (will auto-configure if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        # Setup device and fp16 settings
        self.device, self.use_fp16 = self.setup_device(device, use_fp16)

        # Initialize the BGE reranker
        self.reranker = BGERerankFunction(
            model_name=self.model_name,
            use_fp16=self.use_fp16,
            batch_size=self.batch_size,
            normalize=self.normalize,
            device=self.device,
        )

    def setup_device(
        self, device: Optional[str] = None, use_fp16: Optional[bool] = None
    ) -> Tuple[str, bool]:
        """
        Setup the computation device and fp16 settings.

        Args:
            device: Optional device specification
            use_fp16: Optional fp16 setting

        Returns:
            Tuple of (device_string, use_fp16_boolean)
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if use_fp16 is None:
            use_fp16 = "cpu" not in device
        else:
            if "cpu" in device and use_fp16:
                use_fp16 = False

        return device, use_fp16

    def rerank(self, query: str, docs: List[str], top_k: int = 10) -> List[str]:
        """
        Rerank documents using the BGE reranker.

        Args:
            query: The query string
            docs: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents
        """
        return self.reranker(query, docs, top_k)

    def rerank_milvus_results(
        self, query: str, hits: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank Milvus search results.

        Args:
            query: Original query string
            hits: List of hit results from Milvus search
            top_k: Number of top results to return

        Returns:
            Reranked list of hits with updated distances based on reranker scores
        """
        if not hits:
            return []

        # Extract documents from hits
        docs = []
        doc_to_hit_map = {}

        for hit in hits:
            doc = hit["entity"].get("text", "")
            docs.append(doc)
            doc_to_hit_map[doc] = hit

        if not docs:
            return []

        # BGERerankFunction returns a list of objects with text and score attributes
        reranked_results = self.reranker(query, docs, min(len(docs), top_k))

        reranked_hits = []
        for result in reranked_results:
            # Access the text and score attributes directly
            doc = result.text
            score = result.score

            if doc in doc_to_hit_map:
                original_hit = doc_to_hit_map[doc].copy()
                original_hit["distance"] = float(score)
                reranked_hits.append(original_hit)

        return reranked_hits
