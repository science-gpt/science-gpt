from typing import Any, Dict, List, Optional, Tuple

import torch
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

    def rerank(self, query: str, results: List[Any], top_k: int = 10) -> List[Any]:
        """
        Rerank SearchResult objects.

        Args:
            query: The search query string
            results: List of SearchResult objects to rerank
            top_k: Number of top results to return

        Returns:
            List of reranked SearchResult objects
        """
        if not results:
            return []

        docs = [result.document for result in results]
        doc_to_result = {result.document: result for result in results}

        reranked_docs = self.reranker(query, docs, min(top_k, len(docs)))

        reranked_results = []
        for doc in reranked_docs:
            original = doc_to_result[doc.text]
            reranked_results.append(
                type(original)(
                    id=original.id,
                    distance=float(doc.score),
                    metadata=original.metadata,
                    document=original.document,
                    embedding=original.embedding,
                )
            )

        return reranked_results
