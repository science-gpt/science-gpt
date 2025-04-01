import torch
from typing import Optional, List, Dict, Any, Tuple

# LLM-based layerwise reranker (e.g., BAAI/bge-reranker-v2-minicpm-layerwise)
from FlagEmbedding import LayerWiseFlagLLMReranker

# LLM-based reranker (e.g., BAAI/bge-reranker-v2-gemma)
from FlagEmbedding import FlagLLMReranker

# Normal reranker (e.g., BAAI/bge-reranker-base, BAAI/bge-reranker-large, BAAI/bge-reranker-v2-m3)
from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 8,
        normalize: bool = True,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ):
        """
        Initialize the BGE reranker with given parameters.

        Args:
            model_name: The name of the BGE reranker model to use
                - "BAAI/bge-reranker-base"       - Normal reranker (lightweight for Chinese/English)
                - "BAAI/bge-reranker-large"      - Normal reranker (larger model for Chinese/English)
                - "BAAI/bge-reranker-v2-m3"      - Normal reranker (multilingual)
                - "BAAI/bge-reranker-v2-gemma"   - LLM-based reranker (multilingual, better performance)
                - "BAAI/bge-reranker-v2-minicpm-layerwise" - LLM-based layerwise reranker (multilingual,
                                                             allows layer selection)
            batch_size: The batch size used for computation
            normalize: Whether to normalize reranking scores to the 0-1 range using sigmoid
                       If False, scores may be negative (especially for layerwise models)
            device: Optional device specification (will auto-detect if None)
            use_fp16: Whether to use 16-bit floating-point precision (will auto-configure if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self.device, self.use_fp16 = self.setup_device(device, use_fp16)
        print("Current model name: ", self.model_name)

        if "layerwise" in model_name:
            print("start to load layerwise reranker")
            self.reranker = LayerWiseFlagLLMReranker(
                model_name_or_path=self.model_name,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                normalize=self.normalize,
                trust_remote_code=True,
            )
            print("layerwise reranker loaded")
            self.is_layerwise = True
            self.is_llm_based = True
            self.cutoff_layers = [28]
            print(f"Initialized LayerWiseFlagLLMReranker with model {model_name}")

        elif "gemma" in model_name:
            self.reranker = FlagLLMReranker(
                model_name_or_path=self.model_name,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                normalize=self.normalize,
            )
            self.is_layerwise = False
            self.is_llm_based = True
            print(f"Initialized FlagLLMReranker with model {model_name}")

        else:
            self.reranker = FlagReranker(
                model_name_or_path=self.model_name,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                normalize=self.normalize,
            )
            self.is_layerwise = False
            self.is_llm_based = False
            print(f"Initialized FlagReranker with model {model_name}")

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
        pairs = [[query, doc] for doc in docs]
        reranked_results = []

        if self.is_layerwise:
            # For LLM-based layerwise reranker: BAAI/bge-reranker-v2-minicpm-layerwise
            scores = self.reranker.compute_score(
                pairs, cutoff_layers=self.cutoff_layers, normalize=self.normalize
            )
        elif self.is_llm_based:
            # For LLM-based reranker: BAAI/bge-reranker-v2-gemma
            scores = self.reranker.compute_score(pairs, normalize=self.normalize)
        else:
            # Standard reranking for normal models (bge-reranker-base / bge-reranker-large / bge-reranker-v2-m3)
            scores = self.reranker.compute_score(pairs, normalize=self.normalize)

        print("Current model is ", self.model_name, " and scores are: ", scores)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        for doc, score in scored_docs[:top_k]:
            original = doc_to_result[doc]
            reranked_results.append(
                type(original)(
                    id=original.id,
                    distance=float(score),
                    metadata=original.metadata,
                    document=original.document,
                    embedding=original.embedding,
                )
            )

        return reranked_results
