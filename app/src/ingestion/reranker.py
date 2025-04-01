import torch
from typing import Optional, List, Dict, Any, Tuple
from pymilvus.model.reranker import BGERerankFunction

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

        # Setup device and fp16 settings
        self.device, self.use_fp16 = self.setup_device(device, use_fp16)

        if "layerwise" in model_name:
            self.reranker = LayerWiseFlagLLMReranker(
                model_name_or_path=self.model_name,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                normalize=self.normalize,
                trust_remote_code=True,
            )
            self.is_layerwise = True
            self.is_llm_based = True
            # Default cutoff layer based on model docs
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

        if self.is_layerwise:
            pairs = [[query, doc] for doc in docs]
            scores = self.reranker.compute_score(pairs, cutoff_layers=self.cutoff_layers, normalize=self.normalize)

            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Scores: {scores}")
            
            reranked_results = []
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

        
        elif self.is_llm_based:
            # For LLM-based rerankers like BAAI/bge-reranker-v2-gemma
            pairs = [[query, doc] for doc in docs]
            scores = self.reranker.compute_score(pairs, normalize=self.normalize)
            
            # Sort the documents by score
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Create reranked results
            reranked_results = []
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
        else:
            # Standard reranking for normal models (or as fallback)
            
            # First try using compute_score if available (for FlagReranker)
            pairs = [[query, doc] for doc in docs]
            scores = self.reranker.compute_score(pairs, normalize=self.normalize)
            
            # Sort the documents by score
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Create reranked results
            reranked_results = []
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
        