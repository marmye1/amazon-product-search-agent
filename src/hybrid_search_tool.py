"""混合检索工具：BM25 + 向量召回 + RRF。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Union

from .embedding_client import EmbeddingClient, EmbeddingError
from .fuse_rrf import fuse_rrf
from .models import ContractError, SearchRequest
from .opensearch_client import BackendError
from .retrieve_bm25 import retrieve_bm25
from .retrieve_vector import retrieve_vector
from .search_config import config_value
from .hybrid_models import HybridSearchRequest, HybridSearchResponse


@dataclass(frozen=True)
class HybridSearchTool:
    """Agent 使用的唯一混合检索工具，不在工具内调用 LLM。"""

    client: Any
    search_config: Mapping[str, Any]
    embedding_client: EmbeddingClient
    name: str = "hybrid_search"
    is_hybrid: bool = True

    def _request(self, request: Union[SearchRequest, HybridSearchRequest]) -> HybridSearchRequest:
        if isinstance(request, HybridSearchRequest):
            return request
        if isinstance(request, SearchRequest):
            return HybridSearchRequest.from_search_request(
                request,
                bm25_k=int(config_value(self.search_config, "retrieval", "bm25_k", 50)),
                vector_k=int(config_value(self.search_config, "retrieval", "vector_k", 50)),
            )
        raise ContractError("invalid_tool_input", "hybrid_search 工具输入必须是 HybridSearchRequest")

    def invoke(self, request: Union[SearchRequest, HybridSearchRequest]) -> HybridSearchResponse:
        hybrid_request = self._request(request)
        bm25_candidates = retrieve_bm25(hybrid_request, self.client, self.search_config)
        warnings: List[str] = []
        try:
            vector_candidates = retrieve_vector(
                hybrid_request,
                self.client,
                self.embedding_client,
                self.search_config,
            )
        except (EmbeddingError, BackendError, ContractError, ValueError) as exc:
            fallback = bool(config_value(self.search_config, "retrieval", "fallback_on_vector_error", True))
            if not fallback:
                raise BackendError("vector_retrieval_failed", str(exc)) from exc
            warnings.append("vector_fallback_bm25:%s" % getattr(exc, "code", "vector_retrieval_failed"))
            if not bm25_candidates:
                warnings.append("no_results")
            return HybridSearchResponse(
                query=hybrid_request.query,
                results=bm25_candidates[: hybrid_request.top_k],
                total=len(bm25_candidates[: hybrid_request.top_k]),
                retrieval_method="bm25",
                retrieval_channels={item.product_id: ["bm25"] for item in bm25_candidates[: hybrid_request.top_k]},
                fusion_config={
                    "fallback": "bm25",
                    "bm25_k": hybrid_request.bm25_k,
                    "vector_k": hybrid_request.vector_k,
                },
                query_embedding_model=self.embedding_client.model,
                warnings=warnings,
            )

        rrf_k = int(config_value(self.search_config, "retrieval", "rrf_k", 60))
        results, channels = fuse_rrf(
            bm25_candidates,
            vector_candidates,
            top_k=hybrid_request.top_k,
            rrf_k=rrf_k,
        )
        if not results:
            warnings.append("no_results")
        return HybridSearchResponse(
            query=hybrid_request.query,
            results=results,
            total=len(results),
            retrieval_method="hybrid_rrf",
            retrieval_channels=channels,
            fusion_config={
                "algorithm": "rrf",
                "rrf_k": rrf_k,
                "bm25_k": hybrid_request.bm25_k,
                "vector_k": hybrid_request.vector_k,
            },
            query_embedding_model=self.embedding_client.model,
            warnings=warnings,
        )
