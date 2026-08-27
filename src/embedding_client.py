"""通过 LM Studio OpenAI 兼容接口生成文本向量。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests


class EmbeddingError(RuntimeError):
    """Embedding 服务不可用、返回非法响应或向量维度不正确。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "text-embedding-nomic-embed-text-v1.5"
    dimension: int = 768
    timeout_seconds: float = 60.0
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EmbeddingConfig":
        section = config.get("embedding", {})
        if not isinstance(section, Mapping):
            raise ValueError("embedding 配置必须是对象")
        dimension = int(section.get("dimension", cls.dimension))
        if dimension < 1:
            raise ValueError("embedding.dimension 必须是大于 0 的整数")
        return cls(
            base_url=str(section.get("base_url", cls.base_url)),
            model=str(section.get("model", cls.model)),
            dimension=dimension,
            timeout_seconds=float(section.get("timeout_seconds", cls.timeout_seconds)),
            document_prefix=str(section.get("document_prefix", cls.document_prefix)),
            query_prefix=str(section.get("query_prefix", cls.query_prefix)),
        )

    @property
    def embeddings_url(self) -> str:
        return "%s/embeddings" % self.base_url.rstrip("/")


class EmbeddingClient:
    """最小 Embedding 客户端；文档和查询使用不同任务前缀。"""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not config.base_url.startswith(("http://", "https://")):
            raise ValueError("Embedding base_url 必须是 http/https URL")
        if not config.model.strip():
            raise ValueError("Embedding model 不能为空")
        self.config = config
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _embed(self, texts: Sequence[str], prefix: str) -> List[List[float]]:
        if not texts:
            return []
        prepared = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("invalid_embedding_input", "Embedding 输入文本不能为空")
            prepared.append("%s%s" % (prefix, text.strip()))

        try:
            response = self.session.post(
                self.config.embeddings_url,
                headers={"Authorization": "Bearer lm-studio"},
                json={"model": self.config.model, "input": prepared},
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise EmbeddingError("embedding_timeout", "Embedding 请求超时") from exc
        except requests.ConnectionError as exc:
            raise EmbeddingError("embedding_unavailable", "无法连接本地 Embedding 服务") from exc
        except requests.RequestException as exc:
            raise EmbeddingError("embedding_request_error", "Embedding 请求失败") from exc

        if response.status_code >= 400:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise EmbeddingError("embedding_http_error", "Embedding 服务返回 HTTP %s: %s" % (response.status_code, detail))
        try:
            body = response.json()
        except ValueError as exc:
            raise EmbeddingError("embedding_invalid_response", "Embedding 服务返回的不是合法 JSON") from exc

        raw_data = body.get("data") if isinstance(body, Mapping) else None
        if not isinstance(raw_data, list) or len(raw_data) != len(prepared):
            raise EmbeddingError("embedding_invalid_response", "Embedding 返回数量与输入数量不一致")

        ordered = sorted(raw_data, key=lambda item: item.get("index", 0) if isinstance(item, Mapping) else -1)
        vectors: List[List[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(vector, list) or len(vector) != self.config.dimension:
                actual = len(vector) if isinstance(vector, list) else 0
                raise EmbeddingError(
                    "embedding_dimension_mismatch",
                    "Embedding 维度不匹配：期望 %s，实际 %s" % (self.config.dimension, actual),
                )
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
                raise EmbeddingError("embedding_invalid_vector", "Embedding 向量包含非数字值")
            vectors.append([float(value) for value in vector])
        if len(vectors) != len(prepared):
            raise EmbeddingError("embedding_invalid_response", "Embedding 返回顺序或数量无效")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts, self.config.document_prefix)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text], self.config.query_prefix)
        return vectors[0]
