"""真实验收：使用真实 OpenSearch 和 LM Studio 验证混合检索链路。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .embedding_client import EmbeddingClient, EmbeddingConfig, EmbeddingError
from .evaluate_bm25 import load_examples
from .evaluate_retrieval import compare_retrieval_strategies
from .hybrid_search_tool import HybridSearchTool
from .index_embeddings import build_hybrid_index_body, index_embeddings
from .index_products import iter_jsonl_documents
from .models import ContractError, SearchRequest, SearchResponse
from .opensearch_client import BackendError, OpenSearchClient
from .retrieve_bm25 import retrieve_bm25
from .retrieve_vector import retrieve_vector
from .search_config import config_value, load_search_config
from .hybrid_models import HybridSearchRequest


class AcceptanceError(RuntimeError):
    """真实验收步骤未通过。"""

    def __init__(self, code: str, message: str, *, details: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def validate_hybrid_mapping(
    mapping_response: Mapping[str, Any],
    index_name: str,
    dimension: int,
) -> Dict[str, Any]:
    """确认索引同时具备文本字段和正确的 HNSW 向量字段。"""

    errors: List[str] = []
    index_body = mapping_response.get(index_name)
    if not isinstance(index_body, Mapping):
        return {
            "passed": False,
            "errors": ["返回结果中找不到索引映射: %s" % index_name],
        }

    mappings = index_body.get("mappings")
    properties = mappings.get("properties") if isinstance(mappings, Mapping) else None
    if not isinstance(properties, Mapping):
        return {"passed": False, "errors": ["索引映射缺少 mappings.properties"]}

    expected_properties = build_hybrid_index_body(dimension)["mappings"]["properties"]
    required_fields = ("product_id", "locale", "title", "brand", "category", "source_ref", "embedding_model_id", "embedding")
    for field_name in required_fields:
        if field_name not in properties:
            errors.append("缺少字段: %s" % field_name)

    for field_name in ("product_id", "locale", "title", "source_ref", "embedding_model_id"):
        actual = properties.get(field_name)
        expected = expected_properties[field_name]
        if isinstance(actual, Mapping) and actual.get("type") != expected["type"]:
            errors.append("字段 %s 类型错误: %s" % (field_name, actual.get("type")))

    for field_name in ("brand", "category"):
        actual = properties.get(field_name)
        keyword = actual.get("fields", {}).get("keyword") if isinstance(actual, Mapping) else None
        if not isinstance(actual, Mapping) or actual.get("type") != "text":
            errors.append("字段 %s 不是 text" % field_name)
        if not isinstance(keyword, Mapping) or keyword.get("type") != "keyword":
            errors.append("字段 %s.keyword 不是 keyword" % field_name)

    embedding = properties.get("embedding")
    expected_embedding = expected_properties["embedding"]
    if isinstance(embedding, Mapping):
        for key in ("type", "dimension"):
            if embedding.get(key) != expected_embedding[key]:
                errors.append("embedding.%s 错误: %s" % (key, embedding.get(key)))
        method = embedding.get("method")
        expected_method = expected_embedding["method"]
        if not isinstance(method, Mapping):
            errors.append("embedding 缺少 HNSW method")
        else:
            for key in ("name", "engine", "space_type"):
                if method.get(key) != expected_method[key]:
                    errors.append("embedding.method.%s 错误: %s" % (key, method.get(key)))

    return {
        "passed": not errors,
        "errors": errors,
        "expected": {
            "index_name": index_name,
            "embedding_dimension": dimension,
            "embedding_method": expected_embedding["method"],
        },
    }


def _check_indexed_embedding(
    body: Mapping[str, Any],
    *,
    expected_dimension: int,
    expected_model: str,
) -> Dict[str, Any]:
    hits = body.get("hits", {}).get("hits", []) if isinstance(body.get("hits"), Mapping) else []
    if not isinstance(hits, list) or not hits:
        return {"passed": False, "errors": ["索引中没有找到带 embedding 的商品"]}
    first = hits[0]
    source = first.get("_source") if isinstance(first, Mapping) else None
    if not isinstance(source, Mapping):
        return {"passed": False, "errors": ["embedding 探针命中缺少 _source"]}

    errors: List[str] = []
    model = source.get("embedding_model_id")
    vector = source.get("embedding")
    if model != expected_model:
        errors.append("embedding_model_id 错误: %s" % model)
    if not isinstance(vector, list) or len(vector) != expected_dimension:
        actual_dimension = len(vector) if isinstance(vector, list) else 0
        errors.append("embedding 维度错误: %s" % actual_dimension)
    return {
        "passed": not errors,
        "errors": errors,
        "product_id": source.get("product_id") or first.get("_id"),
        "embedding_model_id": model,
        "embedding_dimension": len(vector) if isinstance(vector, list) else 0,
    }


def _check_results(results: Sequence[Any]) -> Dict[str, Any]:
    errors: List[str] = []
    product_ids: List[str] = []
    for result in results:
        product_id = getattr(result, "product_id", None)
        if not isinstance(product_id, str) or not product_id:
            errors.append("结果缺少 product_id")
            continue
        product_ids.append(product_id)
        if not isinstance(getattr(result, "title", None), str) or not result.title:
            errors.append("商品 %s 缺少 title" % product_id)
        if not isinstance(getattr(result, "source_ref", None), str) or not result.source_ref:
            errors.append("商品 %s 缺少 source_ref" % product_id)
    if len(product_ids) != len(set(product_ids)):
        errors.append("结果包含重复 product_id")
    return {
        "passed": not errors,
        "errors": errors,
        "count": len(product_ids),
        "product_ids": product_ids,
    }


def _result_summary(results: Sequence[Any]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        summary.append(
            {
                "rank": rank,
                "product_id": result.product_id,
                "title": result.title,
                "score": result.score,
                "source_ref": result.source_ref,
                "bm25_rank": getattr(result, "bm25_rank", None),
                "vector_rank": getattr(result, "vector_rank", None),
                "rrf_score": getattr(result, "rrf_score", None),
                "retrieval_channels": list(getattr(result, "retrieval_channels", [])),
            }
        )
    return summary


def _require_passed(check: Mapping[str, Any], code: str, message: str) -> None:
    if not bool(check.get("passed")):
        raise AcceptanceError(code, message, details=check)


def _search_response(
    query: str,
    results: Sequence[Any],
    retrieval_method: str,
) -> SearchResponse:
    return SearchResponse(
        query=query,
        results=list(results),
        total=len(results),
        retrieval_method=retrieval_method,
    )


def run_real_search_acceptance(
    config_path: Path,
    *,
    input_path: Optional[Path] = None,
    query: Optional[str] = None,
    locale: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    top_k: int = 5,
    bm25_k: Optional[int] = None,
    vector_k: Optional[int] = None,
    batch_size: int = 8,
    start_record: int = 1,
    index_limit: Optional[int] = 10,
    full_index: bool = False,
    ablation_split: str = "test",
    ablation_limit: Optional[int] = 10,
    full_ablation: bool = False,
    skip_ablation: bool = False,
) -> Dict[str, Any]:
    """执行真实验收；默认只写入 10 条商品，避免误触发全量建库。"""

    started = time.monotonic()
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    report: Dict[str, Any] = {
        "version": "",
        "status": "failed",
        "checks": {},
        "retrieval": {},
        "ablation": {},
        "indexing": {"batch_size": batch_size, "start_record": start_record},
    }

    try:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        if start_record < 1:
            raise ValueError("start_record 必须大于 0")
        if not full_index and (index_limit is None or index_limit < 1):
            raise ValueError("smoke 验收的 index_limit 必须大于 0；全量建库请使用 --full-index")
        if not full_ablation and not skip_ablation and (ablation_limit is None or ablation_limit < 1):
            raise ValueError("smoke 消融的 ablation_limit 必须大于 0；全量消融请使用 --full-ablation")

        config = load_search_config(config_path)
        resolved_input = input_path or Path(str(config_value(config, "data", "products_jsonl")))
        if not resolved_input.is_absolute():
            resolved_input = project_root / resolved_input
        sample_document = next(iter_jsonl_documents(resolved_input))
        query_text = query.strip() if isinstance(query, str) and query.strip() else sample_document.title
        query_locale = (locale or str(config_value(config, "data", "locale", sample_document.locale))).strip().lower()
        resolved_bm25_k = bm25_k or int(config_value(config, "retrieval", "bm25_k", 50))
        resolved_vector_k = vector_k or int(config_value(config, "retrieval", "vector_k", 50))
        index_name = str(config_value(config, "retrieval", "hybrid_index_name", "amazon_products_v4"))
        embedding_config = EmbeddingConfig.from_config(config)

        report["input"] = {
            "config_path": str(config_path),
            "products_jsonl": str(resolved_input),
            "index_name": index_name,
            "query": query_text,
            "query_source": "--query" if query else "first_product_title",
            "locale": query_locale,
            "top_k": top_k,
            "bm25_k": resolved_bm25_k,
            "vector_k": resolved_vector_k,
        }
        report["embedding"] = {
            "base_url": embedding_config.base_url,
            "model": embedding_config.model,
            "dimension": embedding_config.dimension,
        }

        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        embedding_client = EmbeddingClient(embedding_config)
        index_status = client.ensure_index(index_name, build_hybrid_index_body(embedding_client.dimension))
        report["checks"]["backend_and_index"] = {
            "passed": True,
            "index_status": index_status,
            "index_name": index_name,
        }

        mapping_check = validate_hybrid_mapping(
            client.get_mapping(index_name),
            index_name,
            embedding_client.dimension,
        )
        report["checks"]["mapping"] = mapping_check
        _require_passed(mapping_check, "search_mapping_invalid", "索引映射不符合验收要求")

        index_report = index_embeddings(
            resolved_input,
            client,
            embedding_client,
            config,
            batch_size=batch_size,
            limit=None if full_index else index_limit,
            start_record=start_record,
            max_text_chars=int(config_value(config, "embedding", "max_text_chars", 6000)),
        )
        report["checks"]["embedding_index"] = index_report.to_dict()
        if index_report.successful < 1 or index_report.failed > 0:
            raise AcceptanceError(
                "search_embedding_index_failed",
                "向量建库未通过：存在失败商品或没有成功写入商品",
                details=index_report.to_dict(),
            )

        client.refresh(index_name)
        count_body = client.search(
            index_name,
            {"size": 0, "track_total_hits": True, "query": {"match_all": {}}},
        )
        total = count_body.get("hits", {}).get("total", 0) if isinstance(count_body.get("hits"), Mapping) else 0
        if isinstance(total, Mapping):
            total = total.get("value", 0)
        expected_documents = index_report.source_total if full_index else index_report.successful
        report["checks"]["refresh_and_count"] = {
            "passed": isinstance(total, int) and total >= expected_documents,
            "documents_after_refresh": total,
            "new_successful_documents": index_report.successful,
            "expected_source_documents": expected_documents,
        }
        _require_passed(report["checks"]["refresh_and_count"], "search_refresh_failed", "刷新后没有看到已写入的商品")

        embedding_probe = client.search(
            index_name,
            {
                "size": 1,
                "_source": ["product_id", "embedding_model_id", "embedding"],
                "query": {"exists": {"field": "embedding"}},
            },
        )
        embedding_check = _check_indexed_embedding(
            embedding_probe,
            expected_dimension=embedding_client.dimension,
            expected_model=embedding_client.model,
        )
        report["checks"]["embedding_field"] = embedding_check
        _require_passed(embedding_check, "search_embedding_field_invalid", "商品没有可查询的正确维度向量")

        request = HybridSearchRequest(
            query=query_text,
            locale=query_locale or None,
            category=category,
            brand=brand,
            top_k=top_k,
            bm25_k=resolved_bm25_k,
            vector_k=resolved_vector_k,
        )
        bm25_results = retrieve_bm25(request, client, config)
        vector_results = retrieve_vector(request, client, embedding_client, config)
        hybrid_response = HybridSearchTool(client, config, embedding_client).invoke(request)

        bm25_check = _check_results(bm25_results)
        vector_check = _check_results(vector_results)
        hybrid_check = _check_results(hybrid_response.results)
        if not bm25_check["passed"] or not vector_check["passed"] or not hybrid_check["passed"]:
            raise AcceptanceError(
                "search_result_contract_failed",
                "三路检索结果存在重复或缺少核心字段",
                details={"bm25": bm25_check, "vector": vector_check, "hybrid_rrf": hybrid_check},
            )
        if not bm25_results:
            raise AcceptanceError("search_bm25_no_results", "真实 BM25 没有返回候选商品")
        if not vector_results:
            raise AcceptanceError("search_vector_no_results", "真实向量检索没有返回候选商品")
        if hybrid_response.retrieval_method != "hybrid_rrf":
            raise AcceptanceError(
                "search_hybrid_fallback",
                "Hybrid 检索发生了 BM25 降级，没有完成真实 RRF 融合",
                details=hybrid_response.to_dict(),
            )
        if any(warning.startswith("vector_fallback_bm25:") for warning in hybrid_response.warnings):
            raise AcceptanceError(
                "search_hybrid_fallback",
                "Hybrid 检索发生了向量失败降级",
                details={"warnings": hybrid_response.warnings},
            )
        channel_errors: List[str] = []
        for result in hybrid_response.results:
            channels = hybrid_response.retrieval_channels.get(result.product_id)
            if not channels:
                channel_errors.append("商品 %s 缺少 retrieval_channels" % result.product_id)
            if not getattr(result, "retrieval_channels", None):
                channel_errors.append("商品 %s 结果缺少 retrieval_channels" % result.product_id)
            if getattr(result, "rrf_score", None) is None:
                channel_errors.append("商品 %s 缺少 rrf_score" % result.product_id)
        if channel_errors:
            raise AcceptanceError(
                "search_rrf_trace_failed",
                "RRF 结果无法追踪召回通道",
                details={"errors": channel_errors},
            )

        report["retrieval"] = {
            "query": query_text,
            "bm25": {"passed": True, "results": _result_summary(bm25_results)},
            "vector": {"passed": True, "results": _result_summary(vector_results)},
            "hybrid_rrf": {
                "passed": True,
                "retrieval_method": hybrid_response.retrieval_method,
                "retrieval_channels": hybrid_response.retrieval_channels,
                "fusion_config": hybrid_response.fusion_config,
                "warnings": list(hybrid_response.warnings),
                "results": _result_summary(hybrid_response.results),
            },
        }

        if skip_ablation:
            report["ablation"] = {
                "status": "skipped",
                "reason": "使用 --skip-ablation，仅验证真实索引和单 query 三路检索",
            }
            report["status"] = "incomplete"
        else:
            examples_path = Path(str(config_value(config, "data", "examples_parquet")))
            if not examples_path.is_absolute():
                examples_path = project_root / examples_path
            examples = load_examples(
                examples_path,
                locale=query_locale or None,
                split=ablation_split,
                limit=None if full_ablation else ablation_limit,
            )
            if examples.empty:
                raise AcceptanceError(
                    "search_ablation_no_examples",
                    "指定 split 没有可用于消融的 query",
                    details={"split": ablation_split, "examples_path": str(examples_path)},
                )

            def bm25_search(search_request: SearchRequest) -> SearchResponse:
                hybrid_request = HybridSearchRequest.from_search_request(
                    search_request,
                    bm25_k=resolved_bm25_k,
                    vector_k=resolved_vector_k,
                )
                candidates = retrieve_bm25(hybrid_request, client, config)
                return _search_response(search_request.query, candidates, "bm25")

            def vector_search(search_request: SearchRequest) -> SearchResponse:
                hybrid_request = HybridSearchRequest.from_search_request(
                    search_request,
                    bm25_k=resolved_bm25_k,
                    vector_k=resolved_vector_k,
                )
                candidates = retrieve_vector(hybrid_request, client, embedding_client, config)
                return _search_response(search_request.query, candidates, "vector")

            ablation = compare_retrieval_strategies(
                examples,
                {
                    "bm25": bm25_search,
                    "vector": vector_search,
                    "hybrid_rrf": HybridSearchTool(client, config, embedding_client).invoke,
                },
                top_k=top_k,
                positive_labels={str(item).upper() for item in config_value(config, "evaluation", "positive_labels", ["E", "S"])},
                grades={
                    str(key).upper(): int(value)
                    for key, value in config_value(
                        config,
                        "evaluation",
                        "graded_relevance",
                        {"E": 3, "S": 2, "C": 1, "I": 0},
                    ).items()
                },
            )
            ablation["scope"] = "full" if full_ablation else "smoke"
            ablation["split"] = ablation_split
            ablation["dataset"] = {
                "examples_path": str(examples_path),
                "rows": len(examples),
                "query_count": int(examples["query_id"].nunique()),
            }
            report["ablation"] = ablation
            failed_strategies = {
                name: value.get("failed_queries", 0)
                for name, value in ablation.get("strategies", {}).items()
                if value.get("failed_queries", 0) or value.get("aborted_on_backend_error")
            }
            if failed_strategies:
                raise AcceptanceError(
                    "search_ablation_failed",
                    "三路消融中存在失败查询",
                    details={"failed_strategies": failed_strategies},
                )
            report["status"] = "passed" if full_index and full_ablation else "smoke_passed"

    except (AcceptanceError, BackendError, ContractError, EmbeddingError, FileNotFoundError, OSError, ValueError) as exc:
        report["status"] = "failed"
        report["error"] = (
            exc.to_dict()
            if hasattr(exc, "to_dict")
            else {"code": getattr(exc, "code", "search_acceptance_failed"), "message": str(exc)}
        )

    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(".%s.tmp" % path.name)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(str(temp_path), str(path))
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=" 真实 OpenSearch + LM Studio 验收")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--input", type=Path, help="覆盖配置中的 products_jsonl")
    parser.add_argument("--query", help="验收 query；不填时使用首个商品 title，保证 smoke 数据可命中")
    parser.add_argument("--locale")
    parser.add_argument("--category")
    parser.add_argument("--brand")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-k", type=int)
    parser.add_argument("--vector-k", type=int)
    parser.add_argument("--batch-size", type=int, default=8, help="每批请求 LM Studio 的商品数，默认 8")
    parser.add_argument("--start-record", type=int, default=1, help="从第 N 条商品开始处理（从 1 开始，默认 1）")
    parser.add_argument("--limit", type=int, default=10, help="smoke 建库商品数，默认 10")
    parser.add_argument("--full-index", action="store_true", help="处理全部商品；耗时较长，需显式开启")
    parser.add_argument("--split", dest="ablation_split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--ablation-limit", type=int, default=10, help="smoke 消融 query 数，默认 10")
    parser.add_argument("--full-ablation", action="store_true", help="评估整个指定 split")
    parser.add_argument("--skip-ablation", action="store_true", help="只验证真实索引和单 query 三路检索")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parent.parent
    report_path = args.report or project_root / "reports" / "" / "real-acceptance.json"
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report = run_real_search_acceptance(
        config_path,
        input_path=args.input,
        query=args.query,
        locale=args.locale,
        category=args.category,
        brand=args.brand,
        top_k=args.top_k,
        bm25_k=args.bm25_k,
        vector_k=args.vector_k,
        batch_size=args.batch_size,
        start_record=args.start_record,
        index_limit=args.limit,
        full_index=args.full_index,
        ablation_split=args.ablation_split,
        ablation_limit=args.ablation_limit,
        full_ablation=args.full_ablation,
        skip_ablation=args.skip_ablation,
    )
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"passed", "smoke_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
