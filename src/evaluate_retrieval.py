"""比较 BM25、纯向量和混合检索的离线指标。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

import pandas as pd

from .embedding_client import EmbeddingClient, EmbeddingConfig, EmbeddingError
from .evaluate_bm25 import _mean, _normalise_labels, evaluate_one_query, load_examples
from .hybrid_search_tool import HybridSearchTool
from .models import ContractError, SearchRequest, SearchResponse
from .opensearch_client import BackendError, OpenSearchClient
from .progress import ProgressBar
from .retrieve_vector import retrieve_vector
from .search_config import config_value, load_search_config
from .retrieve_bm25 import retrieve_bm25
from .hybrid_models import HybridSearchRequest


SearchCallable = Callable[[SearchRequest], SearchResponse]


def compare_retrieval_strategies(
    examples: pd.DataFrame,
    searches: Mapping[str, SearchCallable],
    *,
    top_k: int,
    positive_labels: Set[str],
    grades: Mapping[str, int],
    sample_count: int = 10,
) -> Dict[str, Any]:
    """在完全相同的 query/标注集合上分别统计三种召回策略。"""

    required = {"query_id", "query", "product_id", "esci_label"}
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError("examples 缺少评估字段: %s" % ", ".join(missing))

    output: Dict[str, Any] = {}
    for strategy_name, search in searches.items():
        metric_values: Dict[str, List[Optional[float]]] = {"recall": [], "mrr": [], "ndcg": []}
        failure_samples: List[Dict[str, Any]] = []
        query_count = 0
        failed_count = 0
        aborted = False
        progress = ProgressBar(int(examples["query_id"].nunique()), " 消融 %s" % strategy_name)
        progress.update(0, status="准备中")

        try:
            for query_id, group in examples.groupby("query_id", sort=True):
                query_count += 1
                query = str(group.iloc[0]["query"])
                labels = _normalise_labels(group, grades)
                request_values = {"query": query, "top_k": top_k}
                locale = str(group.iloc[0]["product_locale"]) if "product_locale" in group.columns else None
                if locale and locale != "<NA>":
                    request_values["locale"] = locale
                try:
                    request = SearchRequest.from_mapping(request_values, default_top_k=top_k, max_top_k=top_k)
                    response = search(request)
                    metrics = evaluate_one_query(
                        labels,
                        response,
                        top_k=top_k,
                        positive_labels=positive_labels,
                        grades=grades,
                    )
                    for metric_name in metric_values:
                        metric_values[metric_name].append(metrics[metric_name])
                    progress.update(1)
                except (ContractError, BackendError, EmbeddingError, ValueError) as exc:
                    failed_count += 1
                    if len(failure_samples) < sample_count:
                        failure_samples.append(
                            {
                                "query_id": str(query_id),
                                "query": query,
                                "error": getattr(
                                    exc,
                                    "to_dict",
                                    lambda: {"code": "retrieval_evaluation_failed", "message": str(exc)},
                                )(),
                            }
                        )
                    progress.update(1, status="后端错误" if isinstance(exc, BackendError) else "查询失败")
                    if isinstance(exc, BackendError):
                        aborted = True
                        break
        finally:
            progress.close()

        output[strategy_name] = {
            "query_count": query_count,
            "successful_queries": query_count - failed_count,
            "failed_queries": failed_count,
            "aborted_on_backend_error": aborted,
            "metrics": {
                "query_level_average": {
                    "recall": _mean(metric_values["recall"]),
                    "mrr": _mean(metric_values["mrr"]),
                    "ndcg": _mean(metric_values["ndcg"]),
                }
            },
            "failure_samples": failure_samples,
        }
    return {"strategies": output, "top_k": top_k}


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
    parser = argparse.ArgumentParser(description="比较 BM25、vector 和 hybrid_rrf 检索")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--limit", type=int, help="只评估前 N 个 query，用于 smoke test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_search_config(config_path)
    project_root = config_path.parent.parent
    examples_path = _resolve_path(str(config_value(config, "data", "examples_parquet")), project_root)
    locale = str(config_value(config, "data", "locale", "us"))
    report_path = args.report or _resolve_path(
        str(config_value(config, "data", "reports_dir", "reports/evaluation/")), project_root
    ) / "retrieval-ablation.json"

    raw_positive = config_value(config, "evaluation", "positive_labels", ["E", "S"])
    raw_grades = config_value(config, "evaluation", "graded_relevance", {"E": 3, "S": 2, "C": 1, "I": 0})
    positive_labels = {str(item).upper() for item in raw_positive}
    grades = {str(key).upper(): int(value) for key, value in raw_grades.items()}
    examples = load_examples(examples_path, locale=locale, split=args.split, limit=args.limit)

    try:
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        embedding_client = EmbeddingClient(EmbeddingConfig.from_config(config))
        hybrid_tool = HybridSearchTool(client, config, embedding_client)
        vector_k = int(config_value(config, "retrieval", "vector_k", 50))
        bm25_k = int(config_value(config, "retrieval", "bm25_k", 50))

        def bm25_search(request: SearchRequest) -> SearchResponse:
            hybrid_request = HybridSearchRequest.from_search_request(
                request,
                bm25_k=bm25_k,
                vector_k=vector_k,
            )
            candidates = retrieve_bm25(hybrid_request, client, config)
            return SearchResponse(
                query=request.query,
                results=candidates,
                total=len(candidates),
                retrieval_method="bm25",
            )

        def vector_search(request: SearchRequest) -> SearchResponse:
            hybrid_request = HybridSearchRequest.from_search_request(
                request,
                bm25_k=bm25_k,
                vector_k=vector_k,
            )
            candidates = retrieve_vector(hybrid_request, client, embedding_client, config)
            return SearchResponse(
                query=request.query,
                results=candidates,
                total=len(candidates),
                retrieval_method="vector",
            )

        summary = compare_retrieval_strategies(
            examples,
            {
                "bm25": bm25_search,
                "vector": vector_search,
                "hybrid_rrf": hybrid_tool.invoke,
            },
            top_k=args.top_k,
            positive_labels=positive_labels,
            grades=grades,
            sample_count=args.sample_count,
        )
    except (BackendError, EmbeddingError, ValueError, OSError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "retrieval_evaluation_failed"), "message": str(exc)}}, ensure_ascii=False))
        return 2

    summary.update(
        {
            "dataset": {
                "examples_path": str(examples_path),
                "locale": locale,
                "split": args.split,
                "rows": len(examples),
            },
            "evaluation": {
                "positive_labels": sorted(positive_labels),
                "graded_relevance": grades,
            },
        }
    )
    _write_json(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
