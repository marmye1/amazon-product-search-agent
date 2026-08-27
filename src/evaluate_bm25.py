"""使用 ESCI 标注对 BM25 搜索结果做离线评估。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from .models import ContractError, SearchRequest, SearchResponse
from .opensearch_client import BackendError, OpenSearchClient
from .search_config import config_value, load_search_config
from .search_products import search_products


SearchCallable = Callable[[SearchRequest], SearchResponse]


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return round(sum(valid) / len(valid), 6) if valid else None


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def _normalise_labels(group: pd.DataFrame, grades: Mapping[str, int]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in group[["product_id", "esci_label"]].itertuples(index=False):
        product_id = str(row.product_id)
        label = str(row.esci_label).upper()
        if product_id not in labels or grades.get(label, 0) > grades.get(labels[product_id], 0):
            labels[product_id] = label
    return labels


def evaluate_one_query(
    expected_labels: Mapping[str, str],
    response: SearchResponse,
    *,
    top_k: int,
    positive_labels: Set[str],
    grades: Mapping[str, int],
) -> Dict[str, Any]:
    ranked_ids = [result.product_id for result in response.results[:top_k]]
    relevant_ids = {product_id for product_id, label in expected_labels.items() if label in positive_labels}
    relevant_ranked_grades = [grades.get(expected_labels.get(product_id, "I"), 0) for product_id in ranked_ids]

    if relevant_ids:
        recall = len(set(ranked_ids) & relevant_ids) / len(relevant_ids)
    else:
        recall = None

    reciprocal_rank = 0.0
    for rank, product_id in enumerate(ranked_ids, start=1):
        if product_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break
    if not relevant_ids:
        reciprocal_rank_value: Optional[float] = None
    else:
        reciprocal_rank_value = reciprocal_rank

    ideal_grades = sorted((grades.get(label, 0) for label in expected_labels.values()), reverse=True)[:top_k]
    ideal_dcg = _dcg(ideal_grades)
    ndcg = _dcg(relevant_ranked_grades) / ideal_dcg if ideal_dcg else None

    return {
        "recall": recall,
        "mrr": reciprocal_rank_value,
        "ndcg": ndcg,
        "returned": len(ranked_ids),
        "relevant_count": len(relevant_ids),
    }


def _sample_success(
    query_id: str,
    query: str,
    labels: Mapping[str, str],
    response: SearchResponse,
    *,
    top_k: int,
) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "expected": [{"product_id": key, "esci_label": value} for key, value in list(labels.items())[:10]],
        "returned": [
            {"product_id": result.product_id, "score": result.score, "title": result.title}
            for result in response.results[:top_k]
        ],
    }


def evaluate_examples(
    examples: pd.DataFrame,
    search: SearchCallable,
    *,
    top_k: int,
    positive_labels: Set[str],
    grades: Mapping[str, int],
    sample_count: int = 10,
) -> Dict[str, Any]:
    required = {"query_id", "query", "product_id", "esci_label"}
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError("examples 缺少评估字段: %s" % ", ".join(missing))

    metric_values: Dict[str, List[Optional[float]]] = {"recall": [], "mrr": [], "ndcg": []}
    success_samples: List[Dict[str, Any]] = []
    failure_samples: List[Dict[str, Any]] = []
    query_count = 0
    failed_count = 0
    aborted = False

    for query_id, group in examples.groupby("query_id", sort=True):
        query_count += 1
        query = str(group.iloc[0]["query"])
        labels = _normalise_labels(group, grades)
        locale = str(group.iloc[0]["product_locale"]) if "product_locale" in group.columns else None
        request_values = {"query": query, "top_k": top_k}
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
            if len(success_samples) < sample_count:
                success_samples.append(_sample_success(str(query_id), query, labels, response, top_k=top_k))
        except (ContractError, BackendError, ValueError) as exc:
            failed_count += 1
            failure_samples.append(
                {"query_id": str(query_id), "query": query, "error": getattr(exc, "to_dict", lambda: {"code": "evaluation_failed", "message": str(exc)})()}
            )
            # 后端不可用时继续请求只会重复同一个故障；保留首个失败并停止。
            if isinstance(exc, BackendError):
                aborted = True
                break

    return {
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
        "success_samples": success_samples,
        "failure_samples": failure_samples[:sample_count],
    }


def load_examples(path: Path, *, locale: Optional[str], split: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    columns = ["query", "query_id", "product_id", "product_locale", "esci_label", "split"]
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
    if locale and locale.lower() not in ("all", "*"):
        frame = frame[frame["product_locale"].astype("string").str.lower().eq(locale.lower())]
    if split and split.lower() not in ("all", "*"):
        frame = frame[frame["split"].astype("string").str.lower().eq(split.lower())]
    if limit is not None:
        query_ids = frame["query_id"].drop_duplicates().head(limit)
        frame = frame[frame["query_id"].isin(set(query_ids))]
    return frame.reset_index(drop=True)


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
    parser = argparse.ArgumentParser(description="使用 ESCI 评估 OpenSearch BM25")
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
    ) / "bm25-evaluation.json"

    raw_positive = config_value(config, "evaluation", "positive_labels", ["E", "S"])
    raw_grades = config_value(config, "evaluation", "graded_relevance", {"E": 3, "S": 2, "C": 1, "I": 0})
    positive_labels = {str(item).upper() for item in raw_positive}
    grades = {str(key).upper(): int(value) for key, value in raw_grades.items()}
    examples = load_examples(examples_path, locale=locale, split=args.split, limit=args.limit)
    try:
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
    except (BackendError, ValueError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "evaluation_failed"), "message": str(exc)}}, ensure_ascii=False))
        return 2

    summary = evaluate_examples(
        examples,
        lambda request: search_products(request, client, config),
        top_k=args.top_k,
        positive_labels=positive_labels,
        grades=grades,
        sample_count=args.sample_count,
    )
    summary.update(
        {
            "dataset": {
                "examples_path": str(examples_path),
                "locale": locale,
                "split": args.split,
                "rows": len(examples),
            },
            "evaluation": {
                "top_k": args.top_k,
                "positive_labels": sorted(positive_labels),
                "graded_relevance": grades,
            },
        }
    )
    _write_json(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
