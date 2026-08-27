"""比较检索入口和服务入口的离线评估结果。

比较函数本身不访问网络，可由测试注入两个搜索函数；命令行入口使用当前
已配置的 HybridSearchTool，检查服务包装是否改变检索结果。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set

import pandas as pd

from ..config.settings import AppSettings
from ..evaluate_bm25 import evaluate_examples, load_examples
from ..models import SearchRequest, SearchResponse
from ..observability.tracing import new_trace_id
from ..search_config import config_value
from ..service.agent_service import AgentService


SearchCallable = Callable[[SearchRequest], SearchResponse]


def _metric_delta(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return round(after - before, 6)


def compare_search_paths(
    examples: pd.DataFrame,
    before_search: SearchCallable,
    after_search: SearchCallable,
    *,
    top_k: int,
    positive_labels: Set[str],
    grades: Mapping[str, int],
    sample_count: int = 10,
) -> Dict[str, Any]:
    """在同一份 ESCI query/label 集合上比较两个入口。"""

    before = evaluate_examples(
        examples,
        before_search,
        top_k=top_k,
        positive_labels=positive_labels,
        grades=grades,
        sample_count=sample_count,
    )
    after = evaluate_examples(
        examples,
        after_search,
        top_k=top_k,
        positive_labels=positive_labels,
        grades=grades,
        sample_count=sample_count,
    )
    before_metrics = before["metrics"]["query_level_average"]
    after_metrics = after["metrics"]["query_level_average"]
    return {
        "before": before,
        "after": after,
        "delta": {
            key: _metric_delta(before_metrics.get(key), after_metrics.get(key))
            for key in ("recall", "mrr", "ndcg")
        },
        "top_k": top_k,
    }


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(".%s.tmp" % path.name)
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temp_path), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="比较检索入口和服务入口")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--limit", type=int, help="只评估前 N 个 query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    settings = AppSettings.from_env()
    settings = AppSettings(
        app_version=settings.app_version,
        agent_version=settings.agent_version,
        search_config_path=config_path,
        retrieval_method=settings.retrieval_method,
        default_chat_top_k=settings.default_chat_top_k,
        api_host=settings.api_host,
        api_port=settings.api_port,
        log_level=settings.log_level,
        health_timeout_seconds=settings.health_timeout_seconds,
    )
    service = AgentService.from_settings(settings)
    examples_path = _resolve_path(
        str(config_value(service.runtime.search_config, "data", "examples_parquet")),
        config_path.parent.parent,
    )
    locale = str(config_value(service.runtime.search_config, "data", "locale", "us"))
    examples = load_examples(examples_path, locale=locale, split=args.split, limit=args.limit)
    raw_positive = config_value(service.runtime.search_config, "evaluation", "positive_labels", ["E", "S"])
    raw_grades = config_value(
        service.runtime.search_config,
        "evaluation",
        "graded_relevance",
        {"E": 3, "S": 2, "C": 1, "I": 0},
    )
    positive_labels = {str(item).upper() for item in raw_positive}
    grades = {str(key).upper(): int(value) for key, value in raw_grades.items()}

    def before_search(request: SearchRequest) -> SearchResponse:
        return service.runtime.search_tool.invoke(request)

    def after_search(request: SearchRequest) -> SearchResponse:
        return service.search(request, trace_id=new_trace_id())

    summary = compare_search_paths(
        examples,
        before_search,
        after_search,
        top_k=args.top_k,
        positive_labels=positive_labels,
        grades=grades,
        sample_count=args.sample_count,
    )
    summary["dataset"] = {
        "examples_path": str(examples_path),
        "locale": locale,
        "split": args.split,
        "rows": len(examples),
    }
    report_path = args.report or config_path.parent.parent / "reports/evaluation/search-path-comparison.json"
    _write_json(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
