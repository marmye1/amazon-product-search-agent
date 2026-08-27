"""检索策略和服务入口评估测试。"""

from __future__ import annotations

import pandas as pd

from src.evaluate_retrieval import compare_retrieval_strategies
from src.evaluation.run_regression import compare_search_paths
from src.models import SearchResponse, SearchResult


def _response(product_id: str) -> SearchResponse:
    return SearchResponse(
        query="wireless mouse",
        results=[
            SearchResult(
                product_id=product_id,
                title="Wireless Mouse",
                brand="Brand A",
                category=None,
                description="A mouse",
                bullet_points=[],
                score=1.0,
                matched_fields=["title"],
                source_ref="esci:v0:products:us:%s" % product_id,
            )
        ],
        total=1,
    )


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q-1",
                "query": "wireless mouse",
                "product_id": "p-1",
                "product_locale": "us",
                "esci_label": "E",
            },
            {
                "query_id": "q-1",
                "query": "wireless mouse",
                "product_id": "p-2",
                "product_locale": "us",
                "esci_label": "I",
            },
        ]
    )


def test_ablation_compares_all_strategies_on_same_query_set() -> None:
    summary = compare_retrieval_strategies(
        _examples(),
        {
            "bm25": lambda request: _response("p-1"),
            "vector": lambda request: _response("p-2"),
            "hybrid_rrf": lambda request: _response("p-1"),
        },
        top_k=1,
        positive_labels={"E", "S"},
        grades={"E": 3, "S": 2, "C": 1, "I": 0},
    )

    assert set(summary["strategies"]) == {"bm25", "vector", "hybrid_rrf"}
    assert summary["strategies"]["bm25"]["metrics"]["query_level_average"]["recall"] == 1.0
    assert summary["strategies"]["vector"]["metrics"]["query_level_average"]["recall"] == 0.0


def test_compare_search_paths_uses_same_labels_and_reports_metric_delta() -> None:
    summary = compare_search_paths(
        _examples(),
        lambda request: _response("p-2"),
        lambda request: _response("p-1"),
        top_k=1,
        positive_labels={"E", "S"},
        grades={"E": 3, "S": 2, "C": 1, "I": 0},
    )

    assert summary["before"]["metrics"]["query_level_average"]["recall"] == 0.0
    assert summary["after"]["metrics"]["query_level_average"]["recall"] == 1.0
    assert summary["delta"]["recall"] == 1.0
