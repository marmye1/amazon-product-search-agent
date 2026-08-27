"""BM25 搜索入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .models import ContractError, SearchRequest, SearchResponse
from .normalize_search_result import normalize_search_response
from .opensearch_client import BackendError, OpenSearchClient
from .search_config import config_value, load_search_config


def build_search_body(request: SearchRequest, config: Mapping[str, Any]) -> Dict[str, Any]:
    raw_weights = config_value(config, "search", "field_weights", {})
    if not isinstance(raw_weights, Mapping) or not raw_weights:
        raise ValueError("search.field_weights 必须是非空对象")

    fields = []
    highlight_fields: Dict[str, Dict[str, Any]] = {}
    for field_name, raw_weight in raw_weights.items():
        weight = float(raw_weight)
        if weight == 1:
            fields.append(field_name)
        elif weight.is_integer():
            fields.append("%s^%s" % (field_name, int(weight)))
        else:
            fields.append("%s^%s" % (field_name, weight))
        highlight_fields[field_name] = {}

    filters = []
    if request.locale:
        filters.append({"term": {"locale": request.locale.lower()}})
    if request.category:
        filters.append({"term": {"category.keyword": request.category}})
    if request.brand:
        filters.append({"term": {"brand.keyword": request.brand}})

    body: Dict[str, Any] = {
        "size": request.top_k,
        "track_total_hits": True,
        "_source": [
            "product_id",
            "locale",
            "title",
            "brand",
            "category",
            "description",
            "bullet_points",
            "source_ref",
        ],
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": request.query,
                            "fields": fields,
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filters,
            }
        },
        "highlight": {"fields": highlight_fields},
    }
    return body


def search_products(
    request: SearchRequest,
    client: Any,
    config: Mapping[str, Any],
    *,
    index_name: Optional[str] = None,
) -> SearchResponse:
    """通过注入的后端执行搜索；真实运行时 client 必须是 OpenSearchClient。"""

    resolved_index_name = index_name or str(config_value(config, "opensearch", "index_name", "amazon_products_v1"))
    body = build_search_body(request, config)
    raw_response = client.search(resolved_index_name, body)
    return normalize_search_response(request.query, raw_response)


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description="执行 OpenSearch BM25 搜索")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--query")
    parser.add_argument("--locale")
    parser.add_argument("--category")
    parser.add_argument("--brand")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_search_config(config_path)
    default_top_k = int(config_value(config, "search", "default_top_k", 10))
    max_top_k = int(config_value(config, "search", "max_top_k", 100))
    request_values = {
        "query": args.query,
        "locale": args.locale,
        "category": args.category,
        "brand": args.brand,
        "top_k": default_top_k if args.top_k is None else args.top_k,
    }

    try:
        request = SearchRequest.from_mapping(
            request_values,
            default_top_k=default_top_k,
            max_top_k=max_top_k,
        )
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        response = search_products(request, client, config)
    except (ContractError, BackendError, ValueError) as exc:
        error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "invalid_request", "message": str(exc)}
        print(json.dumps({"error": error}, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
