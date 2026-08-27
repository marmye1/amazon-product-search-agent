"""把 ProductDocument JSONL 批量写入 OpenSearch。"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .models import ProductDocument
from .opensearch_client import BackendError, OpenSearchClient
from .search_config import config_value, load_search_config


def build_index_body() -> Dict[str, Any]:
    """返回固定的商品索引设置和字段映射。"""

    text_with_keyword = {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    }
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "product_id": {"type": "keyword"},
                "locale": {"type": "keyword"},
                "title": {"type": "text"},
                "brand": text_with_keyword,
                "category": text_with_keyword,
                "description": {"type": "text"},
                "bullet_points": {"type": "text"},
                "source_url": {"type": "keyword", "index": False},
                "data_provenance": {"type": "keyword"},
                "source_ref": {"type": "keyword"},
                "color": text_with_keyword,
            },
        },
    }


def iter_jsonl_documents(path: Path) -> Iterable[ProductDocument]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(" 商品 JSONL 不存在或为空: %s" % path)
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("JSONL 第 %s 行无法解析: %s" % (line_number, exc)) from exc
            if not isinstance(value, Mapping):
                raise ValueError("JSONL 第 %s 行必须是对象" % line_number)
            try:
                document = ProductDocument.from_mapping(value, line_number=line_number)
            except ValueError:
                raise
            if document.product_id in seen_ids:
                raise ValueError("product_id 重复，不允许静默覆盖: %s" % document.product_id)
            seen_ids.add(document.product_id)
            yield document


def _bulk_payload(index_name: str, documents: Sequence[ProductDocument]) -> str:
    lines: List[str] = []
    for document in documents:
        lines.append(json.dumps({"index": {"_index": index_name, "_id": document.product_id}}, separators=(",", ":")))
        lines.append(json.dumps(document.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _bulk_counts(
    body: Mapping[str, Any],
    documents: Sequence[ProductDocument],
) -> Dict[str, Any]:
    items = body.get("items")
    if not isinstance(items, list) or len(items) != len(documents):
        raise BackendError("backend_invalid_response", "OpenSearch bulk items 数量与请求不一致")

    failed_ids: List[str] = []
    successful = 0
    for document, item in zip(documents, items):
        if not isinstance(item, Mapping) or not item:
            failed_ids.append(document.product_id)
            continue
        action = next(iter(item.values()))
        if isinstance(action, Mapping) and action.get("error") is not None:
            failed_ids.append(document.product_id)
        else:
            successful += 1
    return {"successful": successful, "failed": len(failed_ids), "failed_ids": failed_ids}


@dataclass
class IndexingReport:
    index_name: str
    input_path: str
    total: int = 0
    successful: int = 0
    failed: int = 0
    failed_ids: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    index_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def index_products(
    input_path: Path,
    client: OpenSearchClient,
    config: Mapping[str, Any],
    *,
    batch_size: int = 500,
) -> IndexingReport:
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")
    index_name = str(config_value(config, "opensearch", "index_name", "amazon_products_v1"))
    started = time.monotonic()
    report = IndexingReport(index_name=index_name, input_path=str(input_path))
    report.index_status = client.ensure_index(index_name, build_index_body())

    batch: List[ProductDocument] = []
    for document in iter_jsonl_documents(input_path):
        batch.append(document)
        if len(batch) >= batch_size:
            counts = _bulk_counts(client.bulk(_bulk_payload(index_name, batch)), batch)
            report.total += len(batch)
            report.successful += counts["successful"]
            report.failed += counts["failed"]
            report.failed_ids.extend(counts["failed_ids"])
            batch = []
    if batch:
        counts = _bulk_counts(client.bulk(_bulk_payload(index_name, batch)), batch)
        report.total += len(batch)
        report.successful += counts["successful"]
        report.failed += counts["failed"]
        report.failed_ids.extend(counts["failed_ids"])

    report.elapsed_seconds = round(time.monotonic() - started, 3)
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
    parser = argparse.ArgumentParser(description="导入商品到 OpenSearch")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--input", type=Path, help="覆盖配置中的 products_jsonl")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_search_config(config_path)
    project_root = config_path.parent.parent
    input_path = args.input or _resolve_path(str(config_value(config, "data", "products_jsonl")), project_root)
    report_path = args.report or _resolve_path(
        str(config_value(config, "data", "reports_dir", "reports/evaluation/")), project_root
    ) / "index-report.json"

    try:
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        report = index_products(input_path, client, config, batch_size=args.batch_size)
        _write_json(report_path, report.to_dict())
    except (BackendError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "index_failed"), "message": str(exc)}}, ensure_ascii=False))
        return 2

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
