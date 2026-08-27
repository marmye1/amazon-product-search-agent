"""为商品生成 Embedding，并写入独立的 OpenSearch 索引。"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .embedding_client import EmbeddingClient, EmbeddingConfig, EmbeddingError
from .index_products import iter_jsonl_documents
from .models import ProductDocument
from .opensearch_client import BackendError, OpenSearchClient
from .progress import ProgressBar
from .search_config import config_value, load_search_config


def build_hybrid_index_body(dimension: int) -> Dict[str, Any]:
    """返回保留文本字段、增加 knn_vector 字段的索引映射。"""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("向量维度必须是大于 0 的整数")
    text_with_keyword = {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    }
    return {
        "settings": {
            "index": {"knn": True},
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
                "embedding_model_id": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
            },
        },
    }


def product_embedding_text(document: ProductDocument, *, max_chars: int = 6000) -> str:
    """按稳定顺序拼接商品字段，避免向量索引依赖未定义的字段顺序。"""

    parts = [
        "title: %s" % document.title,
        "brand: %s" % document.brand if document.brand else None,
        "category: %s" % document.category if document.category else None,
        "description: %s" % document.description if document.description else None,
        "bullet_points: %s" % " ; ".join(document.bullet_points) if document.bullet_points else None,
        "color: %s" % document.color if document.color else None,
    ]
    text = "\n".join(part for part in parts if part)
    return text[:max_chars].rstrip()


def _bulk_payload(
    index_name: str,
    documents: Sequence[ProductDocument],
    vectors: Sequence[Sequence[float]],
    embedding_model_id: str,
) -> str:
    if len(documents) != len(vectors):
        raise ValueError("商品数量与向量数量不一致")
    lines: List[str] = []
    for document, vector in zip(documents, vectors):
        lines.append(json.dumps({"index": {"_index": index_name, "_id": document.product_id}}, separators=(",", ":")))
        body = document.to_dict()
        body["embedding_model_id"] = embedding_model_id
        body["embedding"] = list(vector)
        lines.append(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _bulk_counts(body: Mapping[str, Any], documents: Sequence[ProductDocument]) -> Dict[str, Any]:
    items = body.get("items")
    if not isinstance(items, list) or len(items) != len(documents):
        raise BackendError("backend_invalid_response", "OpenSearch bulk items 数量与请求不一致")
    failed_ids: List[str] = []
    successful = 0
    for document, item in zip(documents, items):
        action = next(iter(item.values())) if isinstance(item, Mapping) and item else None
        if not isinstance(action, Mapping) or action.get("error") is not None:
            failed_ids.append(document.product_id)
        else:
            successful += 1
    return {"successful": successful, "failed": len(failed_ids), "failed_ids": failed_ids}


def count_jsonl_documents(path: Path) -> int:
    """快速统计非空 JSONL 行数，用于显示建库总进度。"""

    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


@dataclass
class EmbeddingIndexReport:
    index_name: str
    input_path: str
    embedding_model_id: str
    dimension: int
    start_record: int = 1
    source_total: int = 0
    total: int = 0
    successful: int = 0
    failed: int = 0
    failed_ids: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    index_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def index_embeddings(
    input_path: Path,
    client: OpenSearchClient,
    embedding_client: EmbeddingClient,
    config: Mapping[str, Any],
    *,
    batch_size: int = 8,
    limit: Optional[int] = None,
    start_record: int = 1,
    max_text_chars: int = 6000,
) -> EmbeddingIndexReport:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是大于 0 的整数")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit 必须是大于 0 的整数")
    if isinstance(start_record, bool) or not isinstance(start_record, int) or start_record < 1:
        raise ValueError("start_record 必须是大于 0 的整数")
    if isinstance(max_text_chars, bool) or not isinstance(max_text_chars, int) or max_text_chars < 1:
        raise ValueError("max_text_chars 必须是大于 0 的整数")

    index_name = str(config_value(config, "retrieval", "hybrid_index_name", "amazon_products_v4"))
    started = time.monotonic()
    report = EmbeddingIndexReport(
        index_name=index_name,
        input_path=str(input_path),
        embedding_model_id=embedding_client.model,
        dimension=embedding_client.dimension,
        start_record=start_record,
    )
    report.index_status = client.ensure_index(index_name, build_hybrid_index_body(embedding_client.dimension))

    total_documents = count_jsonl_documents(input_path)
    report.source_total = total_documents
    if start_record > total_documents:
        raise ValueError("start_record 超出商品文件范围：文件共 %s 条，起始条目为 %s" % (total_documents, start_record))
    remaining_documents = total_documents - start_record + 1
    progress_total = min(limit, remaining_documents) if limit is not None else remaining_documents
    progress = ProgressBar(progress_total, " 向量建库")
    progress.update(0, status="从第 %s 条开始" % start_record)

    batch: List[ProductDocument] = []
    processed = 0
    batch_number = 0
    try:
        for source_record, document in enumerate(iter_jsonl_documents(input_path), start=1):
            if source_record < start_record:
                continue
            batch.append(document)
            processed += 1
            if len(batch) >= batch_size or (limit is not None and processed >= limit):
                batch_number += 1
                report.total += len(batch)
                progress.set_status("第 %s 批：正在请求 LM Studio" % batch_number)
                texts = [product_embedding_text(item, max_chars=max_text_chars) for item in batch]
                try:
                    vectors = embedding_client.embed_documents(texts)
                    counts = _bulk_counts(
                        client.bulk(_bulk_payload(index_name, batch, vectors, embedding_client.model)),
                        batch,
                    )
                    report.successful += counts["successful"]
                    report.failed += counts["failed"]
                    report.failed_ids.extend(counts["failed_ids"])
                except EmbeddingError:
                    # 不写入空向量；首个批次失败即停止，报告已处理的失败商品。
                    report.failed += len(batch)
                    report.failed_ids.extend(item.product_id for item in batch)
                    progress.update(len(batch), status="第 %s 批失败，停止" % batch_number)
                    batch = []
                    break
                progress.update(len(batch), status="已写入 OpenSearch")
                batch = []
                if limit is not None and processed >= limit:
                    break

        if batch:
            batch_number += 1
            report.total += len(batch)
            progress.set_status("第 %s 批：正在请求 LM Studio" % batch_number)
            texts = [product_embedding_text(item, max_chars=max_text_chars) for item in batch]
            try:
                vectors = embedding_client.embed_documents(texts)
                counts = _bulk_counts(
                    client.bulk(_bulk_payload(index_name, batch, vectors, embedding_client.model)),
                    batch,
                )
                report.successful += counts["successful"]
                report.failed += counts["failed"]
                report.failed_ids.extend(counts["failed_ids"])
                progress.update(len(batch), status="已写入 OpenSearch")
            except EmbeddingError:
                report.failed += len(batch)
                report.failed_ids.extend(item.product_id for item in batch)
                progress.update(len(batch), status="第 %s 批失败" % batch_number)
    finally:
        progress.close()

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
    parser = argparse.ArgumentParser(description="为商品生成向量并写入  OpenSearch 索引")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--input", type=Path, help="覆盖配置中的 products_jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="只处理前 N 条商品，用于 smoke test")
    parser.add_argument("--start-record", type=int, default=1, help="从第 N 条商品开始处理（从 1 开始，默认 1）")
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_search_config(config_path)
    project_root = config_path.parent.parent
    input_path = args.input or _resolve_path(str(config_value(config, "data", "products_jsonl")), project_root)
    report_path = args.report or _resolve_path(
        str(config_value(config, "data", "reports_dir", "reports/evaluation/")), project_root
    ) / "embedding-index-report.json"

    try:
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        embedding_client = EmbeddingClient(EmbeddingConfig.from_config(config))
        report = index_embeddings(
            input_path,
            client,
            embedding_client,
            config,
            batch_size=args.batch_size,
            limit=args.limit,
            start_record=args.start_record,
            max_text_chars=args.max_text_chars,
        )
        _write_json(report_path, report.to_dict())
    except (BackendError, EmbeddingError, FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"error": {"code": getattr(exc, "code", "embedding_index_failed"), "message": str(exc)}}, ensure_ascii=False))
        return 2

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
