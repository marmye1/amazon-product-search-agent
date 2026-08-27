"""把商品 Parquet 流式适配为 ProductDocument JSONL。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pyarrow.parquet as pq

from .models import ProductDocument
from .search_config import config_value, load_search_config


_PRODUCT_COLUMNS = [
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def iter_product_documents(parquet_path: Path, *, data_provenance: str = "amazon_esci_v0") -> Iterable[ProductDocument]:
    """按批读取商品表，保留原文，不把 120 万行一次性放入内存。"""

    parquet = pq.ParquetFile(parquet_path)
    available = set(parquet.schema_arrow.names)
    missing = [column for column in _PRODUCT_COLUMNS if column not in available]
    if missing:
        raise ValueError("products.parquet 缺少必需字段: %s" % ", ".join(missing))

    seen_ids = set()
    for batch in parquet.iter_batches(columns=_PRODUCT_COLUMNS, batch_size=10000):
        for line_number, row in enumerate(batch.to_pylist(), start=1):
            product_id = _text(row.get("product_id"))
            locale = _text(row.get("product_locale"))
            title = _text(row.get("product_title"))
            if not product_id:
                raise ValueError("product_id 不能为空（批次内第 %s 条）" % line_number)
            if product_id in seen_ids:
                raise ValueError("product_id 重复，不允许静默覆盖: %s" % product_id)
            seen_ids.add(product_id)
            if not locale:
                raise ValueError("product_locale 不能为空: %s" % product_id)
            if not title:
                raise ValueError("product_title 不能为空: %s" % product_id)

            normalized_locale = locale.lower()
            yield ProductDocument(
                product_id=product_id,
                locale=normalized_locale,
                title=title,
                brand=_text(row.get("product_brand")),
                category=None,
                description=_text(row.get("product_description")),
                #  是一个原始 bullet-point 文本字段。为了不猜测分隔规则，
                # 这里保留为只有一个元素的数组，搜索时仍可全文匹配。
                bullet_points=([_text(row.get("product_bullet_point"))]
                               if _text(row.get("product_bullet_point")) else []),
                source_url=None,
                data_provenance=data_provenance,
                source_ref="esci:v0:products:%s:%s" % (normalized_locale, product_id),
                color=_text(row.get("product_color")),
            )


def export_products_jsonl(
    parquet_path: Path,
    jsonl_path: Path,
    *,
    data_provenance: str = "amazon_esci_v0",
) -> Dict[str, Any]:
    """原子写出 ProductDocument JSONL，并返回导出统计。"""

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = jsonl_path.with_name(".%s.tmp" % jsonl_path.name)
    total = 0
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for document in iter_product_documents(parquet_path, data_provenance=data_provenance):
                handle.write(json.dumps(document.to_dict(), ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                total += 1
        os.replace(str(temp_path), str(jsonl_path))
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "input": str(parquet_path),
        "output": str(jsonl_path),
        "rows": total,
        "data_provenance": data_provenance,
        "mapping": {
            "product_locale": "locale",
            "product_title": "title",
            "product_brand": "brand",
            "product_description": "description",
            "product_bullet_point": "bullet_points[0]（保留原文，不拆分）",
            "product_color": "color",
            "category": "null（ 无类目字段）",
            "source_url": "null（ 未提供）",
        },
    }


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main() -> int:
    parser = argparse.ArgumentParser(description="导出  ProductDocument JSONL")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--input", type=Path, help="覆盖配置中的 products_parquet")
    parser.add_argument("--output", type=Path, help="覆盖配置中的 products_jsonl")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_search_config(config_path)
    project_root = config_path.parent.parent
    input_path = args.input or _resolve_path(str(config_value(config, "data", "products_parquet")), project_root)
    output_path = args.output or _resolve_path(str(config_value(config, "data", "products_jsonl")), project_root)
    summary = export_products_jsonl(input_path, output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
