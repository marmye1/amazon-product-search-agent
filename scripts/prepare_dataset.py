"""把 ESCI 原始文件转换为  标准 Parquet 数据。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

try:
    from .data_common import (
        EXAMPLE_STRING_COLUMNS,
        PRODUCT_TEXT_COLUMNS,
        SOURCE_STRING_COLUMNS,
        TABLE_REQUIRED_COLUMNS,
        config_project_root,
        json_records,
        load_yaml,
        normalize_enum,
        normalize_locale,
        normalize_nullable_text,
        read_parquet,
        resolve_project_path,
        write_json_atomic,
        write_parquet_atomic,
    )
except ImportError:  # 允许直接执行 python scripts/prepare_dataset.py
    from data_common import (  # type: ignore
        EXAMPLE_STRING_COLUMNS,
        PRODUCT_TEXT_COLUMNS,
        SOURCE_STRING_COLUMNS,
        TABLE_REQUIRED_COLUMNS,
        config_project_root,
        json_records,
        load_yaml,
        normalize_enum,
        normalize_locale,
        normalize_nullable_text,
        read_parquet,
        resolve_project_path,
        write_json_atomic,
        write_parquet_atomic,
    )


RAW_FILES = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
    "sources": "shopping_queries_dataset_sources.csv",
}


def _assert_columns(frame: pd.DataFrame, table_name: str) -> None:
    missing = [column for column in TABLE_REQUIRED_COLUMNS[table_name] if column not in frame.columns]
    if missing:
        raise ValueError("%s 缺少必需字段: %s" % (table_name, ", ".join(missing)))


def _preserve_column_order(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    required = TABLE_REQUIRED_COLUMNS[table_name]
    extra = [column for column in frame.columns if column not in required]
    return frame[required + extra]


def _read_raw_tables(raw_dir: Path, locale_scope: str) -> Dict[str, pd.DataFrame]:
    examples_path = raw_dir / RAW_FILES["examples"]
    products_path = raw_dir / RAW_FILES["products"]
    sources_path = raw_dir / RAW_FILES["sources"]
    for path in (examples_path, products_path, sources_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("原始文件不存在或为空: %s" % path)

    locale = str(locale_scope or "all").lower()
    parquet_filter = None if locale in ("all", "*") else [("product_locale", "==", locale)]
    examples = read_parquet(examples_path, parquet_filter)
    products = read_parquet(products_path, parquet_filter)
    sources = pd.read_csv(sources_path, dtype={"query_id": "string", "source": "string"})
    return {"examples": examples, "products": products, "sources": sources}


def standardize_products(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_columns(frame, "products")
    result = frame.copy()
    result["product_id"] = result["product_id"].astype("string")
    result["product_locale"] = normalize_locale(result["product_locale"])
    for column in PRODUCT_TEXT_COLUMNS:
        result[column] = normalize_nullable_text(result[column])
    result = _preserve_column_order(result, "products")
    return result.sort_values(["product_locale", "product_id"], kind="mergesort").reset_index(drop=True)


def standardize_examples(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_columns(frame, "examples")
    result = frame.copy()
    for column in EXAMPLE_STRING_COLUMNS:
        result[column] = normalize_nullable_text(result[column])
    result["product_locale"] = normalize_locale(result["product_locale"])
    result["esci_label"] = normalize_enum(result["esci_label"], upper=True)
    result["split"] = normalize_enum(result["split"], upper=False)
    for column in ("small_version", "large_version"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("Int8")
    result = _preserve_column_order(result, "examples")
    return result.sort_values(["product_locale", "example_id"], kind="mergesort").reset_index(drop=True)


def standardize_sources(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_columns(frame, "sources")
    result = frame.copy()
    for column in SOURCE_STRING_COLUMNS:
        result[column] = normalize_nullable_text(result[column])
    result = _preserve_column_order(result, "sources")
    return result.sort_values(["query_id"], kind="mergesort").reset_index(drop=True)


def apply_example_scope(
    examples: pd.DataFrame,
    locale_scope: str,
    version_scope: str,
) -> pd.DataFrame:
    """按配置选择 locale 和 small/large 版本，保留原始标签字段。"""

    mask = pd.Series(True, index=examples.index)
    locale = str(locale_scope or "all").lower()
    if locale not in ("all", "*"):
        mask &= examples["product_locale"].eq(locale)

    version = str(version_scope or "all").lower()
    if version in ("small_version", "large_version"):
        if version not in examples.columns:
            raise ValueError("examples 缺少版本字段: %s" % version)
        mask &= examples[version].eq(1)
    elif version not in ("all", "*"):
        raise ValueError("version_scope 只支持 all、small_version 或 large_version")
    return examples.loc[mask].copy()


def prepare(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    project_root = config_project_root(config_path)
    raw_dir = resolve_project_path(config["raw_dir"], project_root)
    processed_dir = resolve_project_path(config["processed_dir"], project_root)
    reports_dir = resolve_project_path(config["reports_dir"], project_root)

    raw_tables = _read_raw_tables(raw_dir, str(config.get("locale_scope", "all")))
    examples = standardize_examples(raw_tables["examples"])
    products = standardize_products(raw_tables["products"])
    sources = standardize_sources(raw_tables["sources"])

    examples = apply_example_scope(
        examples,
        str(config.get("locale_scope", "all")),
        str(config.get("version_scope", "all")),
    )

    # sources 只保留当前 examples 范围内的 query，避免把未选版本的来源带入标准数据。
    selected_query_ids = set(examples["query_id"].dropna().astype(str))
    sources = sources[sources["query_id"].astype("string").isin(selected_query_ids)].copy()
    sources = sources.sort_values(["query_id"], kind="mergesort").reset_index(drop=True)

    output_paths = {
        "products": processed_dir / "products.parquet",
        "examples": processed_dir / "examples.parquet",
        "sources": processed_dir / "sources.parquet",
    }
    write_parquet_atomic(products, output_paths["products"])
    write_parquet_atomic(examples, output_paths["examples"])
    write_parquet_atomic(sources, output_paths["sources"])

    summary: Dict[str, Any] = {
        "schema_version": "",
        "locale_scope": config.get("locale_scope", "all"),
        "version_scope": config.get("version_scope", "all"),
        "tables": {
            "products": {"rows": len(products), "sample": json_records(products)},
            "examples": {"rows": len(examples), "sample": json_records(examples)},
            "sources": {"rows": len(sources), "sample": json_records(sources)},
        },
        "outputs": {name: str(path.relative_to(project_root)) for name, path in output_paths.items()},
    }
    write_json_atomic(reports_dir / "-preparation-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ESCI  标准数据")
    parser.add_argument("--config", required=True, type=Path, help="data.yaml 路径")
    args = parser.parse_args()
    try:
        summary = prepare(args.config)
    except (KeyError, ValueError, FileNotFoundError, OSError, ImportError) as exc:
        print("错误: %s" % exc)
        return 1

    print("标准化完成:")
    for table_name, table in summary["tables"].items():
        print("  %s: %s 行" % (table_name, table["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
