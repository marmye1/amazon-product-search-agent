"""执行  数据契约、文件一致性和关联完整性校验。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

try:
    from .data_common import (
        EXAMPLE_STRING_COLUMNS,
        PRODUCT_TEXT_COLUMNS,
        SOURCE_STRING_COLUMNS,
        TABLE_NON_NULL_COLUMNS,
        TABLE_REQUIRED_COLUMNS,
        json_records,
        load_yaml,
        read_parquet,
        resolve_project_path,
        sha256_file,
        utc_now,
        write_json_atomic,
    )
except ImportError:  # 允许直接执行 python scripts/validate_dataset.py
    from data_common import (  # type: ignore
        EXAMPLE_STRING_COLUMNS,
        PRODUCT_TEXT_COLUMNS,
        SOURCE_STRING_COLUMNS,
        TABLE_NON_NULL_COLUMNS,
        TABLE_REQUIRED_COLUMNS,
        json_records,
        load_yaml,
        read_parquet,
        resolve_project_path,
        sha256_file,
        utc_now,
        write_json_atomic,
    )


PROCESSED_FILES = {
    "products": "products.parquet",
    "examples": "examples.parquet",
    "sources": "sources.parquet",
}


def issue(
    severity: str,
    check: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "message": message,
        "details": dict(details or {}),
    }


def _is_missing(value: Any) -> bool:
    result = pd.isna(value)
    return bool(result) if not hasattr(result, "__len__") else False


def _check_string_series(
    table_name: str,
    column: str,
    series: pd.Series,
) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    if is_numeric_dtype(series):
        problems.append(
            issue(
                "error",
                "%s.%s.type" % (table_name, column),
                "字段不能是数值类型，必须保留字符串",
                {"dtype": str(series.dtype)},
            )
        )
        return problems

    non_strings = [
        value
        for value in series.dropna().tolist()
        if not isinstance(value, str)
    ]
    if non_strings:
        problems.append(
            issue(
                "error",
                "%s.%s.type" % (table_name, column),
                "字段包含非字符串值",
                {"example_type": type(non_strings[0]).__name__},
            )
        )
    return problems


def validate_table_contract(table_name: str, frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """校验单表字段、非空规则、类型和枚举值，供 CLI 和 pytest 共用。"""

    problems: List[Dict[str, Any]] = []
    required = TABLE_REQUIRED_COLUMNS[table_name]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        problems.append(
            issue(
                "error",
                "%s.schema.required_columns" % table_name,
                "缺少必需字段",
                {"missing_columns": missing_columns},
            )
        )
        return problems

    for column in TABLE_NON_NULL_COLUMNS[table_name]:
        null_count = int(frame[column].isna().sum())
        if null_count:
            problems.append(
                issue(
                    "error",
                    "%s.%s.not_null" % (table_name, column),
                    "必填字段存在空值",
                    {"null_count": null_count},
                )
            )

    if table_name == "products":
        string_columns = ["product_id", "product_locale"] + PRODUCT_TEXT_COLUMNS
    elif table_name == "examples":
        string_columns = EXAMPLE_STRING_COLUMNS + ["product_locale", "esci_label", "split"]
    else:
        string_columns = SOURCE_STRING_COLUMNS

    for column in string_columns:
        problems.extend(_check_string_series(table_name, column, frame[column]))

    if table_name == "products":
        locale_values = set(frame["product_locale"].dropna().astype(str))
        uppercase_values = sorted(value for value in locale_values if value != value.lower())
        if uppercase_values:
            problems.append(
                issue(
                    "error",
                    "products.product_locale.normalized",
                    "product_locale 必须统一为小写",
                    {"values": uppercase_values[:10]},
                )
            )

    if table_name == "examples":
        labels = set(frame["esci_label"].dropna().astype(str))
        invalid_labels = sorted(labels - {"E", "S", "C", "I"})
        if invalid_labels:
            problems.append(
                issue(
                    "error",
                    "examples.esci_label.enum",
                    "esci_label 存在非法值",
                    {"invalid_values": invalid_labels},
                )
            )

        splits = set(frame["split"].dropna().astype(str))
        invalid_splits = sorted(splits - {"train", "test"})
        if invalid_splits:
            problems.append(
                issue(
                    "error",
                    "examples.split.enum",
                    "split 存在非法值",
                    {"invalid_values": invalid_splits},
                )
            )
        for column in ("small_version", "large_version"):
            if not is_numeric_dtype(frame[column]):
                problems.append(
                    issue(
                        "error",
                        "examples.%s.type" % column,
                        "版本标识必须是 integer 或 bool",
                        {"dtype": str(frame[column].dtype)},
                    )
                )

    return problems


def _table_statistics(table_name: str, frame: pd.DataFrame) -> Dict[str, Any]:
    nulls: Dict[str, Dict[str, Any]] = {}
    for column in frame.columns:
        count = int(frame[column].isna().sum())
        nulls[column] = {
            "count": count,
            "ratio": (count / len(frame)) if len(frame) else 0.0,
        }

    lengths: Dict[str, int] = {}
    for column in frame.columns:
        if frame[column].dtype == "object" or pd.api.types.is_string_dtype(frame[column]):
            values = frame[column].dropna().astype(str)
            lengths[column] = int(values.str.len().max()) if not values.empty else 0

    result: Dict[str, Any] = {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "nulls": nulls,
        "max_text_lengths": lengths,
        "sample": json_records(frame),
    }
    if "product_locale" in frame.columns:
        result["locale_distribution"] = {
            str(key): int(value) for key, value in frame["product_locale"].value_counts(dropna=False).items()
        }
    if "esci_label" in frame.columns:
        result["esci_label_distribution"] = {
            str(key): int(value) for key, value in frame["esci_label"].value_counts(dropna=False).items()
        }
    if "split" in frame.columns:
        result["split_distribution"] = {
            str(key): int(value) for key, value in frame["split"].value_counts(dropna=False).items()
        }
    return result


def validate_relationships(
    products: pd.DataFrame,
    examples: pd.DataFrame,
    sources: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """校验主键、复合关联键，并统计 sources 的可关联情况。"""

    problems: List[Dict[str, Any]] = []
    product_duplicates = int(products.duplicated(["product_locale", "product_id"]).sum())
    if product_duplicates:
        problems.append(
            issue(
                "error",
                "products.logical_primary_key.unique",
                "products 逻辑主键重复",
                {"duplicate_count": product_duplicates},
            )
        )

    example_duplicates = int(examples.duplicated(["example_id"]).sum())
    if example_duplicates:
        problems.append(
            issue(
                "error",
                "examples.example_id.unique",
                "examples.example_id 重复",
                {"duplicate_count": example_duplicates},
            )
        )

    product_keys = products[["product_locale", "product_id"]].drop_duplicates()
    example_keys = examples[["product_locale", "product_id"]].drop_duplicates()
    joined = example_keys.merge(
        product_keys,
        how="left",
        on=["product_locale", "product_id"],
        indicator=True,
    )
    unmatched = int((joined["_merge"] == "left_only").sum())
    if unmatched:
        problems.append(
            issue(
                "error",
                "examples.products.referential_integrity",
                "examples 存在无法关联的商品复合键",
                {"unmatched_key_count": unmatched},
            )
        )

    source_query_ids = set(sources["query_id"].dropna().astype(str))
    example_query_ids = set(examples["query_id"].dropna().astype(str))
    missing_sources = example_query_ids - source_query_ids
    if missing_sources:
        problems.append(
            issue(
                "warning",
                "examples.sources.query_id.coverage",
                "部分 examples.query_id 在 sources 中没有对应记录",
                {
                    "missing_query_id_count": len(missing_sources),
                    "sample_query_ids": sorted(missing_sources)[:10],
                },
            )
        )
    return problems


def _manifest_checks(manifest_path: Path, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    if manifest.get("processing", {}).get("schema_version") != "":
        problems.append(
            issue(
                "error",
                "manifest.schema_version",
                "Manifest schema_version 不是 ",
                {"actual": manifest.get("processing", {}).get("schema_version")},
            )
        )

    raw_dir_value = manifest.get("raw_dir", "data/raw/esci")
    raw_dir = resolve_project_path(raw_dir_value, manifest_path.parent.parent)
    raw_files = manifest.get("raw_files")
    if not isinstance(raw_files, list) or not raw_files:
        return problems + [issue("error", "manifest.raw_files", "Manifest 缺少 raw_files")]

    for item in raw_files:
        if not isinstance(item, dict):
            problems.append(issue("error", "manifest.raw_files.item", "raw_files 中存在非法项"))
            continue
        name = item.get("name")
        expected_hash = item.get("sha256")
        expected_size = item.get("size_bytes")
        if not isinstance(name, str) or not name:
            problems.append(issue("error", "manifest.raw_files.name", "原始文件缺少 name"))
            continue
        path = raw_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            problems.append(
                issue("error", "manifest.raw_files.exists", "原始文件不存在或为空", {"file": str(path)})
            )
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected_size:
            problems.append(
                issue(
                    "error",
                    "manifest.raw_files.size",
                    "原始文件大小与 Manifest 不一致",
                    {"file": name, "expected": expected_size, "actual": actual_size},
                )
            )
        if actual_hash != expected_hash:
            problems.append(
                issue(
                    "error",
                    "manifest.raw_files.sha256",
                    "原始文件 SHA-256 与 Manifest 不一致",
                    {"file": name, "expected": expected_hash, "actual": actual_hash},
                )
            )
    return problems


def validate(manifest_path: Path, data_dir: Path, report_path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "",
        "generated_at_utc": utc_now(),
        "status": "failed",
        "manifest": {},
        "tables": {},
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    try:
        manifest = load_yaml(manifest_path)
        report["manifest"] = {
            "dataset_name": manifest.get("dataset_name"),
            "downloaded_at_utc": manifest.get("downloaded_at_utc"),
            "locale_scope": manifest.get("processing", {}).get("locale_scope"),
            "version_scope": manifest.get("processing", {}).get("version_scope"),
        }
        manifest_problems = _manifest_checks(manifest_path, manifest)
    except Exception as exc:
        manifest = {}
        manifest_problems = [issue("error", "manifest.read", "Manifest 无法读取", {"error": str(exc)})]

    for problem in manifest_problems:
        report["checks"].append(problem)

    tables: Dict[str, pd.DataFrame] = {}
    for table_name, filename in PROCESSED_FILES.items():
        path = data_dir / filename
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError("文件不存在或为空: %s" % path)
            frame = read_parquet(path)
            tables[table_name] = frame
            report["tables"][table_name] = _table_statistics(table_name, frame)
            for problem in validate_table_contract(table_name, frame):
                report["checks"].append(problem)
        except Exception as exc:
            report["checks"].append(
                issue("error", "%s.read" % table_name, "标准数据无法读取", {"file": str(path), "error": str(exc)})
            )

    if set(tables) == {"products", "examples", "sources"}:
        for problem in validate_relationships(tables["products"], tables["examples"], tables["sources"]):
            report["checks"].append(problem)

    report["errors"] = [item for item in report["checks"] if item["severity"] == "error"]
    report["warnings"] = [item for item in report["checks"] if item["severity"] == "warning"]
    report["blocking_error_count"] = len(report["errors"])
    report["warning_count"] = len(report["warnings"])
    report["status"] = "passed" if not report["errors"] else "failed"
    write_json_atomic(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 ESCI  数据契约")
    parser.add_argument("--manifest", required=True, type=Path, help="MANIFEST.yaml 路径")
    parser.add_argument("--data-dir", required=True, type=Path, help="processed 数据目录")
    parser.add_argument("--report", required=True, type=Path, help="质量报告路径")
    args = parser.parse_args()
    try:
        report = validate(args.manifest.resolve(), args.data_dir.resolve(), args.report.resolve())
    except (OSError, ValueError) as exc:
        print("错误: %s" % exc)
        return 1

    print(
        " 校验 %s：%s 个阻塞错误，%s 个警告"
        % ("通过" if report["status"] == "passed" else "失败", report["blocking_error_count"], report["warning_count"])
    )
    print("质量报告: %s" % args.report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
