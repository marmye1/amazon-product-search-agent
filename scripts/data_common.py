""" 脚本共享的路径、配置、数据契约和文件工具。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


TABLE_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "products": [
        "product_id",
        "product_locale",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ],
    "examples": [
        "example_id",
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "esci_label",
        "small_version",
        "large_version",
        "split",
    ],
    "sources": ["query_id", "source"],
}

TABLE_NON_NULL_COLUMNS: Dict[str, List[str]] = {
    "products": ["product_id", "product_locale", "product_title"],
    "examples": [
        "example_id",
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "esci_label",
        "small_version",
        "large_version",
        "split",
    ],
    "sources": ["query_id"],
}

PRODUCT_TEXT_COLUMNS = [
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]
EXAMPLE_STRING_COLUMNS = [
    "example_id",
    "query",
    "query_id",
    "product_id",
]
SOURCE_STRING_COLUMNS = ["query_id", "source"]


def utc_now() -> str:
    """返回机器可读的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML，并确保顶层是对象。"""

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("YAML 顶层必须是对象")
    return value


def write_yaml_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """以临时文件加原子替换写入 YAML，避免留下半份清单。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                dict(value),
                handle,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        os.replace(temp_name, str(path))
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """以临时文件加原子替换写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
            handle.write("\n")
        os.replace(temp_name, str(path))
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def json_default(value: Any) -> Any:
    """把常见的 NumPy/Pandas 值转换成 JSON 基础类型。"""

    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("无法序列化类型: %s" % type(value).__name__)


def resolve_project_path(value: str, project_root: Path) -> Path:
    """将配置中的相对路径解析到项目根目录。"""

    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def config_project_root(config_path: Path) -> Path:
    """根据 configs/data.yaml 推导项目根目录。"""

    return config_path.resolve().parent.parent


def file_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """读取并校验 data.yaml 中的文件配置。"""

    raw_specs = config.get("files")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("files 必须是非空列表")

    result: List[Dict[str, Any]] = []
    for raw_spec in raw_specs:
        if isinstance(raw_spec, str):
            spec = {"name": raw_spec}
        elif isinstance(raw_spec, dict):
            spec = dict(raw_spec)
        else:
            raise ValueError("files 中的每一项必须是文件名或对象")

        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("每个数据文件必须有非空 name")
        name_path = Path(name)
        if name_path.is_absolute() or ".." in name_path.parts or name_path.name != name:
            raise ValueError("数据文件名必须是单层相对文件名: %s" % name)
        url = spec.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("数据文件必须配置 http/https URL: %s" % name)
        result.append(spec)
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256，不把整个文件载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_nullable_text(series: pd.Series) -> pd.Series:
    """统一文本列类型，并把空字符串和纯空白识别为缺失值。"""

    values = series.astype("string")
    blank = values.str.strip().eq("")
    return values.mask(blank, pd.NA)


def normalize_locale(series: pd.Series) -> pd.Series:
    """将 locale 标准化为小写字符串。"""

    values = series.astype("string").str.strip().str.lower()
    return values.mask(values.eq(""), pd.NA)


def normalize_enum(series: pd.Series, upper: bool = False) -> pd.Series:
    """标准化枚举字段的空白和大小写。"""

    values = series.astype("string").str.strip()
    if upper:
        values = values.str.upper()
    else:
        values = values.str.lower()
    return values.mask(values.eq(""), pd.NA)


def read_parquet(path: Path, filters: Optional[Sequence[Tuple[str, str, Any]]] = None) -> pd.DataFrame:
    """通过 PyArrow 读取 Parquet，必要时下推过滤条件。"""

    kwargs: Dict[str, Any] = {"engine": "pyarrow"}
    if filters:
        kwargs["filters"] = list(filters)
    return pd.read_parquet(path, **kwargs)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """写入 Parquet，并在成功后替换目标文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(".%s.tmp" % path.name)
    try:
        frame.to_parquet(temp_path, engine="pyarrow", index=False)
        os.replace(str(temp_path), str(path))
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def json_records(frame: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    """生成报告用的 JSON 记录，正确处理缺失值。"""

    if frame.empty:
        return []
    return json.loads(frame.head(limit).to_json(orient="records", force_ascii=False))
