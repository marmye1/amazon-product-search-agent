"""搜索配置读取。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_search_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("搜索配置顶层必须是对象")
    for section in ("opensearch", "search", "data"):
        if not isinstance(value.get(section), Mapping):
            raise ValueError("搜索配置缺少对象字段: %s" % section)
    return value


def config_value(config: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {})
    if not isinstance(value, Mapping):
        return default
    return value.get(key, default)
