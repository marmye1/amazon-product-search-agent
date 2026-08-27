"""用户可见回答的语言兜底规则。"""

from __future__ import annotations

import re
from typing import Iterable, List


_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def contains_chinese(value: str) -> bool:
    return bool(_CJK_PATTERN.search(value or ""))


def chinese_or_fallback(value: str, fallback: str) -> str:
    """保留模型的中文表达；模型返回纯英文时使用不涉及商品事实的中文兜底。"""

    cleaned = value.strip() if isinstance(value, str) else ""
    return cleaned if cleaned and contains_chinese(cleaned) else fallback


def chinese_list_or_fallback(values: Iterable[str], fallback: str) -> List[str]:
    return [chinese_or_fallback(value, fallback) for value in values]
