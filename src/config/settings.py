"""类型化应用配置。

配置层只读取环境变量和现有 search.yaml，不保存用户请求，也不保存账号密码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..generate_recommendation import LocalQwenConfig
from ..search_config import load_search_config


class SettingsError(RuntimeError):
    """配置缺失或不满足固定架构时抛出。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError("invalid_config", "%s 必须是整数" % name) from exc
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise SettingsError("invalid_config", "%s 必须是布尔值" % name)


@dataclass(frozen=True)
class AppSettings:
    """服务运行所需的非敏感配置。"""

    app_version: str = "production"
    agent_version: str = "production"
    search_config_path: Path = Path("config/search.yaml")
    retrieval_method: str = "hybrid_rrf"
    default_chat_top_k: int = 5
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    health_timeout_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "AppSettings":
        retrieval_method = os.getenv("RETRIEVAL_METHOD", cls.retrieval_method).strip().lower()
        if retrieval_method not in {"hybrid_rrf"}:
            raise SettingsError(
                "invalid_config",
                "当前 Agent 必须使用 hybrid_rrf 检索，RETRIEVAL_METHOD 必须是 hybrid_rrf",
            )
        search_config_path = Path(
            os.getenv("SEARCH_CONFIG_PATH", str(cls.search_config_path))
        ).expanduser()
        return cls(
            app_version=os.getenv("APP_VERSION", cls.app_version),
            agent_version=os.getenv("AGENT_VERSION", cls.agent_version),
            search_config_path=search_config_path,
            retrieval_method=retrieval_method,
            default_chat_top_k=_env_int("DEFAULT_CHAT_TOP_K", cls.default_chat_top_k),
            api_host=os.getenv("API_HOST", cls.api_host),
            api_port=_env_int("API_PORT", cls.api_port),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
            health_timeout_seconds=float(
                os.getenv("HEALTH_TIMEOUT_SECONDS", str(cls.health_timeout_seconds))
            ),
        )

    def resolved_search_config_path(self, project_root: Path | None = None) -> Path:
        path = self.search_config_path
        if path.is_absolute():
            return path
        return (project_root or Path.cwd()) / path

    def load_search_config(self, project_root: Path | None = None):
        path = self.resolved_search_config_path(project_root)
        if not path.exists():
            raise SettingsError("missing_config", "找不到搜索配置文件: %s" % path)
        try:
            config = load_search_config(path)
            opensearch: Dict[str, Any] = dict(config.get("opensearch", {}))
            retrieval: Dict[str, Any] = dict(config.get("retrieval", {}))
            if os.getenv("OPENSEARCH_BASE_URL"):
                opensearch["base_url"] = os.getenv("OPENSEARCH_BASE_URL")
            if os.getenv("OPENSEARCH_VERIFY_SSL") is not None:
                opensearch["verify_ssl"] = _env_bool(
                    "OPENSEARCH_VERIFY_SSL",
                    bool(opensearch.get("verify_ssl", True)),
                )
            if os.getenv("OPENSEARCH_INDEX_NAME"):
                opensearch["index_name"] = os.getenv("OPENSEARCH_INDEX_NAME")
            if os.getenv("HYBRID_INDEX_NAME"):
                retrieval["hybrid_index_name"] = os.getenv("HYBRID_INDEX_NAME")
            config["opensearch"] = opensearch
            config["retrieval"] = retrieval
            return config
        except (OSError, ValueError) as exc:
            raise SettingsError("invalid_config", "搜索配置不可用: %s" % exc) from exc

    def qwen_config(self) -> LocalQwenConfig:
        config = LocalQwenConfig.from_env()
        if not config.base_url.strip() or not config.model.strip():
            raise SettingsError("missing_llm_config", "Qwen 的 base_url 和 model 不能为空")
        return config
