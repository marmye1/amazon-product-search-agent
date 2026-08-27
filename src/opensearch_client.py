"""OpenSearch REST API 的最小客户端。"""

from __future__ import annotations

import os
import getpass
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import requests
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning


class BackendError(RuntimeError):
    """后端不可用、超时或返回非法响应。"""

    def __init__(self, code: str, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.status_code is not None:
            result["status_code"] = self.status_code
        return result


def _resolve_credentials(
    username_env: str,
    password_env: str,
    *,
    prompt_for_missing: bool,
    input_func: Optional[Callable[[str], str]] = None,
    password_func: Optional[Callable[[str], str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """读取凭证；命令行入口可要求对缺失项进行交互式补录。"""

    username = os.getenv(username_env) or None
    password = os.getenv(password_env) or None
    if not prompt_for_missing or (username is not None and password is not None):
        return username, password

    # 真实命令行需要可交互终端，测试可以注入输入函数而不触碰用户终端。
    if input_func is None and password_func is None and not sys.stdin.isatty():
        raise BackendError(
            "missing_credentials",
            "缺少 OpenSearch 账号密码，当前不是可交互终端；请设置 %s 和 %s 后重试"
            % (username_env, password_env),
        )

    if username is None:
        username_reader = input_func or input
        username = username_reader("请输入 OpenSearch 用户名: ").strip() or None
    if password is None:
        password_reader = password_func or getpass.getpass
        password = password_reader("请输入 OpenSearch 密码: ") or None

    if username is None or password is None:
        raise BackendError(
            "missing_credentials",
            "OpenSearch 用户名和密码不能为空；也可以设置 %s 和 %s 后重试"
            % (username_env, password_env),
        )
    return username, password


class OpenSearchClient:
    """只封装  需要的索引、批量写入和搜索操作。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: bool = True,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OpenSearch base_url 必须是 http/https URL")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl
        self.session = session or requests.Session()
        self.auth = (username, password) if username is not None and password is not None else None
        if self.base_url.startswith("https://") and not self.verify_ssl:
            # 本项目使用本地 OpenSearch 的自签名证书；验证已按配置关闭，避免每个请求重复刷屏。
            disable_warnings(InsecureRequestWarning)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        prompt_for_missing: bool = False,
        input_func: Optional[Callable[[str], str]] = None,
        password_func: Optional[Callable[[str], str]] = None,
    ) -> "OpenSearchClient":
        section = config.get("opensearch", {})
        if not isinstance(section, Mapping):
            raise ValueError("opensearch 配置必须是对象")
        username_env = str(section.get("username_env", "OPENSEARCH_USERNAME"))
        password_env = str(section.get("password_env", "OPENSEARCH_PASSWORD"))
        username, password = _resolve_credentials(
            username_env,
            password_env,
            prompt_for_missing=prompt_for_missing,
            input_func=input_func,
            password_func=password_func,
        )
        return cls(
            str(section.get("base_url", "http://localhost:9200")),
            timeout_seconds=float(section.get("request_timeout_seconds", 10)),
            username=username,
            password=password,
            verify_ssl=bool(section.get("verify_ssl", True)),
        )

    def _send(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = "%s/%s" % (self.base_url, path.lstrip("/"))
        kwargs.setdefault("timeout", self.timeout_seconds)
        kwargs.setdefault("auth", self.auth)
        kwargs.setdefault("verify", self.verify_ssl)
        try:
            return self.session.request(method, url, **kwargs)
        except requests.Timeout as exc:
            raise BackendError("backend_timeout", "OpenSearch 请求超时") from exc
        except requests.ConnectionError as exc:
            raise BackendError("backend_unavailable", "无法连接 OpenSearch: %s" % self.base_url) from exc
        except requests.RequestException as exc:
            raise BackendError("backend_request_error", "OpenSearch 请求失败: %s" % exc) from exc

    @staticmethod
    def _response_body(response: requests.Response) -> Any:
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise BackendError("backend_invalid_response", "OpenSearch 返回的不是合法 JSON") from exc

    def _json_request(self, method: str, path: str, *, json_body: Optional[Mapping[str, Any]] = None) -> Any:
        response = self._send(method, path, json=json_body)
        if response.status_code >= 400:
            body = response.text.strip().replace("\n", " ")[:500]
            if response.status_code in (401, 403):
                raise BackendError(
                    "backend_auth_error",
                    "OpenSearch 认证失败（HTTP %s），请检查账号密码" % response.status_code,
                    status_code=response.status_code,
                )
            raise BackendError(
                "backend_http_error",
                "OpenSearch 返回 HTTP %s: %s" % (response.status_code, body),
                status_code=response.status_code,
            )
        return self._response_body(response)

    def index_exists(self, index_name: str) -> bool:
        response = self._send("HEAD", index_name)
        if response.status_code == 404:
            return False
        if response.status_code >= 400:
            if response.status_code in (401, 403):
                raise BackendError(
                    "backend_auth_error",
                    "OpenSearch 认证失败（HTTP %s），请检查账号密码" % response.status_code,
                    status_code=response.status_code,
                )
            raise BackendError(
                "backend_http_error",
                "检查索引失败，HTTP %s" % response.status_code,
                status_code=response.status_code,
            )
        return True

    def create_index(self, index_name: str, body: Mapping[str, Any]) -> Any:
        return self._json_request("PUT", index_name, json_body=body)

    def get_mapping(self, index_name: str) -> Any:
        """读取索引映射，用于真实验收时确认字段类型和向量维度。"""

        return self._json_request("GET", "%s/_mapping" % index_name)

    def refresh(self, index_name: str) -> Any:
        """刷新索引，使刚写入的商品立即进入搜索视图。"""

        return self._json_request("POST", "%s/_refresh" % index_name)

    def ensure_index(self, index_name: str, body: Mapping[str, Any]) -> str:
        if self.index_exists(index_name):
            return "exists"
        try:
            self.create_index(index_name, body)
        except BackendError as exc:
            # 多个导入进程同时首次创建时，409 表示另一个进程已经创建成功。
            if exc.status_code != 409:
                raise
            return "exists"
        return "created"

    def bulk(self, payload: str) -> Mapping[str, Any]:
        response = self._send(
            "POST",
            "_bulk",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        if response.status_code >= 400:
            body = response.text.strip().replace("\n", " ")[:500]
            if response.status_code in (401, 403):
                raise BackendError(
                    "backend_auth_error",
                    "OpenSearch 认证失败（HTTP %s），请检查账号密码" % response.status_code,
                    status_code=response.status_code,
                )
            raise BackendError(
                "backend_http_error",
                "OpenSearch bulk 失败，HTTP %s: %s" % (response.status_code, body),
                status_code=response.status_code,
            )
        body = self._response_body(response)
        if not isinstance(body, Mapping):
            raise BackendError("backend_invalid_response", "OpenSearch bulk 返回结构无效")
        return body

    def search(self, index_name: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        result = self._json_request("POST", "%s/_search" % index_name, json_body=body)
        if not isinstance(result, Mapping):
            raise BackendError("backend_invalid_response", "OpenSearch search 返回结构无效")
        return result
