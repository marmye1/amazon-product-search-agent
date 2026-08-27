"""Agent 服务层。

API 只把 HTTP 请求转换成这里的输入；这里负责创建并调用单 Agent，
再把内部状态转换成稳定的服务结果。不会在模块级保存用户请求。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import requests

from ..config.settings import AppSettings
from ..conversation_memory import ConversationMemoryError, extract_conversation_memory
from ..embedding_client import EmbeddingClient, EmbeddingConfig
from ..generate_recommendation import LocalQwenConfig
from ..graph.build_graph import build_graph
from ..graph.state import to_serializable
from ..hybrid_search_tool import HybridSearchTool
from ..models import ContractError, SearchRequest, SearchResponse
from ..opensearch_client import BackendError, OpenSearchClient
from ..observability.tracing import log_event, query_fingerprint
from ..session_context import SessionContextStore


class ServiceError(RuntimeError):
    """服务配置或依赖异常；由 API 层转换为稳定错误响应。"""

    def __init__(self, code: str, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class AgentRuntime:
    """一次性构造的共享运行时；每次请求只向图传入独立 state。"""

    search_config: Mapping[str, Any]
    client: Any
    search_tool: Any
    graph: Any
    llm_config: LocalQwenConfig
    app_version: str
    agent_version: str
    retrieval_method: str
    index_name: str
    embedding_model_id: str
    health_timeout_seconds: float = 2.0

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AgentRuntime":
        search_config = settings.load_search_config()
        if settings.retrieval_method != "hybrid_rrf":
            raise ServiceError(
                "invalid_retrieval_config",
                "Agent 需要 hybrid_rrf 检索配置",
                status_code=500,
            )

        try:
            client = OpenSearchClient.from_config(search_config, prompt_for_missing=False)
            embedding_config = EmbeddingConfig.from_config(search_config)
            embedding_client = EmbeddingClient(embedding_config)
            llm_config = settings.qwen_config()
            search_tool = HybridSearchTool(
                client=client,
                search_config=search_config,
                embedding_client=embedding_client,
            )
            graph = build_graph(
                search_tool,
                search_config=search_config,
                llm_config=llm_config,
                use_agent_pipeline=True,
                related_category_search=search_tool.invoke,
            )
        except (BackendError, ContractError, ValueError, ServiceError):
            raise
        except Exception as exc:
            raise ServiceError("runtime_init_failed", "Agent 运行时初始化失败") from exc

        index_name = str(search_config.get("retrieval", {}).get("hybrid_index_name", "amazon_products_v4"))
        return cls(
            search_config=search_config,
            client=client,
            search_tool=search_tool,
            graph=graph,
            llm_config=llm_config,
            app_version=settings.app_version,
            agent_version=settings.agent_version,
            retrieval_method=settings.retrieval_method,
            index_name=index_name,
            embedding_model_id=embedding_config.model,
            health_timeout_seconds=settings.health_timeout_seconds,
        )


@dataclass(frozen=True)
class ChatExecutionResult:
    """Agent 服务输出；API 层再把它序列化成 ChatResponse。"""

    state: Mapping[str, Any]
    final_response: str
    answer: Optional[Any]
    agent_version: str
    retrieval_method: str
    trace_nodes: List[str]
    trace_snapshots: List[Dict[str, Any]]
    attempt_count: int
    errors: List[str]


class AgentService:
    """对外服务入口。"""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        default_chat_top_k: int = 5,
        context_store: Optional[SessionContextStore] = None,
        invoke_memory_model: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.runtime = runtime
        self.default_chat_top_k = default_chat_top_k
        self.context_store = context_store or SessionContextStore()
        self.invoke_memory_model = invoke_memory_model

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AgentService":
        return cls(
            AgentRuntime.from_settings(settings),
            default_chat_top_k=settings.default_chat_top_k,
        )

    def search(self, request: SearchRequest, *, trace_id: str) -> SearchResponse:
        """输入 SearchRequest，输出 SearchResponse/HybridSearchResponse。"""

        started = time.monotonic()
        log_event(
            "search.started",
            trace_id=trace_id,
            fields={
                "query_sha256": query_fingerprint(request.query),
                "top_k": request.top_k,
                "retrieval_method": self.runtime.retrieval_method,
                "index_version": self.runtime.index_name,
            },
        )
        try:
            response = self.runtime.search_tool.invoke(request)
        except (BackendError, ContractError, ValueError):
            log_event(
                "search.failed",
                trace_id=trace_id,
                fields={
                    "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                    "index_version": self.runtime.index_name,
                },
            )
            raise

        log_event(
            "search.completed",
            trace_id=trace_id,
            fields={
                "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                "returned": len(response.results),
                "retrieval_method": response.retrieval_method,
                "model_id": self.runtime.embedding_model_id,
                "index_version": self.runtime.index_name,
            },
        )
        return response

    def chat(
        self,
        *,
        message: str,
        locale: Optional[str],
        top_k: Optional[int],
        trace_id: str,
        session_id: Optional[str] = None,
    ) -> ChatExecutionResult:
        """输入 ChatRequest 业务字段，输出 Agent 的稳定执行结果。"""

        resolved_top_k = top_k or self.default_chat_top_k
        started = time.monotonic()
        previous_context = self.context_store.get(session_id)
        initial_state: Dict[str, Any] = {
            "user_query": message,
            "previous_context": previous_context or {},
            "errors": [],
            "top_k": resolved_top_k,
            "max_products": resolved_top_k,
            "trace_id": trace_id,
            "trace_nodes": [],
        }
        if locale:
            initial_state["request_locale"] = locale

        log_event(
            "chat.started",
            trace_id=trace_id,
            fields={
                "query_sha256": query_fingerprint(message),
                "top_k": resolved_top_k,
                "model_id": self.runtime.llm_config.model,
                "index_version": self.runtime.index_name,
                "retrieval_method": self.runtime.retrieval_method,
            },
        )
        try:
            state = self.runtime.graph.invoke(initial_state)
        except (BackendError, ContractError, ValueError):
            log_event(
                "chat.failed",
                trace_id=trace_id,
                fields={
                    "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                    "index_version": self.runtime.index_name,
                },
            )
            raise
        except Exception as exc:
            log_event(
                "chat.failed",
                trace_id=trace_id,
                fields={
                    "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                    "error_type": type(exc).__name__,
                    "index_version": self.runtime.index_name,
                },
            )
            raise ServiceError("agent_execution_failed", "Agent 执行失败") from exc

        trace_nodes = list(state.get("trace_nodes", []))
        trace_snapshots = list(state.get("trace_snapshots", []))
        errors = [str(item) for item in state.get("errors", [])]
        if session_id:
            memory_status, memory = self._update_conversation_memory(
                session_id=session_id,
                user_query=message,
                state=state,
                previous_context=previous_context,
                trace_snapshots=trace_snapshots,
            )
            trace_nodes.append("memory_update")
            state = dict(state)
            state["memory_update_status"] = memory_status
            if memory is not None:
                state["conversation_memory"] = memory
        result = ChatExecutionResult(
            state=state,
            final_response=str(state.get("final_response") or "当前没有可输出的回答。"),
            answer=state.get("answer"),
            agent_version=str(state.get("agent_version") or self.runtime.agent_version),
            retrieval_method=str(state.get("retrieval_method") or self.runtime.retrieval_method),
            trace_nodes=trace_nodes,
            trace_snapshots=trace_snapshots,
            attempt_count=int(state.get("attempt_count", 0)),
            errors=errors,
        )
        log_event(
            "chat.completed",
            trace_id=trace_id,
            fields={
                "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                "nodes": trace_nodes,
                "attempt_count": result.attempt_count,
                "error_count": len(errors),
                "retrieval_method": result.retrieval_method,
                "model_id": self.runtime.llm_config.model,
                "index_version": self.runtime.index_name,
                "context_used": bool(state.get("context_used")),
                "memory_update_status": state.get("memory_update_status", "not_requested"),
            },
        )
        return result

    def _update_conversation_memory(
        self,
        *,
        session_id: str,
        user_query: str,
        state: Mapping[str, Any],
        previous_context: Optional[Mapping[str, Any]],
        trace_snapshots: List[Dict[str, Any]],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """提取并保存本轮问答记忆；失败时不影响本轮回答，但清除旧记忆。"""

        started = time.monotonic()
        parsed_constraints = state.get("parsed_constraints")
        if not isinstance(parsed_constraints, Mapping):
            parsed_constraints = {}
        answer_value = state.get("answer")
        answer_payload = to_serializable(answer_value) if answer_value is not None else {}
        if not isinstance(answer_payload, Mapping):
            answer_payload = {"value": answer_payload}
        assistant_answer = str(state.get("final_response") or "当前没有可输出的回答。")
        topic_relation = str(state.get("topic_relation") or "none")
        next_action = str(state.get("next_action") or "unknown")
        input_payload = {
            "user_query": user_query,
            "assistant_answer": assistant_answer,
            "parsed_constraints": dict(parsed_constraints),
            "answer_payload": dict(answer_payload),
            "topic_relation": topic_relation,
        }
        try:
            memory = extract_conversation_memory(
                user_query,
                assistant_answer,
                parsed_constraints,
                dict(answer_payload),
                effective_query=str(state.get("effective_query") or user_query),
                topic_relation=topic_relation,
                next_action=next_action,
                previous_context=previous_context,
                config=self.runtime.llm_config,
                invoke_model=self.invoke_memory_model,
            )
            self.context_store.put(session_id, memory)
            output_payload = {"stored": True, "memory": memory}
            status = "stored"
            trace_status = "completed"
        except ConversationMemoryError as exc:
            # 不把模型失败伪装成成功记忆；清掉旧话题，避免下一轮继续污染。
            self.context_store.clear(session_id)
            output_payload = {
                "stored": False,
                "status": "degraded",
                "error": {"code": exc.code, "message": exc.message},
            }
            status = "degraded"
            trace_status = "degraded"

        trace_snapshots.append(
            {
                "sequence": len(trace_snapshots) + 1,
                "node_name": "memory_update",
                "display_name": "对话记忆更新",
                "input_description": "本轮用户问题、答案、结构化条件和商品证据",
                "output_description": "供下一轮判断话题和解析指代的短期记忆",
                "input_format": "JSON",
                "output_format": "JSON state patch",
                "input_keys": list(input_payload),
                "output_keys": list(output_payload),
                "input": input_payload,
                "output": output_payload,
                "duration_ms": int(round((time.monotonic() - started) * 1000)),
                "status": trace_status,
            }
        )
        return status, memory if status == "stored" else None

    def health(self) -> Dict[str, str]:
        """检查 API 依赖状态，不返回账号、密码、Prompt 或原始响应。"""

        opensearch_status = "ok"
        try:
            if not self.runtime.client.index_exists(self.runtime.index_name):
                opensearch_status = "index_missing"
        except BackendError:
            opensearch_status = "unavailable"

        llm_status = "ready"
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.get(
                "%s/models" % self.runtime.llm_config.base_url.rstrip("/"),
                timeout=self.runtime.health_timeout_seconds,
            )
            if response.status_code >= 400:
                llm_status = "unavailable"
        except requests.RequestException:
            llm_status = "unavailable"
        finally:
            session.close()

        return {
            "status": "ok" if opensearch_status == "ok" and llm_status == "ready" else "degraded",
            "api": "ok",
            "opensearch": opensearch_status,
            "llm": llm_status,
            "index_version": self.runtime.index_name,
            "app_version": self.runtime.app_version,
        }
