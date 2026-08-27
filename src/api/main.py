""" FastAPI 应用入口。

路由只负责 HTTP、校验、trace_id 和序列化；检索和 Agent 逻辑全部委托给
``AgentService``。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config.settings import AppSettings, SettingsError
from ..models import ContractError, SearchRequest as DomainSearchRequest
from ..observability.tracing import configure_logging, new_trace_id
from ..opensearch_client import BackendError
from ..output_language import chinese_or_fallback
from ..rag_models import RecommendationResponse
from ..service.agent_service import AgentService, ChatExecutionResult, ServiceError
from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    Evidence,
    HealthResponse,
    Recommendation,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ModuleTrace,
)


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def _error_payload(code: str, message: str, trace_id: str) -> dict:
    return {"error": {"code": code, "message": message, "trace_id": trace_id}}


def _trace_id(request: Request) -> str:
    return str(getattr(request.state, "trace_id", None) or new_trace_id())


def _serialize_search(response: Any, trace_id: str) -> SearchResponse:
    payload = response.to_dict()
    return SearchResponse(
        query=payload["query"],
        results=[SearchResult.model_validate(item) for item in payload.get("results", [])],
        total=int(payload.get("total", 0)),
        search_backend=str(payload.get("search_backend", "opensearch")),
        retrieval_method=str(payload.get("retrieval_method", "bm25")),
        warnings=list(payload.get("warnings", [])),
        retrieval_channels={
            str(key): list(value) for key, value in payload.get("retrieval_channels", {}).items()
        },
        fusion_config=dict(payload.get("fusion_config", {})),
        query_embedding_model=payload.get("query_embedding_model") or None,
        trace_id=trace_id,
    )


def _serialize_chat(result: ChatExecutionResult, trace_id: str) -> ChatResponse:
    answer = result.answer
    result_state = getattr(result, "state", {})
    if isinstance(answer, RecommendationResponse) and result_state.get("next_action") != "reject":
        recommendations = [
            Recommendation.model_validate(
                {
                    **item.to_dict(),
                    "reason": chinese_or_fallback(
                        item.reason,
                        "该商品的相关字段与当前检索条件匹配。",
                    ),
                }
            )
            for item in answer.recommendations
        ]
        evidence = [Evidence.model_validate(item.to_dict()) for item in answer.evidence]
        limitations = [
            chinese_or_fallback(
                item,
                "仅依据商品字段，未包含实时价格、库存、评分和配送信息。",
            )
            for item in answer.limitations
        ]
    else:
        recommendations = []
        evidence = []
        limitations = ["本次请求没有生成可验证的商品推荐。"]

    related_categories = [
        str(item)
        for item in result_state.get("related_categories", [])
        if isinstance(item, str) and item.strip()
    ]

    if result.errors and not limitations:
        limitations = ["本次请求包含未完成的 Agent 步骤。"]

    return ChatResponse(
        answer=result.final_response,
        recommendations=recommendations,
        evidence=evidence,
        limitations=limitations,
        related_categories=related_categories,
        trace_id=trace_id,
        agent_version=result.agent_version,
        retrieval_method=result.retrieval_method,
        execution_trace=[ModuleTrace.model_validate(item) for item in getattr(result, "trace_snapshots", [])],
    )


def create_app(
    *,
    service: Optional[AgentService] = None,
    settings: Optional[AppSettings] = None,
) -> FastAPI:
    """创建应用；测试可注入 service，真实启动时从配置构造 Agent。"""

    app_settings = settings or AppSettings.from_env()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            try:
                app.state.service = AgentService.from_settings(app_settings)
            except (SettingsError, BackendError, ContractError, ValueError, ServiceError):
                # 让 uvicorn 启动日志保留真实异常；API 不会在半初始化状态下接收请求。
                raise
        yield

    app = FastAPI(
        title="Amazon Retrieval Agent API",
        version=app_settings.app_version,
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    app.state.service = service
    app.state.settings = app_settings
    app.mount(
        "/ui/assets",
        StaticFiles(directory=str(FRONTEND_DIR / "assets")),
        name="ui-assets",
    )

    @app.middleware("http")
    async def attach_trace_id(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID") or new_trace_id()
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload("request_validation_error", "请求字段校验失败", _trace_id(request)),
        )

    @app.exception_handler(BackendError)
    async def backend_error(request: Request, exc: BackendError):
        return JSONResponse(
            status_code=503,
            content=_error_payload(exc.code, exc.message, _trace_id(request)),
        )

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, _trace_id(request)),
        )

    @app.exception_handler(ContractError)
    async def contract_error(request: Request, exc: ContractError):
        return JSONResponse(
            status_code=400,
            content=_error_payload(exc.code, exc.message, _trace_id(request)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", "服务内部错误", _trace_id(request)),
        )

    def get_service() -> AgentService:
        current = getattr(app.state, "service", None)
        if current is None:
            raise ServiceError("service_not_ready", "Agent 服务尚未就绪")
        return current

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse.model_validate(get_service().health())

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(request_body: SearchRequest, request: Request) -> SearchResponse:
        trace_id = _trace_id(request)
        domain_request = DomainSearchRequest.from_mapping(
            request_body.model_dump(),
            default_top_k=request_body.top_k,
            max_top_k=100,
        )
        result = get_service().search(domain_request, trace_id=trace_id)
        return _serialize_search(result, trace_id)

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request_body: ChatRequest, request: Request) -> ChatResponse:
        trace_id = _trace_id(request)
        result = get_service().chat(
            message=request_body.message,
            locale=request_body.locale,
            top_k=request_body.top_k,
            session_id=request_body.session_id,
            trace_id=trace_id,
        )
        return _serialize_chat(result, trace_id)

    return app


app = create_app()
