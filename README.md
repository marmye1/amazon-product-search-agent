# 亚马逊商品检索与导购 Agent

这是一个本地运行的商品检索与导购 Agent，提供 FastAPI 接口和可观测流程界面。核心链路由一个 LangGraph Agent 编排：需求解析 → 混合检索 → 候选重排与硬约束 → 本地 Qwen 生成 → 字段级证据校验 → 稳定 JSON 输出。

## 技术组成

- FastAPI + Pydantic：HTTP 接口、请求校验和响应契约。
- LangGraph：编排上下文合并、需求解析、检索、回答、证据校验和拒答分支。
- OpenSearch：BM25、向量检索和 RRF 融合；基础索引为 `amazon_products_v1`，混合检索索引为 `amazon_products_v4`。
- LM Studio：提供 OpenAI-compatible 的本地 Qwen 和 Embedding 接口。
- 短期对话记忆：只保存在当前 API 进程内，不写数据库。
- 原生 HTML/CSS/JavaScript：展示聊天结果和本次请求的可观测执行链路。

## 运行依赖

运行 API 前，需要准备：

1. OpenSearch 3.x，并准备好商品索引或按项目脚本建立索引。
2. LM Studio，以及可用的 Qwen 模型和 `text-embedding-nomic-embed-text-v1.5` Embedding 模型。
3. Python 3.9 虚拟环境和 `requirements.txt` 中的依赖。

复制 `.env.example` 为 `.env` 后填写本机配置。直接运行 API 时先在当前 shell 加载它：

```bash
set -a
source .env
set +a
```

`.env`、原始数据、处理后数据、OpenSearch 目录和评估报告都不会提交到 Git。

## 统一测试入口

在项目根目录运行一次即可执行全部离线测试：

```bash
.venv/bin/python -m pytest -q
```

这组测试覆盖数据契约、搜索契约、混合检索、RAG、Agent 图、API、对话记忆、认证边界和评估函数。它验证代码和接口契约，不会替你启动或代替真实检查 OpenSearch、LM Studio 的在线状态。

## 启动 API

```bash
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

启动后访问流程界面：

```text
http://127.0.0.1:8000/
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

商品检索：

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"wireless mouse","top_k":5}'
```

导购回答：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"推荐无线鼠标","locale":"us","top_k":3}'
```

每个成功响应和错误响应都有 `trace_id`，并通过 `X-Trace-ID` 响应头返回。`/v1/chat` 的回答会保留 `answer_version: "v2"` 这一现有响应契约。

## 数据与真实验收

数据来源标识使用 `esci:v0`，表示数据来源契约。原始数据和处理后数据默认由 `.gitignore` 排除。

需要检查真实 OpenSearch、Embedding 和混合检索链路时，可使用：

```bash
.venv/bin/python -m src.search_acceptance --help
```

真实验收需要本机依赖在线，并可能写入本地 OpenSearch；统一测试命令不会执行这类外部写操作。
