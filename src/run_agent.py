"""LangGraph 单 Agent 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_recommendation import LocalQwenConfig
from .embedding_client import EmbeddingClient, EmbeddingConfig
from .hybrid_search_tool import HybridSearchTool
from .models import ContractError
from .opensearch_client import BackendError, OpenSearchClient
from .search_config import config_value, load_search_config
from .tools.search_products_tool import SearchProductsTool
from .graph.build_graph import build_graph
from .graph.state import state_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="执行 LangGraph 单 Agent 商品检索")
    parser.add_argument("--config", type=Path, default=Path("config/search.yaml"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-products", type=int, default=5)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--retrieval-method", choices=["bm25", "hybrid_rrf"])
    parser.add_argument("--pipeline", choices=["standard", "agent"], default="agent")
    args = parser.parse_args()

    config = load_search_config(args.config.resolve())
    default_top_k = int(config_value(config, "search", "default_top_k", 10))
    llm_config = LocalQwenConfig.from_env()
    if args.qwen_model:
        from dataclasses import replace

        llm_config = replace(llm_config, model=args.qwen_model)

    try:
        client = OpenSearchClient.from_config(config, prompt_for_missing=True)
        retrieval_method = args.retrieval_method or str(config_value(config, "retrieval", "method", "bm25"))
        if args.pipeline == "agent" and retrieval_method != "hybrid_rrf":
            raise ValueError("Agent 流程需要 --retrieval-method hybrid_rrf，以接收混合候选")
        if retrieval_method == "hybrid_rrf":
            embedding_client = EmbeddingClient(EmbeddingConfig.from_config(config))
            tool = HybridSearchTool(
                client=client,
                search_config=config,
                embedding_client=embedding_client,
            )
        elif retrieval_method == "bm25":
            tool = SearchProductsTool(client=client, search_config=config)
        else:
            raise ValueError("retrieval.method 必须是 bm25 或 hybrid_rrf")
        graph = build_graph(
            tool,
            search_config=config,
            llm_config=llm_config,
            use_agent_pipeline=args.pipeline == "agent",
        )
        result = graph.invoke(
            {
                "user_query": args.query,
                "errors": [],
                "top_k": default_top_k if args.top_k is None else args.top_k,
                "max_products": args.max_products,
            }
        )
    except (BackendError, ContractError, ValueError) as exc:
        print(
            json.dumps(
                {"error": {"code": getattr(exc, "code", "agent_failed"), "message": str(exc)}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    print(json.dumps(state_to_dict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
