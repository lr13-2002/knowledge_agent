"""FastAPI 应用工厂。

create_app() 是整个平台的入口，负责：
1. 初始化所有存储（向量库、图数据库、提案库）
2. 创建 Worker 后台线程消费任务队列
3. 根据环境变量选择 trace 提供者和 LLM 客户端
4. 注册 HTTP 路由

环境变量：
    TRACE_MCP_USERNAME — 有值时使用真实 observe API 获取 trace
    ANTHROPIC_API_KEY  — 有值时使用 Claude LLM 分析，否则降级到规则引擎
    LLM_MODEL          — 指定 LLM 模型（默认 claude-sonnet-4-6）
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from .ask import AskService
from .community import CommunityDetector
from .ingestor import handle_openclaw_trace
from .loader import RAGLoader
from .persistent_stores import init_persistent_stores
from .propagation import ConfidencePropagator
from .review import ReviewService
from .sampler import AdaptiveSampler, SamplingConfig
from .stores import InMemoryIdempotencyStore, InMemoryRawArtifactStore, InMemoryTaskQueue
from .trace import FixtureTraceProvider, ObserveTraceProvider
from .worker import AgentWorker


def _default_trace_provider() -> Any:
    """根据环境变量决定使用真实 observe API 还是 fixture 测试数据。"""
    if os.environ.get("TRACE_MCP_USERNAME"):
        return ObserveTraceProvider()
    return FixtureTraceProvider({})


def create_app(
    trace_provider: Any = None,
    repo_roots: dict[str, str] | None = None,
    data_dir: str = "data",
) -> Any:
    """创建 FastAPI 应用实例。

    参数:
        trace_provider: trace 数据来源（None 时自动检测环境变量）
        repo_roots: 仓库名 → 本地路径的映射，用于代码索引
        data_dir: 持久化数据存储目录
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is optional for module tests; install fastapi to run HTTP APIs") from exc

    # 初始化持久化存储：Chroma 向量库 + SQLite 图/提案库
    vector, graph, proposals = init_persistent_stores(data_dir)
    queue = InMemoryTaskQueue()  # 任务队列（内存，重启丢失无影响）
    raw_artifacts = InMemoryRawArtifactStore()  # 原始 trace 存储
    loader = RAGLoader(vector, graph)  # 数据加载器：将知识写入向量库和图
    propagator = ConfidencePropagator(graph, proposals, loader)  # 置信度传播器
    review = ReviewService(proposals, loader, propagator)  # 人工审核服务（regen_client 在下面注入）
    community_detector = CommunityDetector(graph, vector, loader, llm_client=None)  # 社区检测器（LLM 在下面注入）
    ask_service = AskService(vector, graph)  # 知识库问答服务（LLM 在下面注入，启用 agent 模式）
    sampler = AdaptiveSampler(SamplingConfig())  # 自适应采样器
    idempotency = InMemoryIdempotencyStore()  # 幂等去重

    if trace_provider is None:
        trace_provider = _default_trace_provider()

    # 有 API key 时使用真实 LLM，否则 worker 默认用规则引擎
    # 支持两种 key：ANTHROPIC_API_KEY（官方）或 ANTHROPIC_AUTH_TOKEN（公司代理）
    llm_client = None
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        from .llm import AnthropicLLMClient
        llm_client = AnthropicLLMClient()
        community_detector.llm_client = llm_client  # 社区检测也用同一个 LLM
        review._regen = llm_client  # reject 反馈环用它做结构化重生成
        # 给 AskService 注入原始 openai 兼容 client，启用 ReAct agent 问答模式
        # （AskAgent 需要裸 client 做 function calling，复用 AnthropicLLMClient 内部的 client）
        try:
            import openai
            from .llm.config import LLMConfig
            _ask_cfg = LLMConfig.load()
            ask_service.llm_client = openai.OpenAI(api_key=_ask_cfg.api_key, base_url=_ask_cfg.base_url or None)
            ask_service.model = _ask_cfg.model
        except Exception:
            pass  # 注入失败则 AskService 自动走五路召回降级

    worker = AgentWorker(queue, trace_provider, vector, graph, proposals, raw_artifacts, llm=llm_client, loader=loader, repo_roots=repo_roots or {})

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """应用生命周期：启动时开始后台 worker，关闭时停止。"""
        worker.start_background()
        yield
        worker.stop()

    app = FastAPI(title="Trace Business Understanding Agent", lifespan=lifespan)

    # ==================== HTTP 路由 ====================

    @app.post("/webhooks/openclaw/trace")
    def openclaw_trace(payload: dict[str, Any]) -> dict[str, Any]:
        """接收 trace 事件（来自 OpenClaw 或手动推送），经采样后入队。"""
        try:
            return handle_openclaw_trace(payload, sampler, queue, idempotency)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/reviews/{proposal_id}")
    def get_review(proposal_id: str) -> dict[str, Any]:
        """获取知识提案详情。"""
        try:
            return review.get(proposal_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="proposal not found") from exc

    @app.post("/reviews/{proposal_id}/messages")
    def review_message(proposal_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """向提案发送审核消息（提问/讨论）。"""
        return asdict(review.message(proposal_id, str(payload.get("content", ""))))

    @app.post("/reviews/{proposal_id}/approve")
    def approve(proposal_id: str) -> dict[str, Any]:
        """批准提案，知识正式入库。"""
        return review.approve(proposal_id)

    @app.post("/reviews/{proposal_id}/reject")
    def reject(proposal_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """驳回提案。

        payload 可带 {"reason": "..."}：
        - 提供 reason 且有 LLM → 触发反馈环，LLM 基于理由重生成（最多 3 次），返回 regenerated
        - 否则 → 终态 reject
        """
        reason = str((payload or {}).get("reason", ""))
        return review.reject(proposal_id, reason)

    @app.post("/reviews/{proposal_id}/revise")
    def revise(proposal_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """修订提案内容（修改 summary 和 claims）。"""
        return review.revise(proposal_id, str(payload.get("summary", "")), list(payload.get("claims", [])))

    @app.post("/ask")
    def ask(payload: dict[str, Any]) -> dict[str, Any]:
        """向知识库提问，返回已确认知识 + trace 观察 + 图谱关联。"""
        result = ask_service.ask(str(payload.get("repo", "")), str(payload.get("question", "")), payload.get("trace_id"))
        return asdict(result)

    # ==================== 推送与管理接口 ====================

    @app.get("/proposals/pending")
    def pending_proposals(repo: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """获取待审核 proposal 列表，按优先级排序。

        优先级打分：新领域 > 置信度低 > 创建时间新
        """
        pending = proposals.list_by_status("pending_review", repo=repo, limit=100)
        approved_interfaces = {p.interface_key for p in proposals.list_by_status("approved", limit=1000)}

        def priority_score(p: Any) -> float:
            score = 0.0
            if p.interface_key not in approved_interfaces:
                score += 100  # 新领域
            score += (1.0 - p.confidence_score) * 50  # 置信度越低越优先
            return score

        pending.sort(key=priority_score, reverse=True)
        return [asdict(p) for p in pending[:limit]]

    @app.post("/proposals/{proposal_id}/withdraw")
    def withdraw(proposal_id: str) -> dict[str, Any]:
        """撤回已自动入库的知识，退回草稿层。"""
        try:
            proposal = proposals.get(proposal_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="proposal not found") from exc
        if proposal.status != "approved":
            raise HTTPException(status_code=400, detail="只能撤回 approved 状态的提案")
        proposals.update_status(proposal_id, "pending_review")
        return {"status": "withdrawn", "proposal_id": proposal_id}

    @app.get("/stats/coverage")
    def coverage_stats(repo: str | None = None) -> dict[str, Any]:
        """覆盖率统计：已覆盖/待审核/总接口数。"""
        all_interfaces = graph.find_nodes("Interface", **({"repo": repo} if repo else {}))
        approved = proposals.list_by_status("approved", repo=repo, limit=10000)
        pending = proposals.list_by_status("pending_review", repo=repo, limit=10000)
        approved_keys = {p.interface_key for p in approved}
        pending_keys = {p.interface_key for p in pending}
        total_keys = {n.get("id", "") for n in all_interfaces}
        return {
            "total_interfaces": len(total_keys),
            "approved": len(approved_keys),
            "pending_review": len(pending_keys - approved_keys),
            "uncovered": len(total_keys - approved_keys - pending_keys),
            "coverage_percent": round(len(approved_keys) / max(len(total_keys), 1) * 100, 1),
        }

    @app.post("/collector/run")
    def collector_run(payload: dict[str, Any]) -> dict[str, Any]:
        """手动触发一次 trace 分析。

        payload: {"repo": "...", "trace_id": "...", "service": "...", "method": "...", "path": "..."}
        """
        try:
            result = handle_openclaw_trace({**payload, "manual": True}, sampler, queue, idempotency)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    # ==================== 实体与社区接口（Phase 2 + 3）====================

    @app.get("/entities/search")
    def entities_search(q: str, repo: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """按关键词在业务实体库中检索。"""
        filters = {"repo": repo} if repo else None
        hits = vector.search("entities", q, filters, limit=limit)
        return [
            {
                "id": h["id"],
                "name": h["payload"].get("name", ""),
                "type": h["payload"].get("type", ""),
                "description": h["payload"].get("description", ""),
                "mentions": h["payload"].get("mentions", 0),
                "score": h.get("score", 0.0),
            }
            for h in hits
        ]

    @app.get("/communities")
    def list_communities(limit: int = 20) -> list[dict[str, Any]]:
        """列出所有社区（按 rank 倒序）。"""
        communities = graph.find_nodes("Community")
        communities.sort(key=lambda c: -float(c.get("rank", 0)))
        return [
            {
                "id": c.get("id", ""),
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
                "level": c.get("level", 0),
                "rank": c.get("rank", 0),
                "member_count": c.get("member_count", 0),
            }
            for c in communities[:limit]
        ]

    @app.post("/communities/refresh")
    def refresh_communities() -> dict[str, Any]:
        """触发社区检测重算。

        从所有 Entity 节点出发跑 Leiden 算法，每个社区调 LLM 生成领域摘要。
        """
        reports = community_detector.detect_and_summarize()
        return {
            "status": "ok",
            "community_count": len(reports),
            "communities": [{"id": r.community_id, "title": r.title} for r in reports],
        }

    return app
