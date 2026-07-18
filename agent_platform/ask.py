"""知识库问答服务。

两种模式（自动选择）：
- **Agent 模式**（有 LLM 时）：用 ReAct agent 自主决定调哪些检索工具，
  多轮检索后由 LLM 生成连贯答案。见 llm/ask_agent.py。
- **降级模式**（无 LLM 或 agent 异常时）：固定四路召回 + 模板拼接（本文件下半部分）。

降级模式的四路召回（参考 Microsoft GraphRAG 的 Local + Global Search）:
1. confirmed     — 已审核知识（向量匹配 knowledge_claims）
2. entity_path   — 业务实体（命中实体后展开它被哪些接口提及）
3. community_path — 业务领域社区（适合"支付域整体架构"这类宽泛问题）
4. graph_relate  — 图关联（从命中的实体/社区/TraceCase 出发展开关联接口）

为什么 agent 化：原来固定跑五路 + 模板拼接，返回的是"检索碎片罗列"而非"答案"。
agent 模式让 LLM 按问题类型动态选路、可多轮深挖、生成自然语言答案。
旧四路保留作降级，保证无 LLM（测试 / 无 key）时仍可用。

2026-06-23 砍掉旧"trace_obs 向量路"：trace 本质是结构化数据，不需要语义检索
（旧实现 vector.search 用 trace_id 当 query+filter，根本不走 embedding）。
有 trace_id 时直接走"路 4 图扩展"读 TraceCase 节点的关联（neighbors 查图），
更快更准且不依赖向量库。
"""
from __future__ import annotations

import logging
from typing import Any

from .schemas import AskResponse, Evidence

logger = logging.getLogger(__name__)


class AskService:
    """知识库问答服务。"""

    def __init__(
        self,
        vector: Any,
        graph: Any | None = None,
        llm_client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self.vector = vector
        self.graph = graph
        # llm_client: openai 兼容 client（有则走 agent 模式）
        self.llm_client = llm_client
        self.model = model
        self._agent = None  # 懒加载 AskAgent

    def ask(self, repo: str, question: str, trace_id: str | None = None) -> AskResponse:
        """问答入口：有 LLM 走 agent 模式，否则降级到五路召回。

        参数:
            repo: 仓库名，限定检索范围
            question: 用户问题（自然语言）
            trace_id: 可选，提供时额外检索该 trace 的关联信息
        """
        # Agent 模式：有 LLM 时优先用，异常则降级
        if self.llm_client and self.model:
            try:
                return self._ask_with_agent(repo, question, trace_id)
            except Exception:
                logger.exception("Ask agent 失败，降级到五路召回")
        # 降级模式：固定五路召回
        return self._ask_fallback(repo, question, trace_id)

    def _ask_with_agent(self, repo: str, question: str, trace_id: str | None) -> AskResponse:
        """Agent 模式：ReAct 多轮检索 + LLM 生成答案。"""
        if self._agent is None:
            from .llm.ask_agent import AskAgent
            self._agent = AskAgent(self.llm_client, self.model, self.vector, self.graph)
        return self._agent.run(repo, question, trace_id)

    def _ask_fallback(self, repo: str, question: str, trace_id: str | None = None) -> AskResponse:
        """降级模式：固定五路召回 + 模板拼接（原 ask 逻辑）。"""
        # 路 1: 已审核知识
        confirmed_hits = self.vector.search(
            "knowledge_claims", question, {"repo": repo, "status": "approved"}, limit=5
        )

        # 路 2: 业务实体（Phase 2 新增）
        # Reinforcement（Day 5）+ recency 加权（Day 8）：mentions 强化 + last_seen_at 新近度
        from .entities import rerank_hits
        entity_hits = rerank_hits(
            self.vector.search("entities", question, {"repo": repo}, limit=5),
            recency_field="last_seen_at",
        )

        # 路 3: 社区摘要（Phase 3 新增）
        community_hits = self.vector.search(
            "community_summaries", question, limit=3
        )

        # 路 4: 图关联展开（从命中的实体/社区出发；有 trace_id 时也展开 TraceCase 关联）
        # 2026-06-23 起：trace 信息只走图层 TraceCase 节点（loader 不再写 trace_cases 向量库），
        # 原"路 4 trace_obs 向量召回"已删除。
        graph_context = self._expand_graph_context(entity_hits, community_hits, trace_id, repo)

        # 全部为空时返回兜底
        if not confirmed_hits and not entity_hits and not community_hits and not graph_context:
            return AskResponse(
                answer="暂无足够证据回答该问题。",
                evidence=Evidence(trace_ids=[trace_id] if trace_id else []),
                sections={"confirmed": [], "candidate": [], "observation": [], "inference": [],
                          "domain": [], "entities": []},
            )

        # 组装结构化结果
        confirmed = [item["text"] for item in confirmed_hits]
        domain = [
            f"【{item['payload'].get('title', '?')}】{item['payload'].get('summary', '')}"
            for item in community_hits
        ]
        entities = [
            f"{item['payload'].get('name', '?')}({item['payload'].get('type', '?')}): {item['payload'].get('description', '')}"
            for item in entity_hits
        ]

        evidence = Evidence(
            trace_ids=[trace_id] if trace_id else [],
            business_rule_ids=[item["payload"].get("business_rule_id", item["id"]) for item in confirmed_hits],
        )

        # 组装答案：按宽泛程度排序，社区领域 > 已确认知识 > 实体 > 图谱关联
        parts = []
        if domain:
            parts.append("领域全景: " + " | ".join(domain[:2]))
        if confirmed:
            parts.append("已确认知识: " + "；".join(confirmed[:3]))
        if entities:
            parts.append("相关业务实体: " + "；".join(entities[:3]))
        if graph_context:
            parts.append("图谱关联: " + "；".join(graph_context[:5]))

        answer = "。".join(parts) if parts else "未检索到相关知识。"

        return AskResponse(
            answer=answer,
            evidence=evidence,
            sections={
                "confirmed": confirmed,
                "candidate": [],
                "observation": [],  # 旧 trace_obs 路已废弃，保留 key 兼容老调用方
                "inference": graph_context,
                "domain": domain,
                "entities": entities,
            },
        )

    def _expand_graph_context(
        self,
        entity_hits: list[dict],
        community_hits: list[dict],
        trace_id: str | None,
        repo: str,
    ) -> list[str]:
        """图关联展开 — 从实体/社区命中出发，找相关接口和关联实体。"""
        if not self.graph:
            return []

        graph_context: list[str] = []

        # 从命中的实体反向查：哪些接口提到了这个实体
        for hit in entity_hits[:3]:
            entity_id = hit.get("id", "")
            if not entity_id:
                continue
            try:
                interfaces = self.graph.reverse_neighbors("Entity", entity_id, "MENTIONS")
                for iface in interfaces[:3]:
                    name = hit["payload"].get("name", entity_id)
                    path = iface.get("path", iface.get("id", ""))
                    graph_context.append(f"{name} → {path}")
            except Exception:
                continue

        # 从命中的社区出发，列出该领域下的关键接口
        for hit in community_hits[:2]:
            community_id = hit.get("id", "")
            try:
                # 社区的成员实体 → 这些实体被哪些接口提及
                members = self.graph.reverse_neighbors("Community", community_id, "BELONGS_TO")
                for member in members[:3]:
                    member_id = member.get("id", "")
                    member_name = member.get("name", "?")
                    ifaces = self.graph.reverse_neighbors("Entity", member_id, "MENTIONS")
                    for iface in ifaces[:2]:
                        path = iface.get("path", iface.get("id", ""))
                        graph_context.append(f"{member_name} → {path}")
            except Exception:
                continue

        # 兼容老逻辑：trace_id 模式下查 trace 的图关联
        if trace_id:
            try:
                for neighbor in self.graph.neighbors("TraceCase", trace_id):
                    graph_context.append(
                        f"{neighbor.get('_relation', '?')} → {neighbor.get('name', neighbor.get('id', ''))}"
                    )
            except Exception:
                pass

        # 去重
        return list(dict.fromkeys(graph_context))
