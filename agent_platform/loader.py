"""RAG 数据加载器。

负责将各种来源的数据写入向量库和图数据库：
- 代码索引 → 图（CodeSymbol 节点 + CALLS 边）+ 向量（code_chunks）
- Trace 索引 → 图（TraceCase/Service 节点 + 调用边）⚠️ 不再写向量库
- 知识提案 → 向量（knowledge_claims）+ 图（BusinessRule，evidence 内嵌为属性）

图中的节点和边关系（精简后）：
    CodeSymbol --CALLS--> CodeSymbol           (commit/repo 作为 CodeSymbol 属性)
    Interface --HAS_TRACE--> TraceCase         (流量层主链路)
    Interface --CALLS_SERVICE--> Service
    Interface --MENTIONS--> Entity             (跨层 L2 → L3)
    Entity --RELATED_TO--> Entity              (LLM 抽取的业务关系)
    Entity --BELONGS_TO--> Community           (L3 → L4)
    BusinessRule (evidence 作为节点属性，不再单独建 Evidence 节点)

设计原则：
- Repo/Commit/Span/Evidence 节点信息扁平化为 properties，避免 ontology bloat
- 一种关系表达一种语义，CO_MENTIONS 已合并到 RELATED_TO 的共现兜底边（worker 写双向 weight=0.4，已启用）
"""
from __future__ import annotations

from typing import Any


class RAGLoader:
    """数据加载器，统一管理向量库和图数据库的写入。"""

    def __init__(self, vector: Any, graph: Any) -> None:
        self.vector = vector  # 向量存储（ChromaVectorStore）
        self.graph = graph  # 图存储（SQLiteGraphStore）

    def load_code_index(self, index: dict[str, Any]) -> None:
        """加载代码索引到图和向量库。

        输入格式（来自 CodeIndexer.index_repo().as_dict()）：
            repo_name, commit, symbols: [{id, qualified_name, calls: [{callee}]}], chunks: [{id, text}]

        精简版：repo/commit 作为 CodeSymbol 节点的属性写入，不再建 Repo/Commit 节点也不再建 HAS_* 边。
        想按 commit/repo 检索时，直接 find_nodes("CodeSymbol", commit=..., repo=...) 即可。
        """
        repo = index["repo_name"]
        commit = index.get("commit", "")
        # 创建代码符号节点（commit/repo 作为属性）和调用关系边
        for symbol in index.get("symbols", []):
            symbol_id = symbol["id"]
            self.graph.upsert_node("CodeSymbol", symbol_id, {**symbol, "repo": repo, "commit": commit})
            for call in symbol.get("calls", []):
                callee = call.get("callee", "")
                if callee:
                    self.graph.add_edge("CodeSymbol", symbol_id, "CALLS", "CodeSymbol", callee)
        # 代码片段写入向量库（用于后续语义检索）
        for chunk in index.get("chunks", []):
            self.vector.upsert("code_chunks", chunk["id"], chunk.get("text", ""), {**chunk, "repo": repo, "commit": commit})

    # 2026-06-23 删除 load_trace_index 与 load_pending_knowledge（死代码）:
    # - load_trace_index 是早期离线 trace 灌入接口，主流程从未调用；trace 写入由 worker.process_one
    #   直接 graph.upsert_node("TraceCase", ...) 完成（schemas 与本函数旧实现不一致）。
    # - load_pending_knowledge 是早期"先入库再审"模式的产物；当前流程是 ProposalStore 管草稿状态、
    #   只有 approved 才写向量库（load_approved_knowledge）。无人调用且会产生
    #   id=proposal_id 的冗余记录与 id=rule_id 的 approved 记录并存的问题。
    # 两个函数同属"看似有用其实没用且 schema 与现行不一致"，直接删除以消除认知噪音。

    def load_approved_knowledge(self, proposal: dict[str, Any]) -> None:
        """审核通过的知识正式入库。

        同时写入图（BusinessRule 节点，evidence 作为属性而非独立节点）和向量库（status=approved）。
        这些知识会被 AskService 检索到，也会作为 LLM 的 few-shot 正样本。

        精简版：原 Evidence 节点 + SUPPORTED_BY 边已合并为 BusinessRule 节点的属性。
        要查证据时直接读 BusinessRule 节点的 evidence_* 字段即可。
        """
        rule_id = "rule:" + proposal["proposal_id"]
        evidence = proposal.get("evidence", {}) or {}
        # 把复杂的嵌套结构展平给图节点（图节点 properties 是 JSON）
        rule_props = {
            "summary": proposal.get("summary", ""),
            "interface_key": proposal.get("interface_key", ""),
            "repo": proposal.get("repo", ""),
            "trace_id": proposal.get("trace_id", ""),
            "confidence": proposal.get("confidence", "low"),
            "status": "approved",
            # created_at 供检索时矛盾消解判断新旧（Day 6a）
            "created_at": proposal.get("created_at", ""),
            # evidence 内嵌为属性（原 Evidence 节点扁平化）
            "evidence_trace_ids": evidence.get("trace_ids", []),
            "evidence_code_symbols": evidence.get("code_symbols", []),
            "evidence_commit": evidence.get("commit", ""),
        }
        self.graph.upsert_node("BusinessRule", rule_id, rule_props)
        self.vector.upsert(
            "knowledge_claims",
            rule_id,
            proposal.get("summary", ""),
            {**rule_props, "business_rule_id": rule_id},
        )

    def load_community(self, report: dict[str, Any]) -> None:
        """加载一个社区报告到图和向量库。

        Phase 3 新增：社区检测产出的领域摘要，用于 Ask 的全局问答。
        """
        community_id = report["community_id"]
        # 写入图节点
        props = {
            "title": report.get("title", ""),
            "summary": report.get("summary", ""),
            "level": report.get("level", 0),
            "rank": report.get("rank", 0.0),
            "member_count": len(report.get("member_entity_ids", [])),
            "created_at": report.get("created_at", ""),
        }
        self.graph.upsert_node("Community", community_id, props)
        # 成员实体 → 社区的 BELONGS_TO 边
        for entity_id in report.get("member_entity_ids", []):
            self.graph.add_edge("Entity", entity_id, "BELONGS_TO", "Community", community_id, weight=1.0)
        # 写入向量库（用于 Ask 时的语义检索）
        text = f"{report.get('title', '')} {report.get('summary', '')}"
        self.vector.upsert("community_summaries", community_id, text, {
            "title": report.get("title", ""),
            "summary": report.get("summary", ""),
            "level": report.get("level", 0),
            "rank": report.get("rank", 0.0),
            "member_count": len(report.get("member_entity_ids", [])),
        })
