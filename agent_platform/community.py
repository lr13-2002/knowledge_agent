"""社区检测引擎（Phase 3）— Microsoft GraphRAG 风格。

目标：从 Entity 子图中自动聚类出业务领域（Community），
每个领域由 LLM 生成领域级摘要，供 AskService 做全局问答时使用。

核心步骤：
1. 从 GraphStore 拉取所有 Entity 节点和它们之间的 RELATED_TO 边
   （CO_MENTIONS 已被合并到 RELATED_TO 的不同权重，参见 worker.py:_write_entities_to_graph）
2. 用 igraph 构建图，跑 leidenalg 社区检测
3. 每个社区构造 prompt 输入：成员实体 + 关系 + 关联接口
4. 调 LLM 生成 CommunityReport
5. 写回图（Community 节点 + BELONGS_TO 边）和向量库（community_summaries）

触发时机：
- 冷启动：/communities/refresh?force=true
- 增量定时：后台线程每 N 分钟检查 dirty flag
- 审核驱动：approve 后置 dirty
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from typing import Any

from .llm.community_prompts import (
    COMMUNITY_REPORT_PROMPT,
    COMMUNITY_REPORT_SYSTEM,
    COMMUNITY_REPORT_TOOL,
    format_entities_for_community,
    format_interfaces_for_community,
    format_relations_for_community,
)
from .schemas import CommunityReport, utc_now

logger = logging.getLogger(__name__)

# 社区检测的边权 relation 列表（仅基于实体子图）
# 仅 RELATED_TO 一种关系，包含原 CO_MENTIONS（合并后的低权重边）和 LLM 显式抽取的业务关系
ENTITY_EDGE_RELATIONS = ["RELATED_TO"]

# 单社区送给 LLM 的实体/关系/接口数量上限（防 token 爆炸）
MAX_ENTITIES_PER_COMMUNITY = 30
MAX_RELATIONS_PER_COMMUNITY = 50
MAX_INTERFACES_PER_COMMUNITY = 10

# 孤立小社区过滤：成员数小于此值不生成报告
MIN_COMMUNITY_SIZE = 2

# Leiden 算法的随机种子（保证每次跑结果一致）
LEIDEN_SEED = 42


class CommunityDetector:
    """社区检测主流程。"""

    def __init__(self, graph: Any, vector: Any, loader: Any = None, llm_client: Any = None) -> None:
        self.graph = graph  # GraphStore（读 Entity 子图，写 Community 节点）
        self.vector = vector  # VectorStore（写 community_summaries）
        self.loader = loader  # RAGLoader（写社区报告）
        self.llm_client = llm_client  # LLM 客户端（生成社区摘要，无则降级到模板）

    def detect_and_summarize(self) -> list[CommunityReport]:
        """完整流程：检测社区 + 生成摘要 + 持久化。

        返回所有生成的 CommunityReport 列表。
        """
        # 1. 拉取实体子图
        entities = self.graph.all_nodes("Entity")
        if not entities:
            logger.info("没有实体，跳过社区检测")
            return []

        edges = self.graph.all_edges(ENTITY_EDGE_RELATIONS)

        # 2. 跑 Leiden
        partitions = self._run_leiden(entities, edges)
        if not partitions:
            logger.info("Leiden 无有效分区，跳过")
            return []

        # 3. 清旧社区数据（重算时全量替换）
        self.graph.delete_nodes("Community")

        # 4. 每个社区生成报告
        reports = []
        for community_idx, member_ids in partitions.items():
            if len(member_ids) < MIN_COMMUNITY_SIZE:
                continue
            try:
                report = self._build_community_report(community_idx, member_ids, entities, edges)
                if report:
                    reports.append(report)
                    if self.loader:
                        self.loader.load_community(asdict(report))
            except Exception:
                logger.exception("生成社区报告失败 idx=%s", community_idx)

        logger.info("社区检测完成: %d 个社区，%d 个有效报告", len(partitions), len(reports))
        return reports

    # ------------------------------------------------------------------------
    # 内部：Leiden 算法
    # ------------------------------------------------------------------------

    def _run_leiden(self, entities: list[dict], edges: list[dict]) -> dict[int, list[str]]:
        """跑 Leiden 算法，返回 {community_idx: [entity_id, ...]}。"""
        try:
            import igraph
            import leidenalg
        except ImportError:
            logger.warning("igraph/leidenalg 未安装，跳过社区检测。pip install python-igraph leidenalg")
            return {}

        # 实体 id → 节点 idx 的映射
        entity_ids = [e.get("id", "") for e in entities if e.get("id")]
        if not entity_ids:
            return {}
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # 构建 igraph
        g = igraph.Graph(directed=False)
        g.add_vertices(len(entity_ids))
        g.vs["name"] = entity_ids

        # 聚合边权：同一对实体的多条边累加权重
        edge_weights: dict[tuple[int, int], float] = {}
        for edge in edges:
            src_idx = id_to_idx.get(edge.get("from_id", ""))
            tgt_idx = id_to_idx.get(edge.get("to_id", ""))
            if src_idx is None or tgt_idx is None or src_idx == tgt_idx:
                continue
            # 无向边：永远用 (min, max) 作为 key，避免双向重复
            key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
            edge_weights[key] = edge_weights.get(key, 0.0) + float(edge.get("weight", 1.0))

        if not edge_weights:
            # 没有边，每个实体自成一个社区
            return {i: [eid] for i, eid in enumerate(entity_ids)}

        edge_list = list(edge_weights.keys())
        weights = [edge_weights[e] for e in edge_list]
        g.add_edges(edge_list)
        g.es["weight"] = weights

        # 跑 Leiden
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=LEIDEN_SEED,
        )

        # 转为 {community_idx: [entity_id, ...]}
        result: dict[int, list[str]] = {}
        for idx, comm_idx in enumerate(partition.membership):
            result.setdefault(comm_idx, []).append(entity_ids[idx])
        return result

    # ------------------------------------------------------------------------
    # 内部：社区报告生成
    # ------------------------------------------------------------------------

    def _build_community_report(
        self,
        community_idx: int,
        member_ids: list[str],
        all_entities: list[dict],
        all_edges: list[dict],
    ) -> CommunityReport | None:
        """为一个社区构造报告。"""
        # 用成员 hash 做 community_id（保证幂等：同一组成员永远同一 ID）
        member_hash = hashlib.sha1(("|".join(sorted(member_ids))).encode()).hexdigest()[:12]
        community_id = f"community:lvl0:{member_hash}"

        # 拿到成员实体的完整数据
        member_set = set(member_ids)
        member_entities = [e for e in all_entities if e.get("id") in member_set]

        # 社区内的边
        community_edges = [
            e for e in all_edges
            if e.get("from_id") in member_set and e.get("to_id") in member_set
        ]

        # 社区关联的接口（通过 Entity 反向查 MENTIONS 边）
        related_interfaces = self._find_related_interfaces(member_ids)

        # 计算 rank（成员 mentions 总和）
        rank = sum(int(e.get("mentions", 1)) for e in member_entities)

        # 调 LLM 或降级到模板
        if self.llm_client:
            llm_result = self._llm_summarize(member_entities, community_edges, related_interfaces)
        else:
            llm_result = None

        if llm_result:
            return CommunityReport(
                community_id=community_id,
                level=0,
                title=llm_result.get("title", f"领域 {community_idx}"),
                summary=llm_result.get("summary", ""),
                findings=llm_result.get("findings", []),
                member_entity_ids=member_ids,
                rank=rank,
                created_at=utc_now(),
            )

        # 降级模板（LLM 不可用时）
        names = [e.get("name", "?") for e in member_entities[:5]]
        return CommunityReport(
            community_id=community_id,
            level=0,
            title=f"领域-{names[0] if names else community_idx}",
            summary=f"本领域包含 {len(member_ids)} 个业务实体，主要包括：{'、'.join(names)} 等。",
            findings=[],
            member_entity_ids=member_ids,
            rank=rank,
            created_at=utc_now(),
        )

    def _find_related_interfaces(self, member_ids: list[str]) -> list[dict]:
        """找出社区内实体被哪些接口提及（按提及次数排序取 top）。"""
        interface_counts: dict[str, dict] = {}
        for entity_id in member_ids:
            try:
                ifaces = self.graph.reverse_neighbors("Entity", entity_id, "MENTIONS")
                for iface in ifaces:
                    iface_id = iface.get("id", "")
                    if not iface_id:
                        continue
                    if iface_id not in interface_counts:
                        interface_counts[iface_id] = {**iface, "_count": 0}
                    interface_counts[iface_id]["_count"] += 1
            except Exception:
                continue
        sorted_ifaces = sorted(interface_counts.values(), key=lambda x: -x.get("_count", 0))
        return sorted_ifaces[:MAX_INTERFACES_PER_COMMUNITY]

    def _llm_summarize(
        self,
        member_entities: list[dict],
        community_edges: list[dict],
        related_interfaces: list[dict],
    ) -> dict | None:
        """调 LLM 生成社区摘要。"""
        prompt = COMMUNITY_REPORT_PROMPT.format(
            n_entities=len(member_entities),
            entities_text=format_entities_for_community(member_entities, MAX_ENTITIES_PER_COMMUNITY),
            relations_text=format_relations_for_community(community_edges, MAX_RELATIONS_PER_COMMUNITY),
            n_interfaces=MAX_INTERFACES_PER_COMMUNITY,
            interfaces_text=format_interfaces_for_community(related_interfaces, MAX_INTERFACES_PER_COMMUNITY),
        )

        try:
            return self.llm_client.summarize_community(prompt)
        except Exception:
            logger.exception("LLM 社区摘要失败，降级到模板")
            return None
