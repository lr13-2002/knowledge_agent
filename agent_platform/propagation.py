"""置信度传播引擎（Belief Propagation）。

当一条知识被 approve 后，与它共享服务/接口/代码符号的 pending proposal
的置信度会自动提升。提升到阈值（0.8）时自动 approve。

传播公式：
    new_score = old_score + source_score × edge_weight × decay

传播约束：
    - 最多 3 跳
    - 变化量 < 0.01 时停止
    - score > 0.8 自动 approve（触发递归传播）
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from .schemas import KnowledgeProposal

logger = logging.getLogger(__name__)

# 置信度等级 → 数值的映射（也用在 client.py / review.py 重生成时回填，保持一致）
# 0.85 而非 0.8：让 LLM 主动给 "high" 时刚好超过 AUTO_APPROVE_THRESHOLD，
# 既能触发自动入库，又留 0.15 余量给后续传播补强。
CONFIDENCE_MAP = {"low": 0.3, "medium": 0.6, "high": 0.85}

# 自动 approve 阈值。score ≥ 0.8 视为"证据充分到不需要人审"。
# 调高会让人审压力小但漏过坏知识；调低会让自动入库激进。0.8 是 conservative default。
AUTO_APPROVE_THRESHOLD = 0.8

# 传播 BFS 跳数上限。3 跳够覆盖"同接口→同服务→共享代码符号"三层关联，
# 再远的关联噪音大于信号。GraphRAG 的 community 探测也基本是 2-3 跳。
MAX_DEPTH = 3

# 每跳的衰减系数。0.5 = 每跳信号减半，3 跳后 ≈ 12.5%，配合 MIN_CHANGE 自然收敛。
DECAY_FACTOR = 0.5

# 单次传播最小增量。低于此值认为"传不动了"，跳过持久化（避免大量微小写盘）。
MIN_CHANGE = 0.01


def score_to_level(score: float) -> str:
    """数值置信度 → 离散等级。

    注意：阈值 0.75/0.5 与 CONFIDENCE_MAP 的等级数值 0.85/0.6/0.3 **故意不对称**。
    CONFIDENCE_MAP 是"等级 → 默认数值"（LLM 输出离散等级时回填），
    本函数是"传播后数值 → 等级"（边界更宽松，避免传播微调就跨级）。
    例：从 medium=0.6 传播到 0.7，仍归 medium；到 0.78 才升 high。
    """
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def level_to_score(level: str) -> float:
    """离散等级 → 数值置信度。"""
    return CONFIDENCE_MAP.get(level, 0.3)


class ConfidencePropagator:
    """置信度传播器。

    在 approve 之后调用 propagate()，自动提升关联 proposal 的置信度。
    """

    def __init__(self, graph: Any, proposals: Any, loader: Any = None) -> None:
        self.graph = graph  # GraphStore（查关联关系）
        self.proposals = proposals  # ProposalStore（读写 proposal）
        self.loader = loader  # RAGLoader（自动 approve 时写入知识库）

    def propagate(self, confirmed_proposal_id: str) -> list[str]:
        """确认一条知识后，传播置信度。

        返回: 被自动 approve 的 proposal_id 列表
        """
        confirmed = self.proposals.get(confirmed_proposal_id)
        auto_approved: list[str] = []

        # 找到确认 proposal 关联的所有实体（服务、接口）
        entities = self._extract_entities(confirmed)
        if not entities:
            return auto_approved

        # BFS 传播
        queue: list[tuple[str, str, float, int]] = [
            (label, node_id, 1.0, 0)
            for label, node_id in entities
        ]
        visited: set[tuple[str, str]] = set()

        while queue:
            label, node_id, source_score, depth = queue.pop(0)
            if depth >= MAX_DEPTH:
                continue
            key = (label, node_id)
            if key in visited:
                continue
            visited.add(key)

            # 找到共享该实体的 pending proposals
            affected = self._find_affected_proposals(label, node_id, confirmed_proposal_id)

            for proposal in affected:
                # 计算传播量
                edge_weight = self._get_edge_weight(label, node_id, proposal)
                boost = source_score * edge_weight * DECAY_FACTOR

                old_score = proposal.confidence_score
                new_score = min(1.0, old_score + boost)

                if new_score - old_score < MIN_CHANGE:
                    continue

                # 更新 proposal 的置信度
                self._update_confidence(proposal, new_score)
                logger.info(
                    "传播: %s score %.2f → %.2f (from %s)",
                    proposal.proposal_id, old_score, new_score, confirmed_proposal_id,
                )

                # 超过阈值，自动 approve
                if new_score >= AUTO_APPROVE_THRESHOLD and proposal.status == "pending_review":
                    self._auto_approve(proposal)
                    auto_approved.append(proposal.proposal_id)
                    # 自动 approve 的也触发传播（递归）
                    next_entities = self._extract_entities(proposal)
                    for nl, nid in next_entities:
                        queue.append((nl, nid, new_score, depth + 1))

        return auto_approved

    def _extract_entities(self, proposal: KnowledgeProposal) -> list[tuple[str, str]]:
        """从 proposal 中提取关联的图实体。"""
        entities: list[tuple[str, str]] = []
        # 接口
        if proposal.interface_key:
            entities.append(("Interface", proposal.interface_key))
        # 关联代码符号
        for sym in proposal.related_code_symbols:
            entities.append(("CodeSymbol", sym))
        # 从图中找该接口调用的服务
        if proposal.interface_key:
            for neighbor in self.graph.neighbors("Interface", proposal.interface_key, "CALLS_SERVICE"):
                entities.append(("Service", neighbor.get("id", "")))
        return [(l, n) for l, n in entities if n]

    def _find_affected_proposals(self, label: str, node_id: str, exclude_id: str) -> list[KnowledgeProposal]:
        """找到与该实体关联的 pending proposals（排除已确认的那条）。"""
        affected: list[KnowledgeProposal] = []
        pending = self.proposals.list_by_status("pending_review", limit=100)

        for proposal in pending:
            if proposal.proposal_id == exclude_id:
                continue
            # 检查该 proposal 是否与目标实体有关联
            if label == "Service":
                if node_id in (proposal.related_code_symbols or []):
                    affected.append(proposal)
                    continue
                # 检查 interface 是否调用了这个 service
                if proposal.interface_key:
                    neighbors = self.graph.neighbors("Interface", proposal.interface_key, "CALLS_SERVICE")
                    if any(n.get("id") == node_id for n in neighbors):
                        affected.append(proposal)
                        continue
            elif label == "Interface":
                if proposal.interface_key == node_id:
                    affected.append(proposal)
            elif label == "CodeSymbol":
                if node_id in (proposal.related_code_symbols or []):
                    affected.append(proposal)

        return affected

    def _get_edge_weight(self, label: str, node_id: str, proposal: KnowledgeProposal) -> float:
        """传播时的关联强度系数（独立于图存储边的 weight）。

        注意：这里的数值是**传播衰减系数**，跟 loader 写图时的边 weight（HAS_TRACE=0.9 /
        CALLS_SERVICE=0.7 / MENTIONS=0.6 等）**不是同一回事**——只是数值上凑巧接近。
        图边 weight 是描述"关系强度"的数据属性；这里的 0.9/0.7/0.5/0.3 是
        "通过这种关联传播多少信号"的算法系数。
        """
        if label == "Interface" and proposal.interface_key == node_id:
            return 0.9  # 同接口：最强关联，等价于"几乎就是同一条知识"
        if label == "Service":
            return 0.7  # 共享服务：强关联（接口都调用了同一个下游）
        if label == "CodeSymbol":
            return 0.5  # 共享代码符号：中等（可能是工具函数复用，不一定业务相关）
        return 0.3      # 其他：弱关联兜底

    def _update_confidence(self, proposal: KnowledgeProposal, new_score: float) -> None:
        """更新 proposal 的置信度分数和等级。"""
        proposal.confidence_score = new_score
        proposal.confidence = score_to_level(new_score)
        # 持久化更新
        self.proposals.update_confidence(proposal.proposal_id, new_score, proposal.confidence)

    def _auto_approve(self, proposal: KnowledgeProposal) -> None:
        """自动 approve 高置信度的 proposal。"""
        self.proposals.update_status(proposal.proposal_id, "approved")
        if self.loader:
            self.loader.load_approved_knowledge(asdict(proposal))
        logger.info("自动 approve: %s (score=%.2f)", proposal.proposal_id, proposal.confidence_score)
