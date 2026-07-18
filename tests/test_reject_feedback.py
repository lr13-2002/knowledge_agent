"""Reject 反馈环测试（Day 3）。

验证 reject 时基于反馈让 LLM 重生成 v2 的闭环：
- 有 reason + 有 regen client → 重生成，原地更新，状态回 pending_review
- 重试达上限（3 次）→ 转终态 reject
- 无 reason / 无 regen client → 直接终态 reject
- 重生成抛异常 → 降级为终态 reject
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent_platform.review import MAX_REJECT_RETRY, ReviewService
from agent_platform.schemas import Evidence, KnowledgeProposal
from agent_platform.stores import InMemoryGraphStore, InMemoryProposalStore, InMemoryVectorStore
from agent_platform.loader import RAGLoader


def _make_proposal(repo: str = "demo", trace_id: str = "t1") -> KnowledgeProposal:
    return KnowledgeProposal(
        repo=repo,
        trace_id=trace_id,
        interface_key=f"{repo}:POST:/x",
        summary="原始摘要（有问题）",
        flow_steps=["步骤1"],
        related_code_symbols=[],
        candidate_claims=["原始结论"],
        evidence=Evidence(trace_ids=[trace_id]),
        confidence="medium",
        confidence_score=0.6,
    )


def _regen_client(summary: str = "修正后的摘要", confidence: str = "medium") -> MagicMock:
    """构造一个 mock 的结构化重生成客户端。"""
    client = MagicMock()
    client.regenerate_from_feedback.return_value = {
        "summary": summary,
        "flow_steps": ["修正步骤1", "修正步骤2"],
        "candidate_claims": ["修正后的结论"],
        "confidence": confidence,
        "reasoning": "已根据反馈修正",
    }
    return client


class RejectFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposals = InMemoryProposalStore()
        self.loader = RAGLoader(InMemoryVectorStore(), InMemoryGraphStore())

    def test_reject_with_reason_regenerates(self) -> None:
        """有 reason + regen client → 重生成，状态回 pending_review。"""
        proposal = self.proposals.save(_make_proposal())
        review = ReviewService(self.proposals, self.loader, regen_client=_regen_client())

        result = review.reject(proposal.proposal_id, reason="把订单说成了乘客")

        self.assertEqual(result["reject_action"], "regenerated")
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual(result["summary"], "修正后的摘要")
        self.assertEqual(result["retry_count"], 1)
        self.assertIn("把订单说成了乘客", result["reject_history"])

    def test_reject_without_reason_terminal(self) -> None:
        """无 reason → 直接终态 reject（即使有 regen client）。"""
        proposal = self.proposals.save(_make_proposal())
        review = ReviewService(self.proposals, self.loader, regen_client=_regen_client())

        result = review.reject(proposal.proposal_id)  # 无 reason

        self.assertEqual(result["reject_action"], "rejected")
        self.assertEqual(result["status"], "rejected")

    def test_reject_no_regen_client_terminal(self) -> None:
        """无 regen client → 即使有 reason 也终态 reject。"""
        proposal = self.proposals.save(_make_proposal())
        review = ReviewService(self.proposals, self.loader)  # 无 regen_client

        result = review.reject(proposal.proposal_id, reason="有问题")

        self.assertEqual(result["reject_action"], "rejected")
        self.assertEqual(result["status"], "rejected")

    def test_retry_limit_then_terminal(self) -> None:
        """连续 reject 超过上限后转终态。"""
        proposal = self.proposals.save(_make_proposal())
        review = ReviewService(self.proposals, self.loader, regen_client=_regen_client())

        # 前 MAX_REJECT_RETRY 次都重生成
        for i in range(MAX_REJECT_RETRY):
            result = review.reject(proposal.proposal_id, reason=f"第{i+1}次驳回")
            self.assertEqual(result["reject_action"], "regenerated")
            self.assertEqual(result["retry_count"], i + 1)

        # 第 MAX_REJECT_RETRY+1 次：retry_count 已达上限 → 终态
        result = review.reject(proposal.proposal_id, reason="还是不行")
        self.assertEqual(result["reject_action"], "rejected")
        self.assertEqual(result["status"], "rejected")

    def test_regen_exception_falls_back_to_terminal(self) -> None:
        """重生成抛异常 → 降级为终态 reject，不崩。"""
        proposal = self.proposals.save(_make_proposal())
        bad_client = MagicMock()
        bad_client.regenerate_from_feedback.side_effect = RuntimeError("LLM down")
        review = ReviewService(self.proposals, self.loader, regen_client=bad_client)

        result = review.reject(proposal.proposal_id, reason="有问题")

        self.assertEqual(result["reject_action"], "rejected")
        self.assertEqual(result["status"], "rejected")

    def test_reject_history_accumulates(self) -> None:
        """多次 reject 的理由累加，全部喂给 LLM。"""
        proposal = self.proposals.save(_make_proposal())
        client = _regen_client()
        review = ReviewService(self.proposals, self.loader, regen_client=client)

        review.reject(proposal.proposal_id, reason="理由A")
        review.reject(proposal.proposal_id, reason="理由B")

        # 第二次调用时，传入的 reject_history 应包含两条
        last_call_args = client.regenerate_from_feedback.call_args_list[-1]
        passed_reasons = last_call_args[0][1]  # 第二个位置参数 reject_reasons
        self.assertIn("理由A", passed_reasons)
        self.assertIn("理由B", passed_reasons)


if __name__ == "__main__":
    unittest.main()
