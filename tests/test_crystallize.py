"""Approve 结晶测试（Day 4）。

验证 approve 时把审核对话的改进固化进提案：
- 有对话 + 有 regen client → 结晶，入库精炼版
- 无对话 → 入库原版（不调 LLM，防 over-processing）
- 无 regen client → 入库原版
- 结晶异常 → 降级入库原版，不崩
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent_platform.review import ReviewService
from agent_platform.schemas import Evidence, KnowledgeProposal, ReviewMessage
from agent_platform.stores import InMemoryGraphStore, InMemoryProposalStore, InMemoryVectorStore
from agent_platform.loader import RAGLoader


def _make_proposal(repo: str = "demo", trace_id: str = "t1") -> KnowledgeProposal:
    return KnowledgeProposal(
        repo=repo,
        trace_id=trace_id,
        interface_key=f"{repo}:POST:/x",
        summary="原始摘要",
        flow_steps=["步骤1"],
        related_code_symbols=[],
        candidate_claims=["原始结论"],
        evidence=Evidence(trace_ids=[trace_id]),
        confidence="medium",
        confidence_score=0.6,
    )


def _crystallize_client(summary: str = "结晶后的精炼摘要") -> MagicMock:
    client = MagicMock()
    client.crystallize_from_discussion.return_value = {
        "summary": summary,
        "flow_steps": ["精炼步骤1"],
        "candidate_claims": ["精炼结论"],
        "confidence": "high",
        "reasoning": "已固化讨论共识",
    }
    return client


class CrystallizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposals = InMemoryProposalStore()
        self.loader = RAGLoader(InMemoryVectorStore(), InMemoryGraphStore())

    def test_approve_with_discussion_crystallizes(self) -> None:
        """有对话 + regen client → 结晶，入库精炼版。"""
        proposal = self.proposals.save(_make_proposal())
        self.proposals.add_message(proposal.proposal_id, ReviewMessage(role="user", content="这里应该是订单不是乘客"))
        self.proposals.add_message(proposal.proposal_id, ReviewMessage(role="assistant", content="已更正为订单"))

        client = _crystallize_client()
        review = ReviewService(self.proposals, self.loader, regen_client=client)
        result = review.approve(proposal.proposal_id)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["summary"], "结晶后的精炼摘要")
        self.assertEqual(result["confidence"], "high")
        client.crystallize_from_discussion.assert_called_once()

    def test_approve_no_discussion_keeps_original(self) -> None:
        """无对话 → 入库原版，不调 LLM。"""
        proposal = self.proposals.save(_make_proposal())
        client = _crystallize_client()
        review = ReviewService(self.proposals, self.loader, regen_client=client)

        result = review.approve(proposal.proposal_id)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["summary"], "原始摘要")  # 没变
        client.crystallize_from_discussion.assert_not_called()

    def test_approve_no_regen_client_keeps_original(self) -> None:
        """无 regen client → 入库原版（即使有对话）。"""
        proposal = self.proposals.save(_make_proposal())
        self.proposals.add_message(proposal.proposal_id, ReviewMessage(role="user", content="讨论"))
        review = ReviewService(self.proposals, self.loader)  # 无 regen_client

        result = review.approve(proposal.proposal_id)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["summary"], "原始摘要")

    def test_crystallize_exception_falls_back(self) -> None:
        """结晶抛异常 → 降级入库原版，不崩。"""
        proposal = self.proposals.save(_make_proposal())
        self.proposals.add_message(proposal.proposal_id, ReviewMessage(role="user", content="讨论"))
        bad = MagicMock()
        bad.crystallize_from_discussion.side_effect = RuntimeError("LLM down")
        review = ReviewService(self.proposals, self.loader, regen_client=bad)

        result = review.approve(proposal.proposal_id)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["summary"], "原始摘要")  # 降级保原版


if __name__ == "__main__":
    unittest.main()
