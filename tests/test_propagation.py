"""置信度传播引擎测试。"""
from __future__ import annotations

import unittest

from agent_platform.propagation import ConfidencePropagator, score_to_level, level_to_score
from agent_platform.schemas import Evidence, KnowledgeProposal
from agent_platform.stores import InMemoryGraphStore, InMemoryProposalStore


def _make_proposal(
    proposal_id: str,
    interface_key: str = "repo:svc:POST:/api",
    confidence: str = "low",
    confidence_score: float = 0.3,
    related_code_symbols: list | None = None,
    repo: str = "test-repo",
) -> KnowledgeProposal:
    return KnowledgeProposal(
        repo=repo,
        trace_id=f"trace-{proposal_id}",
        interface_key=interface_key,
        summary=f"summary for {proposal_id}",
        flow_steps=["step1"],
        related_code_symbols=related_code_symbols or [],
        candidate_claims=["claim1"],
        evidence=Evidence(trace_ids=[f"trace-{proposal_id}"]),
        confidence=confidence,
        confidence_score=confidence_score,
        proposal_id=proposal_id,
    )


class ScoreConversionTest(unittest.TestCase):
    def test_score_to_level(self) -> None:
        self.assertEqual(score_to_level(0.9), "high")
        self.assertEqual(score_to_level(0.75), "high")
        self.assertEqual(score_to_level(0.6), "medium")
        self.assertEqual(score_to_level(0.5), "medium")
        self.assertEqual(score_to_level(0.3), "low")
        self.assertEqual(score_to_level(0.0), "low")

    def test_level_to_score(self) -> None:
        self.assertEqual(level_to_score("high"), 0.85)
        self.assertEqual(level_to_score("medium"), 0.6)
        self.assertEqual(level_to_score("low"), 0.3)


class PropagationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = InMemoryGraphStore()
        self.proposals = InMemoryProposalStore()

    def test_shared_service_propagation(self) -> None:
        """确认 A 后，共享同一个 service 的 B 的置信度应提升。"""
        # A 和 B 都调用 pay-svc
        self.graph.upsert_node("Interface", "repo:svc:POST:/order", {"repo": "test-repo"})
        self.graph.upsert_node("Interface", "repo:svc:POST:/refund", {"repo": "test-repo"})
        self.graph.upsert_node("Service", "pay-svc", {"name": "pay-svc"})
        self.graph.add_edge("Interface", "repo:svc:POST:/order", "CALLS_SERVICE", "Service", "pay-svc", weight=0.7)
        self.graph.add_edge("Interface", "repo:svc:POST:/refund", "CALLS_SERVICE", "Service", "pay-svc", weight=0.7)

        proposal_a = _make_proposal("pa", interface_key="repo:svc:POST:/order", confidence="high", confidence_score=0.85)
        proposal_b = _make_proposal("pb", interface_key="repo:svc:POST:/refund", confidence="low", confidence_score=0.3)

        self.proposals.save(proposal_a)
        self.proposals.save(proposal_b)
        # Approve A
        self.proposals.update_status("pa", "approved")

        propagator = ConfidencePropagator(self.graph, self.proposals)
        auto_approved = propagator.propagate("pa")

        # B 的 score 应该提升了
        b = self.proposals.get("pb")
        self.assertGreater(b.confidence_score, 0.3)

    def test_same_interface_propagation(self) -> None:
        """同一个接口的两条 proposal，确认一条后另一条提升。"""
        iface = "repo:svc:POST:/api"
        self.graph.upsert_node("Interface", iface, {"repo": "test-repo"})

        proposal_a = _make_proposal("pa", interface_key=iface, confidence="high", confidence_score=0.85)
        proposal_b = _make_proposal("pb", interface_key=iface, confidence="low", confidence_score=0.3)

        self.proposals.save(proposal_a)
        self.proposals.save(proposal_b)
        self.proposals.update_status("pa", "approved")

        propagator = ConfidencePropagator(self.graph, self.proposals)
        propagator.propagate("pa")

        b = self.proposals.get("pb")
        self.assertGreater(b.confidence_score, 0.3)
        # 同接口权重 0.9，传播量 = 1.0 × 0.9 × 0.5 = 0.45，新值 = 0.75
        self.assertAlmostEqual(b.confidence_score, 0.75, places=1)

    def test_auto_approve_on_threshold(self) -> None:
        """score 超过 0.8 时自动 approve。"""
        iface = "repo:svc:POST:/api"
        self.graph.upsert_node("Interface", iface, {"repo": "test-repo"})

        proposal_a = _make_proposal("pa", interface_key=iface, confidence="high", confidence_score=0.85)
        # B 的初始 score 已经很高，传播后会超过 0.8
        proposal_b = _make_proposal("pb", interface_key=iface, confidence="medium", confidence_score=0.6)

        self.proposals.save(proposal_a)
        self.proposals.save(proposal_b)
        self.proposals.update_status("pa", "approved")

        propagator = ConfidencePropagator(self.graph, self.proposals)
        auto_approved = propagator.propagate("pa")

        # B 应该被自动 approve: 0.6 + 1.0 × 0.9 × 0.5 = 1.05 → capped at 1.0
        self.assertIn("pb", auto_approved)
        b = self.proposals.get("pb")
        self.assertEqual(b.status, "approved")

    def test_no_infinite_loop(self) -> None:
        """传播不会无限循环（depth 限制）。"""
        # 创建环形依赖
        self.graph.upsert_node("Interface", "if1", {"repo": "r"})
        self.graph.upsert_node("Interface", "if2", {"repo": "r"})
        self.graph.upsert_node("Service", "s1", {"name": "s1"})
        self.graph.add_edge("Interface", "if1", "CALLS_SERVICE", "Service", "s1", weight=0.7)
        self.graph.add_edge("Interface", "if2", "CALLS_SERVICE", "Service", "s1", weight=0.7)

        pa = _make_proposal("pa", interface_key="if1", confidence="high", confidence_score=0.85)
        pb = _make_proposal("pb", interface_key="if2", confidence="low", confidence_score=0.3)
        self.proposals.save(pa)
        self.proposals.save(pb)
        self.proposals.update_status("pa", "approved")

        propagator = ConfidencePropagator(self.graph, self.proposals)
        # 不应该抛异常或无限循环
        auto_approved = propagator.propagate("pa")
        self.assertIsInstance(auto_approved, list)


if __name__ == "__main__":
    unittest.main()
