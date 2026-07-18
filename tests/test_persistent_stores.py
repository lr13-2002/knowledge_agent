"""Tests for Chroma + SQLite persistent stores."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_platform.persistent_stores import init_persistent_stores
from agent_platform.schemas import Evidence, KnowledgeProposal, ReviewMessage


class PersistentVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.vector, self.graph, self.proposals = init_persistent_stores(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_upsert_and_search(self) -> None:
        self.vector.upsert("docs", "d1", "用户登录认证流程", {"repo": "auth-svc"})
        self.vector.upsert("docs", "d2", "订单支付结算逻辑", {"repo": "pay-svc"})
        results = self.vector.search("docs", "登录", limit=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], "d1")

    def test_search_with_filter(self) -> None:
        self.vector.upsert("docs", "d1", "登录流程", {"repo": "auth"})
        self.vector.upsert("docs", "d2", "登录接口", {"repo": "gateway"})
        results = self.vector.search("docs", "登录", filters={"repo": "auth"}, limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "d1")

    def test_persistence_across_instances(self) -> None:
        self.vector.upsert("docs", "d1", "持久化测试文档", {"repo": "test"})
        vector2, _, _ = init_persistent_stores(self._tmp.name)
        results = vector2.search("docs", "持久化", limit=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], "d1")


class PersistentGraphStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        _, self.graph, _ = init_persistent_stores(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_upsert_and_find(self) -> None:
        self.graph.upsert_node("Service", "pay-svc", {"name": "pay-svc", "repo": "pay"})
        nodes = self.graph.find_nodes("Service", repo="pay")
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["name"], "pay-svc")

    def test_edges_and_neighbors(self) -> None:
        self.graph.upsert_node("Interface", "/pay", {"path": "/pay"})
        self.graph.upsert_node("Service", "redis", {"name": "redis"})
        self.graph.add_edge("Interface", "/pay", "CALLS_SERVICE", "Service", "redis")
        neighbors = self.graph.neighbors("Interface", "/pay", "CALLS_SERVICE")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["name"], "redis")

    def test_persistence_across_instances(self) -> None:
        self.graph.upsert_node("Service", "s1", {"name": "s1"})
        self.graph.add_edge("Service", "s1", "DEPENDS", "Service", "s1")
        _, graph2, _ = init_persistent_stores(self._tmp.name)
        nodes = graph2.find_nodes("Service", name="s1")
        self.assertEqual(len(nodes), 1)


class PersistentProposalStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        _, _, self.proposals = init_persistent_stores(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_proposal(self, repo: str = "r", trace_id: str = "t1") -> KnowledgeProposal:
        return KnowledgeProposal(
            repo=repo, trace_id=trace_id, interface_key=f"{repo}:svc:GET:/path",
            summary="test summary", flow_steps=["step1"], related_code_symbols=[],
            candidate_claims=["claim1"], evidence=Evidence(trace_ids=[trace_id]),
        )

    def test_save_and_get(self) -> None:
        p = self._make_proposal()
        saved = self.proposals.save(p)
        got = self.proposals.get(saved.proposal_id)
        self.assertEqual(got.summary, "test summary")

    def test_duplicate_returns_existing(self) -> None:
        p1 = self._make_proposal()
        saved1 = self.proposals.save(p1)
        p2 = self._make_proposal()
        saved2 = self.proposals.save(p2)
        self.assertEqual(saved1.proposal_id, saved2.proposal_id)

    def test_update_status(self) -> None:
        p = self._make_proposal()
        saved = self.proposals.save(p)
        updated = self.proposals.update_status(saved.proposal_id, "approved")
        self.assertEqual(updated.status, "approved")

    def test_revise(self) -> None:
        p = self._make_proposal()
        saved = self.proposals.save(p)
        revised = self.proposals.revise(saved.proposal_id, "new summary", ["new claim"])
        self.assertEqual(revised.summary, "new summary")
        self.assertEqual(revised.version, 2)

    def test_persistence_across_instances(self) -> None:
        p = self._make_proposal()
        saved = self.proposals.save(p)
        _, _, proposals2 = init_persistent_stores(self._tmp.name)
        got = proposals2.get(saved.proposal_id)
        self.assertEqual(got.summary, "test summary")


if __name__ == "__main__":
    unittest.main()
