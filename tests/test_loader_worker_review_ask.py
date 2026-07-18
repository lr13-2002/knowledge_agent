import unittest
from dataclasses import asdict

from agent_platform.ask import AskService
from agent_platform.ingestor import handle_openclaw_trace
from agent_platform.loader import RAGLoader
from agent_platform.review import ReviewService
from agent_platform.sampler import AdaptiveSampler, SamplingConfig, SamplingRule
from agent_platform.stores import (
    InMemoryGraphStore,
    InMemoryIdempotencyStore,
    InMemoryProposalStore,
    InMemoryRawArtifactStore,
    InMemoryTaskQueue,
    InMemoryVectorStore,
)
from agent_platform.trace import FixtureTraceProvider
from agent_platform.worker import AgentWorker


class PlatformFlowTest(unittest.TestCase):
    def setUp(self):
        self.vector = InMemoryVectorStore()
        self.graph = InMemoryGraphStore()
        self.loader = RAGLoader(self.vector, self.graph)
        self.queue = InMemoryTaskQueue()
        self.proposals = InMemoryProposalStore()
        self.loader.load_code_index(
            {
                "repo_name": "spruce",
                "commit": "abc",
                "symbols": [
                    {
                        "id": "go:logics/active/utils.go:SendNotice:1",
                        "language": "go",
                        "qualified_name": "spruce.logics.active.SendNotice",
                        "calls": [],
                    }
                ],
                "chunks": [
                    {
                        "id": "code:send",
                        "language": "go",
                        "repo": "spruce",
                        "text": "driver active larix audit SendNotice",
                    }
                ],
            }
        )

    def test_loader_writes_graph_and_vector(self):
        self.assertIn("code_chunks", self.vector.collections)
        self.assertTrue(self.graph.find_nodes("CodeSymbol", language="go"))

    def test_worker_review_and_ask_flow(self):
        sampler = AdaptiveSampler(SamplingConfig(default=SamplingRule(percent=100, max_per_minute=10, min_per_day=0), cold_start_first_n=0))
        handle_openclaw_trace(
            {"repo": "spruce", "trace_id": "trace-1", "service": "spruce", "method": "POST", "path": "/driver/active"},
            sampler,
            self.queue,
            InMemoryIdempotencyStore(),
        )
        trace_provider = FixtureTraceProvider(
            {
                ("spruce", "trace-1"): {
                    "traceId": "trace-1",
                    "commit": "abc",
                    "spans": [
                        {"service": "gateway", "method": "POST", "path": "/driver/active"},
                        {"service": "larix", "operation": "query", "hasError": False},
                    ],
                }
            }
        )
        raw_store = InMemoryRawArtifactStore()
        worker = AgentWorker(self.queue, trace_provider, self.vector, self.graph, self.proposals, raw_store)
        proposal = worker.process_one()
        self.assertIsNotNone(proposal)
        self.assertIn("spruce/traces/raw/trace-1.json", raw_store.artifacts)
        self.assertEqual(proposal.status, "pending_review")
        self.assertFalse(self.graph.find_nodes("BusinessRule", status="approved"))

        review = ReviewService(self.proposals, self.loader)
        reply = review.message(proposal.proposal_id, "证据在哪？")
        self.assertIn("trace-1", reply.content)
        approved = review.approve(proposal.proposal_id)
        self.assertEqual(approved["status"], "approved")
        self.assertTrue(self.graph.find_nodes("BusinessRule", status="approved"))

        ask = AskService(self.vector)
        response = ask.ask("spruce", "driver active")
        self.assertIn("已确认知识", response.answer)
        self.assertTrue(response.evidence.business_rule_ids)

    def test_reject_and_revise(self):
        proposal = self.proposals.save(
            __import__("agent_platform.schemas", fromlist=["KnowledgeProposal"]).KnowledgeProposal(
                repo="spruce",
                trace_id="t2",
                interface_key="spruce:spruce:POST:/x",
                summary="old",
                flow_steps=[],
                related_code_symbols=[],
                candidate_claims=["old"],
                evidence=__import__("agent_platform.schemas", fromlist=["Evidence"]).Evidence(trace_ids=["t2"]),
            )
        )
        review = ReviewService(self.proposals, self.loader)
        revised = review.revise(proposal.proposal_id, "new", ["new claim"])
        self.assertEqual(revised["version"], 2)
        rejected = review.reject(proposal.proposal_id)
        self.assertEqual(rejected["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
