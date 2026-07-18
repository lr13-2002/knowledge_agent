import unittest

from agent_platform.ingestor import STREAM_NAME, handle_openclaw_trace
from agent_platform.sampler import AdaptiveSampler, SamplingConfig, SamplingRule
from agent_platform.stores import InMemoryIdempotencyStore, InMemoryTaskQueue


class IngestorTest(unittest.TestCase):
    def setUp(self):
        self.queue = InMemoryTaskQueue()
        self.idempotency = InMemoryIdempotencyStore()
        self.sampler = AdaptiveSampler(SamplingConfig(default=SamplingRule(percent=100, max_per_minute=10, min_per_day=0), cold_start_first_n=0))

    def test_accepts_and_enqueues(self):
        result = handle_openclaw_trace({"repo": "spruce", "trace_id": "t1", "service": "spruce", "method": "POST", "path": "/driver/active"}, self.sampler, self.queue, self.idempotency)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(self.queue.messages), 1)
        self.assertEqual(self.queue.messages[0]["stream"], STREAM_NAME)

    def test_missing_fields_raise(self):
        with self.assertRaises(ValueError):
            handle_openclaw_trace({"repo": "spruce"}, self.sampler, self.queue, self.idempotency)

    def test_duplicate(self):
        payload = {"repo": "spruce", "trace_id": "t1", "service": "spruce", "method": "POST", "path": "/driver/active"}
        handle_openclaw_trace(payload, self.sampler, self.queue, self.idempotency)
        result = handle_openclaw_trace(payload, self.sampler, self.queue, self.idempotency)
        self.assertEqual(result["status"], "duplicate")

    def test_skipped_not_enqueued(self):
        sampler = AdaptiveSampler(SamplingConfig(default=SamplingRule(percent=0, max_per_minute=10, min_per_day=0), cold_start_first_n=0))
        result = handle_openclaw_trace({"repo": "spruce", "trace_id": "t9", "service": "spruce", "method": "POST", "path": "/skip"}, sampler, self.queue, self.idempotency)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(self.queue.messages), 0)


if __name__ == "__main__":
    unittest.main()
