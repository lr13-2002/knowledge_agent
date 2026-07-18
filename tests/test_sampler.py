import unittest
from datetime import datetime, timezone

from agent_platform.sampler import AdaptiveSampler, SamplingConfig, SamplingRule
from agent_platform.schemas import OpenClawEvent


class SamplerTest(unittest.TestCase):
    def test_per_interface_minute_limit(self):
        config = SamplingConfig(default=SamplingRule(percent=100, max_per_minute=2, min_per_day=0), cold_start_first_n=0)
        sampler = AdaptiveSampler(config)
        now = datetime(2026, 5, 11, 1, 1, tzinfo=timezone.utc)
        event = OpenClawEvent.from_dict({"repo": "spruce", "trace_id": "t1", "service": "spruce", "method": "POST", "path": "/a"})
        self.assertTrue(sampler.decide(event, now).accepted)
        self.assertTrue(sampler.decide(OpenClawEvent.from_dict({**event.raw_event, "repo": "spruce", "trace_id": "t2", "service": "spruce", "method": "POST", "path": "/a"}), now).accepted)
        self.assertFalse(sampler.decide(OpenClawEvent.from_dict({"repo": "spruce", "trace_id": "t3", "service": "spruce", "method": "POST", "path": "/a"}), now).accepted)

    def test_low_traffic_min_per_day(self):
        config = SamplingConfig(default=SamplingRule(percent=0, max_per_minute=10, min_per_day=2), cold_start_first_n=0)
        sampler = AdaptiveSampler(config)
        now = datetime(2026, 5, 11, 1, 1, tzinfo=timezone.utc)
        base = {"repo": "spruce", "service": "spruce", "method": "GET", "path": "/rare"}
        self.assertEqual(sampler.decide(OpenClawEvent.from_dict({**base, "trace_id": "a"}), now).reason, "min_per_day")
        self.assertEqual(sampler.decide(OpenClawEvent.from_dict({**base, "trace_id": "b"}), now).reason, "min_per_day")

    def test_error_boost(self):
        config = SamplingConfig(default=SamplingRule(percent=10, max_per_minute=10, min_per_day=0), boost={"error": 10}, cold_start_first_n=0)
        sampler = AdaptiveSampler(config)
        event = OpenClawEvent.from_dict({"repo": "spruce", "trace_id": "error-1", "service": "spruce", "method": "POST", "path": "/x", "status": "error"})
        decision = sampler.decide(event, datetime(2026, 5, 11, tzinfo=timezone.utc))
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.effective_percent, 100)


if __name__ == "__main__":
    unittest.main()
