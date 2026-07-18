"""社区检测模块测试。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent_platform.community import CommunityDetector
from agent_platform.stores import InMemoryGraphStore, InMemoryVectorStore


class CommunityDetectionTest(unittest.TestCase):
    """构造一个 12 节点 3 群的图，验证 Leiden 能正确分群。"""

    def setUp(self) -> None:
        self.graph = InMemoryGraphStore()
        self.vector = InMemoryVectorStore()

        # 群 1: 支付域（紧密互联）
        for name in ["订单", "支付", "扣款"]:
            self.graph.upsert_node("Entity", f"entity:{name}", {
                "name": name, "type": "business_concept", "description": f"{name}描述",
                "mentions": 5, "repo": "test",
            })
        for a, b in [("订单", "支付"), ("支付", "扣款"), ("订单", "扣款")]:
            self.graph.add_edge("Entity", f"entity:{a}", "RELATED_TO", "Entity", f"entity:{b}", weight=8)
            self.graph.add_edge("Entity", f"entity:{b}", "RELATED_TO", "Entity", f"entity:{a}", weight=8)

        # 群 2: 司机域（紧密互联）
        for name in ["司机", "派单", "导航"]:
            self.graph.upsert_node("Entity", f"entity:{name}", {
                "name": name, "type": "business_concept", "description": f"{name}描述",
                "mentions": 5, "repo": "test",
            })
        for a, b in [("司机", "派单"), ("派单", "导航"), ("司机", "导航")]:
            self.graph.add_edge("Entity", f"entity:{a}", "RELATED_TO", "Entity", f"entity:{b}", weight=8)
            self.graph.add_edge("Entity", f"entity:{b}", "RELATED_TO", "Entity", f"entity:{a}", weight=8)

        # 群 3: 风控域
        for name in ["风控", "黑名单", "反欺诈"]:
            self.graph.upsert_node("Entity", f"entity:{name}", {
                "name": name, "type": "business_concept", "description": f"{name}描述",
                "mentions": 5, "repo": "test",
            })
        for a, b in [("风控", "黑名单"), ("黑名单", "反欺诈"), ("风控", "反欺诈")]:
            self.graph.add_edge("Entity", f"entity:{a}", "RELATED_TO", "Entity", f"entity:{b}", weight=8)
            self.graph.add_edge("Entity", f"entity:{b}", "RELATED_TO", "Entity", f"entity:{a}", weight=8)

        # 跨群弱连接（不应破坏分群）—— 用低权重 RELATED_TO 表达"共现而非业务关系"
        # （CO_MENTIONS 已合并到 RELATED_TO，参见 worker.py:_write_entities_to_graph）
        self.graph.add_edge("Entity", "entity:订单", "RELATED_TO", "Entity", "entity:司机", weight=0.4)
        self.graph.add_edge("Entity", "entity:支付", "RELATED_TO", "Entity", "entity:风控", weight=0.4)

    def test_leiden_finds_three_clusters(self) -> None:
        """Leiden 应该能识别出 3 个明显的社区。"""
        detector = CommunityDetector(self.graph, self.vector, loader=None, llm_client=None)
        partitions = detector._run_leiden(
            self.graph.all_nodes("Entity"),
            self.graph.all_edges(["RELATED_TO"]),
        )
        # 应该有 3 个社区
        self.assertEqual(len(partitions), 3)
        # 每个社区应该是 3 个成员
        sizes = sorted([len(v) for v in partitions.values()])
        self.assertEqual(sizes, [3, 3, 3])

    def test_full_pipeline_with_template_fallback(self) -> None:
        """完整流程，无 LLM 时降级到模板摘要。"""
        # 创建一个 mock loader
        loader = MagicMock()
        detector = CommunityDetector(self.graph, self.vector, loader=loader, llm_client=None)
        reports = detector.detect_and_summarize()

        self.assertEqual(len(reports), 3)
        # 模板摘要会包含成员名
        for report in reports:
            self.assertTrue(len(report.member_entity_ids) >= 2)
            self.assertIn("领域", report.title)
        # 应该写了 loader
        self.assertEqual(loader.load_community.call_count, 3)

    def test_empty_graph(self) -> None:
        """空图时直接返回，不报错。"""
        empty_graph = InMemoryGraphStore()
        empty_vector = InMemoryVectorStore()
        detector = CommunityDetector(empty_graph, empty_vector, loader=None, llm_client=None)
        reports = detector.detect_and_summarize()
        self.assertEqual(reports, [])

    def test_community_id_is_deterministic(self) -> None:
        """同一组成员永远生成同一个 community_id（幂等）。"""
        detector = CommunityDetector(self.graph, self.vector, loader=MagicMock(), llm_client=None)
        reports1 = detector.detect_and_summarize()
        reports2 = detector.detect_and_summarize()
        ids1 = sorted([r.community_id for r in reports1])
        ids2 = sorted([r.community_id for r in reports2])
        self.assertEqual(ids1, ids2)


if __name__ == "__main__":
    unittest.main()
