"""业务实体合并模块测试。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent_platform.entities import (
    is_stopword,
    merge_into_graph,
    normalize_name,
    slug_to_entity_id,
)
from agent_platform.schemas import BusinessEntity
from agent_platform.stores import InMemoryGraphStore


def _make_entity(name: str, type_: str = "business_concept", description: str = "") -> BusinessEntity:
    return BusinessEntity(
        name=name,
        type=type_,
        description=description or f"{name}的描述",
    )


class NormalizationTest(unittest.TestCase):
    def test_basic_normalize(self) -> None:
        self.assertEqual(normalize_name("订单"), "订单")
        self.assertEqual(normalize_name(" 订单 "), "订单")
        self.assertEqual(normalize_name("Order"), "order")

    def test_trim_suffix(self) -> None:
        # "订单数据" -> "订单"
        self.assertEqual(normalize_name("订单数据"), "订单")
        self.assertEqual(normalize_name("订单信息"), "订单")
        self.assertEqual(normalize_name("订单服务"), "订单")

    def test_punctuation(self) -> None:
        self.assertEqual(normalize_name("订单-状态"), "订单状态")
        self.assertEqual(normalize_name("Order_Service"), "order")

    def test_stopword(self) -> None:
        self.assertTrue(is_stopword("Controller"))
        self.assertTrue(is_stopword("OrderController"))
        self.assertTrue(is_stopword("Repository"))
        self.assertFalse(is_stopword("订单"))

    def test_slug_to_entity_id(self) -> None:
        self.assertEqual(slug_to_entity_id("订单"), "entity:订单")
        self.assertEqual(slug_to_entity_id(""), "")


class EntityMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = InMemoryGraphStore()
        # mock 向量库（不让相似度合并触发，单独测）
        self.vector = MagicMock()
        self.vector.search.return_value = []
        self.vector.upsert = MagicMock()

    def test_create_new_entity(self) -> None:
        ent = _make_entity("订单")
        merged = merge_into_graph(self.graph, self.vector, ent, "p1", "t1", "repo")
        self.assertEqual(merged.entity_id, "entity:订单")
        self.assertEqual(merged.mentions, 1)
        # 验证图里写了
        nodes = self.graph.find_nodes("Entity")
        self.assertEqual(len(nodes), 1)

    def test_exact_match_merge(self) -> None:
        # 第一次创建
        merge_into_graph(self.graph, self.vector, _make_entity("订单"), "p1", "t1", "repo")
        # 第二次同名 → 合并
        merged = merge_into_graph(self.graph, self.vector, _make_entity("订单"), "p2", "t2", "repo")
        self.assertEqual(merged.mentions, 2)
        self.assertIn("p1", merged.source_proposal_ids)
        self.assertIn("p2", merged.source_proposal_ids)

    def test_normalized_name_merge(self) -> None:
        # "订单" 和 "订单数据" 归一化后都是"订单"
        merge_into_graph(self.graph, self.vector, _make_entity("订单"), "p1", "t1", "repo")
        merged = merge_into_graph(self.graph, self.vector, _make_entity("订单数据"), "p2", "t2", "repo")
        self.assertEqual(merged.entity_id, "entity:订单")
        self.assertEqual(merged.mentions, 2)
        self.assertIn("订单数据", merged.aliases)

    def test_stopword_skipped(self) -> None:
        merged = merge_into_graph(
            self.graph, self.vector,
            _make_entity("Controller", type_="other"),
            "p1", "t1", "repo",
        )
        # 停用词不写图
        nodes = self.graph.find_nodes("Entity")
        self.assertEqual(len(nodes), 0)

    def test_substring_match_short_wins(self) -> None:
        # 先有 "订单" → 再来"订单状态机"，类型不同时不应合并
        merge_into_graph(self.graph, self.vector, _make_entity("订单", type_="business_concept"),
                         "p1", "t1", "repo")
        merged = merge_into_graph(
            self.graph, self.vector,
            _make_entity("订单状态机", type_="capability"),
            "p2", "t2", "repo",
        )
        # 类型不同 -> 不合并 -> 新建
        self.assertEqual(merged.entity_id, "entity:订单状态机")


if __name__ == "__main__":
    unittest.main()
