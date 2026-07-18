"""Reinforcement（Day 5）+ recency 加权（Day 8）测试。

- rerank_by_mentions / rerank_hits：被反复确认（mentions 高）的实体排更前
- recency 加权：最近确认的排更前，老的温和降权但不归零
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from agent_platform.entities import rerank_by_mentions, rerank_hits


def _hit(hit_id: str, score: float, mentions: int) -> dict:
    return {"id": hit_id, "score": score, "payload": {"name": hit_id, "mentions": mentions}}


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class RerankByMentionsTest(unittest.TestCase):
    def test_high_mentions_boosts_ranking(self) -> None:
        """base_score 相同时，mentions 高的排前。"""
        hits = [
            _hit("长尾实体", score=1.0, mentions=1),
            _hit("核心实体", score=1.0, mentions=50),
        ]
        ranked = rerank_by_mentions(hits)
        self.assertEqual(ranked[0]["id"], "核心实体")
        self.assertEqual(ranked[1]["id"], "长尾实体")

    def test_relevance_still_dominates_when_gap_large(self) -> None:
        """相关性差距足够大时，不应被 mentions 反超（ln 阻尼）。"""
        hits = [
            _hit("强相关低频", score=5.0, mentions=1),
            _hit("弱相关高频", score=1.0, mentions=50),
        ]
        ranked = rerank_by_mentions(hits)
        # 5.0 × (1+0.3·ln2)=~6.04  vs  1.0 × (1+0.3·ln51)=~2.18
        self.assertEqual(ranked[0]["id"], "强相关低频")

    def test_missing_mentions_defaults_to_one(self) -> None:
        """payload 无 mentions 时按 1 算，不报错。"""
        hits = [
            {"id": "无mentions", "score": 2.0, "payload": {"name": "x"}},
            _hit("有mentions", score=2.0, mentions=10),
        ]
        ranked = rerank_by_mentions(hits)
        self.assertEqual(ranked[0]["id"], "有mentions")  # 同 score，有 mentions 的排前

    def test_reranked_score_annotated(self) -> None:
        """重排后标注 _reranked_score，便于调试。"""
        hits = [_hit("a", score=1.0, mentions=5)]
        ranked = rerank_by_mentions(hits)
        self.assertIn("_reranked_score", ranked[0])
        self.assertGreater(ranked[0]["_reranked_score"], 1.0)  # 有 mentions 加成

    def test_empty_list(self) -> None:
        """空列表不报错。"""
        self.assertEqual(rerank_by_mentions([]), [])

    def test_original_score_preserved(self) -> None:
        """不改原 score 字段。"""
        hits = [_hit("a", score=3.0, mentions=2)]
        ranked = rerank_by_mentions(hits)
        self.assertEqual(ranked[0]["score"], 3.0)


class RecencyWeightTest(unittest.TestCase):
    """Day 8：recency 加权。"""

    def _hit_t(self, hit_id: str, score: float, days_ago: int, mentions: int = 1) -> dict:
        return {
            "id": hit_id, "score": score,
            "payload": {"name": hit_id, "mentions": mentions, "last_seen_at": _iso_days_ago(days_ago)},
        }

    def test_recent_ranks_higher(self) -> None:
        """base/mentions 相同时，最近确认的排前。"""
        hits = [
            self._hit_t("老知识", score=1.0, days_ago=180),
            self._hit_t("新知识", score=1.0, days_ago=1),
        ]
        ranked = rerank_hits(hits, recency_field="last_seen_at")
        self.assertEqual(ranked[0]["id"], "新知识")

    def test_old_knowledge_not_zeroed(self) -> None:
        """老知识降权但不归零（180 天仍保留 ~33%）。"""
        hits = [self._hit_t("老知识", score=1.0, days_ago=180)]
        ranked = rerank_hits(hits, recency_field="last_seen_at")
        # 1/(1+180/90) = 1/3 ≈ 0.33，远大于 0
        self.assertGreater(ranked[0]["_reranked_score"], 0.3)

    def test_recency_gentle_not_exponential(self) -> None:
        """recency 是温和降权：强相关老知识不被弱相关新知识反超。"""
        hits = [
            self._hit_t("强相关老", score=5.0, days_ago=180),
            self._hit_t("弱相关新", score=1.0, days_ago=1),
        ]
        ranked = rerank_hits(hits, recency_field="last_seen_at")
        # 强相关老: 5.0 × ~0.33 = ~1.67  vs  弱相关新: 1.0 × ~1.0 = ~1.0
        self.assertEqual(ranked[0]["id"], "强相关老")

    def test_missing_time_no_penalty(self) -> None:
        """无时间字段 → 按 0 天算，不降权，不报错。"""
        hits = [
            {"id": "无时间", "score": 1.0, "payload": {"mentions": 1}},
            self._hit_t("有时间老", score=1.0, days_ago=180),
        ]
        ranked = rerank_hits(hits, recency_field="last_seen_at")
        self.assertEqual(ranked[0]["id"], "无时间")  # 无时间不降权，排前

    def test_no_recency_field_skips_weighting(self) -> None:
        """不传 recency_field → 只按 mentions，等价于 Day 5 行为。"""
        hits = [
            self._hit_t("老高频", score=1.0, days_ago=180, mentions=50),
            self._hit_t("新低频", score=1.0, days_ago=1, mentions=1),
        ]
        ranked = rerank_hits(hits, recency_field=None)
        self.assertEqual(ranked[0]["id"], "老高频")  # 不看时间，只看 mentions


if __name__ == "__main__":
    unittest.main()
