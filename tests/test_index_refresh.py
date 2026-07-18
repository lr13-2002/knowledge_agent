"""代码索引增量更新测试（Day 7）。

验证按 git commit hash 检测变化：
- 首次遇到仓库 → 索引
- commit 没变 → 跳过（不重复解析）
- commit 变了 → 重索引
- refresh_all_repos 批量按 commit 刷新
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agent_platform.worker import AgentWorker
from agent_platform.stores import (
    InMemoryGraphStore,
    InMemoryProposalStore,
    InMemoryTaskQueue,
    InMemoryVectorStore,
)


class _FakeIndex:
    """假的 CodeIndex，as_dict 返回最小结构。"""
    def __init__(self, commit: str) -> None:
        self.commit = commit
        self.chunks = []

    def as_dict(self) -> dict:
        return {"repo_name": "demo", "commit": self.commit, "symbols": [], "chunks": []}


def _make_worker() -> AgentWorker:
    vector = InMemoryVectorStore()
    graph = InMemoryGraphStore()
    proposals = InMemoryProposalStore()
    queue = InMemoryTaskQueue()
    indexer = MagicMock()
    indexer.index_repo.side_effect = lambda root, name: _FakeIndex("c-" + name)
    worker = AgentWorker(
        queue, MagicMock(), vector, graph, proposals,
        indexer=indexer, repo_roots={"demo": "/fake/demo"},
    )
    return worker


class IndexRefreshTest(unittest.TestCase):
    def test_first_time_indexes(self) -> None:
        worker = _make_worker()
        with patch("agent_platform.indexer.index._git_head", return_value="commit1"):
            worker._ensure_repo_indexed("demo")
        self.assertEqual(worker._indexed_commits["demo"], "commit1")
        self.assertEqual(worker.indexer.index_repo.call_count, 1)

    def test_unchanged_commit_skips(self) -> None:
        worker = _make_worker()
        with patch("agent_platform.indexer.index._git_head", return_value="commit1"):
            worker._ensure_repo_indexed("demo")  # 首次
            worker._ensure_repo_indexed("demo")  # commit 没变
        # 只索引一次
        self.assertEqual(worker.indexer.index_repo.call_count, 1)

    def test_changed_commit_reindexes(self) -> None:
        worker = _make_worker()
        with patch("agent_platform.indexer.index._git_head", return_value="commit1"):
            worker._ensure_repo_indexed("demo")
        with patch("agent_platform.indexer.index._git_head", return_value="commit2"):
            worker._ensure_repo_indexed("demo")  # commit 变了
        self.assertEqual(worker.indexer.index_repo.call_count, 2)
        self.assertEqual(worker._indexed_commits["demo"], "commit2")

    def test_unknown_repo_no_root_skips(self) -> None:
        worker = _make_worker()
        worker._ensure_repo_indexed("not_configured")  # 不在 repo_roots
        self.assertEqual(worker.indexer.index_repo.call_count, 0)

    def test_refresh_all_only_reindexes_changed(self) -> None:
        worker = _make_worker()
        with patch("agent_platform.indexer.index._git_head", return_value="commit1"):
            worker._ensure_repo_indexed("demo")
            n = worker.refresh_all_repos()  # commit 没变
        self.assertEqual(n, 0)
        self.assertEqual(worker.indexer.index_repo.call_count, 1)

        with patch("agent_platform.indexer.index._git_head", return_value="commit2"):
            n = worker.refresh_all_repos()  # commit 变了
        self.assertEqual(n, 1)
        self.assertEqual(worker.indexer.index_repo.call_count, 2)


if __name__ == "__main__":
    unittest.main()
