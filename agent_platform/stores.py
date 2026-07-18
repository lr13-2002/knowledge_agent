"""内存存储实现。

提供全套内存存储，用于：
1. 单元测试（无需外部依赖）
2. 不需要持久化的组件（TaskQueue、IdempotencyStore、SamplerState）
3. 开发时快速验证

生产环境中，VectorStore/GraphStore/ProposalStore 使用 persistent_stores.py 的实现替代。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import KnowledgeProposal, ReviewMessage


class InMemoryTaskQueue:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.dead_letters: list[dict[str, Any]] = []

    def enqueue(self, stream: str, message: dict[str, Any]) -> str:
        message_id = f"{stream}-{len(self.messages) + 1}"
        self.messages.append({"id": message_id, "stream": stream, "message": dict(message)})
        return message_id

    def pop(self, stream: str) -> dict[str, Any] | None:
        for index, item in enumerate(self.messages):
            if item["stream"] == stream:
                return self.messages.pop(index)["message"]
        return None

    def dead_letter(self, stream: str, message: dict[str, Any], reason: str) -> None:
        self.dead_letters.append({"stream": stream, "message": dict(message), "reason": reason})


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    def seen(self, key: str) -> bool:
        return key in self.keys

    def mark(self, key: str) -> None:
        self.keys.add(key)


class InMemorySamplerState:
    def __init__(self) -> None:
        self.total_by_interface: dict[str, int] = {}
        self.accepted_by_interface: dict[str, int] = {}
        self.minute_by_interface: dict[tuple[str, str], int] = {}
        self.day_by_interface: dict[tuple[str, str], int] = {}
        self.skipped: list[dict[str, Any]] = []

    def record_total(self, interface_key: str) -> int:
        value = self.total_by_interface.get(interface_key, 0) + 1
        self.total_by_interface[interface_key] = value
        return value

    def record_accepted(self, interface_key: str, minute_bucket: str, day_bucket: str) -> None:
        self.accepted_by_interface[interface_key] = self.accepted_by_interface.get(interface_key, 0) + 1
        self.minute_by_interface[(interface_key, minute_bucket)] = self.minute_by_interface.get((interface_key, minute_bucket), 0) + 1
        self.day_by_interface[(interface_key, day_bucket)] = self.day_by_interface.get((interface_key, day_bucket), 0) + 1


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def upsert(self, collection: str, item_id: str, text: str, payload: dict[str, Any]) -> None:
        items = self.collections.setdefault(collection, [])
        for item in items:
            if item["id"] == item_id:
                item.update({"text": text, "payload": dict(payload)})
                return
        items.append({"id": item_id, "text": text, "payload": dict(payload)})

    def search(self, collection: str, query: str, filters: dict[str, Any] | None = None, limit: int = 5) -> list[dict[str, Any]]:
        filters = filters or {}
        terms = [term.lower() for term in query.split() if term.strip()]
        results = []
        for item in self.collections.get(collection, []):
            payload = item["payload"]
            if any(payload.get(key) != value for key, value in filters.items()):
                continue
            text = item["text"].lower()
            score = sum(1 for term in terms if term in text)
            if score > 0 or not terms:
                results.append({**item, "score": score})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str, str, str], float] = {}

    def upsert_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        current = self.nodes.get((label, node_id), {})
        current.update(properties)
        current["id"] = node_id
        current["label"] = label
        self.nodes[(label, node_id)] = current

    def add_edge(self, from_label: str, from_id: str, relation: str, to_label: str, to_id: str, weight: float = 1.0) -> None:
        key = (from_label, from_id, relation, to_label, to_id)
        self.edges[key] = weight

    def find_nodes(self, label: str, **properties: Any) -> list[dict[str, Any]]:
        out = []
        for (node_label, _), node in self.nodes.items():
            if node_label != label:
                continue
            if all(node.get(key) == value for key, value in properties.items()):
                out.append(node)
        return out

    def neighbors(self, from_label: str, from_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        out = []
        for (fl, fid, rel, tl, tid), weight in self.edges.items():
            if fl == from_label and fid == from_id:
                if relation and rel != relation:
                    continue
                node = self.nodes.get((tl, tid))
                if node:
                    out.append({**node, "_relation": rel, "_weight": weight})
        return out

    def all_nodes(self, label: str | None = None) -> list[dict[str, Any]]:
        """遍历所有节点（可按 label 过滤）。"""
        out = []
        for (l, _), node in self.nodes.items():
            if label is None or l == label:
                out.append(node)
        return out

    def all_edges(self, relations: list[str] | None = None) -> list[dict[str, Any]]:
        """遍历所有边（可按 relation 列表过滤）。"""
        out = []
        for (fl, fid, rel, tl, tid), weight in self.edges.items():
            if relations is None or rel in relations:
                out.append({
                    "from_label": fl,
                    "from_id": fid,
                    "relation": rel,
                    "to_label": tl,
                    "to_id": tid,
                    "weight": weight,
                })
        return out

    def delete_nodes(self, label: str) -> None:
        """删除指定 label 的所有节点及相关边。"""
        self.nodes = {k: v for k, v in self.nodes.items() if k[0] != label}
        self.edges = {k: w for k, w in self.edges.items() if k[0] != label and k[3] != label}

    def reverse_neighbors(self, to_label: str, to_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        """反向查询：谁指向了这个节点。"""
        out = []
        for (fl, fid, rel, tl, tid), weight in self.edges.items():
            if tl == to_label and tid == to_id:
                if relation and rel != relation:
                    continue
                node = self.nodes.get((fl, fid))
                if node:
                    out.append({**node, "_relation": rel, "_weight": weight})
        return out


class InMemoryProposalStore:
    def __init__(self) -> None:
        self.proposals: dict[str, KnowledgeProposal] = {}
        self.by_trace: dict[tuple[str, str], str] = {}
        self.messages: dict[str, list[ReviewMessage]] = {}

    def save(self, proposal: KnowledgeProposal) -> KnowledgeProposal:
        key = (proposal.repo, proposal.trace_id)
        existing = self.by_trace.get(key)
        if existing:
            return self.proposals[existing]
        self.proposals[proposal.proposal_id] = proposal
        self.by_trace[key] = proposal.proposal_id
        self.messages.setdefault(proposal.proposal_id, [])
        return proposal

    def get(self, proposal_id: str) -> KnowledgeProposal:
        return self.proposals[proposal_id]

    def add_message(self, proposal_id: str, message: ReviewMessage) -> None:
        self.messages.setdefault(proposal_id, []).append(message)

    def get_messages(self, proposal_id: str) -> list[ReviewMessage]:
        return list(self.messages.get(proposal_id, []))

    def update_status(self, proposal_id: str, status: str) -> KnowledgeProposal:
        proposal = self.proposals[proposal_id]
        proposal.status = status
        return proposal

    def revise(self, proposal_id: str, summary: str, claims: list[str]) -> KnowledgeProposal:
        proposal = self.proposals[proposal_id]
        proposal.summary = summary
        proposal.candidate_claims = claims
        proposal.version += 1
        proposal.status = "pending_review"
        return proposal

    def replace(self, proposal: KnowledgeProposal) -> KnowledgeProposal:
        """原地整体替换提案内容（保留 proposal_id / repo / trace_id）。

        用于 reject 反馈环：LLM 重生成后整体覆盖。
        """
        self.proposals[proposal.proposal_id] = proposal
        return proposal

    def update_confidence(self, proposal_id: str, score: float, level: str) -> None:
        proposal = self.proposals[proposal_id]
        proposal.confidence_score = score
        proposal.confidence = level

    def list_by_status(self, status: str, repo: str | None = None, limit: int = 10) -> list[KnowledgeProposal]:
        out = []
        for p in self.proposals.values():
            if p.status != status:
                continue
            if repo and p.repo != repo:
                continue
            out.append(p)
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out[:limit]

    def as_payload(self, proposal: KnowledgeProposal) -> dict[str, Any]:
        return asdict(proposal)


class InMemoryRawArtifactStore:
    def __init__(self) -> None:
        self.artifacts: dict[str, dict[str, Any]] = {}

    def save_trace(self, repo: str, trace_id: str, raw_mcp: dict[str, Any]) -> str:
        key = f"{repo}/traces/raw/{trace_id}.json"
        self.artifacts[key] = dict(raw_mcp)
        return key


class FileRawArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save_trace(self, repo: str, trace_id: str, raw_mcp: dict[str, Any]) -> str:
        path = self.root / repo / "traces" / "raw" / f"{trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw_mcp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(path)
