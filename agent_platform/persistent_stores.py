"""持久化存储实现 — Chroma（向量）+ SQLite（图 + 提案）。

替代 stores.py 中的 InMemory* 实现，数据持久化到磁盘：
- ChromaVectorStore — 基于 Chroma 的语义向量检索，支持 embedding
- SQLiteGraphStore — 基于 SQLite 的图存储（节点 + 边 + 权重）
- SQLiteProposalStore — 基于 SQLite 的知识提案存储

所有数据存在 data/ 目录下（默认），重启不丢失。

数据文件结构：
    data/
      chroma/               ← Chroma 向量库数据
      agent_platform.db     ← SQLite 数据库（图 + 提案）
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schemas import KnowledgeProposal, ReviewMessage


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _sqlite_conn(db_path: str | Path) -> sqlite3.Connection:
    """创建 SQLite 连接，开启 WAL 模式提升并发性能。"""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging，读写不互锁
    conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和安全
    conn.row_factory = sqlite3.Row  # 查询结果可以通过列名访问
    return conn


def init_persistent_stores(
    data_dir: str | Path = "data",
) -> tuple["ChromaVectorStore", "SQLiteGraphStore", "SQLiteProposalStore"]:
    """一次性创建所有持久化存储，共享同一个数据目录。

    返回: (向量存储, 图存储, 提案存储)
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    vector = ChromaVectorStore(str(root / "chroma"))
    conn = _sqlite_conn(root / "agent_platform.db")
    graph = SQLiteGraphStore(conn)
    proposals = SQLiteProposalStore(conn)
    return vector, graph, proposals


# ---------------------------------------------------------------------------
# ChromaVectorStore — 语义向量检索
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """基于 Chroma 的向量存储，支持 embedding 语义检索。

    Chroma 自带 embedding 模型（默认 all-MiniLM-L6-v2），
    upsert 时自动生成向量，search 时自动做语义匹配。

    数据按 collection 分类（懒加载，首次 upsert/search 时创建）：
    - code_chunks         — 代码片段（供 worker 检索关联代码）
    - knowledge_claims    — 知识条目（AskService 检索用）
    - entities            — 业务实体（Phase 2 新增）
    - community_summaries — 业务领域摘要（Phase 3 新增）

    ⚠️ 2026-06-23 砍掉 trace_cases：trace 是结构化数据，不需要语义检索。
       完整 trace 在 raw_artifacts (KV)，trace 关联在图层 TraceCase 节点。
    """

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        import chromadb
        # PersistentClient 自动持久化到磁盘
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict[str, Any] = {}  # collection 缓存

    def _get_collection(self, name: str) -> Any:
        """获取或创建 collection（带缓存）。"""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
        return self._collections[name]

    def upsert(self, collection: str, item_id: str, text: str, payload: dict[str, Any]) -> None:
        """写入或更新一条数据。text 会被自动 embedding。"""
        coll = self._get_collection(collection)
        coll.upsert(
            ids=[item_id],
            documents=[text or " "],  # Chroma 不接受空字符串
            metadatas=[self._flatten_metadata(payload)],
        )

    def search(
        self,
        collection: str,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """语义检索：用 query 的 embedding 在 collection 中找最相似的。

        返回列表中每项包含: id, text, payload, score（0~1，越大越相似）
        """
        coll = self._get_collection(collection)
        if coll.count() == 0:
            return []

        where = self._build_where(filters) if filters else None
        kwargs: dict[str, Any] = {
            "query_texts": [query or " "],
            "n_results": min(limit, coll.count()),
        }
        if where:
            kwargs["where"] = where

        try:
            results = coll.query(**kwargs)
        except Exception:
            return []

        # 解析 Chroma 返回的嵌套列表结构
        out: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, item_id in enumerate(ids):
            out.append({
                "id": item_id,
                "text": documents[i] if i < len(documents) else "",
                "payload": metadatas[i] if i < len(metadatas) else {},
                "score": 1.0 - (distances[i] if i < len(distances) else 1.0),  # 距离转相似度
            })
        return out

    @staticmethod
    def _flatten_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        """Chroma metadata 只支持 str/int/float/bool，复杂类型序列化为 JSON 字符串。"""
        flat: dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif v is None:
                continue
            else:
                flat[k] = json.dumps(v, ensure_ascii=False)
        return flat

    @staticmethod
    def _build_where(filters: dict[str, Any]) -> dict[str, Any] | None:
        """构建 Chroma 的 where 过滤条件。"""
        conditions = []
        for k, v in filters.items():
            if isinstance(v, (str, int, float, bool)):
                conditions.append({k: {"$eq": v}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# SQLiteGraphStore — 图存储（节点 + 带权重的边）
# ---------------------------------------------------------------------------

class SQLiteGraphStore:
    """基于 SQLite 的图存储。

    存储知识之间的关联关系（精简版）：
    - 节点: Interface, Service, TraceCase, CodeSymbol, Entity, Community, BusinessRule
            (commit/repo 作为节点属性写入，不再单独建 Repo/Commit/Span/Evidence 节点)
    - 边: CALLS_SERVICE, HAS_TRACE, CALLS, MENTIONS, RELATED_TO, BELONGS_TO
    - 边权重: 用于置信度传播（HAS_TRACE=0.9, CALLS_SERVICE=0.7,
            MENTIONS=0.6, CALLS=0.5, RELATED_TO=0.1~1.0, BELONGS_TO=1.0）
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.executescript(
            # 节点表：label + node_id 唯一标识，properties 存 JSON
            "CREATE TABLE IF NOT EXISTS graph_nodes ("
            "  label      TEXT NOT NULL,"
            "  node_id    TEXT NOT NULL,"
            "  properties TEXT NOT NULL DEFAULT '{}',"
            "  PRIMARY KEY (label, node_id)"
            ");"
            # 边表：五元组 + 权重
            "CREATE TABLE IF NOT EXISTS graph_edges ("
            "  from_label TEXT NOT NULL,"
            "  from_id    TEXT NOT NULL,"
            "  relation   TEXT NOT NULL,"
            "  to_label   TEXT NOT NULL,"
            "  to_id      TEXT NOT NULL,"
            "  weight     REAL NOT NULL DEFAULT 1.0,"
            "  PRIMARY KEY (from_label, from_id, relation, to_label, to_id)"
            ");"
        )
        self._conn.commit()

    def upsert_node(self, label: str, node_id: str, properties: dict[str, Any]) -> None:
        """创建或更新节点。已有的 properties 会被 merge（不是覆盖）。"""
        existing = self._conn.execute(
            "SELECT properties FROM graph_nodes WHERE label = ? AND node_id = ?",
            (label, node_id),
        ).fetchone()
        if existing:
            merged = json.loads(existing["properties"])
            merged.update(properties)
        else:
            merged = dict(properties)
        merged["id"] = node_id
        merged["label"] = label
        self._conn.execute(
            "INSERT INTO graph_nodes (label, node_id, properties) VALUES (?, ?, ?)"
            " ON CONFLICT(label, node_id) DO UPDATE SET properties=excluded.properties",
            (label, node_id, json.dumps(merged, ensure_ascii=False)),
        )
        self._conn.commit()

    def add_edge(self, from_label: str, from_id: str, relation: str, to_label: str, to_id: str, weight: float = 1.0) -> None:
        """创建或更新边。重复边会更新权重。"""
        self._conn.execute(
            "INSERT INTO graph_edges (from_label, from_id, relation, to_label, to_id, weight)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(from_label, from_id, relation, to_label, to_id) DO UPDATE SET weight=excluded.weight",
            (from_label, from_id, relation, to_label, to_id, weight),
        )
        self._conn.commit()

    def find_nodes(self, label: str, **properties: Any) -> list[dict[str, Any]]:
        """按 label 和 properties 查找节点。"""
        rows = self._conn.execute(
            "SELECT properties FROM graph_nodes WHERE label = ?", (label,)
        ).fetchall()
        out = []
        for row in rows:
            node = json.loads(row["properties"])
            if all(node.get(k) == v for k, v in properties.items()):
                out.append(node)
        return out

    def neighbors(self, from_label: str, from_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        """正向查询：该节点指向了谁。返回目标节点列表，附带 _relation 和 _weight。"""
        if relation:
            edges = self._conn.execute(
                "SELECT relation, to_label, to_id, weight FROM graph_edges"
                " WHERE from_label = ? AND from_id = ? AND relation = ?",
                (from_label, from_id, relation),
            ).fetchall()
        else:
            edges = self._conn.execute(
                "SELECT relation, to_label, to_id, weight FROM graph_edges"
                " WHERE from_label = ? AND from_id = ?",
                (from_label, from_id),
            ).fetchall()
        out = []
        for edge in edges:
            node_row = self._conn.execute(
                "SELECT properties FROM graph_nodes WHERE label = ? AND node_id = ?",
                (edge["to_label"], edge["to_id"]),
            ).fetchone()
            if node_row:
                node = json.loads(node_row["properties"])
                node["_relation"] = edge["relation"]
                node["_weight"] = edge["weight"]
                out.append(node)
        return out

    def all_nodes(self, label: str | None = None) -> list[dict[str, Any]]:
        """遍历所有节点（可按 label 过滤）。

        用于社区检测时构建完整图。注意：图很大时这是一次全表扫，
        但对当前规模（万级节点）够用。
        """
        if label:
            rows = self._conn.execute(
                "SELECT properties FROM graph_nodes WHERE label = ?", (label,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT properties FROM graph_nodes"
            ).fetchall()
        return [json.loads(row["properties"]) for row in rows]

    def all_edges(self, relations: list[str] | None = None) -> list[dict[str, Any]]:
        """遍历所有边（可按 relation 列表过滤）。

        用于社区检测时构建完整图。
        """
        if relations:
            placeholders = ",".join("?" * len(relations))
            rows = self._conn.execute(
                f"SELECT from_label, from_id, relation, to_label, to_id, weight"
                f" FROM graph_edges WHERE relation IN ({placeholders})",
                relations,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT from_label, from_id, relation, to_label, to_id, weight FROM graph_edges"
            ).fetchall()
        return [
            {
                "from_label": r["from_label"],
                "from_id": r["from_id"],
                "relation": r["relation"],
                "to_label": r["to_label"],
                "to_id": r["to_id"],
                "weight": r["weight"],
            }
            for r in rows
        ]

    def delete_nodes(self, label: str) -> None:
        """删除指定 label 的所有节点及相关边（社区重算时清旧数据用）。"""
        self._conn.execute("DELETE FROM graph_nodes WHERE label = ?", (label,))
        self._conn.execute(
            "DELETE FROM graph_edges WHERE from_label = ? OR to_label = ?",
            (label, label),
        )
        self._conn.commit()

    def reverse_neighbors(self, to_label: str, to_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        """反向查询：谁指向了该节点。用于置信度传播时找上游依赖。"""
        if relation:
            edges = self._conn.execute(
                "SELECT from_label, from_id, relation, weight FROM graph_edges"
                " WHERE to_label = ? AND to_id = ? AND relation = ?",
                (to_label, to_id, relation),
            ).fetchall()
        else:
            edges = self._conn.execute(
                "SELECT from_label, from_id, relation, weight FROM graph_edges"
                " WHERE to_label = ? AND to_id = ?",
                (to_label, to_id),
            ).fetchall()
        out = []
        for edge in edges:
            node_row = self._conn.execute(
                "SELECT properties FROM graph_nodes WHERE label = ? AND node_id = ?",
                (edge["from_label"], edge["from_id"]),
            ).fetchone()
            if node_row:
                node = json.loads(node_row["properties"])
                node["_relation"] = edge["relation"]
                node["_weight"] = edge["weight"]
                out.append(node)
        return out


# ---------------------------------------------------------------------------
# SQLiteProposalStore — 知识提案存储
# ---------------------------------------------------------------------------

class SQLiteProposalStore:
    """基于 SQLite 的知识提案存储。

    每条 proposal 序列化为 JSON 存在 data 列中。
    支持按 (repo, trace_id) 去重、按 status 查询、审核消息记录。
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.executescript(
            # 提案主表：proposal_id 主键，(repo, trace_id) 唯一
            "CREATE TABLE IF NOT EXISTS proposals ("
            "  proposal_id TEXT PRIMARY KEY,"
            "  repo        TEXT NOT NULL,"
            "  trace_id    TEXT NOT NULL,"
            "  data        TEXT NOT NULL,"
            "  UNIQUE(repo, trace_id)"
            ");"
            # 审核对话消息表：每条消息关联一个 proposal
            "CREATE TABLE IF NOT EXISTS review_messages ("
            "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  proposal_id TEXT NOT NULL,"
            "  data        TEXT NOT NULL,"
            "  FOREIGN KEY (proposal_id) REFERENCES proposals(proposal_id)"
            ");"
        )
        self._conn.commit()

    def _to_proposal(self, data: dict[str, Any]) -> KnowledgeProposal:
        """从 JSON dict 还原为 KnowledgeProposal dataclass。"""
        fields = {k: v for k, v in data.items() if k in KnowledgeProposal.__dataclass_fields__}
        if "evidence" in fields and isinstance(fields["evidence"], dict):
            from .schemas import Evidence
            fields["evidence"] = Evidence(**fields["evidence"])
        return KnowledgeProposal(**fields)

    def save(self, proposal: KnowledgeProposal) -> KnowledgeProposal:
        """保存提案。同一个 (repo, trace_id) 不重复保存，返回已有的。"""
        existing = self._conn.execute(
            "SELECT proposal_id, data FROM proposals WHERE repo = ? AND trace_id = ?",
            (proposal.repo, proposal.trace_id),
        ).fetchone()
        if existing:
            return self._to_proposal(json.loads(existing["data"]))
        self._conn.execute(
            "INSERT INTO proposals (proposal_id, repo, trace_id, data) VALUES (?, ?, ?, ?)",
            (proposal.proposal_id, proposal.repo, proposal.trace_id, json.dumps(asdict(proposal), ensure_ascii=False)),
        )
        self._conn.commit()
        return proposal

    def get(self, proposal_id: str) -> KnowledgeProposal:
        """按 ID 获取提案，不存在时抛 KeyError。"""
        row = self._conn.execute(
            "SELECT data FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            raise KeyError(proposal_id)
        return self._to_proposal(json.loads(row["data"]))

    def add_message(self, proposal_id: str, message: ReviewMessage) -> None:
        """记录审核对话消息。"""
        self._conn.execute(
            "INSERT INTO review_messages (proposal_id, data) VALUES (?, ?)",
            (proposal_id, json.dumps(asdict(message), ensure_ascii=False)),
        )
        self._conn.commit()

    def get_messages(self, proposal_id: str) -> list[ReviewMessage]:
        """读取某提案的审核对话消息（按插入顺序）。"""
        rows = self._conn.execute(
            "SELECT data FROM review_messages WHERE proposal_id = ? ORDER BY id ASC",
            (proposal_id,),
        ).fetchall()
        return [ReviewMessage(**json.loads(r["data"])) for r in rows]

    def update_status(self, proposal_id: str, status: str) -> KnowledgeProposal:
        """更新提案状态（pending_review / approved / rejected）。"""
        row = self._conn.execute(
            "SELECT data FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            raise KeyError(proposal_id)
        data = json.loads(row["data"])
        data["status"] = status
        self._conn.execute(
            "UPDATE proposals SET data = ? WHERE proposal_id = ?",
            (json.dumps(data, ensure_ascii=False), proposal_id),
        )
        self._conn.commit()
        return self._to_proposal(data)

    def update_confidence(self, proposal_id: str, score: float, level: str) -> None:
        """更新提案的置信度分数和等级（置信度传播时调用）。"""
        row = self._conn.execute(
            "SELECT data FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            raise KeyError(proposal_id)
        data = json.loads(row["data"])
        data["confidence_score"] = score
        data["confidence"] = level
        self._conn.execute(
            "UPDATE proposals SET data = ? WHERE proposal_id = ?",
            (json.dumps(data, ensure_ascii=False), proposal_id),
        )
        self._conn.commit()

    def revise(self, proposal_id: str, summary: str, claims: list[str]) -> KnowledgeProposal:
        """修订提案内容，版本号 +1，状态重置为 pending_review。"""
        row = self._conn.execute(
            "SELECT data FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if not row:
            raise KeyError(proposal_id)
        data = json.loads(row["data"])
        data["summary"] = summary
        data["candidate_claims"] = claims
        data["version"] = data.get("version", 1) + 1
        data["status"] = "pending_review"
        self._conn.execute(
            "UPDATE proposals SET data = ? WHERE proposal_id = ?",
            (json.dumps(data, ensure_ascii=False), proposal_id),
        )
        self._conn.commit()
        return self._to_proposal(data)

    def replace(self, proposal: KnowledgeProposal) -> KnowledgeProposal:
        """原地整体替换提案内容（保留 proposal_id）。

        用于 reject 反馈环：LLM 重生成后整体覆盖。不存在时抛 KeyError。
        """
        exists = self._conn.execute(
            "SELECT 1 FROM proposals WHERE proposal_id = ?", (proposal.proposal_id,)
        ).fetchone()
        if not exists:
            raise KeyError(proposal.proposal_id)
        self._conn.execute(
            "UPDATE proposals SET data = ? WHERE proposal_id = ?",
            (json.dumps(asdict(proposal), ensure_ascii=False), proposal.proposal_id),
        )
        self._conn.commit()
        return proposal

    def list_by_status(self, status: str, repo: str | None = None, limit: int = 10) -> list[KnowledgeProposal]:
        """按状态查询提案列表（支持按 repo 过滤）。"""
        if repo:
            rows = self._conn.execute(
                "SELECT data FROM proposals WHERE data LIKE ? AND repo = ? ORDER BY proposal_id DESC LIMIT ?",
                (f'%"status": "{status}"%', repo, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM proposals WHERE data LIKE ? ORDER BY proposal_id DESC LIMIT ?",
                (f'%"status": "{status}"%', limit),
            ).fetchall()
        return [self._to_proposal(json.loads(r["data"])) for r in rows]

    def as_payload(self, proposal: KnowledgeProposal) -> dict[str, Any]:
        """将提案转为字典（用于 API 返回）。"""
        return asdict(proposal)
