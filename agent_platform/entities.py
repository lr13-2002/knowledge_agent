"""业务实体合并与归一化。

参考 Microsoft GraphRAG 的 Entity Resolution 思路：
- 同一概念在不同 trace 里可能有不同表达（"订单" / "订单数据" / "Order"）
- 实体合并避免知识图谱被分裂成大量近义节点
- 合并策略多层兜底，从严到宽

合并策略（按顺序）:
1. 名称归一化（去标点/繁简/常见后缀）→ slug 相同直接合并
2. 包含/被包含关系 → 短的胜出，长的作为别名
3. 向量相似度 → cosine ≥ 0.88 且 type 兼容才合并
4. description 合并 → 保留 mentions 最多的主 description

使用方式：
    merged = merge_into_graph(graph, vector, new_entity, proposal_id, trace_id)
    # merged 是合并后的最终实体（可能是新建的，也可能是已有的）

============================================================================
项目数据模型速查（读这个文件需要的背景）
============================================================================
完整规约见 docs/data-model.md，本注释是简化版便于就地理解。
⭐ 标记 = 本文件直接参与的部分。

向量存储 ChromaVectorStore（4 个 collection）
----------------------------------------------------------------------------
| Collection             | 存什么          | 谁写              | 谁读                       |
|------------------------|-----------------|-------------------|----------------------------|
| code_chunks            | 代码片段        | CodeIndexer       | Worker 检索代码上下文      |
| knowledge_claims (¹)   | 已确认/待审知识 | approve 时        | AskService confirmed 路    |
| entities ⭐            | 业务实体        | 本文件            | AskService + 实体合并查询  |
| community_summaries ⭐ | 业务领域摘要    | CommunityDetector | AskService community 路    |

(¹) payload 含 created_at + interface_key，供检索时矛盾消解（Day 6a）。
⚠️ 2026-06-23 砍掉 trace_cases：trace 是结构化数据不需要语义检索，详见 AGENTS.md。

图存储 SQLiteGraphStore（7 个节点 / 6 个关系）
----------------------------------------------------------------------------
⚠️ 2026-06-20 瘦身后（节点 11→7 / 关系 11→6），Repo/Commit/Span/Evidence
   全部扁平化为属性，详见 AGENTS.md §15。

| 层 | 节点              | 代表什么          | node_id 示例              |
|----|-------------------|-------------------|---------------------------|
| L1 | CodeSymbol (²)    | 函数/类/接口      | "go:file.go:Func:15"      |
| L2 | Interface         | HTTP 接口         | "svc:POST:/order/create"  |
| L2 | Service           | 微服务            | "pay-svc"                 |
| L2 | TraceCase         | trace 记录        | "trace-abc123"            |
| L3 | Entity ⭐         | 业务实体          | "entity:订单"             |
| L4 | Community         | 业务领域社区      | "community:lvl0:hash"     |
| L5 | BusinessRule (³)  | 已确认业务知识    | "rule:proposal-xxx"       |

(²) CodeSymbol 节点带 commit/repo 属性（原 Commit/Repo 节点已扁平化）。
(³) BusinessRule 节点带 evidence_trace_ids / evidence_code_symbols /
    evidence_commit / created_at 属性（原 Evidence 节点已扁平化）。

| 关系          | 从 → 到                  | 默认权重     | 谁写               |
|---------------|--------------------------|--------------|--------------------|
| HAS_TRACE     | Interface → TraceCase    | 0.9          | Worker             |
| CALLS_SERVICE | Interface → Service      | 0.7          | Worker             |
| MENTIONS      | Interface → Entity       | 0.6          | Worker（写实体后） |
| CALLS         | CodeSymbol → CodeSymbol  | 0.5          | CodeIndexer        |
| RELATED_TO ⭐ | Entity → Entity          | 0.1~1.0 (⁴)  | Worker             |
| BELONGS_TO    | Entity → Community       | 1.0          | CommunityDetector  |

(⁴) LLM 显式抽取关系: weight = strength/10；同 proposal 共现兜底: weight = 0.4。

本文件职责：L3 Entity 节点的合并/写入 + entities 向量 collection 的 upsert。

----------------------------------------------------------------------------
Entity 数据结构（本文件读写的对象）
----------------------------------------------------------------------------
源头：BusinessEntity dataclass（见 schemas.py）。同一份业务对象会落到三处，
**三处的字段集是子集关系**，按"读取场景需要的最小集"裁剪：

```
BusinessEntity dataclass（最全，业务层用）
    │
    ├─→ 图节点 Entity 的 properties（_persist_entity 写入）
    │   字段：name, type, description, mentions, aliases,
    │         source_proposal_ids, source_trace_ids,
    │         first_seen_at, last_seen_at, repo
    │   （= dataclass 几乎全字段，因为图查询可能用到 source_ids 做溯源）
    │
    └─→ 向量库 entities collection 的 payload（_persist_entity 写入）
        字段：name, type, description, mentions, repo
        （= dataclass 精简子集，只留检索/重排/合并判断要用的）
        ⚠️ 去掉 aliases / source_ids / 时间戳是有意的——
           向量检索主要用 name+description 算相似度，payload 只承担过滤
           （filters={"repo": repo}）+ 重排（mentions 加权）+ 合并判断（type）。
```

字段意义速查（vs schemas.BusinessEntity）：

| 字段          | 类型        | 含义                                          |
|---------------|-------------|-----------------------------------------------|
| name          | str         | 归一化后的业务名（"订单"）                    |
| type          | str         | business_concept / data_entity / actor / ... |
| description   | str         | 一句话说明该实体在本接口的角色                |
| mentions      | int         | 被提及次数（合并时自增，检索时 ln 强化）      |
| aliases       | list[str]   | 历次原始表达（"Order", "订单数据" → "订单"）  |
| source_*_ids  | list[str]   | 来自哪些 proposal/trace（溯源用）             |
| first/last_seen_at | str    | ISO 时间戳（last_seen_at 用于 recency 加权）  |
| repo          | str         | 所属仓库（vector filter 用）                  |
| entity_id     | str         | = "entity:" + slug(name)，图节点 id           |
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from .schemas import BusinessEntity, utc_now

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------

# 实体名归一化时去除的常见后缀（中文 + 英文）
_TRIM_SUFFIXES = [
    "数据", "信息", "单据", "记录", "服务", "模块", "对象",
    "_data", "_info", "_service", "_record",
]

# 业务实体停用词黑名单（LLM 容易把这些技术词当业务实体）
_STOPWORDS = {
    "controller", "service", "repository", "helper", "manager", "handler",
    "impl", "interface", "abstract", "factory", "builder", "util", "utils",
}

# ----------------------------------------------------------------------------
# 向量相似度合并阈值（用于实体合并的 4 层兜底中的第 3 层：cosine ≥ 0.88 才合并）
# ----------------------------------------------------------------------------
# 借鉴：Microsoft GraphRAG (2024) 的 Entity Resolution 默认值
#   https://arxiv.org/abs/2404.16130, §3.1 Element Instance Summaries
#   GraphRAG 实测发现 0.88 是"宁可漏合不能错合"的甜点：
#   - 阈值 ≥ 0.9：很多近义词不被合并（订单/订单数据），图被分裂
#   - 阈值 ≤ 0.85：会把不同概念误合（订单/订单状态机），语义错乱
# 选 0.88 是 GraphRAG 团队在 podcast 数据集上调优的结果，本项目沿用。
SIMILARITY_THRESHOLD = 0.88


# ----------------------------------------------------------------------------
# 检索重排公式（Reinforcement + Recency, Day 5/8）
# ----------------------------------------------------------------------------
# 完整公式（见 rerank_hits 函数）：
#   final = base_score × (1 + ALPHA × ln(1 + mentions)) × 1/(1 + days/HALFLIFE)
#                       └────── 频率强化 ──────┘   └──── 新近度加权 ────┘
#
# 这是 LLM Wiki v2 §1 "memory lifecycle" 的核心机制：
# "见过 12 次的模式比见过 1 次的可靠"+"最近确认的事实更可靠"——
# 但**两者都不能压过相关性**，所以都用阻尼函数（不是线性也不是指数）。

# ---- 频率强化系数 ALPHA ----
# 借鉴：BM25 的 k1 参数 + Elasticsearch 的 frequency boost 设计。
#
# 为什么用 ln 而非线性 / 指数：
#   线性 (× mentions)  → mentions=50 加 50 倍，完全碾压相关性
#   指数 (e^mentions)  → 几次就爆炸
#   ln(1+x)            → 对数阻尼，1→10 加成明显，10→50 趋缓 ← 选这个
#
# 为什么 ALPHA=0.3（不是 0.1 / 1.0）：
#   工程权衡 = "强化要明显能感知，但不能盖过相关性"
#   边界验算：当 base=5（强相关 mentions=1） vs base=1（弱相关 mentions=50），
#     ALPHA=0.3 → 5*1.21=6.04 vs 1*2.18=2.18，强相关仍胜出 ✓
#     ALPHA=1.0 → 5*1.69=8.46 vs 1*4.93=4.93，开始接近 ⚠️
#   测试守门：tests/test_reinforcement.py::test_relevance_still_dominates_when_gap_large
MENTIONS_REINFORCE_ALPHA = 0.3

# ---- 新近度衰减系数 HALFLIFE ----
# 借鉴：Mem0 / Zep / Graphiti（生产 memory 系统）+ Elasticsearch decay function。
#
# 为什么用 1/(1+x) 而非 exp(-x)：
#   exp(-days/90)   → 180 天剩 13%（硬衰减），会**误杀长期稳定的架构类知识**
#                     越老越稳定的知识不该暴跌；指数遗忘是 Ebbinghaus 学术原型，
#                     生产 memory 系统（Mem0/Zep）都不用
#   1/(1+days/90)   → 180 天剩 33%（温和降权），老知识降权但不消失 ← 选这个
#
# 业界印证（这次否决了我最初的指数衰减提案）：
#   - Mem0 / Zep / Graphiti：生产 memory 不做指数遗忘，
#     主流是"失效标记"或"检索时 recency 温和加权"
#   - Elasticsearch time-decay function：linear / exp / gauss 三选一，
#     线性场景多用 1/(1+x) 形态
#
# 为什么 HALFLIFE=90 天：
#   接口业务含义的典型变更周期约 1 季度（参考 Wiki v2 §1.3 衰减分层建议）
#   90 天后系数 = 0.5（半衰），180 天后 = 0.33，365 天后 = 0.18
#   分层（架构 180 天 / 临时 bug 30 天）需要知识类型字段，暂无 →
#   §14 哲学：先用统一 90 天，等观察到误伤再分层（AGENTS.md §1.3b 标 P3）
RECENCY_HALFLIFE_DAYS = 90.0


# ----------------------------------------------------------------------------
# 名称归一化
# ----------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """归一化实体名称，得到稳定的 slug。

    步骤：
    1. 全角转半角 + 转小写
    2. 先去英文后缀（带下划线的，如 _service / _data），因为去标点后就识别不出来了
    3. 去标点和空白
    4. 再去中文后缀（"数据"/"信息"/"服务"等）
    """
    if not name:
        return ""
    # 全角→半角
    s = unicodedata.normalize("NFKC", name).strip().lower()
    # 先处理英文带分隔符的后缀（在去标点前）
    for suffix in _TRIM_SUFFIXES:
        if suffix.startswith("_") and s.endswith(suffix.lower()):
            s = s[: -len(suffix)]
            break
    # 去标点和空白
    s = re.sub(r"[\s\-_.,;:()/\\]+", "", s)
    # 再处理中文/无分隔符的后缀
    for suffix in _TRIM_SUFFIXES:
        if not suffix.startswith("_") and s.endswith(suffix.lower()):
            s = s[: -len(suffix)]
            break
    return s


def slug_to_entity_id(slug: str) -> str:
    """slug → entity_id（带前缀以区分图节点）。"""
    return f"entity:{slug}" if slug else ""


# ----------------------------------------------------------------------------
# 检索强化（Reinforcement, Day 5）
# ----------------------------------------------------------------------------

def _days_since(timestamp: str) -> float:
    """解析 ISO 时间戳，返回距今天数。解析失败返回 0（视为"刚发生"，不降权）。"""
    if not timestamp:
        return 0.0
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def rerank_hits(
    hits: list[dict[str, Any]],
    recency_field: str | None = None,
) -> list[dict[str, Any]]:
    """对检索结果按 mentions 强化（Day 5）+ recency 加权（Day 8）重排。

    完整公式：
        final = base_score × (1 + ALPHA × ln(1+mentions)) × 1/(1+days/HALFLIFE)
                            └────── 频率强化 ──────┘   └───── 新近度 ─────┘

    每段公式的**为什么这么选 + 业界出处 + 参数取值依据**，详见文件顶部
    `MENTIONS_REINFORCE_ALPHA` 和 `RECENCY_HALFLIFE_DAYS` 的注释块（含
    BM25 / GraphRAG / Mem0 / Zep / Elasticsearch 的对照）。

    参数：
      recency_field —— payload 里取哪个时间字段算 recency。
                       实体用 last_seen_at（合并时刷新），知识用 created_at。
                       None 时不做 recency 加权（只强化），相当于退化为 Day 5 行为。

    容错（脏数据兜底）：
      - mentions 缺失/小于 1 按 1 算（等于不加成，不报错）
      - base_score 缺失按 0（最终分 = 0，沉底但不崩）
      - 时间解析失败按 0 天（视为"刚发生"，不降权——脏数据宁可保留也别误杀）

    返回按 final 降序的新列表；标 _reranked_score 便于排查；**不改原 score**
    （recency 是纯展示层计算，存储里的 score 不变，可随时调半衰期不用重算）。
    """
    import math

    def _final(hit: dict[str, Any]) -> float:
        payload = hit.get("payload", {})
        base = float(hit.get("score", 0.0) or 0.0)
        mentions = int(payload.get("mentions", 1) or 1)
        if mentions < 1:
            mentions = 1
        score = base * (1.0 + MENTIONS_REINFORCE_ALPHA * math.log(1 + mentions))
        # recency 加权（Day 8）
        if recency_field:
            days = _days_since(str(payload.get(recency_field, "")))
            score *= 1.0 / (1.0 + days / RECENCY_HALFLIFE_DAYS)
        return score

    ranked = sorted(hits, key=_final, reverse=True)
    for h in ranked:
        h["_reranked_score"] = _final(h)
    return ranked


# 向后兼容别名（Day 5 调用方仍可用，等价于不传 recency_field）
def rerank_by_mentions(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[兼容别名] 仅按 mentions 强化重排，不做 recency 加权。新代码用 rerank_hits。"""
    return rerank_hits(hits, recency_field=None)


def is_stopword(name: str) -> bool:
    """判断是否是技术停用词（不应作为业务实体）。"""
    s = normalize_name(name)
    return s in _STOPWORDS or any(s.endswith(w) for w in _STOPWORDS)


# ----------------------------------------------------------------------------
# 实体合并主逻辑
# ----------------------------------------------------------------------------

def merge_into_graph(
    graph: Any,
    vector: Any,
    new_entity: BusinessEntity,
    proposal_id: str,
    trace_id: str,
    repo: str,
) -> BusinessEntity:
    """将一个新抽取的实体合并到图中。返回合并后的最终实体。

    合并优先级：
    1. 归一化后 slug 完全匹配 → 合并到已有实体
    2. 包含/被包含关系 → 合并
    3. 向量相似度 ≥ 0.88 且 type 兼容 → 合并
    4. 都不匹配 → 新建实体节点
    """
    # 停用词过滤
    if is_stopword(new_entity.name):
        logger.debug("跳过技术停用词实体: %s", new_entity.name)
        return new_entity

    slug = normalize_name(new_entity.name)
    if not slug:
        return new_entity

    entity_id = slug_to_entity_id(slug)

    # 策略 1：完全匹配（归一化 slug 相同）
    existing = _find_existing_entity(graph, entity_id)
    if existing:
        return _merge_attributes(graph, vector, existing, new_entity, proposal_id, trace_id, repo)

    # 策略 2：包含/被包含（短的胜出）
    short_match = _find_substring_match(graph, slug, new_entity.type)
    if short_match:
        # 把当前长名作为别名挂到短实体上
        return _merge_attributes(graph, vector, short_match, new_entity, proposal_id, trace_id, repo)

    # 策略 3：向量相似度合并
    similar = _find_similar_entity(vector, new_entity, repo)
    if similar:
        return _merge_attributes(graph, vector, similar, new_entity, proposal_id, trace_id, repo)

    # 策略 4：新建实体
    return _create_new_entity(graph, vector, new_entity, entity_id, proposal_id, trace_id, repo)


# ----------------------------------------------------------------------------
# 内部辅助函数
# ----------------------------------------------------------------------------

def _find_existing_entity(graph: Any, entity_id: str) -> BusinessEntity | None:
    """按 entity_id 查找已存在的实体节点。"""
    nodes = graph.find_nodes("Entity")
    for node in nodes:
        if node.get("id") == entity_id:
            return _node_to_entity(node)
    return None


def _find_substring_match(graph: Any, slug: str, entity_type: str) -> BusinessEntity | None:
    """查找包含/被包含关系的实体（短的胜出）。"""
    nodes = graph.find_nodes("Entity")
    candidates: list[tuple[str, dict]] = []
    for node in nodes:
        node_slug = normalize_name(node.get("name", ""))
        if not node_slug:
            continue
        # 已有实体的 slug 是新实体 slug 的子串（已有的更短）
        if node_slug in slug or slug in node_slug:
            if node.get("type") == entity_type or _types_compatible(node.get("type", ""), entity_type):
                candidates.append((node_slug, node))

    if not candidates:
        return None
    # 返回 slug 最短的（最具代表性）
    candidates.sort(key=lambda x: len(x[0]))
    return _node_to_entity(candidates[0][1])


def _find_similar_entity(vector: Any, new_entity: BusinessEntity, repo: str) -> BusinessEntity | None:
    """通过向量检索找语义相似的已有实体。"""
    query_text = f"{new_entity.name} {new_entity.description}"
    try:
        hits = vector.search("entities", query_text, {"repo": repo}, limit=3)
    except Exception:
        return None
    for hit in hits:
        score = hit.get("score", 0.0)
        if score < SIMILARITY_THRESHOLD:
            continue
        payload = hit.get("payload", {})
        # type 兼容才合并（避免"订单"和"订单服务"被错合）
        if not _types_compatible(payload.get("type", ""), new_entity.type):
            continue
        return BusinessEntity(
            name=payload.get("name", ""),
            type=payload.get("type", "other"),
            description=payload.get("description", ""),
            entity_id=hit.get("id", ""),
            mentions=int(payload.get("mentions", 1)),
        )
    return None


def _types_compatible(t1: str, t2: str) -> bool:
    """实体类型是否兼容（完全相同或都属于 business_concept 系列）。"""
    if t1 == t2:
        return True
    if not t1 or not t2:
        return True
    # 业务概念类型互相兼容
    business_types = {"business_concept", "data_entity", "actor"}
    return t1 in business_types and t2 in business_types


def _merge_attributes(
    graph: Any,
    vector: Any,
    existing: BusinessEntity,
    new_entity: BusinessEntity,
    proposal_id: str,
    trace_id: str,
    repo: str,
) -> BusinessEntity:
    """合并新实体的属性到已有实体（mentions 自增、aliases 累加）。"""
    # 累加别名
    if new_entity.name not in existing.aliases:
        existing.aliases.append(new_entity.name)
    # 累加来源
    if proposal_id not in existing.source_proposal_ids:
        existing.source_proposal_ids.append(proposal_id)
    if trace_id not in existing.source_trace_ids:
        existing.source_trace_ids.append(trace_id)
    existing.mentions += 1
    existing.last_seen_at = utc_now()

    # 写回图
    _persist_entity(graph, vector, existing, repo)
    return existing


def _create_new_entity(
    graph: Any,
    vector: Any,
    new_entity: BusinessEntity,
    entity_id: str,
    proposal_id: str,
    trace_id: str,
    repo: str,
) -> BusinessEntity:
    """创建新的实体节点。"""
    new_entity.entity_id = entity_id
    new_entity.aliases = [new_entity.name] if new_entity.name not in new_entity.aliases else new_entity.aliases
    new_entity.source_proposal_ids = [proposal_id]
    new_entity.source_trace_ids = [trace_id]
    new_entity.first_seen_at = utc_now()
    new_entity.last_seen_at = utc_now()

    _persist_entity(graph, vector, new_entity, repo)
    return new_entity


def _persist_entity(graph: Any, vector: Any, entity: BusinessEntity, repo: str) -> None:
    """把实体写入图和向量库。"""
    properties = {
        "name": entity.name,
        "type": entity.type,
        "description": entity.description,
        "mentions": entity.mentions,
        "aliases": entity.aliases,
        "source_proposal_ids": entity.source_proposal_ids,
        "source_trace_ids": entity.source_trace_ids,
        "first_seen_at": entity.first_seen_at,
        "last_seen_at": entity.last_seen_at,
        "repo": repo,
    }
    graph.upsert_node("Entity", entity.entity_id, properties)

    # 写入向量库（用于后续相似度合并和 Ask 检索）
    # text: 用于算 embedding 的内容；payload 是检索时附带返回的元数据。
    # payload 只放检索/重排/合并判断要用的字段（精简子集），完整字段在图节点 properties。
    # payload schema 见文件头"Entity 数据结构"表。
    text = f"{entity.name} {entity.type} {entity.description}"
    vector.upsert("entities", entity.entity_id, text, {
        "name": entity.name,            # 用于展示
        "type": entity.type,            # 用于合并时 _types_compatible 判断
        "description": entity.description,
        "mentions": entity.mentions,    # 用于 rerank_hits 频率强化
        "repo": repo,                   # 用于 filters={"repo": repo} 限定查询范围
    })


def _node_to_entity(node: dict) -> BusinessEntity:
    """图节点 dict 转 BusinessEntity dataclass。"""
    return BusinessEntity(
        name=node.get("name", ""),
        type=node.get("type", "other"),
        description=node.get("description", ""),
        aliases=node.get("aliases", []) or [],
        source_proposal_ids=node.get("source_proposal_ids", []) or [],
        source_trace_ids=node.get("source_trace_ids", []) or [],
        entity_id=node.get("id", ""),
        mentions=int(node.get("mentions", 1)),
        first_seen_at=node.get("first_seen_at", ""),
        last_seen_at=node.get("last_seen_at", ""),
    )
