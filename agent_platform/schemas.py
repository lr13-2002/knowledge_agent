"""数据模型定义。

所有核心数据结构都定义在这里，使用 Python dataclass：
- OpenClawEvent — 外部推送的 trace 事件
- TraceSpan / NormalizedTrace — 标准化后的 trace 数据
- KnowledgeProposal — LLM 产出的知识提案
- Evidence — 支撑提案的证据（trace_ids, code_symbols）— 仅作为 Proposal/AskResponse
                内嵌字段使用，**不再作为独立图节点**（已扁平化为 BusinessRule 的属性）
- ReviewMessage — 审核对话消息
- AskResponse — 知识库问答返回
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OpenClawEvent:
    repo: str
    trace_id: str
    service: str
    method: str
    path: str
    event_id: str | None = None
    timestamp: str | None = None
    env: str = "prod"
    status: str = "success"
    latency_ms: int = 0
    raw_event: dict[str, Any] = field(default_factory=dict)
    manual: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenClawEvent":
        missing = [key for key in ("repo", "trace_id", "service", "method", "path") if not data.get(key)]
        if missing:
            raise ValueError("missing required fields: " + ", ".join(missing))
        return cls(
            repo=str(data["repo"]),
            trace_id=str(data["trace_id"]),
            service=str(data["service"]),
            method=str(data["method"]).upper(),
            path=str(data["path"]),
            event_id=data.get("event_id"),
            timestamp=data.get("timestamp"),
            env=str(data.get("env", "prod")),
            status=str(data.get("status", "success")),
            latency_ms=int(data.get("latency_ms", 0) or 0),
            raw_event=dict(data.get("raw_event", {})),
            manual=bool(data.get("manual", False)),
        )

    @property
    def interface_key(self) -> str:
        return f"{self.repo}:{self.service}:{self.method}:{self.path}"

    @property
    def idempotency_key(self) -> str:
        return f"{self.repo}:{self.trace_id}"


@dataclass
class TraceSpan:
    service: str = ""
    name: str = ""
    method: str = ""
    path: str = ""
    duration: str = ""
    has_error: bool = False


@dataclass
class InternalStep:
    """主服务内部执行锚点（AGENTS.md §9.6/§9.7）。

    来自 span 日志的 [类名/文件:行号],经 trace_parsers 解析 + join CodeSymbol。
    它还原"主服务内部走了哪段代码",是 trace 拓扑（只有跨服务）还原不了的现场。
    """
    file: str = ""          # 文件名/相对路径
    line: int = 0           # 日志打印行号（函数体内,非声明行）
    func: str = ""          # 函数名（Go 有,Java 无）
    dltag: str = ""         # 日志分类标记,如 _undef / _com_http_success
    symbol_id: str = ""     # join 命中的 CodeSymbol id（join 不上则空）
    symbol_name: str = ""   # join 命中的符号名
    verified: bool = False  # func 名与命中符号一致 → 高置信


@dataclass
class ExternalCall:
    """主服务发起的跨服务调用（AGENTS.md §9.6）。

    来自 span 的 downstream._com_http_success 等结构化字段。对应 OTel 的 CLIENT span。
    只记"调了谁",不深挖下游内部（下游若被索引,会作为独立 trace 自己处理）。
    """
    url: str = ""           # 下游 url / 接口路径
    errno: str = ""         # 返回码
    proc_time: str = ""     # 耗时


@dataclass
class NormalizedTrace:
    repo: str
    trace_id: str
    interface_key: str
    spans: list[TraceSpan]
    raw_mcp: dict[str, Any]
    upstream: list[str]
    downstream: list[str]
    errors: list[str]
    # §9.6 新增：精细现场（默认空,向后兼容；由 normalize_trace 解析 logs/downstream 填充）
    internal_path: list[InternalStep] = field(default_factory=list)
    external_calls: list[ExternalCall] = field(default_factory=list)


@dataclass
class Evidence:
    trace_ids: list[str] = field(default_factory=list)
    code_symbols: list[str] = field(default_factory=list)
    commit: str = ""
    business_rule_ids: list[str] = field(default_factory=list)


@dataclass
class BusinessEntity:
    """业务实体 — LLM 从 trace + 代码中抽取的业务概念。

    例如：订单、支付、风控、优惠券、用户、对账单等业务方能听懂的词。
    技术词（Controller/Service/Repository）不应该作为实体。
    """
    name: str  # 归一化后的中文名，如"订单"
    type: str  # 自由文本：business_concept/external_service/data_entity/actor/event/capability/other
    description: str  # 1-2 句说明该实体在本接口的角色
    aliases: list[str] = field(default_factory=list)  # 被提到的所有原始字符串
    source_proposal_ids: list[str] = field(default_factory=list)
    source_trace_ids: list[str] = field(default_factory=list)
    entity_id: str = ""  # = "entity:" + slug(name)
    mentions: int = 1  # 被提及次数
    first_seen_at: str = field(default_factory=utc_now)
    last_seen_at: str = field(default_factory=utc_now)


@dataclass
class BusinessRelation:
    """业务实体之间的关系（LLM 从 trace 中识别）。"""
    source: str  # 源实体名（或 entity_id）
    target: str  # 目标实体名（或 entity_id）
    relation: str  # 自由文本：调用/触发/包含/写入/依赖/通知/校验
    description: str  # 关系说明
    strength: int = 5  # 1-10，10=核心因果，1=偶发
    source_proposal_ids: list[str] = field(default_factory=list)


@dataclass
class CommunityReport:
    """社区报告 — Leiden 算法聚类后，由 LLM 为每个领域生成的摘要。"""
    community_id: str  # community:lvl{n}:{hash}
    level: int = 0  # Leiden 层级，0=最细
    title: str = ""  # 4-10 字领域名
    summary: str = ""  # 段落级摘要 ≤300 字
    findings: list[dict] = field(default_factory=list)  # [{insight, explanation}]
    member_entity_ids: list[str] = field(default_factory=list)
    rank: float = 0.0  # = sum(member.mentions)，用于 Ask 排序
    created_at: str = field(default_factory=utc_now)


@dataclass
class KnowledgeProposal:
    repo: str
    trace_id: str
    interface_key: str
    summary: str
    flow_steps: list[str]
    related_code_symbols: list[str]
    candidate_claims: list[str]
    evidence: Evidence
    confidence: Literal["low", "medium", "high"] = "low"
    confidence_score: float = 0.3  # 0~1 连续值，用于置信度传播计算
    status: str = "pending_review"
    proposal_id: str = field(default_factory=lambda: "proposal-" + uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    version: int = 1
    # 业务实体抽取（Phase 2 新增）
    entities: list[BusinessEntity] = field(default_factory=list)
    relations: list[BusinessRelation] = field(default_factory=list)
    reasoning: str = ""  # LLM 给出的置信度推理过程
    # Reject 反馈环（Day 3 新增）：
    # reject 时若有 LLM，会基于"原内容 + 历次 reject 理由"反思重生成，原地更新本提案。
    retry_count: int = 0  # 已被反馈重生成的次数（上限 3，超过则 reject 终态）
    reject_history: list[str] = field(default_factory=list)  # 历次 reject 理由，喂给 LLM 反思


@dataclass
class ReviewMessage:
    role: Literal["user", "assistant"]
    content: str
    evidence: Evidence = field(default_factory=Evidence)
    created_at: str = field(default_factory=utc_now)


@dataclass
class AskResponse:
    answer: str
    evidence: Evidence
    sections: dict[str, list[str]]
