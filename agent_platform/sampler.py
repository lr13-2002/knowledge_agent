"""自适应采样器。

控制每个接口被分析的频率，避免高频接口浪费 LLM 资源。

采样策略（优先级从高到低）：
1. 频率限制 — 每分钟同一接口最多采样 N 次
2. 冷启动 — 新接口的前 N 条全部采样（确保覆盖）
3. 每日最低 — 保证每个接口每天至少有 N 条被分析
4. 加权采样 — 错误/慢请求/手动触发的权重更高
5. 概率采样 — 使用 SHA256 确保同一条 trace 的决策稳定（幂等）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import OpenClawEvent
from .stores import InMemorySamplerState


@dataclass(frozen=True)
class SamplingRule:
    """单个接口的采样规则。"""
    percent: int = 1  # 基础采样率（百分比）
    max_per_minute: int = 5  # 每分钟最多采样数
    min_per_day: int = 3  # 每天最少采样数


@dataclass
class SamplingConfig:
    """全局采样配置。"""
    default: SamplingRule = field(default_factory=SamplingRule)  # 默认规则
    # 加权倍数：error=5倍, slow=3倍, manual=100%(全量)
    boost: dict[str, int] = field(default_factory=lambda: {"error": 5, "slow": 3, "manual": 100})
    interfaces: dict[str, SamplingRule] = field(default_factory=dict)  # 接口级别自定义规则
    slow_latency_ms: int = 1000  # 超过此延迟视为慢请求
    cold_start_first_n: int = 3  # 冷启动前 N 条全量采样

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplingConfig":
        """从字典构造配置（用于配置文件加载）。"""
        def rule(value: dict[str, Any]) -> SamplingRule:
            return SamplingRule(
                percent=int(value.get("percent", 1)),
                max_per_minute=int(value.get("max_per_minute", 5)),
                min_per_day=int(value.get("min_per_day", 3)),
            )

        return cls(
            default=rule(data.get("default", {})),
            boost=dict(data.get("boost", {"error": 5, "slow": 3, "manual": 100})),
            interfaces={key: rule(value) for key, value in data.get("interfaces", {}).items()},
            slow_latency_ms=int(data.get("slow_latency_ms", 1000)),
            cold_start_first_n=int(data.get("cold_start_first_n", 3)),
        )


@dataclass(frozen=True)
class SamplingDecision:
    """采样决策结果。"""
    accepted: bool  # 是否被选中进行分析
    reason: str  # 决策原因（用于调试和日志）
    effective_percent: int  # 最终生效的采样率


class AdaptiveSampler:
    """自适应采样器，根据接口流量和事件特征动态调整采样率。"""

    def __init__(self, config: SamplingConfig, state: InMemorySamplerState | None = None) -> None:
        self.config = config
        self.state = state or InMemorySamplerState()

    def decide(self, event: OpenClawEvent, now: datetime | None = None) -> SamplingDecision:
        """对一条事件做采样决策。"""
        now = now or datetime.now(timezone.utc)
        minute_bucket = now.strftime("%Y%m%d%H%M")
        day_bucket = now.strftime("%Y%m%d")
        interface_key = event.interface_key

        # 记录该接口总请求数
        total = self.state.record_total(interface_key)
        rule = self.config.interfaces.get(interface_key, self.config.default)

        # 检查频率限制：每分钟不超过 max_per_minute
        minute_count = self.state.minute_by_interface.get((interface_key, minute_bucket), 0)
        day_count = self.state.day_by_interface.get((interface_key, day_bucket), 0)
        if minute_count >= rule.max_per_minute:
            decision = SamplingDecision(False, "max_per_minute_reached", rule.percent)
            self.state.skipped.append({"interface_key": interface_key, "reason": decision.reason})
            return decision

        # 冷启动：新接口的前 N 条全部采样
        if total <= self.config.cold_start_first_n:
            self.state.record_accepted(interface_key, minute_bucket, day_bucket)
            return SamplingDecision(True, "cold_start", 100)

        # 每日最低保证：确保每个接口每天至少被分析 min_per_day 次
        if day_count < rule.min_per_day:
            self.state.record_accepted(interface_key, minute_bucket, day_bucket)
            return SamplingDecision(True, "min_per_day", 100)

        # 计算有效采样率（根据事件特征加权）
        effective_percent = rule.percent
        if event.manual:
            # 手动触发：全量通过
            effective_percent = max(effective_percent, self.config.boost.get("manual", 100))
        elif event.status.lower() == "error":
            # 错误请求：提升采样率
            effective_percent = min(100, effective_percent * self.config.boost.get("error", 1))
        elif event.latency_ms >= self.config.slow_latency_ms:
            # 慢请求：提升采样率
            effective_percent = min(100, effective_percent * self.config.boost.get("slow", 1))

        # 概率采样：使用稳定哈希，同一条 trace 的决策幂等
        accepted = stable_percent(event.idempotency_key) < effective_percent
        if accepted:
            self.state.record_accepted(interface_key, minute_bucket, day_bucket)
            return SamplingDecision(True, "percent", effective_percent)
        decision = SamplingDecision(False, "percent_skipped", effective_percent)
        self.state.skipped.append({"interface_key": interface_key, "reason": decision.reason})
        return decision


def stable_percent(value: str) -> int:
    """对字符串做稳定哈希，映射到 [0, 100) 区间。

    使用 SHA256 保证分布均匀且对同一输入结果稳定（幂等）。
    """
    import hashlib
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % 100
