"""事件摄入层。

接收外部推送的 trace 事件（OpenClaw webhook / 手动触发），经过：
1. 幂等去重 — 同一个 trace_id 不重复处理
2. 自适应采样 — 根据接口热度、错误率等决定是否分析
3. 入队 — 通过采样的事件进入任务队列，等待 Worker 消费

这是整个流水线的入口点。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .sampler import AdaptiveSampler
from .schemas import OpenClawEvent
from .stores import InMemoryIdempotencyStore, InMemoryTaskQueue

# 任务队列的 stream 名称，Worker 从这个 stream 消费
STREAM_NAME = "trace_analysis_tasks"


def handle_openclaw_trace(
    payload: dict[str, Any],
    sampler: AdaptiveSampler,
    queue: InMemoryTaskQueue,
    idempotency: InMemoryIdempotencyStore,
) -> dict[str, Any]:
    """处理一条 trace 事件，返回处理结果。

    可能的返回状态：
        duplicate — 已处理过，跳过
        skipped  — 采样器决定不分析（频率过高等）
        accepted — 成功入队，等待 Worker 处理
    """
    # 解析并校验事件字段
    event = OpenClawEvent.from_dict(payload)

    # 幂等检查：同一个 repo + trace_id 不重复处理
    if idempotency.seen(event.idempotency_key):
        return {"status": "duplicate", "trace_id": event.trace_id, "interface_key": event.interface_key}
    idempotency.mark(event.idempotency_key)

    # 采样决策：控制每个接口的分析频率
    decision = sampler.decide(event)
    if not decision.accepted:
        return {
            "status": "skipped",
            "trace_id": event.trace_id,
            "interface_key": event.interface_key,
            "reason": decision.reason,
        }

    # 通过采样，入队等待 Worker 消费
    message = asdict(event)
    message["interface_key"] = event.interface_key
    message_id = queue.enqueue(STREAM_NAME, message)
    return {
        "status": "accepted",
        "message_id": message_id,
        "trace_id": event.trace_id,
        "interface_key": event.interface_key,
        "reason": decision.reason,
    }
