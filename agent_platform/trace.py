"""Trace 数据获取和标准化。

提供两种 TraceProvider 实现：
- FixtureTraceProvider — 测试用，从内存字典返回预设数据
- ObserveTraceProvider — 生产用，调用滴滴 observe API 获取真实 trace

同时提供 normalize_trace() 函数，将不同来源的 trace 数据
统一转换为 NormalizedTrace 结构（spans、upstream、downstream、errors）。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Protocol

from .schemas import ExternalCall, InternalStep, NormalizedTrace, OpenClawEvent, TraceSpan

logger = logging.getLogger(__name__)


class TraceProvider(Protocol):
    def fetch(self, repo: str, trace_id: str) -> dict[str, Any]:
        ...


class FixtureTraceProvider:
    def __init__(self, traces: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.traces = traces

    def fetch(self, repo: str, trace_id: str) -> dict[str, Any]:
        key = (repo, trace_id)
        if key not in self.traces:
            raise KeyError(f"trace not found: {repo}/{trace_id}")
        return self.traces[key]


class ObserveTraceProvider:
    """Fetches real traces from the observe API (same backend as trace-mcp).

    Credentials are resolved from environment variables:
      - TRACE_MCP_USERNAME
      - TRACE_MCP_KMS_SECRET_ID (+ KMS access keys for secret resolution)
      - Or OBSERVE_SECRET for direct secret injection (testing/local dev)

    Falls back to OBSERVE_BASE_URL env var for the API endpoint.
    """

    def __init__(
        self,
        base_url: str | None = None,
        username: str | None = None,
        secret: str | None = None,
        idc_num: int = 1,
    ) -> None:
        self._base_url = (base_url or os.environ.get("OBSERVE_BASE_URL") or "http://observeapi-us.intra.xiaojukeji.com").rstrip("/")
        self._username = username or os.environ.get("TRACE_MCP_USERNAME", "")
        self._secret = secret or os.environ.get("OBSERVE_SECRET", "")
        self._idc_num = idc_num

    def _resolve_secret(self) -> str:
        if self._secret:
            return self._secret
        try:
            from trace_mcp.config import load_credentials
            _, secret = load_credentials()
            self._secret = secret
            return secret
        except Exception:
            logger.warning("could not resolve observe secret via KMS, trace fetching will fail")
            return ""

    def _build_headers(self) -> dict[str, str]:
        """observe API 的 HMAC 签名协议（与 trace-mcp 同步）。

        签名规则：HMAC-SHA1(secret, 当前 5 分钟时间桶)。
        bucket_seconds=300（5min）：服务端允许 ±1 桶的时钟误差，
        即同一 secret 算出的 sig 在 [bucket-300, bucket+600] 内都有效。
        如果调成 60s，跨进程时钟稍漂就会鉴权失败。
        """
        secret = self._resolve_secret()
        bucket_seconds = 300  # 5 分钟时间桶，给客户端时钟漂移留缓冲
        ts = int(time.time())
        bucket = ts - ts % bucket_seconds
        digest = hmac.new(
            secret.encode("utf-8"),
            str(bucket).encode("utf-8"),
            hashlib.sha1,
        ).digest()
        sig = base64.b64encode(digest).decode("ascii").rstrip("\n")
        return {
            "authorization": f'hmac username="{self._username}", algorithm="hmac-sha1", signature="{sig}"',
            "TimeZone": "Asia/Shanghai",
            "Content-Type": "application/json",
        }

    def _infer_time_window(self, trace_id: str) -> tuple[int, int]:
        """从 trace_id 中提取生成时间，推断 [start_ms, end_ms] 查询窗口。

        滴滴 trace_id 格式：前 16 个 hex 字符中，第 9-16 位（[8:16]）是生成时间的
        unix 秒数。提取出来后给上下文窗口 [t-1h, t+11h]（共 12h），覆盖跨天 trace。

        sanity check：946684800 = 2000-01-01 epoch 秒（早于此判为非时间，可能是脏数据）；
        time.time() + 366*86400 = 一年后（晚于此也是脏数据，比如全 f 的 trace_id）。
        两道闸过滤掉非法/构造 trace_id，避免下游查询窗口异常。

        提取失败返回 (0, 0)，observe API 会用平台默认窗口兜底。
        """
        try:
            normalized = trace_id.strip().lower()
            if normalized.startswith("0x"):
                normalized = normalized[2:]
            if len(normalized) >= 16:
                seconds = int(normalized[8:16], 16)
                # 边界：必须晚于 2000-01-01 且早于"现在+1年"，否则视为非法
                if 946684800 < seconds < int(time.time()) + 366 * 86400:
                    # 窗口：t-1h 到 t+11h（毫秒），覆盖跨天 + 服务端延迟
                    return (seconds - 3600) * 1000, (seconds + 11 * 3600) * 1000
        except (ValueError, IndexError):
            pass
        return 0, 0

    def fetch(self, repo: str, trace_id: str) -> dict[str, Any]:
        import httpx

        start_ms, end_ms = self._infer_time_window(trace_id)
        params: dict[str, Any] = {
            "traceid": trace_id,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "idcNum": self._idc_num,
        }
        headers = self._build_headers()
        url = f"{self._base_url}/api/v3/traceLink"

        logger.debug("fetching trace %s from %s", trace_id, url)
        response = httpx.get(url, params=params, headers=headers, timeout=15.0)
        if response.status_code >= 400:
            raise RuntimeError(f"observe API returned {response.status_code}: {response.text[:500]}")
        payload = response.json()

        spans = self._extract_spans(payload)
        return {
            "traceId": trace_id,
            "repo": repo,
            "spans": spans,
            "raw_response": payload,
        }

    def _extract_spans(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        data = payload.get("data") or payload
        if isinstance(data, dict):
            root = data.get("root") or data
            self._walk_spans(root, spans)
        elif isinstance(data, list):
            for item in data:
                self._walk_spans(item, spans)
        return spans

    def _walk_spans(self, node: dict[str, Any], out: list[dict[str, Any]]) -> None:
        if not isinstance(node, dict):
            return
        span: dict[str, Any] = {
            "service": node.get("appName") or node.get("appname") or node.get("service") or "",
            "name": node.get("spanName") or node.get("operation") or node.get("name") or "",
            "method": node.get("method") or "",
            "path": node.get("url") or node.get("path") or "",
            "duration": node.get("duration") or "",
            "hasError": bool(node.get("hasError") or node.get("has_error") or node.get("error")),
            "spanId": node.get("spanId") or node.get("spanid") or "",
            "parentSpanId": node.get("parentSpanId") or node.get("parent_spanid") or "",
        }
        if span["service"] or span["name"]:
            out.append(span)
        for child in node.get("children") or node.get("childList") or []:
            self._walk_spans(child, out)


def normalize_trace(event: OpenClawEvent, raw_mcp: dict[str, Any]) -> NormalizedTrace:
    """将不同来源的 trace 统一为 NormalizedTrace。

    两层数据形态（AGENTS.md §9.6）：
    1. span 树（traceLink API）→ spans/upstream/downstream/errors（一直支持）。
    2. span detail（query_span_detail，含 logs[]/downstream）→ internal_path/external_calls。
       detail 需逐 span 拉取,traceLink 拿不到；没有 detail 时这两个字段留空,不报错。

    internal_path 的 join CodeSymbol 不在这里做（normalize 是纯函数,不依赖 graph）,
    由 worker 在有 graph 时调用 trace_parsers.join_anchors 二次填充。
    """
    spans = []
    for item in raw_mcp.get("spans") or raw_mcp.get("call_stack") or []:
        spans.append(
            TraceSpan(
                service=str(item.get("service") or item.get("appname") or item.get("appName") or ""),
                name=str(item.get("name") or item.get("operation") or ""),
                method=str(item.get("method") or ""),
                path=str(item.get("path") or item.get("url") or ""),
                duration=str(item.get("duration") or ""),
                has_error=bool(item.get("hasError") or item.get("has_error") or item.get("error") or False),
            )
        )
    upstream = []
    downstream = []
    if spans:
        root = spans[0]
        upstream = [root.service or root.path or root.name]
        downstream = sorted({span.service or span.path or span.name for span in spans[1:] if span.service or span.path or span.name})
    errors = [span.name or span.service for span in spans if span.has_error]

    # §9.6 精细现场：仅当 raw 携带 span detail（logs/downstream）时才有产物
    internal_path = _extract_internal_path(event.repo, raw_mcp)
    external_calls = _extract_external_calls(raw_mcp)

    return NormalizedTrace(
        repo=event.repo,
        trace_id=event.trace_id,
        interface_key=event.interface_key,
        spans=spans,
        raw_mcp=raw_mcp,
        upstream=upstream,
        downstream=downstream,
        errors=errors,
        internal_path=internal_path,
        external_calls=external_calls,
    )


# 模块级单例 registry,避免每条 trace 重复读 yaml
_PARSER_REGISTRY: Any = None


def _get_registry() -> Any:
    """懒加载 parser 注册表。yaml 缺失/pyyaml 未装时返回 None（降级,不阻塞）。"""
    global _PARSER_REGISTRY
    if _PARSER_REGISTRY is None:
        try:
            from .trace_parsers import LogParserRegistry
            _PARSER_REGISTRY = LogParserRegistry.from_yaml()
        except Exception:
            logger.warning("trace_parsers registry 加载失败,internal_path 将为空", exc_info=True)
            _PARSER_REGISTRY = False  # 标记"试过且失败",不再重试
    return _PARSER_REGISTRY or None


def _detect_lang(appname: str) -> str:
    """从 appname 粗判语言。真实实现应走代码索引的 repo→lang 反查（TODO）。"""
    a = (appname or "").lower()
    return "go" if ("spruce" in a or "cedar" in a) else "java"


def _iter_span_details(raw_mcp: dict[str, Any]) -> list[dict[str, Any]]:
    """收集 raw 里携带 span detail 的节点（含 logs 或 downstream 字段）。

    兼容两种形态：
    - 单 span detail（query_span_detail 的 data）：raw_mcp["span_details"] = [data, ...]
    - 直接把一个 detail 的 data 放在 raw_mcp 顶层（测试常用）
    """
    details = raw_mcp.get("span_details")
    if isinstance(details, list):
        return [d for d in details if isinstance(d, dict)]
    # 顶层就是一个 detail（含 logs/downstream）
    if "logs" in raw_mcp or "downstream" in raw_mcp:
        return [raw_mcp]
    return []


def _extract_internal_path(repo: str, raw_mcp: dict[str, Any]) -> list[InternalStep]:
    """从 span detail 的 logs[] 解析内部锚点。无 detail 时返回空。"""
    registry = _get_registry()
    if registry is None:
        return []
    from .trace_parsers import parse_log_anchors

    steps: list[InternalStep] = []
    for detail in _iter_span_details(raw_mcp):
        logs = detail.get("logs")
        if not isinstance(logs, list) or not logs:
            continue
        appname = (detail.get("appname") or [""])
        appname = appname[0] if isinstance(appname, list) and appname else ""
        parser = registry.pick(appname, _detect_lang(appname))
        for anchor in parse_log_anchors(parser, logs):
            steps.append(
                InternalStep(
                    file=anchor.file,
                    line=anchor.line,
                    func=anchor.func,
                    dltag=anchor.dltag,
                )
            )
    return steps


def _extract_external_calls(raw_mcp: dict[str, Any]) -> list[ExternalCall]:
    """从 span detail 的 downstream._com_http_success 解析跨服务调用。无 detail 时返回空。

    结构（实测）：downstream = {_com_http_success: [{content: {url, errno, proc_time}}], _undef: [...]}
    只取结构化的 _com_http_success（对应 OTel CLIENT span）,丢弃 _undef 业务日志噪音。
    """
    calls: list[ExternalCall] = []
    for detail in _iter_span_details(raw_mcp):
        downstream = detail.get("downstream")
        if not isinstance(downstream, dict):
            continue
        for entry in downstream.get("_com_http_success") or []:
            content = entry.get("content") if isinstance(entry, dict) else None
            if not isinstance(content, dict):
                continue
            url = content.get("url") or content.get("uri") or ""
            if url:
                calls.append(
                    ExternalCall(
                        url=str(url),
                        errno=str(content.get("errno") or ""),
                        proc_time=str(content.get("proc_time") or ""),
                    )
                )
    return calls
