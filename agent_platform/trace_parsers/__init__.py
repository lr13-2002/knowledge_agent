"""Trace 日志解析 + 锚点 → CodeSymbol join（AGENTS.md §9.6 / §9.7）。

背景（2026-06-24 三服务实测确立）：
- 同公司同 observe 平台,不同服务日志格式互不相同（usce Java / cedar Go / spruce Go）。
- 根因：公司未强制日志规范,logback/dlog 各自 pattern 由业务自决。
- 业界做法（Datadog / ARMS / Splunk）：接入是配置动作,不是开发动作。

本模块对齐这条路：
1. LogParserRegistry — 从 parsers.yaml 加载规则,按 appname/lang 匹配 parser。
2. parse_log_anchors() — 对一个 span 的 logs[] 提取 LogAnchor 列表（file/line/func/dltag）。
3. join_anchor_to_symbol() — 用"行号落入 [start_line, end_line] 区间"反查 CodeSymbol。

三层兜底（§9.6）：
- L1 精确匹配 yaml（appname_contains / lang）→ 锚点丰富
- L2 generic_fallback 通用正则 → 未配置服务也能提粗锚点
- L3 完全失败 → internal_path 为空,不阻塞 snapshot（span 骨架 + downstream 仍可用）

join 策略（§9.7,cedar 真实仓库 3/3 实证）：
- ❗ 主键用 file + line 区间,不用 qualified_name 字符串匹配
  （行号是函数体内 log 调用行,不是声明行；qualified_name 前缀与 trace pkg 不一致）
- func 名只作命中后校验/加分,不作主键
"""
from __future__ import annotations

from .anchors import LogAnchor, parse_log_anchors
from .join import join_anchor_to_symbol
from .registry import LogParserRegistry

__all__ = [
    "LogAnchor",
    "LogParserRegistry",
    "join_anchor_to_symbol",
    "parse_log_anchors",
]
