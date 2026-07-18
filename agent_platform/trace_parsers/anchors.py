"""从 span 的 logs[] 提取代码锚点 LogAnchor（AGENTS.md §9.6）。

LogAnchor = 一行日志解析出的"代码位置线索"：file + line(+ func + dltag)。
它是 trace 现场和代码索引之间的桥梁：file+line 可 join CodeSymbol（见 join.py）。

实测要点（2026-06-24 三服务）：
- mcp 返回的 logs[] 元素通常只有 log 全文,没有结构化 line 字段
  → 统一对 log 全文应用 parser 的 pattern 正则。
- Java（usce）：`[com.x.Foo:24]`,无函数名,file 从类全名末段 + .java 推。
  feign.Logger 这类框架噪音因不匹配 `com.x` pattern 被天然过滤。
- Go（cedar）：`[pkg.Func/file.go:N]`,有函数名,Go 闭包后缀 .funcN.M 需剥离。
- Go（spruce）：`[/gaia/.../build/path/file.go:N]`,无函数名,构建前缀在 pattern 内消化。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# dltag 标记:_undef / _com_request_in / _com_http_success / _com_mysql_success ...
# 对所有语言通用,日志里都以 " _xxx||" 形式出现。
_DLTAG_RE = re.compile(r" (_[a-z_]+)\|\|")

# Go 闭包子函数后缀:Init.TraceWithConfig.func4.1 → 顶层 func 是 TraceWithConfig
_GO_CLOSURE_RE = re.compile(r"\.func\d+(\.\d+)*$")


@dataclass
class LogAnchor:
    """一行日志解析出的代码锚点。

    file/line 是 join CodeSymbol 的主键（§9.7 区间匹配）。
    func 仅作命中后校验,不作 join 主键（pkg 前缀与索引 qualified_name 不一致）。
    """

    file: str = ""              # 文件名或相对路径,如 "driver_info.go" / "app/web-api/api/http/formPage.go"
    line: int = 0               # 日志打印行号（函数体内 log 调用行,不是声明行）
    func: str = ""              # 函数名,Go 有 / Java 无;join 命中后用于二次校验
    pkg: str = ""               # 包路径,Go 解析时附带
    func_closure: str = ""      # Go 闭包后缀,如 "func4.1",保留供调试
    pkg_class: str = ""         # Java 类全名,如 "com.x.Foo"
    dltag: str = ""             # 日志分类标记,如 "_undef" / "_com_http_success"
    parser: str = ""            # 命中的 parser 名,便于追溯
    raw_match: str = ""         # 正则匹配到的原文片段
    # join 结果（join.py 填充,初始为空）
    symbol_id: str = ""         # 命中的 CodeSymbol id
    symbol_name: str = ""       # 命中的符号名
    join_verified: bool = False  # func 名是否与命中符号一致（高置信标志）


def parse_log_anchors(parser: dict[str, Any], logs: list[dict[str, Any]]) -> list[LogAnchor]:
    """对一个 span 的 logs[] 逐行应用 parser,返回成功解析的锚点列表。

    解析失败的行直接跳过（不抛异常,§9.6 不变量：降级不阻塞）。
    """
    anchors: list[LogAnchor] = []
    pattern = parser.get("pattern")
    if not pattern:
        return anchors
    compiled = re.compile(pattern)
    for entry in logs:
        anchor = _parse_one(parser, compiled, entry)
        if anchor is not None:
            anchors.append(anchor)
    return anchors


def _parse_one(parser: dict[str, Any], compiled: re.Pattern[str], log_entry: dict[str, Any]) -> LogAnchor | None:
    """对单条日志行应用 parser,返回 LogAnchor 或 None。

    统一对 log 全文应用正则（实测 logs[] 通常只有 log 全文,无结构化 line 字段）。
    """
    raw = log_entry.get("log") or ""
    m = compiled.search(raw)
    if not m:
        return None

    gd = m.groupdict()
    anchor = LogAnchor(
        parser=parser.get("name", ""),
        file=gd.get("file") or "",
        line=int(gd["line"]) if gd.get("line") else 0,
        raw_match=m.group(0),
    )

    # Go：有函数名,处理 pkg.Func 切分 + 闭包后缀剥离
    if parser.get("has_function"):
        if gd.get("func"):
            anchor.func = gd["func"]
        elif gd.get("pkg_func"):
            split = parser.get("func_field_split", ".")
            pkg_func = gd["pkg_func"]
            stripped = _GO_CLOSURE_RE.sub("", pkg_func)
            idx = stripped.rfind(split)
            if idx >= 0:
                anchor.pkg = stripped[:idx]
                anchor.func = stripped[idx + 1:]
                if pkg_func != stripped:
                    anchor.func_closure = pkg_func[len(stripped):].lstrip(".")

    # Java：从类全名末段推 file（com.x.Foo → Foo.java）
    if parser.get("file_field_derive") == "from_pkg_class" and gd.get("pkg_class"):
        anchor.pkg_class = gd["pkg_class"]
        cls_name = gd["pkg_class"].rsplit(".", 1)[-1]
        anchor.file = f"{cls_name}.java"

    # dltag 对所有语言通用,从 log 全文提
    dltag_m = _DLTAG_RE.search(raw)
    if dltag_m:
        anchor.dltag = dltag_m.group(1)

    return anchor
