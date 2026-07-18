"""锚点 → CodeSymbol join（AGENTS.md §9.7,cedar 真实仓库 3/3 实证）。

核心策略（实证坐实,不可改）：
- ❗ 主键用 file + line 区间：start_line ≤ anchor.line ≤ end_line。
  原因 1：trace 日志行号是函数体内 log 调用行（259）,不是函数声明行（247）,
          所以不能用"行号 == 声明行"。
  原因 2：trace 解析出的 pkg 是完整 module path（git.xiaojukeji.com/nuwa/cedar/...）,
          索引产物 qualified_name 是 repo 名 + 简化 module（cedar.app.core.dao.xxx）,
          前缀对不上,qualified_name 不能作 join 主键。
- 命中多个区间时取最内层（区间最小）的符号。
- func 名只作命中后校验：命中符号 symbol_name == anchor.func → join_verified=True（高置信）。
- join 失败（file 找不到对应符号）→ 锚点降级为纯文本附注,不阻塞 snapshot。
"""
from __future__ import annotations

from typing import Any

from .anchors import LogAnchor


def join_anchor_to_symbol(anchor: LogAnchor, symbols: list[dict[str, Any]]) -> LogAnchor:
    """用区间匹配把锚点接到 CodeSymbol,原地填充 anchor 的 symbol_* 字段后返回。

    symbols: 同一文件（或同 repo 候选）的 CodeSymbol 列表,需含 start_line/end_line/symbol_name。
             调用方负责先按 file 过滤候选（见 _file_matches），减少误命中。
    join 不上时 anchor 的 symbol_id/symbol_name 保持空,join_verified=False。
    """
    candidates = [
        s for s in symbols
        if _file_matches(anchor.file, s.get("file", ""))
        and s.get("start_line", 0) <= anchor.line <= s.get("end_line", 10 ** 9)
    ]
    if not candidates:
        return anchor

    # 取最内层（区间最小）的符号——嵌套场景下命中最具体的那个
    candidates.sort(key=lambda s: s.get("end_line", 0) - s.get("start_line", 0))
    hit = candidates[0]
    anchor.symbol_id = hit.get("id", "")
    anchor.symbol_name = hit.get("symbol_name", "")
    # func 名二次校验：一致则高置信（Go 有 func,Java 无 func 时跳过校验）
    if anchor.func and anchor.symbol_name:
        anchor.join_verified = anchor.func == anchor.symbol_name
    return anchor


def _file_matches(anchor_file: str, symbol_file: str) -> bool:
    """判断锚点 file 与索引符号 file 是否指向同一文件。

    锚点 file 可能是相对路径片段（spruce: "app/web-api/api/http/formPage.go"）
    或纯文件名（cedar: "driver_info.go" / Java: "Foo.java"）。
    索引 symbol.file 是 repo 内相对路径（"app/core/dao/driver_info.go"）。
    用后缀匹配兼容两种情况：symbol_file 以 anchor_file 结尾,或反之 basename 相同。
    """
    if not anchor_file or not symbol_file:
        return False
    if symbol_file.endswith(anchor_file) or anchor_file.endswith(symbol_file):
        return True
    # 兜底：basename 相同（纯文件名锚点 join 同名文件,可能多命中,靠区间收敛）
    return symbol_file.rsplit("/", 1)[-1] == anchor_file.rsplit("/", 1)[-1]


def join_anchors(anchors: list[LogAnchor], symbols: list[dict[str, Any]]) -> list[LogAnchor]:
    """批量 join。对每个锚点就地填充 symbol_*,返回同一列表。"""
    for anchor in anchors:
        join_anchor_to_symbol(anchor, symbols)
    return anchors
