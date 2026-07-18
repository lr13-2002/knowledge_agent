"""trace_parsers 生产模块测试（AGENTS.md §9.6 / §9.7）。

复用 tests/parser_samples/ 下的三个真实 trace 样本（usce/cedar/spruce）做回归,
并用 cedar 真实仓库（若存在）验证 file+line 区间 join。

数据来源：真实线上 trace 0a883c8c... / 0ac56cf2...,见各 *_sample.json 的 _note。
"""
import json
import os
import unittest
from pathlib import Path

from agent_platform.trace_parsers import (
    LogParserRegistry,
    join_anchor_to_symbol,
    parse_log_anchors,
)
from agent_platform.trace_parsers.join import join_anchors

SAMPLES = Path(__file__).parent / "parser_samples"
CEDAR_REPO = "/Users/didi/team_project/cedar"


def _load_sample(name: str) -> dict:
    with open(SAMPLES / f"{name}_sample.json", encoding="utf-8") as f:
        return json.load(f)


def _detect_lang(appname: str) -> str:
    a = appname.lower()
    return "go" if ("spruce" in a or "cedar" in a) else "java"


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = LogParserRegistry.from_yaml()

    def test_appname_match_picks_specific_parser(self) -> None:
        self.assertEqual(self.reg.pick("hnb-v.cedar.os.biz.didi.com", "go")["name"], "cedar")
        self.assertEqual(self.reg.pick("hna-v.spruce.os.biz.didi.com", "go")["name"], "spruce")

    def test_lang_fallback_for_unknown_appname(self) -> None:
        # 没有 appname_contains 命中时,按 lang 兜底到 java_default
        self.assertEqual(self.reg.pick("hnb-v.usce-api.usce.biz.didi.com", "java")["name"], "java_default")

    def test_generic_fallback_always_matches(self) -> None:
        # 完全陌生的服务 + 未知语言 → generic_fallback（L3 永不失败）
        chosen = self.reg.pick("totally.unknown.service", "ruby")
        self.assertEqual(chosen["name"], "generic_fallback")


class ParseAnchorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reg = LogParserRegistry.from_yaml()

    def _parse_sample(self, name: str):
        sample = _load_sample(name)
        appname = (sample.get("appname") or [""])[0]
        parser = self.reg.pick(appname, _detect_lang(appname))
        return parser, parse_log_anchors(parser, sample["logs"])

    def test_usce_java_extracts_and_filters_feign(self) -> None:
        parser, anchors = self._parse_sample("usce")
        self.assertEqual(parser["name"], "java_default")
        # 11 行里 2 行是 feign.Logger 噪音,应被 com.x pattern 过滤掉 → 9 个锚点
        self.assertEqual(len(anchors), 9)
        # Java 锚点：file 从类全名末段推 .java,有 pkg_class,无 func
        first = anchors[0]
        self.assertEqual(first.file, "UsceTraceLogFilter.java")
        self.assertEqual(first.line, 71)
        self.assertTrue(first.pkg_class.startswith("com.didichuxing"))
        self.assertEqual(first.func, "")
        self.assertEqual(first.dltag, "_com_request_in")
        # 确认 feign 噪音确实没混进来
        self.assertFalse(any("feign" in a.raw_match.lower() for a in anchors))

    def test_cedar_go_extracts_function_names(self) -> None:
        parser, anchors = self._parse_sample("cedar")
        self.assertEqual(parser["name"], "cedar")
        self.assertEqual(len(anchors), 3)
        funcs = {a.func for a in anchors}
        self.assertEqual(funcs, {"TraceWithConfig", "GetDriverInfoByTuple", "traceRequestOut"})
        # Go 闭包后缀剥离：Init.TraceWithConfig.func4.1 → func=TraceWithConfig + closure 记录
        closure_anchor = next(a for a in anchors if a.func == "TraceWithConfig")
        self.assertEqual(closure_anchor.file, "trace.go")
        self.assertEqual(closure_anchor.line, 81)
        self.assertEqual(closure_anchor.func_closure, "func4.1")

    def test_spruce_go_strips_build_prefix(self) -> None:
        parser, anchors = self._parse_sample("spruce")
        self.assertEqual(parser["name"], "spruce")
        self.assertEqual(len(anchors), 15)
        # spruce 无函数名,file 是 strip 构建前缀后的相对路径
        first = anchors[0]
        self.assertEqual(first.file, "middleware/wardenAuth.go")
        self.assertEqual(first.line, 80)
        self.assertEqual(first.func, "")
        self.assertEqual(first.dltag, "_com_request_in")

    def test_parse_never_raises_on_garbage(self) -> None:
        # §9.6 不变量：解析失败跳过,不抛异常
        parser = self.reg.pick("anything", "java")
        anchors = parse_log_anchors(parser, [{"log": "完全不匹配的垃圾行"}, {"log": ""}, {}])
        self.assertEqual(anchors, [])


@unittest.skipUnless(os.path.isdir(CEDAR_REPO), f"需要真实 cedar 仓库: {CEDAR_REPO}")
class JoinWithRealRepoTest(unittest.TestCase):
    """§9.7 实证：file+line 区间 join 真实 CodeSymbol。"""

    def _symbols(self, rel_file: str) -> list[dict]:
        from agent_platform.indexer.go_parser import index_go_file
        symbols, _ = index_go_file(CEDAR_REPO, rel_file, "cedar")
        return symbols

    def test_join_hits_correct_symbol_by_interval(self) -> None:
        from agent_platform.trace_parsers.anchors import LogAnchor
        cases = [
            ("app/core/dao/driver_info.go", "GetDriverInfoByTuple", 259),
            ("middleware/trace.go", "traceRequestOut", 123),
            ("middleware/trace.go", "TraceWithConfig", 81),
        ]
        for rel_file, exp_func, log_line in cases:
            with self.subTest(file=rel_file, line=log_line):
                symbols = self._symbols(rel_file)
                anchor = LogAnchor(file=rel_file, line=log_line, func=exp_func)
                join_anchor_to_symbol(anchor, symbols)
                self.assertEqual(anchor.symbol_name, exp_func)
                self.assertTrue(anchor.join_verified)  # func 名一致 → 高置信

    def test_join_uses_interval_not_declaration_line(self) -> None:
        # 关键实证：日志行号 259 是函数体内 log 调用行,声明行是 247。
        # 区间 join 仍能命中 → 证明用的是区间而非声明行匹配。
        from agent_platform.trace_parsers.anchors import LogAnchor
        symbols = self._symbols("app/core/dao/driver_info.go")
        anchor = LogAnchor(file="app/core/dao/driver_info.go", line=259, func="GetDriverInfoByTuple")
        join_anchor_to_symbol(anchor, symbols)
        hit = next(s for s in symbols if s["symbol_name"] == "GetDriverInfoByTuple")
        self.assertLess(hit["start_line"], 259)        # 声明行 < 日志行
        self.assertGreaterEqual(hit["end_line"], 259)  # 日志行在区间内
        self.assertEqual(anchor.symbol_name, "GetDriverInfoByTuple")

    def test_join_miss_leaves_anchor_unlinked(self) -> None:
        # join 不上时锚点保持未链接,不报错（降级为纯文本附注）
        from agent_platform.trace_parsers.anchors import LogAnchor
        symbols = self._symbols("middleware/trace.go")
        anchor = LogAnchor(file="middleware/trace.go", line=999999, func="nope")
        join_anchor_to_symbol(anchor, symbols)
        self.assertEqual(anchor.symbol_name, "")
        self.assertFalse(anchor.join_verified)


if __name__ == "__main__":
    unittest.main()
