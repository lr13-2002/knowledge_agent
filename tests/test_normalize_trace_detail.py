"""normalize_trace 解析 span detail → internal_path / external_calls（AGENTS.md §9.6）。

覆盖：
- 带 span detail（logs/downstream）时,解析出内部锚点 + 跨服务调用
- traceLink-only（无 detail）时,新字段为空,旧字段照常（向后兼容）
- span_details 列表形态（多 span）
"""
import unittest

from agent_platform.schemas import OpenClawEvent
from agent_platform.trace import normalize_trace


def _event(repo: str = "cedar") -> OpenClawEvent:
    return OpenClawEvent(repo=repo, trace_id="t1", service=repo, method="POST", path="/cedar/app/web/singleSubmit")


# 模拟 cedar query_span_detail 的 data（含 logs + downstream）
CEDAR_DETAIL = {
    "spans": [{"service": "cedar", "path": "/cedar/app/web/singleSubmit"}],
    "appname": ["hnb-v.cedar.os.biz.didi.com"],
    "logs": [
        {"log": "[INFO][2026][git.xiaojukeji.com/nuwa/cedar/app/core/dao.GetDriverInfoByTuple/driver_info.go:259] _undef||x"},
        {"log": "[INFO][2026][git.xiaojukeji.com/nuwa/cedar/middleware.traceRequestOut/trace.go:123] _com_request_out||x"},
    ],
    "downstream": {
        "_com_http_success": [
            {"content": {"url": "http://x/passport/ticket/v5/validate", "errno": "200", "proc_time": "0.003"}},
        ],
        "_undef": [{"content": {"_msg": "业务日志噪音,不该进 external_calls"}}],
    },
}


class NormalizeWithDetailTest(unittest.TestCase):
    def test_internal_path_parsed_from_logs(self) -> None:
        nt = normalize_trace(_event(), CEDAR_DETAIL)
        got = [(s.file, s.line, s.func, s.dltag) for s in nt.internal_path]
        self.assertIn(("driver_info.go", 259, "GetDriverInfoByTuple", "_undef"), got)
        self.assertIn(("trace.go", 123, "traceRequestOut", "_com_request_out"), got)

    def test_external_calls_only_from_http_success(self) -> None:
        nt = normalize_trace(_event(), CEDAR_DETAIL)
        # 只取 _com_http_success,_undef 噪音不进
        self.assertEqual(len(nt.external_calls), 1)
        call = nt.external_calls[0]
        self.assertEqual(call.url, "http://x/passport/ticket/v5/validate")
        self.assertEqual(call.errno, "200")

    def test_internal_path_not_joined_in_normalize(self) -> None:
        # normalize 是纯函数,不 join CodeSymbol → symbol_id 应为空,由 worker 二次填充
        nt = normalize_trace(_event(), CEDAR_DETAIL)
        self.assertTrue(all(s.symbol_id == "" for s in nt.internal_path))

    def test_legacy_fields_still_populated(self) -> None:
        nt = normalize_trace(_event(), CEDAR_DETAIL)
        self.assertEqual(len(nt.spans), 1)
        self.assertEqual(nt.upstream, ["cedar"])


class NormalizeBackwardCompatTest(unittest.TestCase):
    def test_tracelink_only_yields_empty_new_fields(self) -> None:
        # 无 logs/downstream detail → internal_path/external_calls 空,不报错
        raw = {"spans": [{"service": "cedar", "path": "/x"}, {"service": "passport", "path": "/p"}]}
        nt = normalize_trace(_event(), raw)
        self.assertEqual(nt.internal_path, [])
        self.assertEqual(nt.external_calls, [])
        # 旧字段照常
        self.assertEqual(nt.downstream, ["passport"])

    def test_empty_raw_does_not_crash(self) -> None:
        nt = normalize_trace(_event(), {})
        self.assertEqual(nt.spans, [])
        self.assertEqual(nt.internal_path, [])
        self.assertEqual(nt.external_calls, [])


class NormalizeMultiSpanTest(unittest.TestCase):
    def test_span_details_list_form(self) -> None:
        # span_details 列表形态：多个 span 的 detail
        raw = {
            "spans": [{"service": "cedar", "path": "/x"}],
            "span_details": [
                {
                    "appname": ["hnb-v.cedar.os.biz.didi.com"],
                    "logs": [{"log": "[INFO][2026][git.xiaojukeji.com/nuwa/cedar/middleware.traceRequestOut/trace.go:123] _com_request_out||x"}],
                },
                {
                    "appname": ["hna-v.spruce.os.biz.didi.com"],
                    "logs": [{"log": "[NOTICE][2026][/gaia/workspace-job/git.xiaojukeji.com/pt-arch/spruce/build/middleware/wardenAuth.go:80] _com_request_in||x"}],
                },
            ],
        }
        nt = normalize_trace(_event(), raw)
        files = {s.file for s in nt.internal_path}
        self.assertIn("trace.go", files)
        self.assertIn("middleware/wardenAuth.go", files)


if __name__ == "__main__":
    unittest.main()
