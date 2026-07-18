"""Mock 测试 AnthropicLLMClient。

不实际调用 LLM API，通过 mock 验证：
1. function calling 数据流转正确
2. 校验失败时重试
3. API 异常时降级到 HeuristicLLMClient
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from agent_platform.llm.client import AnthropicLLMClient, LLMOutputError
from agent_platform.llm.validators import (
    validate_business_synthesis,
    validate_code_association,
    validate_trace_summary,
)
from agent_platform.schemas import NormalizedTrace, OpenClawEvent, TraceSpan


_GOOD_ARGS = json.dumps({
    "summary": "用户下单后调用支付服务完成扣款",
    "flow_steps": ["接收下单请求", "调用 pay-svc 完成扣款", "返回订单结果"],
    "candidate_claims": ["该接口会调用 pay-svc 进行实际扣款", "下单流程是同步阻塞的"],
    "confidence": "medium",
    "reasoning": "trace 明确显示 my-svc 调用了 pay-svc，代码中有 Charge 方法",
})

_BAD_ARGS = json.dumps({
    "summary": "",
    "flow_steps": [],
    "candidate_claims": [],
    "confidence": "invalid",
    "reasoning": "",
})


def _make_context() -> dict[str, Any]:
    event = OpenClawEvent(
        repo="test-repo",
        trace_id="abc123",
        service="my-svc",
        method="POST",
        path="/api/v1/order/create",
    )
    trace = NormalizedTrace(
        repo="test-repo",
        trace_id="abc123",
        interface_key="test-repo:my-svc:POST:/api/v1/order/create",
        spans=[
            TraceSpan(service="my-svc", name="handleCreate", method="POST", path="/api/v1/order/create", duration="15ms"),
            TraceSpan(service="pay-svc", name="charge", method="POST", path="/pay/charge", duration="8ms"),
        ],
        raw_mcp={},
        upstream=["my-svc"],
        downstream=["pay-svc"],
        errors=[],
    )
    code_hits = [
        {"id": "sym1", "text": "func HandleCreate(ctx context.Context, req *OrderReq) error { ... }", "payload": {"qualified_name": "order.HandleCreate"}, "score": 3},
        {"id": "sym2", "text": "func (s *PayClient) Charge(amount int) error { ... }", "payload": {"qualified_name": "pay.PayClient.Charge"}, "score": 2},
    ]
    return {"event": event, "trace": trace, "code_hits": code_hits, "commit": "deadbeef"}


def _mock_function(name: str, arguments: str) -> MagicMock:
    """构造一个 mock 的 OpenAI function call 对象。"""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    return fn


def _mock_tool_call(name: str = "business_synthesis", arguments: str = _GOOD_ARGS) -> MagicMock:
    """构造一个 mock 的 OpenAI tool_call 对象。"""
    call = MagicMock()
    call.id = "call_001"
    call.function = _mock_function(name, arguments)
    return call


def _mock_response(tool_calls: list | None = None, content: str = "") -> MagicMock:
    """构造一个 mock 的 OpenAI ChatCompletion response。"""
    message = MagicMock()
    message.tool_calls = tool_calls
    message.content = content
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class ValidatorTest(unittest.TestCase):
    def test_valid_trace_summary(self) -> None:
        data = {"entry_point": "POST /api", "call_chain": [{"service": "a", "operation": "b", "is_error": False}], "error_flags": [], "trace_pattern": "简单转发"}
        self.assertTrue(validate_trace_summary(data).valid)

    def test_invalid_trace_summary(self) -> None:
        data = {"entry_point": "", "call_chain": [], "error_flags": [], "trace_pattern": ""}
        result = validate_trace_summary(data)
        self.assertFalse(result.valid)
        self.assertTrue(len(result.errors) >= 2)

    def test_valid_code_association(self) -> None:
        data = {"related_symbols": [{"qualified_name": "a.B", "role": "entry_handler", "relevance": "high"}], "code_to_trace_mapping": "B 处理入口"}
        self.assertTrue(validate_code_association(data).valid)

    def test_valid_business_synthesis(self) -> None:
        data = {"summary": "下单接口", "flow_steps": ["step1"], "candidate_claims": ["claim1"], "confidence": "high", "reasoning": "ok"}
        self.assertTrue(validate_business_synthesis(data).valid)

    def test_invalid_confidence(self) -> None:
        data = {"summary": "x", "flow_steps": ["s"], "candidate_claims": ["c"], "confidence": "very_high", "reasoning": "r"}
        result = validate_business_synthesis(data)
        self.assertFalse(result.valid)


class AnthropicLLMClientTest(unittest.TestCase):
    @patch("agent_platform.llm.client.openai")
    def test_propose_success(self, mock_openai_module) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            tool_calls=[_mock_tool_call()]
        )
        mock_openai_module.OpenAI.return_value = mock_client

        client = AnthropicLLMClient(api_key="test-key")
        context = _make_context()
        proposal = client.propose(context)

        self.assertEqual(proposal.repo, "test-repo")
        self.assertEqual(proposal.trace_id, "abc123")
        self.assertEqual(proposal.summary, "用户下单后调用支付服务完成扣款")
        self.assertEqual(proposal.confidence, "medium")
        self.assertEqual(len(proposal.flow_steps), 3)
        self.assertEqual(len(proposal.candidate_claims), 2)
        self.assertIn("order.HandleCreate", proposal.related_code_symbols)

    @patch("agent_platform.llm.client.openai")
    def test_propose_fallback_on_error(self, mock_openai_module) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai_module.OpenAI.return_value = mock_client

        client = AnthropicLLMClient(api_key="test-key")
        context = _make_context()
        proposal = client.propose(context)

        self.assertEqual(proposal.repo, "test-repo")
        self.assertIn(proposal.confidence, ("low", "medium"))

    @patch("agent_platform.llm.client.openai")
    def test_retry_on_validation_failure(self, mock_openai_module) -> None:
        bad_response = _mock_response(tool_calls=[_mock_tool_call(arguments=_BAD_ARGS)])
        good_response = _mock_response(tool_calls=[_mock_tool_call()])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [bad_response, good_response]
        mock_openai_module.OpenAI.return_value = mock_client

        client = AnthropicLLMClient(api_key="test-key", max_retries=2)
        context = _make_context()
        proposal = client.propose(context)

        self.assertEqual(proposal.summary, "用户下单后调用支付服务完成扣款")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
