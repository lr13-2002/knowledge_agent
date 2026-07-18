"""Ask Agent 测试。

用 mock LLM client 模拟 ReAct 循环：
- 第一轮：LLM 返回 tool_calls（要求检索）
- 第二轮：LLM 返回最终答案

同时测降级路径（无 LLM 时 AskService 走五路召回）。
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_platform.ask import AskService
from agent_platform.llm.ask_agent import AskAgent
from agent_platform.stores import InMemoryGraphStore, InMemoryVectorStore


def _tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    """构造一个 mock 的 tool_call 对象（模拟 openai SDK 的返回结构）。"""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """构造一个 mock 的 chat completion response。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AskAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = InMemoryVectorStore()
        self.graph = InMemoryGraphStore()
        # 灌一个业务实体进向量库
        self.vector.upsert("entities", "entity:订单", "订单 业务概念 用户下单生成的交易单据", {
            "name": "订单", "type": "business_concept",
            "description": "用户下单生成的交易单据", "repo": "demo",
        })
        # 图里建实体节点 + 一个接口提及它
        self.graph.upsert_node("Entity", "entity:订单", {"name": "订单", "repo": "demo"})
        self.graph.upsert_node("Interface", "demo:POST:/order/create", {
            "repo": "demo", "path": "/order/create", "method": "POST",
        })
        self.graph.add_edge("Interface", "demo:POST:/order/create", "MENTIONS", "Entity", "entity:订单", weight=0.6)

    def test_agent_calls_tool_then_answers(self) -> None:
        """两轮：先调 search_entities，再生成答案。"""
        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            # 第一轮：要求检索实体
            _response(tool_calls=[_tool_call("c1", "search_entities", {"query": "订单"})]),
            # 第二轮：基于检索结果生成答案
            _response(content="订单是用户下单生成的交易单据。证据：entity:订单"),
        ]

        agent = AskAgent(llm, "test-model", self.vector, self.graph)
        resp = agent.run("demo", "订单是什么？")

        self.assertIn("订单", resp.answer)
        self.assertEqual(llm.chat.completions.create.call_count, 2)

    def test_agent_no_retrieval_hit_returns_fallback_answer(self) -> None:
        """检索全空时不让 LLM 编，返回兜底答案。"""
        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            _response(tool_calls=[_tool_call("c1", "search_entities", {"query": "不存在的概念"})]),
            _response(content="我编的一个答案"),  # LLM 想编，但应被覆盖
        ]
        agent = AskAgent(llm, "test-model", self.vector, self.graph)
        resp = agent.run("demo", "完全不相关的问题")

        self.assertIn("暂无足够证据", resp.answer)

    def test_agent_direct_answer_no_tool(self) -> None:
        """LLM 第一轮就直接回答（不调工具）→ 无检索命中 → 兜底。"""
        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            _response(content="直接回答，没检索"),
        ]
        agent = AskAgent(llm, "test-model", self.vector, self.graph)
        resp = agent.run("demo", "随便问问")
        # 没有任何检索命中 → 兜底
        self.assertIn("暂无足够证据", resp.answer)

    def test_expand_graph_tool(self) -> None:
        """命中实体后 expand_graph 能找到提及它的接口。"""
        llm = MagicMock()
        llm.chat.completions.create.side_effect = [
            _response(tool_calls=[_tool_call("c1", "search_entities", {"query": "订单"})]),
            _response(tool_calls=[_tool_call("c2", "expand_graph", {"node_id": "entity:订单"})]),
            _response(content="订单相关接口：/order/create。证据：entity:订单"),
        ]
        agent = AskAgent(llm, "test-model", self.vector, self.graph)
        resp = agent.run("demo", "订单涉及哪些接口？")
        self.assertIn("订单", resp.answer)
        self.assertEqual(llm.chat.completions.create.call_count, 3)

    def test_max_turns_respected(self) -> None:
        """达到最大轮数时，最后一轮强制无工具生成答案。"""
        llm = MagicMock()
        # 前两轮都想调工具，第三轮（最后）被强制无工具，必须 content 收尾
        llm.chat.completions.create.side_effect = [
            _response(tool_calls=[_tool_call("c1", "search_entities", {"query": "订单"})]),
            _response(tool_calls=[_tool_call("c2", "search_entities", {"query": "订单"})]),
            _response(content="综合回答。证据：entity:订单"),
        ]
        agent = AskAgent(llm, "test-model", self.vector, self.graph, max_turns=3)
        resp = agent.run("demo", "订单")
        self.assertEqual(llm.chat.completions.create.call_count, 3)
        # 最后一轮调用应该没有 tools 参数
        last_call = llm.chat.completions.create.call_args_list[-1]
        self.assertIsNone(last_call.kwargs.get("tools"))


class ConflictResolutionTest(unittest.TestCase):
    """Day 6a：同接口多条知识召回时，文本带上时间+接口供 LLM 消解。"""

    def setUp(self) -> None:
        self.vector = InMemoryVectorStore()
        self.graph = InMemoryGraphStore()
        # 同一接口两条矛盾知识，时间不同
        self.vector.upsert("knowledge_claims", "rule:old", "订单超时是30分钟", {
            "repo": "demo", "status": "approved", "business_rule_id": "rule:old",
            "interface_key": "demo:POST:/order", "created_at": "2026-01-01T00:00:00",
        })
        self.vector.upsert("knowledge_claims", "rule:new", "订单超时是15分钟", {
            "repo": "demo", "status": "approved", "business_rule_id": "rule:new",
            "interface_key": "demo:POST:/order", "created_at": "2026-06-01T00:00:00",
        })

    def test_knowledge_hits_surface_time_and_interface(self) -> None:
        """召回文本应带 [接口=... 时间=...]，让 LLM 能判新旧。"""
        captured = {}

        def fake_create(**kwargs):
            # 捕获第二轮（带 tool 结果）的 messages
            captured["messages"] = kwargs.get("messages", [])
            # 第一轮要求检索，第二轮直接答
            n = llm.chat.completions.create.call_count
            if n == 1:
                return _response(tool_calls=[_tool_call("c1", "search_knowledge", {"query": "订单超时"})])
            return _response(content="订单超时是15分钟（较早的30分钟结论已过时）。证据：rule:new")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = fake_create

        agent = AskAgent(llm, "test-model", self.vector, self.graph)
        resp = agent.run("demo", "订单超时多久")

        # 找到 tool 返回消息，确认带了时间和接口
        tool_msgs = [m for m in captured["messages"] if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        tool_text = tool_msgs[0]["content"]
        self.assertIn("时间=", tool_text)
        self.assertIn("接口=", tool_text)
        # 两条都应呈现给 LLM（由 LLM 决定取哪条）
        self.assertIn("rule:old", tool_text)
        self.assertIn("rule:new", tool_text)


class AskServiceFallbackTest(unittest.TestCase):
    """AskService 在无 LLM 时降级到五路召回。"""

    def setUp(self) -> None:
        self.vector = InMemoryVectorStore()
        self.graph = InMemoryGraphStore()
        self.vector.upsert("knowledge_claims", "rule:1", "订单创建流程", {
            "repo": "demo", "status": "approved", "business_rule_id": "rule:1",
        })

    def test_no_llm_uses_fallback(self) -> None:
        """无 llm_client → 走五路召回，不报错。"""
        svc = AskService(self.vector, self.graph)  # 不传 llm_client
        resp = svc.ask("demo", "订单怎么创建")
        self.assertIsNotNone(resp.answer)

    def test_agent_exception_falls_back(self) -> None:
        """agent 抛异常 → 自动降级到五路召回。"""
        llm = MagicMock()
        llm.chat.completions.create.side_effect = RuntimeError("LLM down")
        svc = AskService(self.vector, self.graph, llm_client=llm, model="test-model")
        resp = svc.ask("demo", "订单怎么创建")
        # 没崩，降级返回了结果
        self.assertIsNotNone(resp.answer)


if __name__ == "__main__":
    unittest.main()
