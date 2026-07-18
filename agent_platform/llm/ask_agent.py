"""Ask Agent — 基于 ReAct 的知识库问答 agent。

替代原来 AskService 写死的"5 路召回 + 模板拼接"。
核心区别：LLM 自主决定调哪些检索工具、调几次，最后生成连贯答案。

ReAct 循环：
    1. 把"问题 + 检索工具"给 LLM（tool_choice=auto）
    2. LLM 返回 tool_calls → 执行检索 → 结果回填
    3. LLM 判断证据够不够：不够再调工具，够了生成最终答案
    4. 最多 MAX_TURNS 轮，防止无限循环

设计要点：
- 不依赖任何 agent 框架，纯用 OpenAI 兼容的 function calling
- 接收原始 openai 兼容 client（同 review.py 模式），便于 mock 测试
- 工具执行复用 vector / graph store，跟旧 5 路召回查的是同一批数据
- 召回全空时不让 LLM 编造，直接返回"暂无足够证据"

降级关系：本 agent 由 AskService 在"有 LLM"时调用；无 LLM 或 agent 异常时
AskService 回退到旧的 5 路召回（见 ask.py）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from ..schemas import AskResponse, Evidence
from .ask_tools import ASK_AGENT_SYSTEM, ASK_RETRIEVAL_TOOLS

logger = logging.getLogger(__name__)

# ReAct 最多轮数（每轮 = 一次 LLM 调用 + 可能的工具执行）
# 3 轮足够：先查实体 → 展开图 → 生成答案。超过说明问题没规约清楚。
MAX_TURNS = 3

# 每个检索工具单次返回的最大条数（防止 context 爆炸）
SEARCH_LIMIT = 5
EXPAND_LIMIT = 5


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic 格式 tool → OpenAI 格式（与 client.py 同逻辑，独立一份避免循环依赖）。"""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


class AskAgent:
    """ReAct 问答 agent。

    用法：
        agent = AskAgent(llm_client, model, vector, graph)
        response = agent.run(repo, question, trace_id)
    """

    def __init__(
        self,
        llm_client: Any,  # openai 兼容 client（有 .chat.completions.create）
        model: str,
        vector: Any,
        graph: Any | None = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.vector = vector
        self.graph = graph
        self.max_turns = max_turns
        # 收集本次问答用到的证据，最后塞进 AskResponse
        self._evidence_trace_ids: list[str] = []
        self._evidence_rule_ids: list[str] = []

    # ------------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------------

    def run(self, repo: str, question: str, trace_id: str | None = None) -> AskResponse:
        """跑一次 ReAct 问答，返回 AskResponse。

        任何异常都向上抛，由 AskService 捕获并降级到旧 5 路召回。
        """
        self._evidence_trace_ids = [trace_id] if trace_id else []
        self._evidence_rule_ids = []
        self._repo = repo

        openai_tools = [_to_openai_tool(t) for t in ASK_RETRIEVAL_TOOLS]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ASK_AGENT_SYSTEM},
            {"role": "user", "content": question},
        ]

        any_retrieval_hit = False  # 是否检索到过任何证据

        for turn in range(self.max_turns):
            is_last_turn = turn == self.max_turns - 1
            # 最后一轮强制不给工具，逼 LLM 生成最终答案
            response = self.llm.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=messages,
                tools=openai_tools if not is_last_turn else None,
                tool_choice="auto" if not is_last_turn else None,
            )
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)

            # LLM 没有调工具 → 它给出了最终答案
            if not tool_calls:
                answer = choice.message.content or ""
                return self._finalize(answer, any_retrieval_hit)

            # 把 assistant 的 tool_calls 消息加入对话历史
            messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # 执行每个工具调用，把结果作为 tool 消息回填
            for tc in tool_calls:
                result_text, hit = self._execute_tool(tc.function.name, tc.function.arguments)
                any_retrieval_hit = any_retrieval_hit or hit
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        # 轮数用尽仍没拿到最终答案（理论上最后一轮无工具会强制生成，这里兜底）
        return self._finalize("根据现有检索结果暂时无法形成完整回答。", any_retrieval_hit)

    # ------------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------------

    def _execute_tool(self, name: str, arguments: str) -> tuple[str, bool]:
        """执行一个检索工具调用。返回 (结果文本, 是否命中)。"""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return "工具参数解析失败。", False

        try:
            if name == "search_knowledge":
                return self._search_collection("knowledge_claims", args.get("query", ""),
                                               {"repo": self._repo, "status": "approved"})
            if name == "search_entities":
                return self._search_collection("entities", args.get("query", ""), {"repo": self._repo})
            if name == "search_community":
                return self._search_collection("community_summaries", args.get("query", ""), None)
            if name == "expand_graph":
                return self._expand_graph(args.get("node_id", ""))
        except Exception:
            logger.exception("检索工具执行失败: %s", name)
            return f"工具 {name} 执行出错。", False

        return f"未知工具: {name}", False

    def _search_collection(
        self, collection: str, query: str, filters: dict[str, Any] | None
    ) -> tuple[str, bool]:
        """通用向量检索，把命中结果格式化成给 LLM 读的文本。"""
        hits = self.vector.search(collection, query, filters, limit=SEARCH_LIMIT)
        if not hits:
            return f"[{collection}] 无命中。", False

        # Reinforcement（Day 5）+ recency 加权（Day 8）：
        # 实体按 mentions 强化 + last_seen_at 新近度重排；
        # 知识按 mentions（通常无）+ created_at 新近度重排。
        if collection == "entities":
            from ..entities import rerank_hits
            hits = rerank_hits(hits, recency_field="last_seen_at")
        elif collection == "knowledge_claims":
            from ..entities import rerank_hits
            hits = rerank_hits(hits, recency_field="created_at")

        lines = []
        for h in hits:
            payload = h.get("payload", {})
            hit_id = h.get("id", "")
            # 收集证据来源
            if collection == "knowledge_claims":
                self._evidence_rule_ids.append(payload.get("business_rule_id", hit_id))
                tid = payload.get("trace_id", "")
                if tid and tid not in self._evidence_trace_ids:
                    self._evidence_trace_ids.append(tid)
            # 格式化（带 id，方便 agent 后续 expand_graph 引用）
            text = h.get("text", "") or payload.get("summary", "") or payload.get("description", "")
            name = payload.get("name") or payload.get("title") or ""
            # Day 6a 矛盾消解：知识条目召回时把"接口"和"时间"露在文本里给 LLM 看。
            # 配合 ASK_AGENT_SYSTEM 的消解指引（同接口多条按时间新旧判断）→ LLM 看到
            # 例如 "[接口=demo:POST:/order 时间=2026-06-01]" 就能识别"这是同一接口的两版本结论"。
            # 不在召回时按接口去重，是因为同接口的"互补结论"也要保留，由 LLM 判断。
            meta = ""
            if collection == "knowledge_claims":
                iface = payload.get("interface_key", "")
                created = payload.get("created_at", "")
                meta = f" [接口={iface} 时间={created}]" if (iface or created) else ""
            lines.append(f"- id={hit_id} {name} {text}{meta}".strip())

        return f"[{collection}] 命中 {len(hits)} 条:\n" + "\n".join(lines), True

    def _expand_graph(self, node_id: str) -> tuple[str, bool]:
        """从实体/社区节点出发，展开关联接口。"""
        if not self.graph or not node_id:
            return "无法展开（图存储不可用或 node_id 为空）。", False

        lines: list[str] = []
        try:
            if node_id.startswith("entity:"):
                # 实体 → 提到它的接口
                interfaces = self.graph.reverse_neighbors("Entity", node_id, "MENTIONS")
                for iface in interfaces[:EXPAND_LIMIT]:
                    path = iface.get("path", iface.get("id", ""))
                    lines.append(f"接口 {path}")
            elif node_id.startswith("community:"):
                # 社区 → 成员实体 → 接口
                members = self.graph.reverse_neighbors("Community", node_id, "BELONGS_TO")
                for member in members[:EXPAND_LIMIT]:
                    member_name = member.get("name", member.get("id", ""))
                    lines.append(f"成员实体 {member_name}")
        except Exception:
            logger.exception("图展开失败: %s", node_id)
            return f"展开 {node_id} 出错。", False

        if not lines:
            return f"{node_id} 没有关联节点。", False
        return f"{node_id} 的关联:\n" + "\n".join(lines), True

    # ------------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------------

    def _finalize(self, answer: str, any_retrieval_hit: bool) -> AskResponse:
        """组装 AskResponse。召回全空时覆盖为兜底答案，防止幻觉。"""
        if not any_retrieval_hit:
            answer = "暂无足够证据回答该问题。"

        evidence = Evidence(
            trace_ids=list(dict.fromkeys(self._evidence_trace_ids)),
            business_rule_ids=list(dict.fromkeys(self._evidence_rule_ids)),
        )
        return AskResponse(
            answer=answer.strip(),
            evidence=evidence,
            sections={"answer": [answer.strip()]},
        )
