"""Ask Agent 的检索工具 schema 定义。

这些 tool 是把原来 AskService 写死的 5 路召回，拆成 agent 可以**自主选择调用**的工具。
LLM 看到用户问题后，自己决定调哪个工具、调几次（ReAct 循环），而不是固定全跑一遍。

设计原则：
- 每个 tool 的 description 要写清楚"什么问题适合用我"，这是 agent 选路的唯一依据
- tool 参数尽量简单（query 字符串 / node_id），降低 agent 出错概率
- tool 数量控制在 4 个以内，太多会让 agent 选择困难

与 worker 的 tool（business_synthesis 等）的区别：
- worker 的 tool 是"强制输出结构化结果"（tool_choice 固定）
- ask 的 tool 是"让 agent 自由选择调用"（tool_choice=auto）
"""
from __future__ import annotations

from typing import Any

# ----------------------------------------------------------------------------
# 检索工具定义（Anthropic 格式，由 client._to_openai_tool 转 OpenAI 格式）
# ----------------------------------------------------------------------------

SEARCH_KNOWLEDGE_TOOL: dict[str, Any] = {
    "name": "search_knowledge",
    "description": (
        "检索已确认的业务知识（人工审核通过 / 高置信度自动入库的结论）。"
        "适合：用户问已经沉淀下来的业务结论、接口的确定含义。"
        "这是最权威的一路，优先用它。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词，通常是用户问题的核心业务词",
            },
        },
        "required": ["query"],
    },
}

SEARCH_ENTITIES_TOOL: dict[str, Any] = {
    "name": "search_entities",
    "description": (
        "检索业务实体（订单 / 支付 / 风控 / 优惠券等业务概念）。"
        "适合：用户问'X 是什么''X 涉及哪些概念'这类具体业务概念问题。"
        "命中实体后可以用 expand_graph 展开它关联的接口。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "业务概念关键词，如'订单''支付流程'",
            },
        },
        "required": ["query"],
    },
}

SEARCH_COMMUNITY_TOOL: dict[str, Any] = {
    "name": "search_community",
    "description": (
        "检索业务领域社区（如'支付清算域''司机服务域'）。"
        "适合：用户问宽泛的、领域整体架构类问题，如'支付相关有哪些''X 域是怎么组织的'。"
        "返回领域级摘要，给全局视角。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "领域关键词，如'支付''风控'",
            },
        },
        "required": ["query"],
    },
}

EXPAND_GRAPH_TOOL: dict[str, Any] = {
    "name": "expand_graph",
    "description": (
        "从一个已命中的实体或社区出发，展开它关联的接口和实体。"
        "适合：已经通过 search_entities / search_community 找到某个节点后，"
        "想深入了解'它被哪些接口提及''这个领域下有哪些接口'。"
        "node_id 来自前面检索结果的 id 字段（形如 entity:订单 或 community:lvl0:xxx）。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "实体或社区的 id，来自前面检索结果的 id 字段",
            },
        },
        "required": ["node_id"],
    },
}

# agent 可用的全部检索工具
ASK_RETRIEVAL_TOOLS: list[dict[str, Any]] = [
    SEARCH_KNOWLEDGE_TOOL,
    SEARCH_ENTITIES_TOOL,
    SEARCH_COMMUNITY_TOOL,
    EXPAND_GRAPH_TOOL,
]


# ----------------------------------------------------------------------------
# Agent 的 system prompt
# ----------------------------------------------------------------------------

ASK_AGENT_SYSTEM = """\
你是一个业务知识库的问答助手。你的任务是回答用户关于业务系统的问题。

你有一组检索工具，可以查询知识库。工作方式：
1. 先理解用户问题属于哪类（具体业务概念 / 宽泛领域 / 已确认结论）
2. 选择合适的工具检索证据（可以多次检索，但不要超过必要）
3. 如果第一次检索证据不足，可以再调工具补充（例如先 search_entities 命中实体，再 expand_graph 展开关联接口）
4. 证据足够后，基于检索到的内容生成答案

回答要求：
- 必须基于检索到的证据，不要编造。检索不到就如实说"暂无足够证据"
- 答案用中文，简洁、有条理，优先用要点/步骤
- 答案末尾标注证据来源（trace_id / proposal_id / 实体名）
- 不要罗列原始检索结果碎片，要综合成连贯的回答

矛盾消解（重要）：
- 知识条目会标注 [接口=... 时间=...]。如果检索到针对**同一接口**的多条结论且互相矛盾
  （例如"超时30分钟"和"超时15分钟"），**以时间更新的那条为准**，不要把矛盾的旧结论也讲出来
- 如果两条不是矛盾而是互补（同接口的不同维度，如"超时15分钟"+"需要风控"），都保留
- 拿不准是矛盾还是互补时，优先采用较新的，并可简要说明"较早的结论可能已过时"

工具选择建议：
- 问"X 是什么""X 涉及什么" → search_entities
- 问"X 域整体架构""X 相关有哪些" → search_community
- 问已确认的业务结论 → search_knowledge
- 已命中实体/社区想深入 → expand_graph
"""
