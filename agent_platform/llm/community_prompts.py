"""社区报告 prompt 和 tool schema。

参考 Microsoft GraphRAG 的 community report 设计：
- 每个社区代表一个业务领域
- LLM 根据社区内的实体、关系、关联接口，生成领域级摘要
- title/summary/findings 三层结构，由粗到细
"""
from __future__ import annotations

from typing import Any

# tool schema：强制 LLM 输出结构化的社区报告
COMMUNITY_REPORT_TOOL: dict[str, Any] = {
    "name": "community_report",
    "description": "为一个业务领域社区生成领域级摘要报告。",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "4-10 字的领域名，如'支付清算域'、'订单履约域'。不要叫'领域A'之类无意义的名字。",
            },
            "summary": {
                "type": "string",
                "description": "一段话（不超过 300 字）描述这个领域在做什么、谁在调谁、关键流程是什么。",
            },
            "findings": {
                "type": "array",
                "description": "3-5 条要点，每条由一句话洞见和 1-2 句解释组成。",
                "items": {
                    "type": "object",
                    "properties": {
                        "insight": {
                            "type": "string",
                            "description": "一句话核心洞见，如'支付与风控强耦合'",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "1-2 句解释，引用社区内的实体或关系作为依据",
                        },
                    },
                    "required": ["insight", "explanation"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "对该领域刻画的置信度",
            },
        },
        "required": ["title", "summary", "findings", "confidence"],
    },
}


COMMUNITY_REPORT_SYSTEM = """\
你是某代码仓库的领域分析专家。你的任务是根据一组在调用链中被频繁一起提及的业务实体，
判断它们是否构成了一个业务领域，并给出该领域的摘要。

核心原则：
1. 不要凭空发明实体里没有的概念
2. title 要能让业务方一眼看懂（如"支付清算域"），不要叫"领域A"
3. summary 要描述这个领域在做什么、有哪些关键流程
4. findings 要引用具体的实体或关系作为依据，不要泛泛而谈
5. 使用中文输出"""


COMMUNITY_REPORT_PROMPT = """\
下面是一组在调用链中被频繁一起提及的业务实体，它们可能共同构成了一个业务领域。
请基于这些实体和它们之间的关系，给出该领域的摘要。

## 实体（共 {n_entities} 个）
{entities_text}

## 实体间关系
{relations_text}

## 关联接口（按提及次数前 {n_interfaces} 个）
{interfaces_text}

请调用 community_report 工具输出该领域的报告。"""


def format_entities_for_community(entities: list[dict[str, Any]], limit: int = 30) -> str:
    """将实体列表格式化为 prompt 可读的文本。

    格式: name(type) — description — 提及{mentions}次
    """
    lines = []
    sorted_entities = sorted(entities, key=lambda e: -int(e.get("mentions", 1)))[:limit]
    for ent in sorted_entities:
        name = ent.get("name", "?")
        type_ = ent.get("type", "?")
        desc = ent.get("description", "")[:80]
        mentions = ent.get("mentions", 1)
        lines.append(f"  - {name}({type_}) — {desc} — 提及{mentions}次")
    return "\n".join(lines) if lines else "  （无）"


def format_relations_for_community(relations: list[dict[str, Any]], limit: int = 50) -> str:
    """将关系列表格式化为 prompt 可读的文本。"""
    lines = []
    for rel in relations[:limit]:
        src = rel.get("source", "?")
        tgt = rel.get("target", "?")
        relation = rel.get("relation", "")
        strength = rel.get("strength", rel.get("weight", 5))
        lines.append(f"  - {src} --[{relation}, strength={strength}]--> {tgt}")
    return "\n".join(lines) if lines else "  （无）"


def format_interfaces_for_community(interfaces: list[dict[str, Any]], limit: int = 10) -> str:
    """将关联接口格式化为文本。"""
    lines = []
    for iface in interfaces[:limit]:
        path = iface.get("path", iface.get("id", "?"))
        method = iface.get("method", "")
        repo = iface.get("repo", "")
        lines.append(f"  - {method} {path} ({repo})")
    return "\n".join(lines) if lines else "  （无）"
