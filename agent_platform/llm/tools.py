"""tool_use schema 定义。

这些 schema 定义了 LLM 必须按什么格式输出结果。
通过 Anthropic API 的 tool_choice 参数，强制模型只能调用指定的 tool，
从而保证输出永远是符合 schema 的结构化 JSON，而不是自由文本。

三个 tool 对应知识生成的三个阶段：
    1. trace_summary — 从原始 span 数据提取链路摘要
    2. code_association — 将代码符号与 trace 调用关联
    3. business_synthesis — 综合产出最终的业务理解（MVP 阶段直接用这个）
"""
from __future__ import annotations

from typing import Any

# ============================================================================
# Tool 1: trace_summary（Step 1 链路摘要）
# 输入：trace 的 span 列表
# 输出：结构化的调用链摘要
# ============================================================================
TRACE_SUMMARY_TOOL: dict[str, Any] = {
    "name": "trace_summary",
    "description": "从 trace 的 span 列表中提取结构化的链路摘要。",
    "input_schema": {
        "type": "object",
        "properties": {
            "entry_point": {
                "type": "string",
                "description": "入口接口，格式: METHOD /path",
            },
            "call_chain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},  # 服务名
                        "operation": {"type": "string"},  # 操作名/方法名
                        "is_error": {"type": "boolean"},  # 是否报错
                    },
                    "required": ["service", "operation", "is_error"],
                },
                "description": "按调用顺序排列的服务调用链",
            },
            "error_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "有错误的 span 的服务名或操作名",
            },
            "trace_pattern": {
                "type": "string",
                "description": "一句话描述这条 trace 的整体调用模式",
            },
        },
        "required": ["entry_point", "call_chain", "error_flags", "trace_pattern"],
    },
}

# ============================================================================
# Tool 2: code_association（Step 2 代码关联）
# 输入：step1 的链路摘要 + 向量检索命中的代码片段
# 输出：代码符号与 trace 的映射关系
# ============================================================================
CODE_ASSOCIATION_TOOL: dict[str, Any] = {
    "name": "code_association",
    "description": "基于代码片段和链路摘要，识别 trace 关联的代码符号和调用关系。",
    "input_schema": {
        "type": "object",
        "properties": {
            "related_symbols": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "qualified_name": {
                            "type": "string",
                            "description": "完整限定名如 pkg.Class.Method",
                        },
                        "role": {
                            "type": "string",
                            # 角色分类：入口处理器/下游调用方/工具函数/数据访问/未知
                            "enum": ["entry_handler", "downstream_caller", "utility", "data_access", "unknown"],
                        },
                        "relevance": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],  # 与本次 trace 的相关度
                        },
                    },
                    "required": ["qualified_name", "role", "relevance"],
                },
            },
            "code_to_trace_mapping": {
                "type": "string",
                "description": "简要说明代码符号与 trace span 之间的对应关系",
            },
        },
        "required": ["related_symbols", "code_to_trace_mapping"],
    },
}

# ============================================================================
# Tool 3: business_synthesis（Step 3 业务归纳 / MVP 直接使用）
# 输入：step1 + step2 的产物 + 历史反馈（如有）
# 输出：最终的业务理解提案，对应 KnowledgeProposal 的核心字段
# ============================================================================
BUSINESS_SYNTHESIS_TOOL: dict[str, Any] = {
    "name": "business_synthesis",
    "description": "综合链路摘要和代码分析，产出最终的业务理解提案。",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                # 核心产出：一句话描述接口的业务含义（不是技术描述）
                "description": "一句话业务含义（不超过200字），描述该接口的业务目的和核心流程",
            },
            "flow_steps": {
                "type": "array",
                "items": {"type": "string"},
                # 按顺序描述调用链中每一步在做什么业务动作
                "description": "按顺序排列的调用步骤描述，每步是一个自然语言句子",
            },
            "candidate_claims": {
                "type": "array",
                "items": {"type": "string"},
                # 可验证的断言，审核时可以回代码/trace 确认真伪
                "description": "可验证的业务结论，每条应当是可通过代码或 trace 验证的断言",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                # high=trace+代码双重证据  medium=单一来源  low=推测
                "description": "high=代码+trace双重证据，medium=单一证据来源，low=推测",
            },
            "reasoning": {
                "type": "string",
                # 为什么给出这个置信度，方便人工审核时判断
                "description": "置信度判断的理由",
            },
            # ============ Phase 2 新增：业务实体抽取 ============
            "entities": {
                "type": "array",
                "description": (
                    "本 trace 体现的业务实体（不是技术类名）。"
                    "包括业务概念（订单/优惠券）、外部服务（支付网关）、"
                    "数据实体（账户）、领域事件（下单成功）等。"
                    "技术工具类（Controller/Service/Repository/Helper）不要提取。"
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "实体名（中文优先，2-8字）。同一概念用同一词，如总用'订单'不混用'订单单据'",
                        },
                        "type": {
                            "type": "string",
                            "description": "实体类型：business_concept / external_service / data_entity / actor / event / capability / other",
                        },
                        "description": {
                            "type": "string",
                            "description": "1-2 句话说明该实体在本接口的角色",
                        },
                    },
                    "required": ["name", "type", "description"],
                },
            },
            "relations": {
                "type": "array",
                "description": (
                    "实体之间的业务关系。只在确实存在调用/依赖/包含/触发关系时输出。"
                    "source 和 target 必须是 entities 中已声明的实体名。"
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "源实体名"},
                        "target": {"type": "string", "description": "目标实体名"},
                        "relation": {
                            "type": "string",
                            "description": "关系类型：调用/触发/包含/写入/依赖/通知/校验",
                        },
                        "description": {"type": "string", "description": "一句话描述这个关系"},
                        "strength": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "10=核心因果，5=支撑步骤，1=偶发",
                        },
                    },
                    "required": ["source", "target", "relation", "description", "strength"],
                },
            },
        },
        # entities/relations 不加入 required，向后兼容降级
        "required": ["summary", "flow_steps", "candidate_claims", "confidence", "reasoning"],
    },
}
