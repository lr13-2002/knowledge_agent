"""Prompt 模板。

设计原则：
- system prompt 定义角色和约束规则（所有步骤共用）
- user prompt 是每次调用的具体输入材料
- 模板使用 Python format string，由 client.py 负责填充变量
- 所有 prompt 最终目的是让 LLM 通过 tool_use 输出结构化 JSON
"""
from __future__ import annotations

# ============================================================================
# System Prompt — 定义 LLM 的角色和行为约束
# 所有步骤共用，确保输出风格一致
# ============================================================================
SYSTEM_PROMPT = """\
你是一个代码与链路分析专家，任务是根据线上 trace 数据和代码片段，产出对接口业务含义的结构化理解。

核心原则：
1. 只基于提供的 trace 和代码片段做判断，不要凭空猜测业务逻辑。
2. 区分"观察到的事实"和"推断的结论"。事实来自 trace span 和代码文本，结论需要标注置信度。
3. 每条 candidate_claim 必须是可通过查看代码或回放 trace 验证的断言，不要输出泛泛而谈的描述。
4. summary 应该是一句话的业务含义，不是技术描述。例如"司机完成活动后发送奖励通知"而不是"调用了 SendNotice 方法"。
5. 使用中文输出。

6. 业务实体抽取（关键）：
   - 在分析过程中，抽取出本 trace 体现的"业务实体"。
   - 业务实体是业务方在评审会上能听懂的概念，例如：订单、用户、优惠券、风控规则、支付网关、对账单、奖励发放。
   - 请不要把 Java/Go 的 Controller、Service、Repository、Helper 这类纯技术词作为实体——除非它在业务上有明确含义（例如 PayGateway 直接代表"支付网关"）。
   - 实体名要稳定：如果两个表达指同一概念（"订单" / "订单数据" / "order entity"），统一用最短的业务名"订单"。
   - 同时抽取实体之间的关系（调用/触发/包含/写入/通知/校验），并给出 1-10 的强度评分。
   - 拿不准是技术还是业务时，宁可不输出。
   - 这些信息会被聚合到全局图中用于做领域聚类，所以"叫什么名字"比"细节多丰富"更重要。"""

# ============================================================================
# MVP Prompt — 单次调用，一次性给出所有材料
# 适用于验证阶段，先跑通再拆步骤
# ============================================================================
MVP_PROMPT = """\
请分析以下接口的 trace 数据和关联代码，产出业务理解提案。

## 接口信息
- 仓库: {repo}
- 服务: {service}
- 方法: {method} {path}
- 接口标识: {interface_key}

## Trace Spans（共 {span_count} 个）
{spans_text}

## 关联代码片段（共 {code_hit_count} 条）
{code_hits_text}

请调用 business_synthesis 工具输出结构化的业务理解提案。

## 实体抽取要求
请在 business_synthesis 的 entities / relations 字段中输出本接口涉及的业务实体与关系。
判断标准：
- 业务方在评审会上会用到的词 → 是实体（订单、支付、风控）
- 只有写代码的人才会念叨的词 → 不是实体（OrderController、PayServiceImpl）
- 拿不准时宁可不输出"""

# ============================================================================
# 三步 Chain 的 Prompt — ⚠️ 当前**未使用**（保留备用）
# ============================================================================
# 当前生产路径是 MVP_PROMPT（一次性调 business_synthesis 一步出结论）。
# 下面三个 STEP_* prompt 是早期设计的"trace_summary → code_association → business_synthesis"
# 三步 chain，对应 tools.py 里同名 3 个 tool schema。
#
# 保留原因：未来如果 MVP 在复杂 trace 上质量不够，可以拆成三步链路。
# 当前不接入：单步在测试场景质量够用，多调一次 LLM 成本翻倍。
# ============================================================================

# Step 1: 链路摘要 — 只看 trace 数据，提炼调用模式
STEP1_TRACE_SUMMARY = """\
请分析以下 trace 数据，提取结构化的链路摘要。

## 接口信息
- 仓库: {repo}
- 服务: {service}
- 方法: {method} {path}
- 接口标识: {interface_key}

## Trace Spans（共 {span_count} 个）
{spans_text}

请调用 trace_summary 工具输出结构化摘要。"""

# Step 2: 代码关联 — 拿 step1 的摘要 + 代码片段，做代码↔trace 映射
STEP2_CODE_ASSOCIATION = """\
请基于链路摘要和代码片段，识别关联的代码符号。

## 链路摘要（上一步产出）
- 入口: {entry_point}
- 调用链: {call_chain_text}
- 链路模式: {trace_pattern}

## 代码片段（向量检索命中，共 {code_hit_count} 条）
{code_hits_text}

请调用 code_association 工具输出代码关联分析。"""

# Step 3: 业务归纳 — 综合 step1 + step2 + 历史反馈，产出最终结论
STEP3_BUSINESS_SYNTHESIS = """\
请综合以下信息，产出最终的业务理解提案。

## 链路摘要
- 入口: {entry_point}
- 调用链: {call_chain_text}
- 错误: {error_flags_text}
- 链路模式: {trace_pattern}

## 代码关联分析
{code_analysis_text}

{feedback_text}\
请调用 business_synthesis 工具输出最终的业务理解提案。注意：
- summary 必须是业务含义而非技术描述
- candidate_claims 每条必须是可验证的断言
- confidence 请根据证据充分程度诚实评估"""


# ============================================================================
# Reject 反馈环 — 基于审核人反馈反思重生成（Day 3 新增）
# ============================================================================
# reflection 模式：不重新拉 trace，而是让 LLM 看着"自己上一版的产出 + 审核人为什么驳回"，
# 产出一个修正版。适合审核人指出了具体问题（"把订单说成了乘客""漏了风控"）的场景。
REGENERATE_PROMPT = """\
你之前对某个接口产出了一份业务理解提案，但审核人驳回了它。
请根据审核反馈，重新产出一份修正后的提案。

## 接口信息
- 仓库: {repo}
- 接口标识: {interface_key}

## 你上一版的提案（被驳回）
- 业务摘要: {prev_summary}
- 调用流程: {prev_flow}
- 业务结论: {prev_claims}
- 置信度: {prev_confidence}

## 审核人的驳回理由（按时间顺序，最后一条是最新的）
{reject_reasons}

## 要求
- 针对审核反馈做**实质性修正**，不要原样重复上一版
- 如果反馈指出了事实错误，纠正它
- 如果反馈说证据不足，下调置信度
- 仍然只基于你已知的信息，不要凭空编造新的 trace 细节
- 调用 business_synthesis 工具输出修正后的提案"""


# ============================================================================
# Approve 结晶 — 把审核对话的改进固化进提案（Day 4 新增）
# ============================================================================
# 审核人和 LLM 在 message 阶段讨论修正了某些点，approve 时把"原提案 + 对话"
# 蒸馏成精炼版再入库，否则对话讨论的改进会被丢弃。
# 只在有对话记录时触发（无讨论的提案已是 LLM 一次产出的结果，不重复加工，防漂移）。
CRYSTALLIZE_PROMPT = """\
一份业务理解提案即将通过审核入库。审核过程中，审核人和你有过讨论。
请把"原提案 + 审核讨论"中达成的最终共识，蒸馏成一份精炼的提案。

## 接口信息
- 仓库: {repo}
- 接口标识: {interface_key}

## 原提案
- 业务摘要: {prev_summary}
- 调用流程: {prev_flow}
- 业务结论: {prev_claims}
- 置信度: {prev_confidence}

## 审核讨论记录（按时间顺序）
{discussion}

## 要求
- 把讨论中确认的修正、补充、纠错**固化**进提案
- 如果讨论只是确认无误（没有实质修改），保持原提案核心内容不变
- 不要引入讨论里没有的新结论
- 调用 business_synthesis 工具输出精炼后的提案"""


# ============================================================================
# 辅助函数 — 将数据结构格式化为 prompt 文本
# ============================================================================

def format_spans(spans: list) -> str:
    """将 TraceSpan 列表格式化为可读的文本，供 LLM 阅读。

    输出格式: 序号. 服务名 | 操作名 | 方法 路径 | 耗时 [ERROR]
    """
    lines = []
    for i, span in enumerate(spans, 1):
        # 兼容 dataclass 和 dict 两种输入
        if isinstance(span, dict):
            svc = span.get("service", "")
            name = span.get("name", "")
            method = span.get("method", "")
            path = span.get("path", "")
            duration = span.get("duration", "")
            has_error = span.get("has_error", False)
        else:
            svc = span.service
            name = span.name
            method = span.method
            path = span.path
            duration = span.duration
            has_error = span.has_error
        error_mark = " [ERROR]" if has_error else ""
        lines.append(f"  {i}. {svc} | {name} | {method} {path} | {duration}{error_mark}")
    return "\n".join(lines) if lines else "  （无 span 数据）"


def format_code_hits(code_hits: list[dict]) -> str:
    """将向量检索命中的代码片段格式化为 markdown 代码块。

    每个命中项显示：qualified_name + 代码文本（截断到 500 字符防止 token 爆炸）
    """
    lines = []
    for i, hit in enumerate(code_hits, 1):
        name = hit.get("payload", {}).get("qualified_name", hit.get("id", "unknown"))
        text = hit.get("text", "")[:500]  # 截断防止单个片段过长
        lines.append(f"### 片段 {i}: {name}\n```\n{text}\n```")
    return "\n\n".join(lines) if lines else "（无关联代码）"
