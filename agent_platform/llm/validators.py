"""tool_use 输出校验器。

LLM 通过 tool_use 返回结构化 JSON 后，这里校验它是否满足业务约束：
- 必填字段非空
- 数组至少有一项
- 枚举值在范围内
- 文本长度合理

校验失败时返回错误列表，调用层会把这些错误信息反馈给模型让其修正重试。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """校验结果：valid=True 表示通过，否则 errors 里有具体原因。"""
    valid: bool = True
    errors: list[str] = field(default_factory=list)


def validate_trace_summary(data: dict) -> ValidationResult:
    """校验 Step 1（链路摘要）的输出。"""
    errors = []
    if not data.get("entry_point"):
        errors.append("entry_point 不能为空")
    chain = data.get("call_chain")
    if not isinstance(chain, list) or len(chain) == 0:
        errors.append("call_chain 至少需要 1 个调用节点")
    elif any(not item.get("service") for item in chain):
        errors.append("call_chain 中每个节点必须有 service 字段")
    if not data.get("trace_pattern"):
        errors.append("trace_pattern 不能为空")
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_code_association(data: dict) -> ValidationResult:
    """校验 Step 2（代码关联）的输出。"""
    errors = []
    symbols = data.get("related_symbols")
    if not isinstance(symbols, list):
        errors.append("related_symbols 必须是数组")
    elif any(not item.get("qualified_name") for item in symbols):
        errors.append("related_symbols 中每项必须有 qualified_name")
    if not data.get("code_to_trace_mapping"):
        errors.append("code_to_trace_mapping 不能为空")
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_business_synthesis(data: dict) -> ValidationResult:
    """校验 Step 3（业务归纳）的输出 — MVP 阶段的核心校验。

    校验规则：
    - summary 非空且 <= 500 字
    - flow_steps 至少 1 个
    - candidate_claims 至少 1 条
    - confidence 在枚举范围内
    - entities（如有）每项必须有 name/type/description
    - relations（如有）source/target 必须在 entities 里出现
    """
    errors = []
    summary = data.get("summary", "")
    if not summary:
        errors.append("summary 不能为空")
    elif len(summary) > 500:
        errors.append(f"summary 过长（{len(summary)}字），请控制在 200 字以内")
    steps = data.get("flow_steps")
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("flow_steps 至少需要 1 个步骤")
    claims = data.get("candidate_claims")
    if not isinstance(claims, list) or len(claims) == 0:
        errors.append("candidate_claims 至少需要 1 条可验证的业务结论")
    confidence = data.get("confidence")
    if confidence not in ("low", "medium", "high"):
        errors.append(f"confidence 必须是 low/medium/high，当前值: {confidence}")

    # entities 校验（可选字段，仅出现时校验）
    entities = data.get("entities")
    entity_names: set[str] = set()
    if entities is not None:
        if not isinstance(entities, list):
            errors.append("entities 必须是数组")
        else:
            for i, ent in enumerate(entities):
                if not isinstance(ent, dict):
                    errors.append(f"entities[{i}] 必须是对象")
                    continue
                if not ent.get("name"):
                    errors.append(f"entities[{i}] 缺少 name")
                if not ent.get("type"):
                    errors.append(f"entities[{i}] 缺少 type")
                if not ent.get("description"):
                    errors.append(f"entities[{i}] 缺少 description")
                if ent.get("name"):
                    entity_names.add(ent["name"])

    # relations 校验（可选字段，source/target 必须在 entities 中）
    relations = data.get("relations")
    if relations is not None:
        if not isinstance(relations, list):
            errors.append("relations 必须是数组")
        else:
            for i, rel in enumerate(relations):
                if not isinstance(rel, dict):
                    errors.append(f"relations[{i}] 必须是对象")
                    continue
                src = rel.get("source")
                tgt = rel.get("target")
                if not src or not tgt:
                    errors.append(f"relations[{i}] 缺少 source 或 target")
                # 引用完整性：关系两端必须在 entities 里声明过
                if entity_names and src and src not in entity_names:
                    errors.append(f"relations[{i}].source '{src}' 未在 entities 中声明")
                if entity_names and tgt and tgt not in entity_names:
                    errors.append(f"relations[{i}].target '{tgt}' 未在 entities 中声明")
                strength = rel.get("strength")
                if strength is not None and not (isinstance(strength, int) and 1 <= strength <= 10):
                    errors.append(f"relations[{i}].strength 必须是 1-10 的整数")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_community_report(data: dict) -> ValidationResult:
    """校验社区报告输出（LLM 为每个社区生成的领域摘要）。"""
    errors = []
    if not data.get("title"):
        errors.append("title 不能为空")
    elif len(data["title"]) > 30:
        errors.append("title 过长，建议 4-10 字")
    if not data.get("summary"):
        errors.append("summary 不能为空")
    findings = data.get("findings")
    if not isinstance(findings, list) or len(findings) == 0:
        errors.append("findings 至少需要 1 条")
    elif findings:
        for i, f in enumerate(findings):
            if not isinstance(f, dict) or not f.get("insight"):
                errors.append(f"findings[{i}] 缺少 insight")
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# 校验器注册表：tool_name → 校验函数
# _call_with_tool 会根据 tool_name 自动查找对应的校验器
VALIDATORS = {
    "trace_summary": validate_trace_summary,
    "code_association": validate_code_association,
    "business_synthesis": validate_business_synthesis,
    "community_report": validate_community_report,
}
