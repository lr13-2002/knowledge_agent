"""LLM 客户端 — 基于 OpenAI 兼容协议的结构化 LLM 交互层。

整体架构：
    AgentWorker 调用 llm.propose(context) → 本模块组装 prompt → 调 LLM API
    → LLM 被强制通过 function calling 返回结构化 JSON → 校验 → 组装 KnowledgeProposal

使用 OpenAI SDK（兼容 LiteLLM 代理），支持公司内部代理和多种模型。

MVP 版本：单次 function calling 调用 business_synthesis，所有材料一次性给 LLM。
完整版（后续迭代）：拆为三步 chain（trace_summary → code_association → business_synthesis）。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai  # OpenAI SDK，兼容 LiteLLM 代理

from ..schemas import BusinessEntity, BusinessRelation, Evidence, KnowledgeProposal, OpenClawEvent
from ..worker import HeuristicLLMClient  # 降级方案：LLM 失败时用规则引擎兜底
from .prompts import (
    MVP_PROMPT,
    SYSTEM_PROMPT,
    format_code_hits,
    format_spans,
)
from .community_prompts import COMMUNITY_REPORT_SYSTEM, COMMUNITY_REPORT_TOOL
from .tools import BUSINESS_SYNTHESIS_TOOL
from .validators import VALIDATORS

logger = logging.getLogger(__name__)


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """将 Anthropic 格式的 tool schema 转为 OpenAI 格式。

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


class LLMOutputError(Exception):
    """当 LLM 输出无法满足结构化要求且重试耗尽时抛出。"""
    pass


class AnthropicLLMClient:
    """知识提案生成客户端（通过 OpenAI 兼容协议调用 LLM）。

    核心机制：
    1. function calling 强制结构化 — 通过 tool_choice 让模型必须以 JSON 格式输出
    2. 校验重试 — 输出不合规时，将错误反馈给模型让其修正
    3. 降级兜底 — LLM 完全失败时降级到 HeuristicLLMClient 保证不中断

    配置方式（优先级从高到低）：
        1. 代码传参: AnthropicLLMClient(config=LLMConfig(...))
        2. 配置文件: llm_config.yaml（项目根目录）
        3. 环境变量: ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, LLM_MODEL 等
    """

    def __init__(
        self,
        config: "LLMConfig | None" = None,
        config_path: str | None = None,
        # 以下参数为向后兼容，优先使用 config
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        from .config import LLMConfig

        # 加载配置：传入 config > config_path > llm_config.yaml > 环境变量
        if config is None:
            config = LLMConfig.load(config_path)

        # 参数覆盖（向后兼容直接传参的用法）
        if api_key:
            config.api_key = api_key
        if base_url:
            config.base_url = base_url
        if model:
            config.model = model
        if max_retries is not None:
            config.max_retries = max_retries

        self._config = config
        # 初始化 OpenAI 客户端（兼容 LiteLLM 代理）
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            timeout=config.timeout,
        )
        self._model = config.model
        self._max_retries = config.max_retries
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        # 降级方案：LLM 调用失败时退回到规则引擎
        self._fallback = HeuristicLLMClient()

    def propose(self, context: dict[str, Any]) -> KnowledgeProposal:
        """入口方法，实现 LLMClient Protocol。

        接收 worker 传入的上下文（trace + 代码），调用 LLM 产出结构化提案。
        任何异常都会被捕获并降级到 heuristic，保证 worker 不中断。
        """
        try:
            raw = self._call_mvp(context)
            return self._build_proposal(context, raw)
        except Exception as exc:
            # 降级：LLM 失败时用规则引擎生成基础提案
            logger.warning("LLM chain 失败，降级到 heuristic: %s", exc)
            return self._fallback.propose(context)

    def _call_mvp(self, context: dict[str, Any]) -> dict[str, Any]:
        """MVP 模式：一次性把所有材料给 LLM，产出业务理解。"""
        event: OpenClawEvent = context["event"]
        trace = context["trace"]
        code_hits = context.get("code_hits", [])

        # 组装 user prompt：接口信息 + trace spans + 代码片段
        user_prompt = MVP_PROMPT.format(
            repo=event.repo,
            service=event.service,
            method=event.method,
            path=event.path,
            interface_key=event.interface_key,
            span_count=len(trace.spans),
            spans_text=format_spans(trace.spans),
            code_hit_count=len(code_hits),
            code_hits_text=format_code_hits(code_hits),
        )

        # 调用 LLM，强制通过 business_synthesis function 输出
        return self._call_with_tool(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tool=BUSINESS_SYNTHESIS_TOOL,
            tool_name="business_synthesis",
        )

    def regenerate_from_feedback(
        self,
        proposal: KnowledgeProposal,
        reject_reasons: list[str],
    ) -> dict[str, Any]:
        """Reject 反馈环：基于「上一版提案 + 审核驳回理由」反思重生成。

        reflection 模式 —— 不重新拉 trace，让 LLM 看着自己上一版产出 + 驳回理由产出修正版。
        返回 business_synthesis 的原始 dict（summary/flow_steps/candidate_claims/confidence/...）。

        调用方（ReviewService）负责把返回值映射回 proposal 并原地更新。
        异常向上抛，由调用方决定降级（通常降级为终态 reject）。
        """
        from .prompts import REGENERATE_PROMPT

        reasons_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reject_reasons)) or "（未提供具体理由）"
        user_prompt = REGENERATE_PROMPT.format(
            repo=proposal.repo,
            interface_key=proposal.interface_key,
            prev_summary=proposal.summary,
            prev_flow=" → ".join(proposal.flow_steps[:6]) or "（无）",
            prev_claims="; ".join(proposal.candidate_claims[:6]) or "（无）",
            prev_confidence=proposal.confidence,
            reject_reasons=reasons_text,
        )
        return self._call_with_tool(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tool=BUSINESS_SYNTHESIS_TOOL,
            tool_name="business_synthesis",
        )

    def crystallize_from_discussion(
        self,
        proposal: KnowledgeProposal,
        messages: list[Any],
    ) -> dict[str, Any]:
        """Approve 结晶：把"原提案 + 审核对话"蒸馏成精炼版。

        只在有对话记录时由 ReviewService 调用。把讨论中确认的修正固化进提案。
        返回 business_synthesis 的原始 dict，调用方映射回 proposal 并入库。
        异常向上抛，由调用方降级为"入库原版"。
        """
        from .prompts import CRYSTALLIZE_PROMPT

        # 把对话历史格式化（role: content）
        discussion_lines = []
        for m in messages:
            role = getattr(m, "role", "?")
            content = getattr(m, "content", "")
            discussion_lines.append(f"[{role}] {content}")
        discussion = "\n".join(discussion_lines) or "（无讨论内容）"

        user_prompt = CRYSTALLIZE_PROMPT.format(
            repo=proposal.repo,
            interface_key=proposal.interface_key,
            prev_summary=proposal.summary,
            prev_flow=" → ".join(proposal.flow_steps[:6]) or "（无）",
            prev_claims="; ".join(proposal.candidate_claims[:6]) or "（无）",
            prev_confidence=proposal.confidence,
            discussion=discussion,
        )
        return self._call_with_tool(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tool=BUSINESS_SYNTHESIS_TOOL,
            tool_name="business_synthesis",
        )

    def _call_with_tool(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        """底层封装：调用 LLM + 强制 function calling + 校验 + 重试。

        通过 tool_choice 强制模型只能调用指定 function，保证返回结构化 JSON。

        重试机制:
            1. 模型未返回 function call → 追加提示消息重试
            2. 输出校验失败 → 将错误信息反馈给模型修正
        """
        # 转为 OpenAI 格式的 tool 定义
        openai_tool = _to_openai_tool(tool)
        # 在 messages 前插入 system prompt
        full_messages = [{"role": "system", "content": system}] + messages

        for attempt in range(1 + self._max_retries):
            # 调用 LLM API
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=full_messages,
                tools=[openai_tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            # 情况 1: 模型没有返回 function call
            if not tool_calls:
                if attempt < self._max_retries:
                    full_messages.append({"role": "assistant", "content": choice.message.content or ""})
                    full_messages.append({"role": "user", "content": "请必须调用工具输出结果，不要以文本形式回复。"})
                    continue
                raise LLMOutputError("模型未返回 function call")

            # 提取 function call 的参数（即结构化输出）
            call = tool_calls[0]
            try:
                tool_input = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                if attempt < self._max_retries:
                    full_messages.append(choice.message)
                    full_messages.append({"role": "tool", "tool_call_id": call.id, "content": f"JSON 解析失败: {exc}。请输出合法 JSON。"})
                    continue
                raise LLMOutputError(f"function call 参数 JSON 解析失败: {exc}")

            # 情况 2: 输出校验（检查必填字段、值范围等）
            validator = VALIDATORS.get(tool_name)
            if validator:
                result = validator(tool_input)
                if not result.valid:
                    if attempt < self._max_retries:
                        # 将校验错误反馈给模型，让它修正后重新输出
                        error_msg = "输出校验失败：" + "; ".join(result.errors) + "。请修正后重新调用工具。"
                        full_messages.append(choice.message)
                        full_messages.append({"role": "tool", "tool_call_id": call.id, "content": error_msg})
                        continue
                    raise LLMOutputError(f"校验重试耗尽: {result.errors}")

            # 校验通过，返回结构化结果
            return tool_input

        raise LLMOutputError("重试次数耗尽")

    def summarize_community(self, prompt: str) -> dict[str, Any]:
        """为一个业务社区生成领域级摘要（Phase 3 用）。

        通过 community_report function calling 强制结构化输出。
        失败时抛异常，由调用方降级到模板。
        """
        return self._call_with_tool(
            system=COMMUNITY_REPORT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            tool=COMMUNITY_REPORT_TOOL,
            tool_name="community_report",
        )

    def _build_proposal(self, context: dict[str, Any], raw: dict[str, Any]) -> KnowledgeProposal:
        """将 LLM 的结构化输出组装为 KnowledgeProposal dataclass。

        raw 是 LLM 通过 function calling 返回的 JSON，包含：
            summary, flow_steps, candidate_claims, confidence, reasoning
            entities, relations（Phase 2 新增，可选）
        """
        event: OpenClawEvent = context["event"]
        code_hits = context.get("code_hits", [])
        # 提取代码符号名称作为 evidence
        code_symbols = [hit.get("payload", {}).get("qualified_name", hit.get("id", "")) for hit in code_hits]

        # ============ Phase 2 新增：解析业务实体和关系 ============
        # entities/relations 是可选字段，LLM 没输出时为空列表（向后兼容）
        entities = []
        for ent in raw.get("entities", []) or []:
            entities.append(BusinessEntity(
                name=ent.get("name", ""),
                type=ent.get("type", "other"),
                description=ent.get("description", ""),
                aliases=[ent.get("name", "")],  # 原始名作为初始别名
                source_proposal_ids=[],  # 待 worker 写入时填充
                source_trace_ids=[event.trace_id],
            ))

        relations = []
        for rel in raw.get("relations", []) or []:
            relations.append(BusinessRelation(
                source=rel.get("source", ""),
                target=rel.get("target", ""),
                relation=rel.get("relation", ""),
                description=rel.get("description", ""),
                strength=int(rel.get("strength", 5)),
                source_proposal_ids=[],
            ))

        return KnowledgeProposal(
            repo=event.repo,
            trace_id=event.trace_id,
            interface_key=event.interface_key,
            summary=raw["summary"],
            flow_steps=raw["flow_steps"],
            related_code_symbols=code_symbols,
            candidate_claims=raw["candidate_claims"],
            evidence=Evidence(
                trace_ids=[event.trace_id],
                code_symbols=code_symbols,
                commit=context.get("commit", ""),
            ),
            confidence=raw["confidence"],
            confidence_score={"low": 0.3, "medium": 0.6, "high": 0.85}.get(raw["confidence"], 0.3),
            entities=entities,
            relations=relations,
            reasoning=raw.get("reasoning", ""),
        )
