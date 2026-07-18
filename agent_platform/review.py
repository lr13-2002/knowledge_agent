"""人工审核服务。

提供知识提案的审核流程：
    查看提案 → 提问讨论 → 批准/驳回/修订

审核对话由 LLM 驱动（有 LLM 时），智能体会：
1. 先亮出自己的理解和不确定点
2. 回应用户的质疑并自我修正
3. 对齐后等待用户 approve

审核通过的知识会被写入正式知识库（向量 + 图），
后续 AskService 可以检索到，LLM 也会将其作为正样本参考。
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from .loader import RAGLoader
from .schemas import Evidence, ReviewMessage

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = """\
你是一个知识库审核助手。用户正在审核一条由 AI 产出的业务理解提案。

你的职责：
1. 回答用户对提案的疑问，基于 trace 和代码证据回应
2. 如果用户指出错误，承认并提出修正方案
3. 主动标出你不确定的点，请用户确认
4. 对话结束时总结修正后的理解，等待用户 approve

使用中文回复，简洁直接。"""


# Reject 反馈环：同一提案最多被 LLM 反思重生成几次（超过则 reject 终态）
MAX_REJECT_RETRY = 3


class ReviewService:
    """知识提案审核服务。"""

    def __init__(
        self,
        proposals: Any,
        loader: RAGLoader,
        propagator: Any = None,
        llm_client: Any = None,
        regen_client: Any = None,
    ) -> None:
        self.proposals = proposals
        self.loader = loader
        self.propagator = propagator
        self._llm = llm_client  # OpenAI 兼容客户端（对话式审核用）
        # 结构化重生成客户端（AnthropicLLMClient，有 regenerate_from_feedback 方法）
        # reject 反馈环用它基于驳回理由重生成提案；为 None 时 reject 退化为纯打标签
        self._regen = regen_client

    def _get_llm(self) -> Any:
        """懒加载 LLM 客户端（避免无 key 时初始化报错）。"""
        if self._llm is not None:
            return self._llm
        if os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import openai
                from .llm.config import LLMConfig
                config = LLMConfig.load()
                self._llm = openai.OpenAI(api_key=config.api_key, base_url=config.base_url or None)
                return self._llm
            except Exception:
                pass
        return None

    def get(self, proposal_id: str) -> dict:
        """获取提案详情。"""
        proposal = self.proposals.get(proposal_id)
        return self.proposals.as_payload(proposal)

    def message(self, proposal_id: str, content: str) -> ReviewMessage:
        """向提案发送消息（对话式审核）。

        有 LLM 时：调 LLM 基于 proposal 上下文生成回复
        无 LLM 时：回退到模板拼接
        """
        proposal = self.proposals.get(proposal_id)
        user_msg = ReviewMessage(role="user", content=content)
        self.proposals.add_message(proposal_id, user_msg)

        # 尝试用 LLM 生成回复
        llm = self._get_llm()
        if llm:
            answer = self._llm_reply(llm, proposal, content)
        else:
            answer = self._template_reply(proposal)

        evidence = proposal.evidence
        assistant_msg = ReviewMessage(role="assistant", content=answer, evidence=evidence)
        self.proposals.add_message(proposal_id, assistant_msg)
        return assistant_msg

    def _llm_reply(self, llm: Any, proposal: Any, user_content: str) -> str:
        """用 LLM 生成审核对话回复。"""
        from .llm.config import LLMConfig
        config = LLMConfig.load()

        # 构建对话上下文
        proposal_context = (
            f"## 当前提案\n"
            f"- 接口: {proposal.interface_key}\n"
            f"- 摘要: {proposal.summary}\n"
            f"- 流程: {' → '.join(proposal.flow_steps[:5])}\n"
            f"- 结论: {'; '.join(proposal.candidate_claims[:5])}\n"
            f"- 置信度: {proposal.confidence} ({proposal.confidence_score:.2f})\n"
            f"- 证据 trace: {', '.join(proposal.evidence.trace_ids)}\n"
            f"- 关联代码: {', '.join(proposal.evidence.code_symbols[:5])}\n"
        )

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": proposal_context + f"\n## 用户问题\n{user_content}"},
        ]

        try:
            response = llm.chat.completions.create(
                model=config.model,
                max_tokens=1024,
                temperature=0.0,
                messages=messages,
            )
            return response.choices[0].message.content or self._template_reply(proposal)
        except Exception as exc:
            logger.warning("审核对话 LLM 调用失败，降级到模板: %s", exc)
            return self._template_reply(proposal)

    def _template_reply(self, proposal: Any) -> str:
        """模板拼接回复（LLM 不可用时的降级方案）。"""
        evidence = proposal.evidence
        parts = [f"该提案基于 trace {', '.join(evidence.trace_ids)}"]
        if evidence.code_symbols:
            parts.append(f"关联代码符号: {', '.join(evidence.code_symbols[:5])}")
        if proposal.flow_steps:
            parts.append(f"流程步骤: {' → '.join(proposal.flow_steps[:5])}")
        if proposal.candidate_claims:
            parts.append(f"候选结论: {'; '.join(proposal.candidate_claims[:3])}")
        parts.append(f"置信度: {proposal.confidence}，状态: {proposal.status}")
        parts.append("如确认无误请 approve，需修改请 revise。")
        return "。".join(parts)

    def approve(self, proposal_id: str) -> dict:
        """批准提案 — 知识正式入库，可被检索，并触发置信度传播。

        Day 4 新增 Approve 结晶：如果审核过程中有对话讨论（review_messages 非空）且有
        regen 客户端，先让 LLM 把"原提案 + 对话"蒸馏成精炼版再入库，把讨论的改进固化进去。
        没有对话则直接入库原版（不重复加工，防 over-processing 漂移）。
        """
        # 先尝试结晶（仅当有对话 + 有 LLM）
        self._maybe_crystallize(proposal_id)

        proposal = self.proposals.update_status(proposal_id, "approved")
        self.loader.load_approved_knowledge(asdict(proposal))
        if self.propagator:
            self.propagator.propagate(proposal_id)
        return self.proposals.as_payload(proposal)

    def _maybe_crystallize(self, proposal_id: str) -> None:
        """approve 前的结晶：有对话讨论时把改进固化进提案。失败则保持原版。"""
        if self._regen is None:
            return
        # 读对话记录，没有则不结晶
        if not hasattr(self.proposals, "get_messages"):
            return
        messages = self.proposals.get_messages(proposal_id)
        if not messages:
            return

        try:
            proposal = self.proposals.get(proposal_id)
            raw = self._regen.crystallize_from_discussion(proposal, messages)
            # 映射回提案（只更新 LLM 产出字段，保留 id/repo/trace/evidence）
            proposal.summary = raw.get("summary", proposal.summary)
            proposal.flow_steps = raw.get("flow_steps", proposal.flow_steps)
            proposal.candidate_claims = raw.get("candidate_claims", proposal.candidate_claims)
            new_conf = raw.get("confidence", proposal.confidence)
            proposal.confidence = new_conf
            proposal.confidence_score = {"low": 0.3, "medium": 0.6, "high": 0.85}.get(
                new_conf, proposal.confidence_score
            )
            proposal.reasoning = raw.get("reasoning", proposal.reasoning)
            proposal.version += 1
            self._persist(proposal)
            logger.info("approve 结晶 v%d: %s（基于 %d 条对话）", proposal.version, proposal_id, len(messages))
        except Exception:
            logger.exception("approve 结晶失败，入库原版: %s", proposal_id)

    def reject(self, proposal_id: str, reason: str = "") -> dict:
        """驳回提案 — 触发 Reject 反馈环（Day 3）。

        行为：
        - 记录本次 reject 理由到 reject_history
        - 若满足重生成条件（有 regen 客户端 + 有理由 + 未超重试上限）：
            LLM 基于"原提案 + 历次驳回理由"反思重生成，**原地更新**该提案，
            状态回 pending_review，重新进入审核循环。返回 {status: "regenerated"}
        - 否则（无 LLM / 无理由 / 重试耗尽 / 重生成失败）：
            标记为 rejected 终态，后续可作为 LLM 负样本。返回 {status: "rejected"}

        为什么原地更新而非新建：proposal 按 (repo, trace_id) 唯一，同一 trace 不能有两条。
        """
        proposal = self.proposals.get(proposal_id)

        # 记录驳回理由
        if reason:
            proposal.reject_history.append(reason)

        # 判断是否触发重生成
        can_regen = (
            self._regen is not None
            and bool(reason)
            and proposal.retry_count < MAX_REJECT_RETRY
        )

        if can_regen:
            try:
                regenerated = self._regenerate(proposal)
                return {**self.proposals.as_payload(regenerated), "reject_action": "regenerated"}
            except Exception:
                logger.exception("reject 反馈重生成失败，降级为终态 reject: %s", proposal_id)

        # 终态驳回
        proposal.status = "rejected"
        self._persist(proposal)
        return {**self.proposals.as_payload(proposal), "reject_action": "rejected"}

    def _regenerate(self, proposal: Any) -> Any:
        """基于历次驳回理由，让 LLM 反思重生成提案内容，原地更新。"""
        raw = self._regen.regenerate_from_feedback(proposal, proposal.reject_history)

        # 把重生成结果映射回提案（只更新 LLM 产出的字段，保留 id/repo/trace/evidence）
        proposal.summary = raw.get("summary", proposal.summary)
        proposal.flow_steps = raw.get("flow_steps", proposal.flow_steps)
        proposal.candidate_claims = raw.get("candidate_claims", proposal.candidate_claims)
        new_conf = raw.get("confidence", proposal.confidence)
        proposal.confidence = new_conf
        proposal.confidence_score = {"low": 0.3, "medium": 0.6, "high": 0.85}.get(new_conf, proposal.confidence_score)
        proposal.reasoning = raw.get("reasoning", proposal.reasoning)
        # 反馈环元数据
        proposal.retry_count += 1
        proposal.version += 1
        proposal.status = "pending_review"  # 重新进入审核循环

        self._persist(proposal)
        logger.info("reject 反馈重生成 v%d: %s (retry=%d)", proposal.version, proposal.proposal_id, proposal.retry_count)
        return proposal

    def _persist(self, proposal: Any) -> None:
        """把内存中改过的 proposal 落库。优先用 replace，退化用 update_status。"""
        if hasattr(self.proposals, "replace"):
            self.proposals.replace(proposal)
        else:
            self.proposals.update_status(proposal.proposal_id, proposal.status)

    def revise(self, proposal_id: str, summary: str, claims: list[str]) -> dict:
        """修订提案 — 人工修改 summary 和 claims，版本号 +1。"""
        proposal = self.proposals.revise(proposal_id, summary, claims)
        proposal.evidence.business_rule_ids = []
        return self.proposals.as_payload(proposal)
