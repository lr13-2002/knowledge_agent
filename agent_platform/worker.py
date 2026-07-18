"""核心 Worker — 知识生产的主循环。

AgentWorker 是整个平台的核心引擎，负责：
1. 从任务队列中消费 trace 事件
2. 获取 trace 原始数据（调用 observe API）
3. 检索关联代码（向量检索）
4. 调用 LLM 产出业务理解提案
5. 将结果存入图和提案库

运行模式：后台守护线程，持续轮询队列。
"""
from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import asdict
from typing import Any, Protocol

from . import entities as entities_mod
from .indexer import CodeIndexer
from .ingestor import STREAM_NAME
from .loader import RAGLoader
from .schemas import Evidence, KnowledgeProposal, OpenClawEvent
from .stores import InMemoryRawArtifactStore, InMemoryTaskQueue
from .trace import TraceProvider, normalize_trace

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """LLM 客户端协议 — 所有 LLM 实现必须遵循此接口。

    只需实现 propose() 方法，接收上下文返回 KnowledgeProposal。
    当前有两个实现：
        - HeuristicLLMClient（规则引擎，不调 LLM）
        - AnthropicLLMClient（真实 Claude API 调用）
    """
    def propose(self, context: dict[str, Any]) -> KnowledgeProposal:
        ...


class HeuristicLLMClient:
    """规则引擎实现 — 不调 LLM，基于 trace 数据直接拼接提案。

    用途：
    1. 开发/测试时不消耗 API 额度
    2. 作为 AnthropicLLMClient 的降级方案
    3. 快速验证流水线连通性
    """
    def propose(self, context: dict[str, Any]) -> KnowledgeProposal:
        event: OpenClawEvent = context["event"]
        trace = context["trace"]
        # 从向量检索结果中提取代码符号名
        code_symbols = [item["payload"].get("qualified_name", item["id"]) for item in context.get("code_hits", [])]
        # 构建调用流程步骤
        flow_steps = [f"入口接口 {event.method} {event.path}"]
        flow_steps.extend([f"调用下游 {name}" for name in trace.downstream])
        if trace.errors:
            flow_steps.append("存在错误 span: " + ", ".join(trace.errors))
        # 生成基础摘要（纯模板，不做推理）
        summary = f"{event.interface_key} 的 trace {event.trace_id} 覆盖 {len(trace.spans)} 个 span。"
        claims = [f"{event.path} 可能关联服务: " + ", ".join(trace.downstream)] if trace.downstream else []
        return KnowledgeProposal(
            repo=event.repo,
            trace_id=event.trace_id,
            interface_key=event.interface_key,
            summary=summary,
            flow_steps=flow_steps,
            related_code_symbols=code_symbols,
            candidate_claims=claims,
            evidence=Evidence(trace_ids=[event.trace_id], code_symbols=code_symbols, commit=context.get("commit", "")),
            confidence="medium" if code_symbols else "low",
        )


class AgentWorker:
    """知识生产 Worker — 消费队列，调用 LLM，产出知识提案。

    架构位置：
        TaskQueue → AgentWorker → ProposalStore
                                → GraphStore（写入调用关系）
    """

    def __init__(
        self,
        queue: Any,  # 任务队列
        trace_provider: TraceProvider,  # trace 数据来源
        vector: Any,  # 向量存储（代码检索）
        graph: Any,  # 图存储（写入调用关系）
        proposals: Any,  # 提案存储
        raw_artifacts: Any | None = None,  # 原始 trace 存储
        llm: LLMClient | None = None,  # LLM 客户端（None 时用规则引擎）
        indexer: CodeIndexer | None = None,  # 代码索引器
        loader: RAGLoader | None = None,  # RAG 数据加载器
        repo_roots: dict[str, str] | None = None,  # 仓库名→本地路径映射
    ) -> None:
        self.queue = queue
        self.trace_provider = trace_provider
        self.vector = vector
        self.graph = graph
        self.proposals = proposals
        self.raw_artifacts = raw_artifacts or InMemoryRawArtifactStore()
        self.llm = llm or HeuristicLLMClient()
        self.indexer = indexer or CodeIndexer()
        self.loader = loader or RAGLoader(vector, graph)
        self.repo_roots = repo_roots or {}
        # 仓库 → 上次索引时的 git commit。按 commit 而非仅 repo 名追踪，
        # 这样 git pull 后 commit 变化能被检测到并触发重索引（Day 7）。
        self._indexed_commits: dict[str, str] = {}
        self._stop = threading.Event()  # 停止信号

    def _ensure_repo_indexed(self, repo: str) -> None:
        """确保仓库已被代码索引。首次遇到时索引；commit 变化时重索引。"""
        repo_root = self.repo_roots.get(repo)
        if not repo_root:
            return
        from .indexer.index import _git_head
        current_commit = _git_head(repo_root)
        # commit 没变（且已索引过）→ 跳过，不重复解析整个仓库
        # 注意：非 git 仓库 _git_head 返回 ""，第一次会进入索引（_indexed_commits[repo]=""），
        # 之后每次比对 ""=="" 也跳过——可接受（非 git 场景索引不刷新，要刷新得删 key）。
        if self._indexed_commits.get(repo) == current_commit and repo in self._indexed_commits:
            return
        code_index = self.indexer.index_repo(repo_root, repo)
        self.loader.load_code_index(code_index.as_dict())
        self._indexed_commits[repo] = current_commit
        logger.info(
            "indexed repo %s (%d chunks, commit=%s)",
            repo, len(code_index.chunks), (current_commit[:8] if current_commit else "none"),
        )

    def refresh_all_repos(self) -> int:
        """轮询所有已配置仓库，commit 变化的重索引。返回实际重索引的仓库数。

        被 15min 定时线程调用。复用 _ensure_repo_indexed 的 commit 检测逻辑。
        """
        refreshed = 0
        for repo, repo_root in self.repo_roots.items():
            try:
                from .indexer.index import _git_head
                before = self._indexed_commits.get(repo)
                current = _git_head(repo_root)
                if before == current and repo in self._indexed_commits:
                    continue  # 无变化
                self._ensure_repo_indexed(repo)
                refreshed += 1
            except Exception:
                logger.exception("刷新仓库索引失败: %s", repo)
        if refreshed:
            logger.info("定时刷新：%d 个仓库代码索引已更新", refreshed)
        return refreshed

    def _join_internal_path(self, trace: Any, repo: str) -> None:
        """把 trace.internal_path 的锚点 join 到该 repo 的 CodeSymbol（§9.7）。

        就地填充每个 InternalStep 的 symbol_id/symbol_name/verified。
        join 不上的锚点保持未链接（降级为纯文本现场,不报错）。
        无 internal_path（traceLink-only,无 span detail）时直接返回。
        """
        if not trace.internal_path:
            return
        try:
            from .trace_parsers.join import join_anchor_to_symbol
            from .trace_parsers.anchors import LogAnchor

            symbols = self.graph.find_nodes("CodeSymbol", repo=repo)
            if not symbols:
                return
            for step in trace.internal_path:
                anchor = LogAnchor(file=step.file, line=step.line, func=step.func)
                join_anchor_to_symbol(anchor, symbols)
                step.symbol_id = anchor.symbol_id
                step.symbol_name = anchor.symbol_name
                step.verified = anchor.join_verified
        except Exception:
            logger.exception("internal_path join 失败,降级为未链接锚点: repo=%s", repo)

    def process_one(self) -> KnowledgeProposal | None:
        """处理一条任务：从队列取出 → 获取 trace → 检索代码 → LLM 分析 → 保存。

        这是整个知识生产流水线的核心方法。
        """
        # 1. 从队列取出一条待处理消息
        message = self.queue.pop(STREAM_NAME)
        if not message:
            return None
        try:
            # 2. 解析事件
            event = OpenClawEvent.from_dict(message)
            # 3. 确保该仓库的代码已被索引
            self._ensure_repo_indexed(event.repo)
            # 4. 获取 trace 原始数据（调用 observe API）
            raw = self.trace_provider.fetch(event.repo, event.trace_id)
            self.raw_artifacts.save_trace(event.repo, event.trace_id, raw)
            # 5. 标准化 trace（统一不同来源的字段格式）
            trace = normalize_trace(event, raw)
            # 5b. 内部锚点 join CodeSymbol（§9.7）。无 span detail 时 internal_path 为空,此处为 no-op。
            self._join_internal_path(trace, event.repo)
            # 6. 向量检索关联代码（用接口路径 + 下游服务名作为查询）
            code_hits = self.vector.search("code_chunks", event.path + " " + " ".join(trace.downstream), {"repo": event.repo}, limit=5)
            # 7. 组装 LLM 上下文并调用
            context = {"event": event, "trace": trace, "code_hits": code_hits, "commit": raw.get("commit", "")}
            proposal = self.llm.propose(context)
            # 8. 按置信度分流：high 直接入库，其他进草稿层
            if proposal.confidence_score >= 0.8:
                proposal.status = "approved"
                saved = self.proposals.save(proposal)
                self.loader.load_approved_knowledge(asdict(proposal))
                logger.info("auto-approved high confidence proposal %s (score=%.2f)", saved.proposal_id, proposal.confidence_score)
            else:
                saved = self.proposals.save(proposal)
            # 9. 更新图：记录 trace、接口、服务调用关系
            self.graph.upsert_node("TraceCase", event.trace_id, {"repo": event.repo, "interface_key": event.interface_key})
            self.graph.upsert_node("Interface", event.interface_key, {"repo": event.repo, "path": event.path, "method": event.method})
            self.graph.add_edge("Interface", event.interface_key, "HAS_TRACE", "TraceCase", event.trace_id, weight=0.9)
            for downstream_svc in trace.downstream:
                self.graph.upsert_node("Service", downstream_svc, {"name": downstream_svc, "repo": event.repo})
                self.graph.add_edge("Interface", event.interface_key, "CALLS_SERVICE", "Service", downstream_svc, weight=0.7)

            # ============ Phase 2 新增：写入业务实体和关系 ============
            self._write_entities_to_graph(proposal, event)

            logger.info("processed trace %s -> proposal %s", event.trace_id, saved.proposal_id)
            return saved
        except KeyError:
            # trace 不存在，送入死信队列
            logger.warning("trace not found for message, sending to dead letter: %s", message.get("trace_id", "?"))
            self.queue.dead_letter(STREAM_NAME, message, "trace_not_found")
            return None
        except Exception:
            # 其他异常，送入死信队列并记录
            logger.exception("worker failed to process message: %s", message.get("trace_id", "?"))
            self.queue.dead_letter(STREAM_NAME, message, "processing_error")
            return None

    def run(self, poll_interval: float = 1.0) -> None:
        """主循环：持续轮询队列，直到收到停止信号。"""
        logger.info("worker started, polling every %.1fs", poll_interval)
        while not self._stop.is_set():
            try:
                self.process_one()
            except Exception:
                logger.exception("unexpected error in worker loop")
            self._stop.wait(poll_interval)
        logger.info("worker stopped")

    def _index_refresh_loop(self, interval: float) -> None:
        """后台定时循环：每 interval 秒检查一次所有仓库的 commit 变化并重索引（Day 7）。

        与主队列循环独立的线程。用 commit hash 检测变化，无变化时 git rev-parse 只花 ~10ms，
        不会浪费 CPU 去重解析整个仓库。
        """
        logger.info("代码索引刷新线程启动，间隔 %.0fs", interval)
        while not self._stop.is_set():
            # 先等再查：启动时主流程已按需索引，不必立即重复
            if self._stop.wait(interval):
                break
            try:
                self.refresh_all_repos()
            except Exception:
                logger.exception("代码索引刷新循环异常")
        logger.info("代码索引刷新线程停止")

    def start_background(self, poll_interval: float = 1.0, index_refresh_interval: float = 900.0) -> threading.Thread:
        """启动后台守护线程运行 worker。

        同时启动代码索引刷新线程（默认 15min = 900s）。index_refresh_interval <= 0 时不启动刷新线程。
        """
        thread = threading.Thread(target=self.run, args=(poll_interval,), daemon=True)
        thread.start()
        if index_refresh_interval and index_refresh_interval > 0 and self.repo_roots:
            refresh_thread = threading.Thread(
                target=self._index_refresh_loop, args=(index_refresh_interval,), daemon=True
            )
            refresh_thread.start()
        return thread

    def stop(self) -> None:
        """发送停止信号，worker（含索引刷新线程）会在当前等待结束后退出。"""
        self._stop.set()

    def proposal_payload(self, proposal: KnowledgeProposal) -> dict[str, Any]:
        """将提案转为字典（用于 API 返回）。"""
        return asdict(proposal)

    # ========================================================================
    # Phase 2: 业务实体落图
    # ========================================================================

    # 共现关系合并到 RELATED_TO 时使用的兜底权重
    # 当两个实体只是"在同一 proposal 共现"但 LLM 没显式给出业务关系时，
    # 用这个权重写一条 RELATED_TO；如果 LLM 已显式给关系，按 strength/10 走。
    COMENTION_FALLBACK_WEIGHT = 0.4

    def _write_entities_to_graph(self, proposal: KnowledgeProposal, event: OpenClawEvent) -> None:
        """将 LLM 抽取的业务实体和关系写入图。

        步骤（精简版，CO_MENTIONS 已合并到 RELATED_TO）：
        1. 每个实体经过合并逻辑（去重 + 归一化）后写入 Entity 节点
        2. 接口 → 实体建立 MENTIONS 边（"该接口提到了哪些业务概念"）
        3. LLM 显式给出的 relations → RELATED_TO 边，weight=strength/10
        4. 同 proposal 内未被 LLM 显式连接的实体对 → 兜底 RELATED_TO 边，
           weight=COMENTION_FALLBACK_WEIGHT（替代原 CO_MENTIONS）

        合并 CO_MENTIONS 的设计依据：
        - 业界主流（GraphRAG/LightRAG/Mem0）只用一种实体间关系
        - "共现"和"业务关系"本质都是"实体相关",区别只是强度
        - 用同一种边类型 + 不同权重表达，模型更简洁
        """
        if not proposal.entities:
            return

        # 1. 合并实体到图
        merged_entities = []  # 合并后的实体列表（保留顺序）
        name_to_entity: dict[str, Any] = {}  # LLM 输出的原始名 → 合并后的实体
        for entity in proposal.entities:
            try:
                merged = entities_mod.merge_into_graph(
                    self.graph, self.vector, entity,
                    proposal.proposal_id, event.trace_id, event.repo,
                )
                if not merged.entity_id:
                    continue
                merged_entities.append(merged)
                name_to_entity[entity.name] = merged
            except Exception:
                logger.exception("写入实体失败: %s", entity.name)

        # 2. 接口 → 实体的 MENTIONS 边
        for ent in merged_entities:
            self.graph.add_edge(
                "Interface", event.interface_key,
                "MENTIONS",
                "Entity", ent.entity_id,
                weight=0.6,
            )

        # 3. LLM 显式给出的业务关系 RELATED_TO（强信号）
        explicit_pairs: set[tuple[str, str]] = set()
        for rel in proposal.relations:
            src_entity = name_to_entity.get(rel.source)
            tgt_entity = name_to_entity.get(rel.target)
            if not src_entity or not tgt_entity:
                # 引用了未声明的实体，跳过（validator 应该已经拦截了）
                continue
            if src_entity.entity_id == tgt_entity.entity_id:
                continue
            # strength (1-10) 转 weight (0.1-1.0)
            weight = max(0.1, min(1.0, rel.strength / 10.0))
            self.graph.add_edge(
                "Entity", src_entity.entity_id,
                "RELATED_TO",
                "Entity", tgt_entity.entity_id,
                weight=weight,
            )
            explicit_pairs.add((src_entity.entity_id, tgt_entity.entity_id))

        # 4. 共现兜底：同 proposal 内未被 LLM 显式连接的实体对
        # 写一条弱权重的 RELATED_TO（双向）替代原 CO_MENTIONS 语义
        for a, b in itertools.combinations(merged_entities, 2):
            if a.entity_id == b.entity_id:
                continue
            if (a.entity_id, b.entity_id) in explicit_pairs or (b.entity_id, a.entity_id) in explicit_pairs:
                continue
            self.graph.add_edge(
                "Entity", a.entity_id, "RELATED_TO", "Entity", b.entity_id,
                weight=self.COMENTION_FALLBACK_WEIGHT,
            )
            self.graph.add_edge(
                "Entity", b.entity_id, "RELATED_TO", "Entity", a.entity_id,
                weight=self.COMENTION_FALLBACK_WEIGHT,
            )
