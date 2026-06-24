"""
ragasEvaluator.rag_runner
~~~~~~~~~~~~~~~~~~~~~~~~~~
封装 RAG 引擎调用，统一输出 (answer, contexts) 格式，供 RAGAS 评估使用。

为什么单独拆一层
----------------
- RAGAS 评估需要的是纯文本 ``contexts``，而非 LlamaIndex 的 ``NodeWithScore`` 对象。
- 不同引擎（CustomRAGChatEngine / SimpleQueryEngine）返回结构不同，
  在这一层统一成 ``(answer: str, contexts: List[str])``。
- 支持批量串行调用，每条问题独立执行，避免多轮对话的 memory 污染评测结果。

使用方式
--------
::

    from ragasEvaluator import RAGRunner

    runner = RAGRunner(index=my_index, similarity_top_k=3)
    results = runner.run(["什么是 RAG？", "Milvus 的优势是什么？"])
    # results -> [{"question": ..., "answer": ..., "contexts": [...]}, ...]
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from llama_index.core.indices.vector_store import VectorStoreIndex

logger = logging.getLogger(__name__)


class RAGRunner:
    """
    给定已构建好的 VectorStoreIndex，串行执行 RAG 查询，产出 (answer, contexts) 列表。

    Args:
        index: 已构建的 Milvus ``VectorStoreIndex``（已包含待评测的文档）。
        similarity_top_k: 每次检索召回的节点数，同时也是 contexts 数量上限。
        use_chat_engine: True 使用 ``build_chat_engine``（有 memory，适合多轮对话评测）；
            False 使用 ``build_simple_query_engine``（无 memory，**推荐用于单轮评测**，
            避免上下文污染）。
        debug: 是否开启 RAGDebugHandler 日志（评测时建议关闭，减少噪音）。
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 5,
        use_chat_engine: bool = False,
        debug: bool = False,
    ) -> None:
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.use_chat_engine = use_chat_engine
        self.debug = debug
        # 懒构造：首次调用 _get_engine 时才创建，避免初始化阶段的不必要开销
        self._engine = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self, questions: Sequence[str]) -> List[dict]:
        """
        串行执行 RAG 查询，逐条获取 answer 和 contexts。

        每条问题独立执行，chat engine 模式下会在每次查询前 reset memory，
        确保评测结果不受历史对话影响。

        Args:
            questions: 问题字符串列表。

        Returns:
            结果列表，每项格式::

                {"question": "...", "answer": "...", "contexts": ["ctx1", "ctx2", ...]}
        """
        results: List[dict] = []
        total = len(questions)
        for idx, q in enumerate(questions, 1):
            if not q or not q.strip():
                logger.debug("问题 %d/%d 为空，跳过", idx, total)
                continue

            try:
                answer, contexts = self._ask(q)
            except Exception as e:
                # 单条失败 → 留空值，不让整个评测崩溃
                logger.warning("问题 %d/%d RAG 查询失败: %s: %s", idx, total, type(e).__name__, e)
                answer, contexts = "", []

            results.append({"question": q, "answer": answer, "contexts": contexts})
            logger.info("问题 %d/%d: answer=%d chars, %d contexts",
                        idx, total, len(answer), len(contexts))

        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _ask(self, question: str) -> tuple[str, List[str]]:
        """
        调用一次 RAG 引擎，返回 (answer, [ctx_str, ...])。

        根据 ``use_chat_engine`` 选择 chat engine 或 query engine：
        - chat engine: 调用 ``engine.chat()``，返回 ``response`` + ``source_nodes``
        - query engine: 调用 ``engine.query()``，返回 ``response`` + ``source_nodes``
        """
        engine = self._get_engine()

        if self.use_chat_engine:
            resp = engine.chat(question)
            answer = resp.response or ""
        else:
            resp = engine.query(question)
            answer = str(resp) if resp else ""

        contexts = self._extract_contexts(getattr(resp, "source_nodes", None))
        return answer, contexts

    @staticmethod
    def _extract_contexts(source_nodes) -> List[str]:
        """
        从 LlamaIndex 的 source_nodes 列表中提取每个节点的纯文本。

        兼容两种输入:
            - ``NodeWithScore`` 对象（有 ``.node`` 属性）
            - 裸 ``BaseNode`` 对象（直接有 ``.get_content()``）

        Args:
            source_nodes: ``source_nodes`` 列表，可能为 None。

        Returns:
            纯文本字符串列表。
        """
        if not source_nodes:
            return []

        out: List[str] = []
        for n in source_nodes:
            try:
                node = n.node if hasattr(n, "node") else n
                txt = node.get_content() if hasattr(node, "get_content") else str(node)
            except Exception:
                txt = ""
            if txt:
                out.append(txt)
        return out

    def _get_engine(self):
        """
        懒构造 RAG 引擎。

        首次调用时根据 ``use_chat_engine`` 选择构建 chat engine 或 query engine，
        后续调用直接复用已构建的实例。
        """
        if self._engine is not None:
            return self._engine

        from vectorStore.chat import (
            build_chat_engine,
            build_simple_query_engine,
        )

        if self.use_chat_engine:
            self._engine = build_chat_engine(
                index=self.index,
                similarity_top_k=self.similarity_top_k,
                debug=self.debug,
            )
        else:
            self._engine = build_simple_query_engine(
                index=self.index,
                similarity_top_k=self.similarity_top_k,
                debug=self.debug,
            )
        return self._engine
