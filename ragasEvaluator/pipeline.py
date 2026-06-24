"""
ragasEvaluator.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~
RAGAS 端到端评测流水线，串联以下步骤::

    原始文档 (Document[])
        │
        ▼  dataLoader.auto_load / 手动构造
    切分后节点 (BaseNode[])
        │
        ├────── A: 喂给 RAG 索引 ──────► VectorStoreIndex
        │                                     │
        └────── B: 喂给 LLM 生成 Q&A ──► (question, ground_truth) pairs
                                              │
                                     RAGRunner.run(questions)
                                              │
                                     (answer, contexts) pairs
                                              │
                              按 question 对齐合并
                                              │
                   {question, answer, contexts, ground_truth}
                                              │
                        RAGASEvaluator.evaluate_batch(...)
                                              │
                              各项 RAGAS 指标分数

对外提供两个入口:
    - :class:`RAGASEvalPipeline`  — 面向对象的流水线，可分步控制
    - :func:`run_ragas_eval`      — 一站式工厂函数，一行代码跑完全流程
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from llama_index.core import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import BaseNode

from .rag_runner import RAGRunner
from .testset_generator import TestsetGenerator

logger = logging.getLogger(__name__)


# ============================================================
# 数据对齐：合并 QA 对和 RAG 结果
# ============================================================

def _align_eval_data(
    qa_pairs: Sequence[dict],
    rag_results: Sequence[dict],
) -> dict:
    """
    将 (question, ground_truth) 与 (answer, contexts) 按 question 对齐，
    拼成 RAGAS 评估所需的标准字典格式。

    对齐逻辑:
        - 以 qa_pairs 为主表，逐条查找 rag_results 中相同 question 的记录
        - 如果某个 question 在 rag_results 中找不到（RAG 查询失败），该条会被跳过
        - rag_results 中多余的记录（不在 qa_pairs 中的）也会被忽略

    Args:
        qa_pairs: :meth:`TestsetGenerator.generate_from_nodes` 的输出，
            每项含 ``question`` 和 ``ground_truth``。
        rag_results: :meth:`RAGRunner.run` 的输出，
            每项含 ``question``、``answer`` 和 ``contexts``。

    Returns:
        RAGAS 评估所需的字典::

            {
                "question":     [str, ...],
                "answer":       [str, ...],
                "contexts":     [List[str], ...],
                "ground_truth": [str, ...],
            }
    """
    # 以 question 为 key 建立 RAG 结果索引，加速查找
    rag_by_q = {r["question"]: r for r in rag_results}

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []

    skipped = 0
    for p in qa_pairs:
        q = p["question"]
        gt = p["ground_truth"]
        if q not in rag_by_q:
            skipped += 1
            continue
        r = rag_by_q[q]
        questions.append(q)
        answers.append(r.get("answer", "") or "")
        contexts_list.append(r.get("contexts", []) or [])
        ground_truths.append(gt)

    if skipped:
        logger.warning("数据对齐: %d 条 QA 对在 RAG 结果中找不到匹配（已跳过）", skipped)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


# ============================================================
# 面向对象流水线
# ============================================================

class RAGASEvalPipeline:
    """
    RAGAS 端到端评测流水线。

    将测试集生成、RAG 执行、数据对齐和评估串联在一起，
    输入切分后的 BaseNode 列表，输出各项 RAGAS 指标分数。

    Args:
        index: 已构建好的 Milvus ``VectorStoreIndex``（已包含待评测的文档）。
        llm: LLM 实例；为 None 时自动调用 ``vectorStore.milvus_store.get_llm()``。
        n_questions_per_chunk: 每个 chunk 生成几个问答对（推荐 1~3）。
        similarity_top_k: RAG 检索时召回的节点数。
        use_chat_engine: True 使用 chat engine（有 memory）；False 使用 query engine（无 memory，推荐）。

    Example::

        pipeline = RAGASEvalPipeline(index=my_index, n_questions_per_chunk=2)
        result = pipeline.run(nodes=nodes)
        print(result["scores"])
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm=None,
        n_questions_per_chunk: int = 1,
        similarity_top_k: int = 5,
        use_chat_engine: bool = False,
    ) -> None:
        self.index = index
        self.generator = TestsetGenerator(
            llm=llm, n_questions_per_chunk=n_questions_per_chunk
        )
        self.runner = RAGRunner(
            index=index,
            similarity_top_k=similarity_top_k,
            use_chat_engine=use_chat_engine,
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(
        self,
        nodes: Sequence[BaseNode],
        evaluator=None,
    ) -> dict:
        """
        执行完整评测流程。

        步骤:
            1. 用 TestsetGenerator 从 nodes 生成 (question, ground_truth) 对
            2. 用 RAGRunner 对每个 question 执行 RAG 查询，获取 (answer, contexts)
            3. 按 question 对齐合并两组数据
            4. 调用 RAGASEvaluator 计算各项指标

        Args:
            nodes: 已切分好的 BaseNode 列表（由 spliter 产出）。
            evaluator: :class:`RAGASEvaluator` 实例；为 None 时使用默认四项指标。

        Returns:
            评测结果字典::

                {
                    "eval_data": {"question": [...], "answer": [...], ...},
                    "scores":    {"faithfulness": 0.85, ...},
                    "n":         10,  # 有效样本数
                }
        """
        # 1) 生成 Q&A 对
        logger.info("步骤 1/4: 生成问答对...")
        qa_pairs = self.generator.generate_from_nodes(nodes)
        if not qa_pairs:
            logger.warning("未生成任何问答对，评测终止")
            return {
                "eval_data": _align_eval_data([], []),
                "scores": {},
                "n": 0,
            }
        logger.info("共生成 %d 个问答对", len(qa_pairs))
        questions = [p["question"] for p in qa_pairs]

        # 2) RAG 查询获取 answer + contexts
        logger.info("步骤 2/4: 执行 RAG 查询 (%d 个问题)...", len(questions))
        rag_results = self.runner.run(questions)
        logger.info("RAG 查询完成，获得 %d 条结果", len(rag_results))

        # 3) 对齐成 RAGAS 格式
        logger.info("步骤 3/4: 对齐数据...")
        eval_data = _align_eval_data(qa_pairs, rag_results)
        n = len(eval_data["question"])
        if n == 0:
            logger.warning("对齐后无有效样本，评测终止")
            return {"eval_data": eval_data, "scores": {}, "n": 0}

        # 4) 调用 RAGAS 评估器
        logger.info("步骤 4/4: 执行 RAGAS 评估 (%d 条样本)...", n)
        if evaluator is None:
            from .RAGASevaluatorUtil import RAGASEvaluator
            evaluator = RAGASEvaluator()
        scores = evaluator.evaluate_batch(eval_data)

        return {
            "eval_data": eval_data,
            "scores": scores,
            "n": n,
        }


# ============================================================
# 一站式工厂函数
# ============================================================

def run_ragas_eval(
    docs: Optional[Sequence[Document]] = None,
    nodes: Optional[Sequence[BaseNode]] = None,
    split_fn=None,
    index: Optional[VectorStoreIndex] = None,
    llm=None,
    n_questions_per_chunk: int = 1,
    similarity_top_k: int = 5,
    use_chat_engine: bool = False,
    evaluator=None,
    rebuild_index: bool = True,
) -> dict:
    """
    一站式 RAGAS 评测：docs → split → index → Q&A → RAG → 评估。

    自动串联文档切分、索引构建、测试集生成、RAG 查询和评估打分，
    适合快速跑通评测流程。如需更细粒度控制，请使用 :class:`RAGASEvalPipeline`。

    Args:
        docs: Document 列表（已加载的文档）。如果同时提供了 nodes，则 docs 仅用于切分。
        nodes: 已切分好的 BaseNode 列表。若提供，则跳过 split_fn 切分步骤。
        split_fn: 切分函数，签名为 ``(docs) -> List[BaseNode]``；
            为 None 时默认使用 ``spliter.split_by_sentence``。
        index: 已构建的 ``VectorStoreIndex``；为 None 时自动调用
            ``vectorStore.milvus_store.create_index`` 构建临时索引。
        llm: 自定义 LLM 实例；为 None 时自动获取。
        n_questions_per_chunk: 每个 chunk 生成几个问答对（推荐 1~3）。
        similarity_top_k: RAG 检索 top_k。
        use_chat_engine: True 使用 chat engine（有 memory）；False 使用 query engine（推荐）。
        evaluator: 自定义 :class:`RAGASEvaluator`；为 None 时使用默认四项指标。
        rebuild_index: True 时重新构建索引（需要提供 docs 或 nodes）；
            False 时使用已有 Milvus 索引，自动从索引中提取 nodes，无需提供 docs。

    Returns:
        评测结果字典::

            {
                "eval_data": {"question": [...], "answer": [...], ...},
                "scores":    {"faithfulness": 0.85, ...},
                "n":         10,
            }

    Example::

        # 模式1: 重新构建索引
        from dataLoader import auto_load
        from ragasEvaluator import run_ragas_eval
        docs = auto_load("data/sample.pdf")
        result = run_ragas_eval(docs=docs, n_questions_per_chunk=2)

        # 模式2: 使用已有索引
        result = run_ragas_eval(rebuild_index=False, n_questions_per_chunk=2)
        print("scores:", result["scores"])
    """
    from vectorStore.milvus_store import configure_settings

    if not rebuild_index:
        # 使用已有索引模式：从 Milvus 加载索引，并提取 nodes 用于生成问答对
        from vectorStore.milvus_store import load_existing_index
        from llama_index.core.schema import TextNode
        from pymilvus import MilvusClient
        from config import MILVUS

        configure_settings()
        index = load_existing_index()

        # 从 Milvus 集合中提取所有文本节点，用于生成问答对
        client = MilvusClient(uri=MILVUS.uri)
        rows = client.query(
            MILVUS.collection_name, output_fields=["text"], limit=500
        )
        nodes = [TextNode(text=r["text"]) for r in rows if r.get("text")]
        logger.info("从已有索引中加载了 %d 个节点", len(nodes))
    else:
        # 重新构建索引模式
        # 1) 切分
        if nodes is None:
            if docs is None:
                raise ValueError("rebuild_index=True 时，docs 和 nodes 至少提供一个")
            if split_fn is None:
                from spliter import split_by_sentence
                split_fn = split_by_sentence
            nodes = split_fn(list(docs))
            logger.info("切分完成: %d 个节点", len(nodes))

        # 2) 构建索引
        if index is None:
            from vectorStore.milvus_store import create_index
            configure_settings()
            index = create_index(nodes)
            logger.info("索引构建完成")

    # 3) 执行评测流水线
    pipeline = RAGASEvalPipeline(
        index=index,
        llm=llm,
        n_questions_per_chunk=n_questions_per_chunk,
        similarity_top_k=similarity_top_k,
        use_chat_engine=use_chat_engine,
    )
    return pipeline.run(nodes=nodes, evaluator=evaluator)
