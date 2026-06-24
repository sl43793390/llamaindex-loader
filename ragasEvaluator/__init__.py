"""
ragasEvaluator — RAGAS 端到端 RAG 评测模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
提供从文档切分到 RAGAS 指标计算的完整评测流水线。

核心组件
--------
- :class:`RAGASEvaluator`     RAGAS 评估器（封装 ragas.evaluate，来自 RAGASevaluatorUtil.py）
- :class:`TestsetGenerator`   用 LLM 从 chunk 生成 (question, ground_truth) 问答对
- :class:`RAGRunner`          封装 RAG 引擎调用，统一输出 (answer, contexts)
- :class:`RAGASEvalPipeline`  端到端评测流水线（面向对象，可分步控制）
- :func:`run_ragas_eval`      一站式工厂函数（一行代码跑完全流程）

快速开始
--------
::

    from ragasEvaluator import run_ragas_eval, RAGASEvaluator

    # 方式 1: 一站式（推荐）
    result = run_ragas_eval(docs=docs, n_questions_per_chunk=2)
    print(result["scores"])

    # 方式 2: 命令行 demo
    # python -m ragasEvaluator.run_eval_demo --file data/sample.pdf
"""
from .RAGASevaluatorUtil import RAGASEvaluator
from .pipeline import RAGASEvalPipeline, run_ragas_eval
from .rag_runner import RAGRunner
from .testset_generator import TestsetGenerator

__all__ = [
    "RAGASEvaluator",
    "TestsetGenerator",
    "RAGRunner",
    "RAGASEvalPipeline",
    "run_ragas_eval",
]
