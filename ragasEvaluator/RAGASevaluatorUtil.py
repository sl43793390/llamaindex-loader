"""
ragasEvaluator.RAGASevaluatorUtil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
基于 Ragas 框架的 RAG 系统评估工具类。

核心能力
--------
- :meth:`evaluate_batch`  批量评估，输入标准字典，输出各项指标分数
- :meth:`evaluate_single` 单条评估，便捷封装

默认评估指标
------------
- **faithfulness**        忠实度：回答是否忠实于检索到的上下文
- **answer_relevancy**    答案相关性：回答是否与问题相关
- **context_precision**   上下文精确率：检索到的上下文中有用信息的比例
- **context_recall**      上下文召回率：标准答案所需信息在上下文中的覆盖度

依赖安装::

    pip install ragas datasets langchain-openai
"""
import logging
from typing import List, Dict, Any, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

logger = logging.getLogger(__name__)


def _build_ragas_llm():
    """
    构造 ragas 可用的 LLM 实例。

    使用 langchain_openai.ChatOpenAI 包装 OpenAI 兼容 API，
    再通过 LangchainLLMWrapper 转换为 ragas 内部格式，
    避免 ragas 默认创建 OpenAI 原生客户端时校验模型名。
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from config import LLM as LLM_CFG

    chat = ChatOpenAI(
        model=LLM_CFG.model,
        api_key=LLM_CFG.api_key,
        base_url=LLM_CFG.api_base,
        temperature=LLM_CFG.temperature,
        max_tokens=LLM_CFG.max_tokens,
        timeout=LLM_CFG.timeout,
    )
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings():
    """
    构造 ragas 可用的 Embeddings 实例。

    使用 langchain_openai.OpenAIEmbeddings 包装 OpenAI 兼容 API，
    再通过 LangchainEmbeddingsWrapper 转换为 ragas 内部格式。
    """
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from config import EMBED

    emb = OpenAIEmbeddings(
        model=EMBED.model,
        api_key=EMBED.api_key,
        base_url=EMBED.api_base,
    )
    return LangchainEmbeddingsWrapper(emb)


class RAGASEvaluator:
    """
    基于 Ragas 框架的 RAG 系统评估工具类。

    Args:
        metrics: 自定义评估指标列表；为 None 时使用默认四项核心指标。
        llm: ragas 可用的 LLM 实例；为 None 时自动通过
            :func:`_build_ragas_llm` 构造（使用项目 config 中的 OpenAI 兼容配置）。
        embeddings: ragas 可用的 Embeddings 实例；为 None 时自动通过
            :func:`_build_ragas_embeddings` 构造。
    """

    # 默认支持的核心评估指标
    DEFAULT_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

    def __init__(self, metrics: Optional[List] = None, llm=None, embeddings=None):
        self.metrics = metrics if metrics else self.DEFAULT_METRICS
        self.llm = llm
        self.embeddings = embeddings

    def evaluate_batch(self, eval_data: Dict[str, List[Any]]) -> Dict[str, float]:
        """
        批量评估 RAG 系统性能。

        Args:
            eval_data: 包含以下四个字段的字典::

                {
                    "question":     [str, ...],      # 问题列表
                    "answer":       [str, ...],      # RAG 生成的回答列表
                    "contexts":     [List[str], ...], # 检索到的上下文列表（每个问题对应一个列表）
                    "ground_truth": [str, ...],      # 标准答案列表
                }

        Returns:
            各项指标得分的字典，例如::

                {"faithfulness": 0.85, "answer_relevancy": 0.72, ...}
        """
        try:
            dataset = Dataset.from_dict(eval_data)

            logger.info(
                "开始评估，共 %d 条数据，使用指标: %s",
                len(dataset),
                [m.name for m in self.metrics],
            )

            # 懒构造 LLM / Embeddings，避免 import 时就触发
            ragas_llm = self.llm or _build_ragas_llm()
            ragas_embeddings = self.embeddings or _build_ragas_embeddings()

            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )

            logger.info("评估完成！")
            return result

        except Exception as e:
            logger.error("RAGAS 评估过程中发生错误: %s", e)
            raise

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        评估单条 RAG 问答数据。

        Args:
            question:     问题
            answer:       RAG 生成的回答
            contexts:     检索到的上下文文本列表
            ground_truth: 标准答案

        Returns:
            同 :meth:`evaluate_batch`。
        """
        single_data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        return self.evaluate_batch(single_data)


# ================= 使用示例 =================
if __name__ == "__main__":
    # 1. 准备测试数据（模拟 RAG 系统的输入输出）
    test_data = {
        "question": ["什么是 RAG 技术？", "如何优化向量检索的精度？"],
        "answer": [
            "RAG（检索增强生成）是一种结合检索和生成的 AI 技术...",
            "可以通过以下方法优化向量检索精度：1. 使用更好的 embedding 模型...",
        ],
        "contexts": [
            ["RAG 全称 Retrieval-Augmented Generation，是..."],
            ["向量检索的精度优化方法包括 Reranking、HyDE..."],
        ],
        "ground_truth": [
            "RAG 是检索增强生成技术，通过检索外部知识库来增强大模型的回答质量。",
            "优化向量检索精度的主要方法有：使用 Reranking 模型、HyDE 查询扩展等。",
        ],
    }

    # 2. 初始化评估器并进行批量评估
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate_batch(test_data)
    print("批量评估结果:", scores)

    # 3. 评估单条数据
    single_score = evaluator.evaluate_single(
        question="RAG 有什么优势？",
        answer="RAG 的优势在于能够利用外部最新知识，减少大模型的幻觉。",
        contexts=["RAG 通过外挂知识库，有效缓解了 LLM 的知识滞后和幻觉问题。"],
        ground_truth="RAG 能够减少幻觉，并让模型获取实时外部知识。",
    )
    print("单条评估结果:", single_score)
