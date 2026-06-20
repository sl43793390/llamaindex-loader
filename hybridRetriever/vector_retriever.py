"""
hybridRetriever.vector_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
稠密向量检索(对已有 ``VectorStoreIndex`` 的薄封装)。

提供:
    - ``VectorRetriever``:从 ``VectorStoreIndex`` 取一个 similarity retriever
    - ``score_normalize`` :把 Milvus 返回的距离 / 内积换算到 [0, 1] 相似度
                            (融合时跨检索器可比)

设计目的:
    HybridRetriever 需要把"向量分数"和"BM25 分数"加权融合。
    但各 retriever 的 score 量纲不同,必须先归一化。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle


def score_normalize(
    nodes_with_score: Sequence[NodeWithScore],
    invert: bool = False,
) -> List[NodeWithScore]:
    """
    把分数归一化到 [0, 1]。

    Args:
        nodes_with_score: 输入。
        invert: True 时按 1 - normalized(把"距离"变成"相似度")。
                Milvus 在用 IP 时返回的是"距离越小越相关" → 需要 invert。
    """
    if not nodes_with_score:
        return []
    scores = [n.score or 0.0 for n in nodes_with_score]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        norm = [1.0] * len(scores)
    else:
        span = hi - lo
        if invert:
            norm = [(hi - s) / span for s in scores]
        else:
            norm = [(s - lo) / span for s in scores]
    return [
        NodeWithScore(node=nws.node, score=float(s))
        for nws, s in zip(nodes_with_score, norm)
    ]


class VectorRetriever:
    """
    包装 ``VectorStoreIndex`` 的稠密向量检索器。

    Args:
        index: 已构建好的 ``VectorStoreIndex``。
        embed_model: 自定义 Embedding,None 时用 ``index`` 内部已注册的。
        similarity_top_k: 每次 retrieve 召回的数量(可被 retrieve(top_k=...) 覆盖)。
        normalize_score: 是否把分数归一化到 [0, 1](融合时强烈建议 True)。
        invert_score: 是否反转(适配"距离越小越相关"的场景)。
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: int = 10,
        normalize_score: bool = True,
        invert_score: bool = False,
    ) -> None:
        self.index = index
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.normalize_score = normalize_score
        self.invert_score = invert_score
        # 预构造一个 retriever
        self._retriever = index.as_retriever(
            similarity_top_k=similarity_top_k,
        )

    def retrieve(
        self,
        query: str | QueryBundle,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        q = query if isinstance(query, QueryBundle) else QueryBundle(query_str=str(query))
        if top_k is not None and top_k != self.similarity_top_k:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
        else:
            retriever = self._retriever
        # 向量索引的 retriever.retrieve 接受 str 或 QueryBundle
        try:
            result = retriever.retrieve(q)
        except TypeError:
            result = retriever.retrieve(q.query_str)
        if self.normalize_score:
            result = score_normalize(result, invert=self.invert_score)
        return result

    def __len__(self) -> int:
        try:
            return len(self.index.docstore.docs)
        except Exception:
            return 0
