"""
hybridRetriever.hybrid_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
混合检索:多个 retriever 并行召回 → 融合排序。

支持三种融合方式:
    - RRF(Reciprocal Rank Fusion):无参数、对量纲不敏感,论文经典。
    - Weighted:按 retriever 权重加权,需要先把分数归一化。
    - Concat:不融合,直接把多路结果拼起来(去重后截 top_k)。

工厂函数 :func:`build_hybrid_retriever` 一次性构建并返回 ``HybridRetriever`` 实例。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from .bm25_retriever import BM25Retriever
from .keyword_retriever import LlamaIndexKeywordRetriever, TFIDFKeywordRetriever
from .utils import dedup_by_id, rrf_fuse, top_k_by_score, weighted_fuse
from .vector_retriever import VectorRetriever


# 任何 ``BaseRetriever``-like(有 .retrieve(query, top_k))的对象都接受
RetrieverLike = object


class HybridRetriever:
    """
    混合检索器:并行调多个 retriever,融合结果。

    Args:
        retrievers: retriever 列表,每个必须有 ``retrieve(query, top_k)`` 方法。
        weights:    每个 retriever 的权重(weighted 模式使用)。
        method:     ``"rrf"`` / ``"weighted"`` / ``"concat"``。
        top_k:      默认返回前 N。
        rrf_k:      RRF 平滑常数,论文推荐 60。
        normalize:  weighted 模式是否先把每个 retriever 分数归一化到 [0, 1]。
    """

    def __init__(
        self,
        retrievers: Sequence[RetrieverLike],
        weights: Optional[Sequence[float]] = None,
        method: str = "rrf",
        top_k: int = 10,
        rrf_k: int = 60,
        normalize: bool = True,
    ) -> None:
        assert method in ("rrf", "weighted", "concat"), f"bad method: {method}"
        if weights is None:
            weights = [1.0] * len(retrievers)
        assert len(retrievers) == len(weights), "weights must match retrievers"
        self.retrievers = list(retrievers)
        self.weights = list(weights)
        self.method = method
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.normalize = normalize

    # ----------------- 主入口 -----------------
    def retrieve(
        self,
        query: str | QueryBundle,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        并行 / 串行调用每个 retriever,融合后返回。

        Args:
            query: 查询。
            top_k: 覆盖默认 top_k。

        Returns:
            融合后的 ``NodeWithScore`` 列表。
        """
        k = top_k or self.top_k
        per_retriever_k = max(k, self.top_k)  # 多召回一些,防止融合后过少

        lists: List[List[NodeWithScore]] = []
        for r in self.retrievers:
            try:
                res = r.retrieve(query, top_k=per_retriever_k)
            except TypeError:
                # retriever 不接受 top_k
                res = r.retrieve(query)
            if not res:
                continue
            # 各 retriever 内部按 score 降序
            res = top_k_by_score(res, per_retriever_k)
            lists.append(res)

        if not lists:
            return []

        if self.method == "rrf":
            fused = rrf_fuse(lists, k=self.rrf_k, top_k=k)
        elif self.method == "weighted":
            fused = weighted_fuse(
                lists, self.weights, top_k=k, normalize=self.normalize
            )
        else:  # concat
            seen: List[NodeWithScore] = []
            for lst in lists:
                seen.extend(lst)
            fused = top_k_by_score(dedup_by_id(seen), k)

        return fused

    # ----------------- 信息 -----------------
    def describe(self) -> str:
        lines = [f"HybridRetriever(method={self.method}, top_k={self.top_k}, rrf_k={self.rrf_k})"]
        for i, r in enumerate(self.retrievers):
            lines.append(f"  [{i}] weight={self.weights[i]} {type(r).__name__}({len(r)})")
        return "\n".join(lines)


# ============================================================
# 工厂:一站式构建
# ============================================================
def build_hybrid_retriever(
    nodes: Sequence[BaseNode],
    index: Optional[VectorStoreIndex] = None,
    retrievers: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    method: str = "rrf",
    top_k: int = 10,
    bm25_kwargs: Optional[dict] = None,
    keyword_kwargs: Optional[dict] = None,
) -> HybridRetriever:
    """
    快速构造一个混合检索器。

    Args:
        nodes: 所有节点(给 BM25 / Keyword 用)。
        index: 已构建的 ``VectorStoreIndex``(给向量用)。
                没传则不会加入向量 retriever。
        retrievers: 要包含的子检索器,可选:
            - ``"vector"``    :稠密向量(需要 index)
            - ``"bm25"``      :BM25
            - ``"keyword"``   :TFIDF 关键词
            - ``"rake"``      :LlamaIndex RAKE 关键词
        weights:  与 retrievers 等长的权重列表;None 时各路 1.0。
        method:  ``"rrf"`` / ``"weighted"`` / ``"concat"``。
        top_k:   默认返回前 N。
        bm25_kwargs: 透传给 ``BM25Retriever``。
        keyword_kwargs: 透传给 ``TFIDFKeywordRetriever`` / ``LlamaIndexKeywordRetriever``。

    Returns:
        :class:`HybridRetriever` 实例。

    Example::

        hybrid = build_hybrid_retriever(
            nodes=nodes,
            index=index,
            retrievers=["vector", "bm25", "keyword"],
            weights=[0.6, 0.3, 0.1],
            method="weighted",
        )
        result = hybrid.retrieve("什么是 UFX?")
    """
    retrievers = list(retrievers or ["vector", "bm25"])
    if weights is None:
        weights = [1.0] * len(retrievers)
    assert len(retrievers) == len(weights), "retrievers / weights length mismatch"

    built: List[RetrieverLike] = []
    actual_weights: List[float] = []
    for r_name, w in zip(retrievers, weights):
        if r_name == "vector":
            if index is None:
                raise ValueError("retrievers contains 'vector' but index is None")
            built.append(VectorRetriever(index=index))
        elif r_name == "bm25":
            built.append(BM25Retriever.from_nodes(
                nodes, **(bm25_kwargs or {})
            ))
        elif r_name == "keyword":
            built.append(TFIDFKeywordRetriever.from_nodes(
                nodes, **(keyword_kwargs or {})
            ))
        elif r_name == "rake":
            built.append(LlamaIndexKeywordRetriever.from_nodes(
                nodes, mode="rake", **(keyword_kwargs or {})
            ))
        else:
            raise ValueError(f"unknown retriever: {r_name}")
        actual_weights.append(w)

    return HybridRetriever(
        retrievers=built,
        weights=actual_weights,
        method=method,
        top_k=top_k,
    )
