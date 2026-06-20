"""
hybridRetriever.bm25_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BM25 检索器(基于 ``rank_bm25`` 第三方库)。

为什么用 rank_bm25:
    - 经典实现、经过大量生产验证
    - 提供 BM25Okapi / BM25Plus / BM25L 三个变体
    - 比手写稳定、bug 更少

API 与上一版完全一致(``from_nodes / retrieve / add_nodes``),可直接替换。

用法::

    retriever = BM25Retriever.from_nodes(nodes, variant="okapi", k1=1.5, b=0.75)
    result = retriever.retrieve("查询语句", top_k=10)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from rank_bm25 import BM25L, BM25Okapi, BM25Plus

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from .utils import tokenize


_VARIANTS = {
    "okapi": BM25Okapi,
    "plus": BM25Plus,
    "l": BM25L,
}


class BM25Retriever:
    """
    BM25 检索器,基于 ``rank_bm25`` 实现。

    Args:
        nodes: 索引的 BaseNode 列表。
        variant: ``"okapi"``(默认)/ ``"plus"`` / ``"l"``。
        k1: BM25 词频饱和参数,通常 1.2 ~ 2.0(BM25L/Plus 也用它)。
        b:  文档长度归一化参数 0 ~ 1(BM25L/Plus 也用它)。
        delta: BM25Plus 专用参数(``rank_bm25`` 用法);其他变体忽略。
        remove_stopwords: 分词时是否去掉中英文常见停用词。
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode],
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        remove_stopwords: bool = True,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown BM25 variant: {variant}; choose from {list(_VARIANTS)}"
            )
        self.nodes: List[BaseNode] = list(nodes)
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.remove_stopwords = remove_stopwords

        # 内部
        self._bm25 = None
        self._docs_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._build_index()

    # ----- 工厂方法 -----
    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[BaseNode],
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        remove_stopwords: bool = True,
    ) -> "BM25Retriever":
        return cls(
            nodes=nodes,
            variant=variant,
            k1=k1,
            b=b,
            delta=delta,
            remove_stopwords=remove_stopwords,
        )

    # ----- 内部索引 -----
    def _build_index(self) -> None:
        self._docs_tokens = [
            tokenize(n.get_content(), remove_stopwords=self.remove_stopwords)
            for n in self.nodes
        ]
        self._doc_len = [len(toks) for toks in self._docs_tokens]
        self._avgdl = (sum(self._doc_len) / max(len(self._doc_len), 1)) or 1.0
        # 构造 rank_bm25 实例
        cls = _VARIANTS[self.variant]
        if self.variant == "plus":
            # BM25Plus 用 (k1, b, delta)
            self._bm25 = cls(
                self._docs_tokens,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
        else:
            # BM25Okapi / BM25L 用 (k1, b)
            self._bm25 = cls(
                self._docs_tokens,
                k1=self.k1,
                b=self.b,
            )

    # ----- 检索 -----
    def retrieve(
        self,
        query: str | QueryBundle,
        top_k: int = 10,
    ) -> List[NodeWithScore]:
        """
        检索 top_k 相关节点。

        Args:
            query: 查询字符串或 ``QueryBundle``。
            top_k: 返回节点数。

        Returns:
            按 BM25 分数降序的 ``NodeWithScore`` 列表。
        """
        q_text = query.query_str if isinstance(query, QueryBundle) else str(query)
        q_tokens = tokenize(q_text, remove_stopwords=self.remove_stopwords)
        if not q_tokens or not self.nodes or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # scores 是 ndarray / list,长度 == len(nodes)
        # 取 top_k(分数 > 0)
        n = len(self.nodes)
        # 用 argsort 取 top_k 索引
        indexed = sorted(
            range(n), key=lambda i: -float(scores[i])
        )[:top_k]
        return [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in indexed
            if float(scores[i]) > 0
        ]

    # ----- 增量更新 -----
    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """追加新节点并重建索引。"""
        for n in nodes:
            self.nodes.append(n)
        self._build_index()

    # ----- 信息 -----
    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"BM25Retriever(variant={self.variant}, n={len(self)}, "
            f"k1={self.k1}, b={self.b}, avgdl={self._avgdl:.1f})"
        )
