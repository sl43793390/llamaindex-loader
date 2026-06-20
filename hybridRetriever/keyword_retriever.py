"""
hybridRetriever.keyword_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
关键词检索器(轻量自实现 + LlamaIndex 内置 KeywordTable 封装)。

提供两种实现:
    1. ``TFIDFKeywordRetriever``(本模块,纯 Python):
        倒排索引 + TF-IDF 评分,够用、可控、不依赖 LLM 提取关键词。
        比 BM25 简单,适合"对 query 中的关键词精确匹配"场景。

    2. ``LlamaIndexKeywordRetriever``(基于 ``KeywordTableIndex``):
        利用 LlamaIndex 的 ``KeywordTableSimpleRetriever`` / ``KeywordTableRAKERetriever``
        做关键词提取 + 检索。功能更强但启动稍慢。

选择建议:
    - 数据 < 5w 文档,query 中含明确关键词 → TFIDFKeywordRetriever
    - 需要 RAKE / GPT 智能关键词 / 中文分词好 → LlamaIndexKeywordRetriever
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from llama_index.core.indices.keyword_table import (
    KeywordTableIndex,
    KeywordTableRAKERetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from .utils import (
    attach_scores,
    build_inverted_index,
    tfidf_score,
    tokenize,
    top_k_by_score,
)


# ============================================================
# 1. 自实现:TF-IDF 关键词检索
# ============================================================
class TFIDFKeywordRetriever:
    """
    倒排索引 + TF-IDF 评分。

    适合"对 query 里的关键词做精确匹配 + 词频权重"的场景。
    不会做语义扩展(找不到同义词),但响应快、可解释、零依赖 LLM。
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode],
        remove_stopwords: bool = True,
    ) -> None:
        self.nodes: List[BaseNode] = list(nodes)
        self.remove_stopwords = remove_stopwords
        self._inv = {}
        self._tf = []
        self._build_index()

    @classmethod
    def from_nodes(
        cls, nodes: Sequence[BaseNode], **kwargs
    ) -> "TFIDFKeywordRetriever":
        return cls(nodes=nodes, **kwargs)

    def _build_index(self) -> None:
        docs_tokens = [
            tokenize(n.get_content(), remove_stopwords=self.remove_stopwords)
            for n in self.nodes
        ]
        self._inv, self._tf = build_inverted_index(docs_tokens)

    def retrieve(
        self,
        query: str | QueryBundle,
        top_k: int = 10,
    ) -> List[NodeWithScore]:
        q_text = query.query_str if isinstance(query, QueryBundle) else str(query)
        q_tokens = tokenize(q_text, remove_stopwords=self.remove_stopwords)
        if not q_tokens or not self.nodes:
            return []
        scores = tfidf_score(
            q_tokens, self._inv, self._tf, n_docs=len(self.nodes)
        )
        # 排序
        top = scores.most_common(top_k)
        return attach_scores(
            [self.nodes[i] for i, _ in top],
            [s for _, s in top],
        )

    def add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        for n in nodes:
            self.nodes.append(n)
        self._build_index()

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"TFIDFKeywordRetriever(n={len(self)}, vocab={len(self._inv)})"


# ============================================================
# 2. LlamaIndex 内置 KeywordTable 检索器
# ============================================================
class LlamaIndexKeywordRetriever:
    """
    包装 :class:`KeywordTableIndex` 的检索器。

    支持两种后端:
        - ``mode="simple"`` → 用 ``KeywordTableSimpleRetriever``(正则切词,英文友好)
        - ``mode="rake"``   → 用 ``KeywordTableRAKERetriever``(RAKE 算法,适合长文)
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode],
        mode: str = "simple",
        llm: Optional[LLM] = None,
    ) -> None:
        assert mode in ("simple", "rake"), f"unsupported mode: {mode}"
        self.mode = mode
        self._index = KeywordTableIndex(
            nodes=list(nodes),
            llm=llm,  # 简单模式不强制需要,但 rake 模式会用到
        )
        if mode == "rake":
            self._retriever = KeywordTableRAKERetriever(self._index)
        else:
            self._retriever = KeywordTableSimpleRetriever(self._index)

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[BaseNode],
        mode: str = "simple",
        llm: Optional[LLM] = None,
    ) -> "LlamaIndexKeywordRetriever":
        return cls(nodes=nodes, mode=mode, llm=llm)

    def retrieve(
        self,
        query: str | QueryBundle,
        top_k: int = 10,
    ) -> List[NodeWithScore]:
        q = query.query_str if isinstance(query, QueryBundle) else str(query)
        # 内部 retriever 接收 query str / QueryBundle 都行
        retriever = self._retriever
        # 设置 top_k
        try:
            retriever._similarity_top_k = top_k  # noqa: SLF001
        except Exception:
            pass
        return retriever.retrieve(q)

    def __len__(self) -> int:
        return len(self._index.docstore.docs)
