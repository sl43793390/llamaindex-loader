"""
hybridRetriever
================

RAG 混合检索模块。

提供:
    - 关键词检索
        - :class:`TFIDFKeywordRetriever`            (倒排索引 + TF-IDF,纯 Python)
        - :class:`LlamaIndexKeywordRetriever`        (KeywordTableIndex,simple/rake)
    - BM25 检索
        - :class:`BM25Retriever`                     (BM25 Okapi,纯 Python)
    - 稠密向量检索
        - :class:`VectorRetriever`                   (VectorStoreIndex 包装)
    - 混合检索
        - :class:`HybridRetriever`                   (多路 RRF / 加权 / 拼接)
        - :func:`build_hybrid_retriever`             (一站式工厂)
    - Milvus 原生混合
        - :func:`build_milvus_hybrid_store`
        - :class:`MilvusHybridRetriever`
    - 工具
        - :func:`rrf_fuse` / :func:`weighted_fuse`
        - :func:`score_normalize` / :func:`tokenize`
"""
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import (
    HybridRetriever,
    build_hybrid_retriever,
)
from .keyword_retriever import (
    LlamaIndexKeywordRetriever,
    TFIDFKeywordRetriever,
)
from .milvus_hybrid import (
    MilvusHybridRetriever,
    build_milvus_hybrid_store,
)
from .utils import (
    dedup_by_id,
    rrf_fuse,
    tokenize,
    weighted_fuse,
)
from .vector_retriever import VectorRetriever, score_normalize

__all__ = [
    # 基础
    "BM25Retriever",
    "TFIDFKeywordRetriever",
    "LlamaIndexKeywordRetriever",
    "VectorRetriever",
    # 混合
    "HybridRetriever",
    "build_hybrid_retriever",
    # Milvus
    "build_milvus_hybrid_store",
    "MilvusHybridRetriever",
    # 工具
    "rrf_fuse",
    "weighted_fuse",
    "score_normalize",
    "tokenize",
    "dedup_by_id",
]
