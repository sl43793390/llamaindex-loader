"""
hybridRetriever.milvus_hybrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Milvus 原生混合检索(稠密 + 稀疏向量在同一 collection 内联合打分)。

原理:
    - Milvus 2.4+ 支持在同一 collection 存多向量字段:
        * ``embedding``(稠密,float 向量)
        * ``sparse_embedding``(稀疏,BGE-M3 / SPLADE 等产生的 {token: weight})
    - MilvusVectorStore 在 0.14 里通过 ``enable_sparse`` + ``sparse_embedding_function``
      启用这个能力。
    - 检索时用 ``hybrid_ranker="rrf"`` 或 ``"weighted"`` 一次召回。

优势:
    - 一次 search 拿混合结果,不用手动 RRF 融合
    - Milvus 内部 ANN 性能高,适合工业级

何时用:
    - 数据量 > 10w 节点
    - 已经有稀疏向量模型(BGE-M3 / SPLADE-v2 等)
    - 不想维护多套索引
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from .vector_retriever import score_normalize


# 两种内置 hybrid_ranker 字符串
MILVUS_RRF = "rrf"
MILVUS_WEIGHTED = "weighted"


def build_milvus_hybrid_store(
    collection_name: str,
    dim: int,
    enable_sparse: bool = True,
    sparse_embedding_function: Optional[Callable] = None,
    hybrid_ranker: str = MILVUS_RRF,
    hybrid_ranker_params: Optional[dict] = None,
    uri: Optional[str] = None,
    token: Optional[str] = None,
    overwrite: Optional[bool] = None,
):
    """
    构造一个开启混合检索的 MilvusVectorStore。

    Args:
        collection_name: 集合名。
        dim: 稠密向量维度。
        enable_sparse: 是否启用稀疏向量字段。
        sparse_embedding_function: BGE-M3 / SPLADE 之类的稀疏向量化函数;
            为 None 时只启用稠密字段(等同普通 Milvus 客户端)。
        hybrid_ranker: ``"rrf"`` 或 ``"weighted"``。
        hybrid_ranker_params: RRF 时通常传 ``{"k": 60}``。
        uri / token / overwrite: 同 ``MilvusVectorStore`` 构造参数。

    Returns:
        :class:`MilvusVectorStore` 实例。
    """
    from llama_index.vector_stores.milvus import MilvusVectorStore

    from config import MILVUS

    params: dict[str, Any] = dict(
        uri=uri or MILVUS.uri,
        token=token or MILVUS.token,
        collection_name=collection_name,
        dim=dim,
        enable_dense=True,
        enable_sparse=enable_sparse,
        overwrite=overwrite if overwrite is not None else MILVUS.overwrite,
        output_fields=["text"],
        hybrid_ranker=hybrid_ranker,
    )
    if hybrid_ranker_params:
        params["hybrid_ranker_params"] = hybrid_ranker_params
    if sparse_embedding_function is not None:
        params["sparse_embedding_function"] = sparse_embedding_function

    return MilvusVectorStore(**params)


class MilvusHybridRetriever:
    """
    在 :class:`MilvusVectorStore` 上做混合检索的薄包装。

    用法::

        store = build_milvus_hybrid_store(
            collection_name="rag",
            dim=1024,
            sparse_embedding_function=bge_m3_sparse,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=store, ...
        )
        retriever = MilvusHybridRetriever(index, similarity_top_k=10)
        nodes = retriever.retrieve("什么是 UFX?")
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        normalize_score: bool = True,
    ) -> None:
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.normalize_score = normalize_score
        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)

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
        try:
            result = retriever.retrieve(q)
        except TypeError:
            result = retriever.retrieve(q.query_str)
        if self.normalize_score:
            result = score_normalize(result)
        return result

    def __len__(self) -> int:
        try:
            return len(self.index.docstore.docs)
        except Exception:
            return 0
