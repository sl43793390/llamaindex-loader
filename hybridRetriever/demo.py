"""
hybridRetriever.demo
~~~~~~~~~~~~~~~~~~~~~~~
混合检索的端到端 Demo。

覆盖 4 个常见场景:
    1. **零依赖 Demo**              —— 只用 BM25 / TF-IDF,不需要 API Key
    2. **接入现有 Milvus 索引**      —— 把 Milvus 当成向量源,加 BM25 / 关键词增强
    3. **Milvus 原生混合检索**       —— 启用 ``enable_sparse``,一次 search 出混合结果
    4. **接 Chat Engine 对话**       —— 把 HybridRetriever 当 retriever 喂给 ``CustomRAGChatEngine``

跑法(任选其一)::

    # 全跑(需要 OPENAI_API_KEY)
    python -m hybridRetriever.demo

    # 只跑前两个 zero-dep / BM25-only demo
    python -m hybridRetriever.demo --mode zero

    # 跑 Milvus 接入
    python -m hybridRetriever.demo --mode milvus

    # 跑 Milvus 原生混合
    python -m hybridRetriever.demo --mode milvus_hybrid

    # 跑混合检索 + 对话
    python -m hybridRetriever.demo --mode chat
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

# ---- 让脚本既能 `python -m hybridRetriever.demo`,也能 `python hybridRetriever/demo.py` 跑 ----
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from hybridRetriever import (
        BM25Retriever,
        HybridRetriever,
        MilvusHybridRetriever,
        TFIDFKeywordRetriever,
        VectorRetriever,
        build_hybrid_retriever,
    )
else:
    from . import (
        BM25Retriever,
        HybridRetriever,
        MilvusHybridRetriever,
        TFIDFKeywordRetriever,
        VectorRetriever,
        build_hybrid_retriever,
    )

from llama_index.core.schema import BaseNode, NodeWithScore, TextNode


# ============================================================
# 公共语料 —— 不需要任何外部文件
# ============================================================
SAMPLE_DOCS = [
    "Milvus 是一个开源的向量数据库,支持亿级向量的毫秒级检索。",
    "BM25 是经典的信息检索算法,基于词频和文档长度归一化,常被用作稀疏检索的 baseline。",
    "LlamaIndex 是一个用于构建 RAG 应用的框架,支持多种数据加载、索引、检索方式。",
    "FAISS 是 Facebook 开发的稠密向量 ANN 检索库,适合单机大规模场景。",
    "Elasticsearch 是基于倒排索引的全文搜索引擎,内置 BM25,适合做 keyword search。",
    "Hybrid Search 通常结合稠密向量检索和稀疏检索,通过 RRF 或加权融合提升召回质量。",
    "Sentence-Transformers 提供多种预训练 embedding 模型,支持中英文双语。",
    "Milvus 2.x 支持多向量字段,可以同时存稠密向量和稀疏向量做混合检索。",
    "RRF(Reciprocal Rank Fusion) 是一种无参数的检索结果融合算法。",
    "Agentic Chunking 使用 LLM 提取命题,然后用 Embedding 聚类成主题块。",
]


def _to_nodes(docs: Sequence[str]) -> List[BaseNode]:
    """把字符串列表转成 BaseNode 列表(给 BM25 / Keyword 用)。"""
    return [TextNode(text=t, metadata={"src": f"sample-{i}"}) for i, t in enumerate(docs)]


def _print_hits(title: str, hits: Sequence[NodeWithScore], top: int = 5) -> None:
    """把召回结果打印出来。"""
    print(f"--- {title} ---")
    for h in hits[:top]:
        s = f"{h.score:.4f}" if h.score is not None else "n/a"
        print(f"  [{s}] {h.node.get_content()[:70]}")
    print()


# ============================================================
# Demo 1:零依赖 —— 不需要 Embedding / LLM / Milvus
# ============================================================
def demo_zero_dependency() -> None:
    """
    用纯 Python 的 BM25 + TF-IDF 构建混合检索。

    适用:
        - 想快速验证混合检索的召回效果
        - 不想花 Embedding / LLM 的钱
        - 数据量 < 1w 节点

    关键点:
        - BM25 对"专业术语 / 短词"友好(BM25 出现)
        - TF-IDF 对"多关键词组合"友好(Hybrid Search 提升)
        - RRF 融合两者,不需要调权重
    """
    print("=" * 70)
    print("Demo 1: 零依赖混合检索(BM25 + TF-IDF,RRF 融合)")
    print("=" * 70)

    nodes = _to_nodes(SAMPLE_DOCS)

    # 1) 构造两个稀疏检索器
    bm25 = BM25Retriever.from_nodes(nodes, variant="okapi", k1=1.5, b=0.75)
    kw = TFIDFKeywordRetriever.from_nodes(nodes)

    # 2) 融合(RRF,无参数)
    hybrid = HybridRetriever(retrievers=[bm25, kw], method="rrf", top_k=5)

    print(hybrid.describe(), "\n")

    # 3) 跑 3 个查询,看融合后的 top-5
    queries = [
        "Milvus 混合检索",
        "什么是 BM25?",
        "怎么用 LlamaIndex 搭 RAG?",
    ]
    for q in queries:
        _print_hits(f"Q: {q}", hybrid.retrieve(q, top_k=5))


# ============================================================
# Demo 2:接入已有 Milvus 索引
# ============================================================
def demo_with_milvus_index() -> None:
    """
    把已有 ``VectorStoreIndex`` 当向量源,叠 BM25 / 关键词做混合检索。

    适用:
        - 项目里已经有写好的 Milvus 索引(``vectorStore.create_index``)
        - 想在不改写库的前提下,给检索结果加一层稀疏增强

    关键点:
        - ``VectorRetriever`` 把 Milvus 的"内积 / 距离"归一化到 [0, 1]
        - ``weighted_fuse`` 给向量更高权重(0.7),稀疏 0.3
    """
    print("=" * 70)
    print("Demo 2: 接入现有 Milvus 索引 + BM25/关键词")
    print("=" * 70)

    # 这段需要 Embedding;拿不到就降级到 mock
    from vectorStore.milvus_store import build_milvus_store, create_index, configure_settings

    configure_settings()  # 注册全局 Embedding / LLM
    store = build_milvus_store(overwrite=True)
    nodes = _to_nodes(SAMPLE_DOCS)
    # 如果已经存在索引,则跳过创建，直接加载已存在的索引
    index = create_index(nodes, milvus_store=store)
    print(f"  Milvus 索引已就绪({len(nodes)} 节点)")

    # 工厂一行构建混合检索
    hybrid = build_hybrid_retriever(
        nodes=nodes,
        index=index,
        retrievers=["vector", "bm25", "keyword"],
        weights=[0.6, 0.3, 0.1],
        method="weighted",
        top_k=5,
    )
    print(hybrid.describe(), "\n")

    for q in ["Milvus 怎么混合检索", "BM25 公式是什么", "Sentence-Transformers 支持中文吗"]:
        _print_hits(f"Q: {q}", hybrid.retrieve(q, top_k=5))


# ============================================================
# Demo 3:Milvus 原生混合检索(enable_sparse)
# ============================================================
def demo_milvus_native_hybrid() -> None:
    """
    用 Milvus 2.4+ 的"双向量字段"做原生混合检索。

    适用:
        - 数据量 > 10w 节点,需要 Milvus 内部的 ANN 性能
        - 有 BGE-M3 / SPLADE-V2 之类的稀疏向量化函数
        - 想一次 search 出混合结果,不在 Python 层做 RRF

    关键点:
        - ``enable_sparse=True`` + ``sparse_embedding_function=...``
        - ``hybrid_ranker="rrf"`` / ``"weighted"``
        - ``MilvusHybridRetriever`` 是薄包装,接口和 ``VectorRetriever`` 一样
    """
    print("=" * 70)
    print("Demo 3: Milvus 原生混合检索(稠密 + 稀疏)")
    print("=" * 70)

    from config import EMBED
    from hybridRetriever.milvus_hybrid import build_milvus_hybrid_store
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 真实项目里 sparse_embedding_function 用 BGE-M3 之类
    # 这里给个占位(返回 {token: weight} dict 即可)
    def fake_sparse_fn(texts):
        return [{t: 1.0 for t in (tx or "").split()[:5]} for tx in texts]

    store = build_milvus_hybrid_store(
        collection_name="hybrid_demo",
        dim=EMBED.dim,
        sparse_embedding_function=fake_sparse_fn,
        hybrid_ranker="rrf",
        hybrid_ranker_params={"k": 60},
        overwrite=True,
    )
    nodes = _to_nodes(SAMPLE_DOCS)
    index = VectorStoreIndex(nodes=nodes, storage_context=None)
    # 把 storage 绑到自定义 store
    from llama_index.core import StorageContext
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
    )
    print(f"  Milvus 混合 collection 已创建(overwrite=True)")

    retriever = MilvusHybridRetriever(index, similarity_top_k=5)
    for q in ["Milvus 混合检索", "BM25"]:
        _print_hits(f"Q: {q}", retriever.retrieve(q, top_k=5))


# ============================================================
# Demo 4:把 HybridRetriever 喂给 Chat Engine
# ============================================================
def demo_hybrid_with_chat() -> None:
    """
    用 ``HybridRetriever`` 替代默认的向量 retriever,做 RAG 对话。

    适用:
        - 已经用 ``vectorStore.chat.build_chat_engine`` 跑通了 baseline
        - 想给问答效果"加 buff",提升召回质量

    关键点:
        - ``CustomRAGChatEngine`` 接受任意 ``retriever`` 对象
        - ``HybridRetriever`` 实现了 ``BaseRetriever`` 兼容的 ``retrieve(query)`` 协议
    """
    print("=" * 70)
    print("Demo 4: HybridRetriever + CustomRAGChatEngine")
    print("=" * 70)

    from vectorStore.chat import build_chat_engine
    from vectorStore.milvus_store import build_milvus_store, create_index, configure_settings

    configure_settings()
    store = build_milvus_store(overwrite=True)
    nodes = _to_nodes(SAMPLE_DOCS)
    index = create_index(nodes, milvus_store=store)

    # 用 HybridRetriever 替换默认 retriever
    hybrid = build_hybrid_retriever(
        nodes=nodes,
        index=index,
        retrievers=["vector", "bm25", "keyword"],
        weights=[0.6, 0.3, 0.1],
        method="rrf",
        top_k=5,
    )

    # 直接拿 hybrid 当 retriever 喂给 Chat Engine
    engine = build_chat_engine(index=index, retriever=hybrid, debug=True)

    questions = [
        "Milvus 是什么?",
        "BM25 用在哪里?",
    ]
    for q in questions:
        print(f">>> Q: {q}")
        resp = engine.chat(q)
        print(f"<<< A: {resp.response}\n")


# ============================================================
# 入口
# ============================================================
def main() -> None:

     demo_zero_dependency()
    # parser = argparse.ArgumentParser(description="混合检索 Demo 集合")
    # parser.add_argument(
    #     "--mode",
    #     choices=["zero", "milvus", "milvus_hybrid", "chat", "all"],
    #     default="zero",
    #     help="要跑的 demo。默认只跑零依赖 demo。",
    # )
    # args = parser.parse_args()

    # # 强制 UTF-8 输出
    # if hasattr(sys.stdout, "reconfigure"):
    #     sys.stdout.reconfigure(encoding="utf-8")

    # if args.mode == "zero" or args.mode == "all":
    #     demo_zero_dependency()

    # if args.mode == "milvus" or args.mode == "all":
    #     try:
    #         demo_with_milvus_index()
    #     except Exception as e:
    #         print(f"[demo2 skipped] {type(e).__name__}: {e}")

    # if args.mode == "milvus_hybrid" or args.mode == "all":
    #     try:
    #         demo_milvus_native_hybrid()
    #     except Exception as e:
    #         print(f"[demo3 skipped] {type(e).__name__}: {e}")

    # if args.mode == "chat" or args.mode == "all":
    #     try:
    #         demo_hybrid_with_chat()
    #     except Exception as e:
    #         print(f"[demo4 skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
